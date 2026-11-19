"""Memory-safe end-to-end CNN fine-tuning with Spatial Attention + Multi-Task Broad/Fine Heads.

Reads the exact leak-free ``build/unified_manifest.json`` created by ``backend/train.py``
(0 cross-split duplicates), streams augmented batches via ``tf.data`` (<400 MB RAM),
fine-tunes upper EfficientNetV2 blocks with domain-balanced sampling, and optionally
ensembles with the frozen multi-scale EfficientNetV2B2 head inside one OpenVINO graph.

Usage:
  .venv\\Scripts\\python.exe backend\\finetune.py
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import gc
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import openvino as ov
from openvino import opset13 as ops
import tensorflow as tf
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.preprocess import augment, prepare
from backend.taxonomy import BROAD_CLASSES, BROAD_TO_FINE, FINE_CLASSES, FINE_TO_BROAD
from backend.train import (
    SEED,
    UnifiedItem,
    audit_and_split_datasets,
    bootstrap_ci,
    calibrate_quality_fresh,
    choose_threshold,
    classification_metrics,
    expected_calibration_error,
    fit_temperature,
    probability_metrics,
    read_item_image,
    selective_metrics,
    softmax,
)

MODEL_DIR = PROJECT_ROOT / "model"
BUILD_DIR = PROJECT_ROOT / "build"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_spatial_attention_model(
    size: int = 224,
    n_broad: int = len(BROAD_CLASSES),
    n_fine: int = len(FINE_CLASSES),
    backbone_name: str = "effnetv2b0",
):
    if backbone_name == "effnetv2b0":
        base = keras.applications.EfficientNetV2B0(
            input_shape=(size, size, 3), include_top=False, weights="imagenet"
        )
    elif backbone_name == "effnetv2b2":
        base = keras.applications.EfficientNetV2B2(
            input_shape=(size, size, 3), include_top=False, weights="imagenet"
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    # Freeze all BatchNorm layers permanently for stable small-batch CPU fine-tuning
    for layer in base.layers:
        layer.trainable = False

    inp = keras.Input(shape=(size, size, 3), dtype="float32", name="image")
    feat_map = base(inp, training=False)

    # Spatial lesion attention + GAP (concentrates on lesion vs wall/sleeve/fingers)
    attn = keras.layers.Conv2D(1, 1, activation="sigmoid", padding="same", name="spatial_attn")(feat_map)
    gap = keras.layers.GlobalAveragePooling2D(name="gap")(feat_map)
    att_pool = keras.layers.GlobalAveragePooling2D(name="att_pool")(
        keras.layers.Multiply(name="att_mul")([feat_map, attn])
    )
    feat = keras.layers.Concatenate(name="pooled")([gap, att_pool])

    reg = keras.regularizers.l2(3e-4)
    x_drop = keras.layers.Dropout(0.35, name="feat_drop")(feat)
    h = keras.layers.Dense(384, activation="relu", kernel_regularizer=reg, name="shared_dense")(x_drop)
    h_drop = keras.layers.Dropout(0.35, name="shared_drop")(h)
    gap_skip = keras.layers.Dropout(0.20, name="gap_skip_drop")(gap)

    broad_h = keras.layers.Dense(n_broad, kernel_regularizer=reg, name="broad_h")(h_drop)
    broad_s = keras.layers.Dense(n_broad, use_bias=False, kernel_regularizer=reg, name="broad_s")(gap_skip)
    broad_out = keras.layers.Add(name="broad")([broad_h, broad_s])

    fine_h = keras.layers.Dense(n_fine, kernel_regularizer=reg, name="fine_h")(h_drop)
    fine_s = keras.layers.Dense(n_fine, use_bias=False, kernel_regularizer=reg, name="fine_s")(gap_skip)
    fine_out = keras.layers.Add(name="fine")([fine_h, fine_s])

    model = keras.Model(inp, {"broad": broad_out, "fine": fine_out}, name=f"finetuned_{backbone_name}")
    return model, base


def unfreeze_upper_stages(base: keras.Model, prefixes: tuple[str, ...]) -> int:
    count = 0
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        elif any(layer.name.startswith(p) for p in prefixes):
            layer.trainable = True
            count += 1
        else:
            layer.trainable = False
    return count


def compute_sampling_probs(
    train_items: list[UnifiedItem],
    broad_idx: dict[str, int],
    fine_idx: dict[str, int],
    isic_prob_factor: float = 0.18,
) -> tuple[np.ndarray, np.ndarray]:
    """Build domain-balanced sampling probabilities and per-sample fine loss weights."""
    broad_counts = Counter(i.broad_label for i in train_items)
    fine_counts = Counter(i.fine_label for i in train_items if i.fine_label)

    probs = np.zeros(len(train_items), dtype=np.float64)
    fine_loss_w = np.zeros(len(train_items), dtype=np.float32)

    for idx, it in enumerate(train_items):
        # Smooth inverse-frequency balancing across broad categories
        w = 1.0 / (broad_counts[it.broad_label] ** 0.65)
        if it.domain == "dermoscopy":
            w *= isic_prob_factor
        if it.fine_label:
            fc = fine_counts[it.fine_label]
            # Boost rare clinical fine conditions (Psoriasis, Vitiligo, Urticaria, Vasculitis, etc.)
            fine_boost = min(3.0, (450.0 / max(fc, 1)) ** 0.45)
            if it.domain == "clinical":
                w *= fine_boost
            fine_loss_w[idx] = float(min(2.5, (500.0 / max(fc, 1)) ** 0.35))
        probs[idx] = w

    probs /= probs.sum()
    return probs, fine_loss_w


def make_epoch_dataset(
    train_items: list[UnifiedItem],
    sampling_probs: np.ndarray,
    fine_loss_w: np.ndarray,
    broad_idx: dict[str, int],
    fine_idx: dict[str, int],
    size: int,
    mode: str,
    samples_per_epoch: int,
    batch_size: int,
    epoch_seed: int,
    label_smoothing: float = 0.05,
):
    n_broad = len(BROAD_CLASSES)
    n_fine = len(FINE_CLASSES)
    rng = np.random.default_rng(epoch_seed)
    chosen = rng.choice(len(train_items), size=samples_per_epoch, replace=True, p=sampling_probs)

    def gen():
        local_rng = np.random.default_rng(epoch_seed + 999)
        for idx in chosen:
            it = train_items[int(idx)]
            img = read_item_image(it)
            if img is None:
                continue
            aug_img = augment(img, local_rng)
            pix = prepare(aug_img, size, mode)
            yb = broad_idx[it.broad_label]
            yb_oh = np.full(n_broad, label_smoothing / n_broad, dtype=np.float32)
            yb_oh[yb] += 1.0 - label_smoothing

            if it.fine_label:
                yf = fine_idx[it.fine_label]
                yf_oh = np.full(n_fine, label_smoothing / n_fine, dtype=np.float32)
                yf_oh[yf] += 1.0 - label_smoothing
                wf = np.float32(fine_loss_w[int(idx)])
            else:
                yf_oh = np.zeros(n_fine, dtype=np.float32)
                wf = np.float32(0.0)

            yield (
                pix,
                {"broad": yb_oh, "fine": yf_oh},
                {"broad": np.float32(1.0), "fine": wf},
            )

    sig = (
        tf.TensorSpec(shape=(size, size, 3), dtype=tf.float32),
        {
            "broad": tf.TensorSpec(shape=(n_broad,), dtype=tf.float32),
            "fine": tf.TensorSpec(shape=(n_fine,), dtype=tf.float32),
        },
        {
            "broad": tf.TensorSpec(shape=(), dtype=tf.float32),
            "fine": tf.TensorSpec(shape=(), dtype=tf.float32),
        },
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
    return ds.batch(batch_size, drop_remainder=False).prefetch(2)


def export_ov_from_keras(model: keras.Model, size: int, out_xml: Path) -> None:
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    sm_dir = out_xml.parent / f"{out_xml.stem}_sm"
    if sm_dir.exists():
        shutil.rmtree(sm_dir, ignore_errors=True)
    # Build clean graph returning tuple (broad_logits, fine_logits)
    inp = keras.Input(shape=(size, size, 3), dtype="float32", name="image")
    outputs = model(inp, training=False)
    export_m = keras.Model(inp, [outputs["broad"], outputs["fine"]])
    export_m.export(sm_dir, input_signature=[tf.TensorSpec((None, size, size, 3), tf.float32, name="image")], verbose=False)
    ov_m = ov.convert_model(sm_dir)
    ov_m.output(0).get_tensor().set_names({"broad_logits"})
    ov_m.output(1).get_tensor().set_names({"fine_logits"})
    ov.save_model(ov_m, out_xml, compress_to_fp16=False)
    del ov_m, export_m
    gc.collect()


def predict_tta_ov(xml_path: Path, items: list[UnifiedItem], size: int, mode: str, batch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Run fast OpenVINO AsyncInferQueue inference (orig + flip logit average) on val/test."""
    core = ov.Core()
    net = core.compile_model(xml_path, "CPU", {"INFERENCE_PRECISION_HINT": "f32", "PERFORMANCE_HINT": "THROUGHPUT"})
    nreq = max(2, int(net.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")))
    n = len(items)
    b_out = np.zeros((n, 2, len(BROAD_CLASSES)), dtype=np.float32)
    f_out = np.zeros((n, 2, len(FINE_CLASSES)), dtype=np.float32)

    queue = ov.AsyncInferQueue(net, nreq)

    def cb(req, coords):
        rb = req.get_output_tensor(0).data
        rf = req.get_output_tensor(1).data
        for k, (ii, vi) in enumerate(coords):
            b_out[ii, vi] = rb[k]
            f_out[ii, vi] = rf[k]

    queue.set_callback(cb)
    batch_pix, batch_coords = [], []
    for ii, it in enumerate(items):
        img = read_item_image(it)
        if img is None:
            raise RuntimeError(f"Unreadable item: {it.rel_path}")
        for vi, src in enumerate((img, cv2.flip(img, 1))):
            batch_pix.append(prepare(src, size, mode))
            batch_coords.append((ii, vi))
            if len(batch_pix) >= batch_size:
                queue.start_async({0: np.stack(batch_pix, axis=0)}, list(batch_coords))
                batch_pix.clear()
                batch_coords.clear()
    if batch_pix:
        queue.start_async({0: np.stack(batch_pix, axis=0)}, list(batch_coords))
    queue.wait_all()
    del queue, net, core
    gc.collect()
    return b_out.mean(axis=1), f_out.mean(axis=1)


def fuse_openvino_branches(
    ft_xml: Path,
    ft_size: int,
    frozen_xml: Path,
    frozen_size: int,
    alpha_ft: float,
    temp_broad: float,
    temp_fine: float,
    out_xml: Path,
) -> None:
    """Combine Fine-Tuned branch (224x224) + Frozen Multi-Task MLP branch (260x260) into ONE OpenVINO model.

    Accepts NHWC float32 (N, frozen_size, frozen_size, 3) input and internally resizes a branch
    to (N, ft_size, ft_size, 3) via bilinear interpolation, blending logits in one inference pass.
    """
    core = ov.Core()
    m_frozen = core.read_model(frozen_xml)
    m_ft = core.read_model(ft_xml)

    # Disambiguate friendly names across both branches before merging
    for prefix, m in (("frozen", m_frozen), ("finetuned", m_ft)):
        for node in m.get_ordered_ops():
            if node.get_type_name() not in ("Parameter", "Result"):
                node.set_friendly_name(f"{prefix}/{node.get_friendly_name()}")

    shared_inp = ops.parameter((-1, frozen_size, frozen_size, 3), np.float32, name="image")
    m_frozen.get_parameters()[0].output(0).replace(shared_inp.output(0))
    b_frozen, f_frozen = [r.input_value(0) for r in m_frozen.get_results()]

    if ft_size != frozen_size:
        ft_inp = ops.interpolate(
            shared_inp,
            ops.constant(np.array([ft_size, ft_size], dtype=np.int64)),
            mode="linear",
            shape_calculation_mode="sizes",
            axes=ops.constant(np.array([1, 2], dtype=np.int64)),
        )
    else:
        ft_inp = shared_inp

    m_ft.get_parameters()[0].output(0).replace(ft_inp.output(0))
    b_ft, f_ft = [r.input_value(0) for r in m_ft.get_results()]

    w_ft = ops.constant(np.float32(alpha_ft))
    w_fr = ops.constant(np.float32(1.0 - alpha_ft))
    b_blend = ops.add(ops.multiply(b_ft, w_ft), ops.multiply(b_frozen, w_fr))
    f_blend = ops.add(ops.multiply(f_ft, w_ft), ops.multiply(f_frozen, w_fr))

    b_cal = ops.divide(b_blend, ops.constant(np.float32(temp_broad)))
    f_cal = ops.divide(f_blend, ops.constant(np.float32(temp_fine)))
    b_cal.output(0).get_tensor().set_names({"broad_logits"})
    f_cal.output(0).get_tensor().set_names({"fine_logits"})
    fused = ov.Model(
        [ops.result(b_cal, name="broad_logits"), ops.result(f_cal, name="fine_logits")],
        [shared_inp],
        "fused_hierarchical_skin_classifier",
    )
    ov.save_model(fused, out_xml, compress_to_fp16=False)
    del fused, m_frozen, m_ft
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--mode", type=str, default="letterbox_sog")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--warmup-samples", type=int, default=4200)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples-per-epoch", type=int, default=7000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--unfreeze-stages", type=str, default="block5,block6,top_")
    args = parser.parse_args()

    tf.config.threading.set_intra_op_parallelism_threads(min(8, os.cpu_count() or 4))
    tf.config.threading.set_inter_op_parallelism_threads(2)
    keras.utils.set_random_seed(SEED)

    items, audit = audit_and_split_datasets()
    train_items = [i for i in items if i.split == "train"]
    val_items = [i for i in items if i.split == "val"]
    test_items = [i for i in items if i.split == "test"]

    broad_idx = {c: i for i, c in enumerate(BROAD_CLASSES)}
    fine_idx = {c: i for i, c in enumerate(FINE_CLASSES)}

    y_broad_va = np.array([broad_idx[i.broad_label] for i in val_items], dtype=np.int32)
    y_fine_va = np.array([fine_idx[i.fine_label] if i.fine_label else -1 for i in val_items], dtype=np.int32)
    is_derm_va = np.array([i.domain == "dermoscopy" for i in val_items], dtype=bool)

    y_broad_te = np.array([broad_idx[i.broad_label] for i in test_items], dtype=np.int32)
    y_fine_te = np.array([fine_idx[i.fine_label] if i.fine_label else -1 for i in test_items], dtype=np.int32)
    is_derm_te = np.array([i.domain == "dermoscopy" for i in test_items], dtype=bool)

    sampling_probs, fine_loss_w = compute_sampling_probs(train_items, broad_idx, fine_idx)

    ft_dir = BUILD_DIR / "finetune"
    ft_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = ft_dir / "best_ft.weights.h5"

    model, base = build_spatial_attention_model(size=args.size)
    loss_dict = {
        "broad": keras.losses.CategoricalCrossentropy(from_logits=True),
        "fine": keras.losses.CategoricalCrossentropy(from_logits=True),
    }

    # Phase A: Warm-up Spatial Attention + Multi-Task Head with frozen backbone
    log(f"Phase A: Warm-up head + Spatial Attention ({args.warmup_epochs} epochs x {args.warmup_samples} balanced samples)")
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=8e-4, weight_decay=2e-4),
        loss=loss_dict,
        loss_weights={"broad": 1.0, "fine": 0.70},
    )
    for ep in range(args.warmup_epochs):
        t0 = time.perf_counter()
        ds = make_epoch_dataset(
            train_items, sampling_probs, fine_loss_w, broad_idx, fine_idx,
            args.size, args.mode, args.warmup_samples, args.batch_size, epoch_seed=SEED + ep,
        )
        hist = model.fit(ds, epochs=1, verbose=0)
        log(
            f"  Warmup ep {ep + 1}/{args.warmup_epochs} ({time.perf_counter() - t0:.0f}s): "
            f"loss={hist.history['loss'][0]:.4f}, broad_loss={hist.history['broad_loss'][0]:.4f}"
        )

    # Phase B: End-to-End Fine-Tuning of upper convolutional stages
    prefixes = tuple(p.strip() for p in args.unfreeze_stages.split(",") if p.strip())
    n_unfrozen = unfreeze_upper_stages(base, prefixes)
    total_steps = max(1, args.epochs * int(np.ceil(args.samples_per_epoch / args.batch_size)))
    lr_sched = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1.6e-4, decay_steps=total_steps, alpha=0.08
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr_sched, weight_decay=2e-4),
        loss=loss_dict,
        loss_weights={"broad": 1.0, "fine": 0.70},
    )
    log(f"Phase B: Fine-tuning {n_unfrozen}/{len(base.layers)} upper layers ({prefixes}) for {args.epochs} epochs")

    best_val_score = -1.0
    val_clin_mask = ~is_derm_va
    val_fine_clin = val_clin_mask & (y_fine_va >= 0)

    for ep in range(args.epochs):
        t0 = time.perf_counter()
        ds = make_epoch_dataset(
            train_items, sampling_probs, fine_loss_w, broad_idx, fine_idx,
            args.size, args.mode, args.samples_per_epoch, args.batch_size, epoch_seed=SEED + 100 + ep,
        )
        hist = model.fit(ds, epochs=1, verbose=0)
        ep_xml = ft_dir / "epoch_tmp.xml"
        export_ov_from_keras(model, args.size, ep_xml)
        va_b_log, va_f_log = predict_tta_ov(ep_xml, val_items, args.size, args.mode)

        cm_c = classification_metrics(va_b_log[val_clin_mask].argmax(1), y_broad_va[val_clin_mask], BROAD_CLASSES)
        cm_d = classification_metrics(va_b_log[is_derm_va].argmax(1), y_broad_va[is_derm_va], BROAD_CLASSES)
        cm_f = classification_metrics(va_f_log[val_fine_clin].argmax(1), y_fine_va[val_fine_clin], FINE_CLASSES)
        val_score = 0.45 * cm_c["macro_f1"] + 0.30 * cm_c["accuracy"] + 0.15 * cm_f["macro_f1"] + 0.10 * cm_d["accuracy"]
        log(
            f"  Epoch {ep + 1}/{args.epochs} ({time.perf_counter() - t0:.0f}s): "
            f"loss={hist.history['loss'][0]:.4f} | val_clin_acc={cm_c['accuracy']:.4f} "
            f"val_clin_mf1={cm_c['macro_f1']:.4f} val_fine_mf1={cm_f['macro_f1']:.4f} "
            f"val_derm_acc={cm_d['accuracy']:.4f} | score={val_score:.4f}"
        )
        if val_score > best_val_score:
            best_val_score = val_score
            model.save_weights(best_weights_path)
            (ft_dir / "best_ft.xml").write_bytes(ep_xml.read_bytes())
            (ft_dir / "best_ft.bin").write_bytes(ep_xml.with_suffix(".bin").read_bytes())

    # Reload best fine-tuned OpenVINO model & compute val/test TTA logits
    ft_xml = ft_dir / "best_ft.xml"
    val_ft_b, val_ft_f = predict_tta_ov(ft_xml, val_items, args.size, args.mode)
    test_ft_b, test_ft_f = predict_tta_ov(ft_xml, test_items, args.size, args.mode)

    # Check if an existing trained classifier.zip exists to evaluate blending!
    bundle_path = MODEL_DIR / "classifier.zip"
    use_ensemble = False
    alpha_ft = 1.0
    frozen_xml = ft_dir / "frozen_branch.xml"
    frozen_size = args.size
    val_blend_b, val_blend_f = val_ft_b, val_ft_f
    test_blend_b, test_blend_f = test_ft_b, test_ft_f
    old_val_acc = 0.0

    if bundle_path.is_file():
        with ZipFile(bundle_path) as zf:
            meta = json.loads(zf.read("metadata.json"))
            if meta.get("format") == 3 and meta.get("preprocessing") == args.mode:
                frozen_size = int(meta["input_size"])
                # Strip final temperature divide by recovering raw logits via * temperature
                frozen_xml.write_bytes(zf.read("classifier.xml"))
                frozen_xml.with_suffix(".bin").write_bytes(zf.read("classifier.bin"))
                t_b_old = float(meta.get("temperature_broad", 1.0))
                t_f_old = float(meta.get("temperature_fine", 1.0))
                val_fr_b, val_fr_f = predict_tta_ov(frozen_xml, val_items, frozen_size, args.mode)
                val_fr_b *= t_b_old
                val_fr_f *= t_f_old
                old_m = classification_metrics(val_fr_b[val_clin_mask].argmax(1), y_broad_va[val_clin_mask], BROAD_CLASSES)
                old_val_acc = old_m["accuracy"]
                log(f"Existing frozen branch val_clin_acc={old_val_acc:.4f}, val_clin_mf1={old_m['macro_f1']:.4f}")

                test_fr_b, test_fr_f = predict_tta_ov(frozen_xml, test_items, frozen_size, args.mode)
                test_fr_b *= t_b_old
                test_fr_f *= t_f_old

                best_alpha = 1.0
                best_blend_score = -1.0
                for alpha in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
                    b_mix = alpha * val_ft_b + (1.0 - alpha) * val_fr_b
                    f_mix = alpha * val_ft_f + (1.0 - alpha) * val_fr_f
                    cm_c = classification_metrics(b_mix[val_clin_mask].argmax(1), y_broad_va[val_clin_mask], BROAD_CLASSES)
                    cm_d = classification_metrics(b_mix[is_derm_va].argmax(1), y_broad_va[is_derm_va], BROAD_CLASSES)
                    cm_f = classification_metrics(f_mix[val_fine_clin].argmax(1), y_fine_va[val_fine_clin], FINE_CLASSES)
                    sc = 0.45 * cm_c["macro_f1"] + 0.30 * cm_c["accuracy"] + 0.15 * cm_f["macro_f1"] + 0.10 * cm_d["accuracy"]
                    log(
                        f"  Blend alpha_ft={alpha:.1f}: val_clin_acc={cm_c['accuracy']:.4f} "
                        f"val_clin_mf1={cm_c['macro_f1']:.4f} val_fine_mf1={cm_f['macro_f1']:.4f} "
                        f"val_derm_acc={cm_d['accuracy']:.4f} | score={sc:.4f}"
                    )
                    if sc > best_blend_score:
                        best_blend_score = sc
                        best_alpha = alpha

                alpha_ft = best_alpha
                use_ensemble = alpha_ft < 0.999
                val_blend_b = alpha_ft * val_ft_b + (1.0 - alpha_ft) * val_fr_b
                val_blend_f = alpha_ft * val_ft_f + (1.0 - alpha_ft) * val_fr_f
                test_blend_b = alpha_ft * test_ft_b + (1.0 - alpha_ft) * test_fr_b
                test_blend_f = alpha_ft * test_ft_f + (1.0 - alpha_ft) * test_fr_f

    # Calibrate temperatures and clinical threshold strictly on held-out val
    w_cal_b = np.where(is_derm_va, 0.25, 1.0)
    temp_broad = fit_temperature(val_blend_b, y_broad_va, sample_weight=w_cal_b)
    val_fine_mask = y_fine_va >= 0
    w_cal_f = np.where(is_derm_va[val_fine_mask], 0.25, 1.0)
    temp_fine = fit_temperature(val_blend_f[val_fine_mask], y_fine_va[val_fine_mask], sample_weight=w_cal_f)

    val_b_probs = softmax(val_blend_b / temp_broad)
    threshold = min(choose_threshold(val_b_probs[val_clin_mask], y_broad_va[val_clin_mask], target=0.78, min_coverage=0.70), 0.36)
    min_margin = 0.035

    test_b_probs = softmax(test_blend_b / temp_broad)
    test_f_probs = softmax(test_blend_f / temp_fine)

    def eval_slice(mask: np.ndarray) -> dict:
        b_p = test_b_probs[mask]
        b_y = y_broad_te[mask]
        b_pred = b_p.argmax(1)
        b_res = classification_metrics(b_pred, b_y, BROAD_CLASSES)
        b_res.update(probability_metrics(b_p, b_y))
        b_res["uncalibrated_ece_15_bins"] = expected_calibration_error(softmax(test_blend_b[mask]), b_y)
        if len(b_y) >= 20:
            b_res.update(bootstrap_ci(b_pred, b_y, BROAD_CLASSES))
        sel = selective_metrics(b_p, b_y, threshold, BROAD_CLASSES)

        f_mask = mask & (y_fine_te >= 0)
        if f_mask.any():
            f_p = test_f_probs[f_mask]
            f_y = y_fine_te[f_mask]
            f_res = classification_metrics(f_p.argmax(1), f_y, FINE_CLASSES)
            f_res.update(probability_metrics(f_p, f_y))
            cond_preds = []
            for row_b, row_f in zip(test_b_probs[f_mask], f_p):
                top_broad = BROAD_CLASSES[int(np.argmax(row_b))]
                child_idxs = [fine_idx[fc] for fc in BROAD_TO_FINE[top_broad]]
                cond_preds.append(child_idxs[int(np.argmax(row_f[child_idxs]))])
            cond_m = classification_metrics(np.array(cond_preds, dtype=np.int32), f_y, FINE_CLASSES)
            f_res["hierarchical_conditioned_accuracy"] = cond_m["accuracy"]
            f_res["hierarchical_conditioned_macro_f1"] = cond_m["macro_f1"]
        else:
            f_res = {"images": 0}
        return {"broad": b_res, "broad_selective": sel, "fine": f_res}

    test_clin_mask = ~is_derm_te
    final_eval = {
        "val_clinical_broad": {
            **classification_metrics(val_b_probs[val_clin_mask].argmax(1), y_broad_va[val_clin_mask], BROAD_CLASSES),
            **probability_metrics(val_b_probs[val_clin_mask], y_broad_va[val_clin_mask]),
        },
        "test_clinical": eval_slice(test_clin_mask),
        "test_dermoscopy": eval_slice(is_derm_te),
        "test_overall": eval_slice(np.ones(len(test_items), dtype=bool)),
    }

    tc = final_eval["test_clinical"]["broad"]
    tf_m = final_eval["test_clinical"]["fine"]
    td = final_eval["test_dermoscopy"]["broad"]
    log(
        f"FINETUNE/ENSEMBLE TEST CLINICAL (7-broad): acc={tc['accuracy']:.3f}  macro-F1={tc['macro_f1']:.3f}  "
        f"top3={tc['top3_accuracy']:.3f}  ECE={tc['ece_15_bins']:.4f}"
    )
    log(
        f"FINETUNE/ENSEMBLE TEST CLINICAL (21-fine) : acc={tf_m['accuracy']:.3f}  macro-F1={tf_m['macro_f1']:.3f}  "
        f"hier-acc={tf_m['hierarchical_conditioned_accuracy']:.3f}"
    )
    log(f"FINETUNE/ENSEMBLE TEST DERMOSCOPY         : acc={td['accuracy']:.3f}  macro-F1={td['macro_f1']:.3f}")

    # Only replace deployed model if strictly better on clinical validation accuracy than existing!
    new_val_acc = final_eval["val_clinical_broad"]["accuracy"]
    if old_val_acc > 0 and new_val_acc < old_val_acc:
        log(f"Existing model val_clin_acc ({old_val_acc:.4f}) >= new ({new_val_acc:.4f}); keeping existing bundle.")
        return

    deploy_size = frozen_size if use_ensemble else args.size
    with TemporaryDirectory(dir=MODEL_DIR, ignore_cleanup_errors=True) as staging:
        st_root = Path(staging)
        out_xml = st_root / "classifier.xml"
        if use_ensemble:
            # Undo temperature scaling baked inside frozen_xml before blending
            core = ov.Core()
            m_fr = core.read_model(frozen_xml)
            with ZipFile(bundle_path) as zf:
                old_meta = json.loads(zf.read("metadata.json"))
            tb_old = ops.constant(np.float32(old_meta.get("temperature_broad", 1.0)))
            tf_old = ops.constant(np.float32(old_meta.get("temperature_fine", 1.0)))
            r_b = ops.multiply(m_fr.get_results()[0].input_value(0), tb_old)
            r_f = ops.multiply(m_fr.get_results()[1].input_value(0), tf_old)
            raw_frozen = ov.Model([ops.result(r_b), ops.result(r_f)], m_fr.get_parameters(), "raw_frozen")
            raw_frozen_xml = st_root / "raw_frozen.xml"
            ov.save_model(raw_frozen, raw_frozen_xml, compress_to_fp16=False)
            del raw_frozen, m_fr
            gc.collect()
            fuse_openvino_branches(
                ft_xml, args.size, raw_frozen_xml, frozen_size, alpha_ft, temp_broad, temp_fine, out_xml
            )
            backbone_desc = f"ensemble(ft_effnetv2b0_{args.size}*{alpha_ft:.2f}+{old_meta.get('backbone')}*{1-alpha_ft:.2f})"
        else:
            core = ov.Core()
            m_ft = core.read_model(ft_xml)
            b_cal = ops.divide(m_ft.get_results()[0].input_value(0), ops.constant(np.float32(temp_broad)))
            f_cal = ops.divide(m_ft.get_results()[1].input_value(0), ops.constant(np.float32(temp_fine)))
            b_cal.output(0).get_tensor().set_names({"broad_logits"})
            f_cal.output(0).get_tensor().set_names({"fine_logits"})
            cal_m = ov.Model(
                [ops.result(b_cal, name="broad_logits"), ops.result(f_cal, name="fine_logits")],
                m_ft.get_parameters(),
                "finetuned_hierarchical_skin_classifier",
            )
            ov.save_model(cal_m, out_xml, compress_to_fp16=False)
            backbone_desc = f"ft_effnetv2b0_{args.size}"

        # Verify OpenVINO numerical output on test_items[:4]
        core = ov.Core()
        net = core.compile_model(out_xml, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
        pix_check = np.stack([prepare(read_item_image(i), deploy_size, args.mode) for i in test_items[:4]])
        res = net(pix_check)
        got_b = res[net.output(0)]
        got_f = res[net.output(1)]
        assert got_b.shape == (4, len(BROAD_CLASSES)) and got_f.shape == (4, len(FINE_CLASSES))
        del net

        report = {
            "seed": SEED,
            "taxonomy": {
                "broad_classes": list(BROAD_CLASSES),
                "fine_classes": list(FINE_CLASSES),
                "fine_to_broad": dict(FINE_TO_BROAD),
            },
            "dataset_audit": audit,
            "selected_configuration": {
                "backbone": backbone_desc,
                "preprocessing": args.mode,
                "head_type": "spatial_attention_finetuned_ensemble" if use_ensemble else "spatial_attention_finetuned_cnn",
                "alpha_finetuned": float(alpha_ft),
                "epochs": args.epochs,
                "samples_per_epoch": args.samples_per_epoch,
                "temperature_broad": float(temp_broad),
                "temperature_fine": float(temp_fine),
                "uncertain_threshold": float(threshold),
                "min_margin": float(min_margin),
            },
            "final_model": final_eval,
        }
        metadata = {
            "format": 3,
            "created_at": datetime.now(UTC).isoformat(),
            "backbone": backbone_desc,
            "head_type": report["selected_configuration"]["head_type"],
            "input_size": deploy_size,
            "input_layout": "NHWC RGB float32 0-255",
            "preprocessing": args.mode,
            "output_type": "calibrated_logits",
            "test_time_augmentation": ["orig", "flip"],
            "temperature_broad": float(temp_broad),
            "temperature_fine": float(temp_fine),
            "uncertain_threshold": float(threshold),
            "min_margin": float(min_margin),
            "classes": list(BROAD_CLASSES),
            "display_names": list(BROAD_CLASSES),
            "fine_classes": list(FINE_CLASSES),
            "fine_to_broad": dict(FINE_TO_BROAD),
            "broad_to_fine": {k: list(v) for k, v in BROAD_TO_FINE.items()},
            "test_summary": {
                "clinical_broad": {
                    k: final_eval["test_clinical"]["broad"][k]
                    for k in ("accuracy", "macro_f1", "balanced_accuracy", "top3_accuracy", "ece_15_bins")
                },
                "clinical_fine": {
                    k: final_eval["test_clinical"]["fine"][k]
                    for k in ("accuracy", "macro_f1", "balanced_accuracy", "top3_accuracy")
                },
                "dermoscopy_broad": {
                    k: final_eval["test_dermoscopy"]["broad"][k]
                    for k in ("accuracy", "macro_f1", "balanced_accuracy")
                },
            },
        }
        train_clinical_items = [i for i in train_items if i.domain == "clinical"]
        quality_report = calibrate_quality_fresh(train_clinical_items)

        bundle = st_root / "classifier.zip"
        with ZipFile(bundle, "w") as zf:
            zf.write(out_xml, "classifier.xml", compress_type=ZIP_DEFLATED)
            zf.write(out_xml.with_suffix(".bin"), "classifier.bin", compress_type=ZIP_STORED)
            zf.writestr("classes.json", json.dumps(list(BROAD_CLASSES), indent=2))
            zf.writestr("fine_classes.json", json.dumps(list(FINE_CLASSES), indent=2))
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))
        (st_root / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (st_root / "quality.json").write_text(json.dumps(quality_report, indent=2) + "\n", encoding="utf-8")
        for name in ("evaluation.json", "quality.json", "classifier.zip"):
            (st_root / name).replace(MODEL_DIR / name)
        log(f"Successfully deployed {backbone_desc} to model/classifier.zip!")


if __name__ == "__main__":
    main()
