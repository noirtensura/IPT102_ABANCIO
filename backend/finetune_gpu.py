"""Fine-tune a DINOv2 Vision Transformer end to end on a GPU and export it as the scanner's classifier.zip.

Built for Google Colab (see ``colab/finetune_colab.ipynb``). ``backend/train.py`` keeps the backbones
frozen because a laptop CPU cannot do more; here every layer is trained, which needs a GPU.

The data contract is the one ``backend/train.py`` uses:
  - ``build/unified_manifest.json`` is read as is, so train/val/test hold exactly the same photos;
  - pixels go through ``backend.preprocess.prepare``, the function the scanner calls at inference;
  - the epoch is picked on the val split only, calibration is fitted on val, and test is scored once;
  - the result is a format-3 ``classifier.zip``: NHWC RGB float32 0-255 in, calibrated broad and fine
    logits out.

Usage:
  python backend/finetune_gpu.py --out build/colab_result            (GPU, about an hour on a Colab T4)
  python backend/finetune_gpu.py --smoke --out build/colab_smoke     (CPU, a few minutes: checks the plumbing)
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import sys
import time
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.preprocess import PREPROCESSING_MODES, augment, prepare
from backend.taxonomy import BROAD_CLASSES, BROAD_TO_FINE, FINE_CLASSES, FINE_TO_BROAD
from backend.train import (
    SEED,
    UnifiedItem,
    build_split_arrays,
    calibrate_on_val,
    compute_domain_balanced_weights,
    evaluate_model,
    headline,
    manifest_fingerprint,
    read_item_image,
    selection_score,
)

MODELS = {
    # key: (timm checkpoint, short name used in the bundle metadata). Both are Apache-2.0.
    "small": ("vit_small_patch14_dinov2.lvd142m", "dinov2_vits14"),
    "base": ("vit_base_patch14_dinov2.lvd142m", "dinov2_vitb14"),
}
MIN_MARGIN = 0.03  # same lead the laptop pipeline requires between the top two categories


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------- data
def random_crop(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random window covering 55-100% of the photo; a third of the photos are left whole."""
    if rng.random() < 0.35:
        return image
    height, width = image.shape[:2]
    area = rng.uniform(0.55, 1.0)
    ratio = math.exp(rng.uniform(math.log(3 / 4), math.log(4 / 3)))
    crop_h = max(8, min(height, round(height * math.sqrt(area / ratio))))
    crop_w = max(8, min(width, round(width * math.sqrt(area * ratio))))
    top = int(rng.integers(0, height - crop_h + 1))
    left = int(rng.integers(0, width - crop_w + 1))
    return image[top : top + crop_h, left : left + crop_w]


class PhotoDataset(torch.utils.data.Dataset):
    """Manifest photos as uint8 NHWC tensors. Training: one augmented view. Otherwise: original + flip."""

    def __init__(self, items: list[UnifiedItem], size: int, mode: str, train: bool):
        self.items, self.size, self.mode, self.train = items, size, mode, train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        image = read_item_image(item)
        if image is None:
            raise RuntimeError(f"Cannot read {item.rel_path}")
        if self.train:
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            image = augment(random_crop(image, rng), rng)
            pixels = prepare(image, self.size, self.mode)
        else:
            pixels = np.stack([prepare(image, self.size, self.mode), prepare(cv2.flip(image, 1), self.size, self.mode)])
        # prepare() returns whole numbers in [0, 255], so uint8 is lossless and four times smaller to move.
        return torch.from_numpy(pixels.astype(np.uint8)), index


def _worker_init(_worker_id: int) -> None:
    cv2.setNumThreads(0)


def make_loader(dataset, batch_size: int, workers: int, sampler=None, drop_last: bool = False):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        worker_init_fn=_worker_init if workers else None,
        persistent_workers=False,
    )


# --------------------------------------------------------------------------- model
class SkinViT(torch.nn.Module):
    """DINOv2 backbone, class token + mean patch token, one linear head per label level."""

    def __init__(self, timm_name: str, size: int, drop_path: float, head_dropout: float, pretrained: bool = True):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            timm_name, pretrained=pretrained, num_classes=0, img_size=size, drop_path_rate=drop_path
        )
        config = self.backbone.pretrained_cfg
        self.pixel_mean = [255.0 * float(v) for v in config["mean"]]
        self.pixel_std = [255.0 * float(v) for v in config["std"]]
        self.register_buffer("mean", torch.tensor(self.pixel_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(self.pixel_std).view(1, 3, 1, 1), persistent=False)
        self.prefix = int(getattr(self.backbone, "num_prefix_tokens", 1))
        width = 2 * int(self.backbone.num_features)
        self.dropout = torch.nn.Dropout(head_dropout)
        self.broad = torch.nn.Linear(width, len(BROAD_CLASSES))
        self.fine = torch.nn.Linear(width, len(FINE_CLASSES))

    def forward_normalised(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """NCHW float input that is already mean/std normalised."""
        tokens = self.backbone.forward_features(x)
        features = torch.cat([tokens[:, 0], tokens[:, self.prefix :].mean(dim=1)], dim=1)
        features = self.dropout(features)
        return self.broad(features), self.fine(features)

    def forward(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """NHWC RGB pixels in [0, 255] (uint8 or float), as produced by ``prepare``."""
        x = pixels.permute(0, 3, 1, 2).float()
        return self.forward_normalised((x - self.mean) / self.std)


def parameter_groups(model: SkinViT, lr: float, layer_decay: float, weight_decay: float, head_lr: float) -> list[dict]:
    """Layer-wise learning-rate decay: early layers move least, so the pretrained features survive."""
    n_blocks = len(model.backbone.blocks)
    groups: dict[tuple, dict] = {}
    for name, param in model.named_parameters():
        if name.startswith(("broad.", "fine.")):
            layer, rate = n_blocks + 1, head_lr
        else:
            if name.startswith("backbone.blocks."):
                layer = int(name.split(".")[2]) + 1
            elif name.startswith("backbone.norm"):
                layer = n_blocks + 1
            else:
                layer = 0  # patch embedding, position embedding, class/register tokens
            rate = lr * layer_decay ** (n_blocks + 1 - layer)
        is_head = name.startswith(("broad.", "fine."))
        no_decay = param.ndim <= 1 or any(key in name for key in ("pos_embed", "cls_token", "reg_token", ".ls1.", ".ls2."))
        decay = 0.0 if no_decay else weight_decay
        group = groups.setdefault(
            (layer, rate, decay, is_head),
            {"params": [], "lr": rate, "base_lr": rate, "weight_decay": decay, "is_head": is_head},
        )
        group["params"].append(param)
    return list(groups.values())


def multitask_loss(logits_b, logits_f, y_b, y_f, w_b, w_f, smoothing: float, fine_weight: float) -> torch.Tensor:
    ce_b = F.cross_entropy(logits_b.float(), y_b, reduction="none", label_smoothing=smoothing)
    loss = (w_b * ce_b).sum() / w_b.sum().clamp_min(1e-6)
    labelled = w_f > 0
    if labelled.any():
        ce_f = F.cross_entropy(logits_f.float()[labelled], y_f[labelled], reduction="none", label_smoothing=smoothing)
        loss = loss + fine_weight * (w_f[labelled] * ce_f).sum() / w_f[labelled].sum()
    return loss


@torch.no_grad()
def predict(model: SkinViT, items: list[UnifiedItem], size: int, mode: str, device, batch_size: int, workers: int, amp: bool):
    """Broad and fine logits averaged over the original and the mirrored photo (what the scanner does)."""
    model.eval()
    out_b = np.zeros((len(items), len(BROAD_CLASSES)), dtype=np.float32)
    out_f = np.zeros((len(items), len(FINE_CLASSES)), dtype=np.float32)
    loader = make_loader(PhotoDataset(items, size, mode, train=False), batch_size, workers)
    for views, index in loader:
        n = views.shape[0]
        flat = views.view(n * 2, *views.shape[2:]).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits_b, logits_f = model(flat)
        out_b[index.numpy()] = logits_b.float().view(n, 2, -1).mean(dim=1).cpu().numpy()
        out_f[index.numpy()] = logits_f.float().view(n, 2, -1).mean(dim=1).cpu().numpy()
    return out_b, out_f


# --------------------------------------------------------------------------- export
class _Calibrated(torch.nn.Module):
    def __init__(self, model: SkinViT, temp_broad: float, temp_fine: float):
        super().__init__()
        self.model, self.temp_broad, self.temp_fine = model, float(temp_broad), float(temp_fine)

    def forward(self, x: torch.Tensor):
        logits_b, logits_f = self.model.forward_normalised(x)
        return logits_b / self.temp_broad, logits_f / self.temp_fine


def export_openvino(model: SkinViT, size: int, temp_broad: float, temp_fine: float, xml_path: Path, fp16: bool) -> None:
    """One graph with the scanner's contract: NHWC RGB float32 0-255 in, calibrated logits out."""
    import openvino as ov

    wrapper = _Calibrated(model, temp_broad, temp_fine).cpu().float().eval()
    with torch.no_grad():
        converted = ov.convert_model(wrapper, example_input=torch.zeros(1, 3, size, size))
    converted.reshape(ov.PartialShape([-1, 3, size, size]))
    ppp = ov.preprocess.PrePostProcessor(converted)
    ppp.input().tensor().set_element_type(ov.Type.f32).set_layout(ov.Layout("NHWC"))
    ppp.input().model().set_layout(ov.Layout("NCHW"))
    ppp.input().preprocess().mean(model.pixel_mean).scale(model.pixel_std)
    converted = ppp.build()
    converted.input(0).get_tensor().set_names({"image"})
    converted.output(0).get_tensor().set_names({"broad_logits"})
    converted.output(1).get_tensor().set_names({"fine_logits"})
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(converted, xml_path, compress_to_fp16=fp16)


def export_gap(model: SkinViT, xml_path: Path, items: list[UnifiedItem], size: int, mode: str, temps: tuple[float, float]) -> float:
    """Largest difference between the saved graph and the PyTorch model on real test photos."""
    import openvino as ov

    pixels = np.stack([prepare(read_item_image(item), size, mode) for item in items])
    net = ov.Core().compile_model(xml_path, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    result = net(pixels)
    got_b, got_f = result[net.output(0)], result[net.output(1)]
    model.cpu().float().eval()
    with torch.no_grad():
        want_b, want_f = model(torch.from_numpy(pixels))
    gap = max(
        float(np.abs(got_b - want_b.numpy() / temps[0]).max()),
        float(np.abs(got_f - want_f.numpy() / temps[1]).max()),
    )
    agree = float((got_b.argmax(1) == want_b.numpy().argmax(1)).mean())
    log(f"Export check on {len(items)} test photos: largest logit difference {gap:.4f}, same top category {agree:.0%}")
    return gap


# --------------------------------------------------------------------------- main
def smoke_subset(items: list[UnifiedItem]) -> list[UnifiedItem]:
    """A few photos per split, category, and domain: enough to run every step of the pipeline quickly."""
    caps = {"train": 14, "val": 6, "test": 6}
    seen: dict[tuple, int] = {}
    kept = []
    for item in items:
        key = (item.split, item.broad_label, item.domain)
        if seen.get(key, 0) < caps[item.split]:
            seen[key] = seen.get(key, 0) + 1
            kept.append(item)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="small", help="small is fast on the scanner's CPU; base is 4x heavier.")
    parser.add_argument("--size", type=int, default=224, help="Input side in pixels; must be a multiple of 14.")
    parser.add_argument("--mode", choices=PREPROCESSING_MODES, default="letterbox_sog")
    parser.add_argument("--epochs", type=int, default=12, help="Epochs with every layer training.")
    parser.add_argument("--head-epochs", type=int, default=2, help="Epochs that train only the new heads first.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate of the last backbone block for a batch of 64; it is scaled in proportion to --batch-size.",
    )
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=float, default=1.5, help="Backbone warm-up after it is unfrozen.")
    parser.add_argument("--layer-decay", type=float, default=0.75)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--head-dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--fine-loss-weight", type=float, default=0.65)
    parser.add_argument("--isic-weight", type=float, default=0.3, help="Share of the dermoscopy images used in each epoch.")
    parser.add_argument("--class-weight", choices=("none", "sqrt", "balanced"), default="sqrt")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2))
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "build" / "colab_result")
    parser.add_argument("--resume", action="store_true", help="Continue from the checkpoint in --out after a disconnect.")
    parser.add_argument("--no-fp16", action="store_true", help="Store full-precision weights (doubles classifier.zip).")
    parser.add_argument("--smoke", action="store_true", help="Tiny run on a few photos to test every step; not a real model.")
    parser.add_argument("--max-train-photos", type=int, default=0, help="Trial run: use only this many random training photos.")
    parser.add_argument("--max-eval-photos", type=int, default=0, help="Trial run: use only this many random val and test photos each.")
    parser.add_argument("--allow-cpu", action="store_true", help="Run without a GPU (very slow; only for trial runs).")
    args = parser.parse_args()
    if args.size % 14:
        parser.error("--size must be a multiple of 14 (the DINOv2 patch size)")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if device.type != "cuda" and not (args.smoke or args.allow_cpu):
        parser.error("No GPU found. In Colab choose Runtime > Change runtime type > T4 GPU, or pass --smoke for a quick CPU check.")
    log(f"Device: {torch.cuda.get_device_name(0) if amp else 'CPU'}")

    manifest = json.loads((PROJECT_ROOT / "build" / "unified_manifest.json").read_text(encoding="utf-8"))
    all_items = [UnifiedItem(**row) for row in manifest["items"]]
    fingerprint = manifest_fingerprint(all_items)
    items = all_items
    if args.smoke:
        items = smoke_subset(all_items)
        args.epochs, args.head_epochs, args.batch_size, args.workers = 1, 1, 8, 0
    trial = bool(args.smoke or args.max_train_photos or args.max_eval_photos)
    if args.max_train_photos or args.max_eval_photos:
        picker = np.random.default_rng(SEED)
        kept = []
        for split, cap in (("train", args.max_train_photos), ("val", args.max_eval_photos), ("test", args.max_eval_photos)):
            subset = [i for i in items if i.split == split]
            if cap and len(subset) > cap:
                subset = [subset[int(k)] for k in sorted(picker.choice(len(subset), size=cap, replace=False))]
            kept.extend(subset)
        items = kept
    data = build_split_arrays(items)
    train_items = data.train_items
    log(f"Photos: train={len(train_items)} val={len(data.val_items)} test={len(data.test_items)} (manifest {fingerprint})")

    # Loss weights. Each epoch uses every clinical photo and a random --isic-weight share of the
    # dermoscopy images, so the class weights are computed for that mix and carry no domain factor.
    is_derm = data.is_derm_tr
    domain_factor = np.where(is_derm, args.isic_weight, 1.0)
    w_broad = compute_domain_balanced_weights(data.y_broad_tr, is_derm, len(BROAD_CLASSES), args.class_weight, args.isic_weight)
    has_fine = data.y_fine_tr >= 0
    w_fine = compute_domain_balanced_weights(
        np.where(has_fine, data.y_fine_tr, 0), is_derm, len(FINE_CLASSES), "sqrt", args.isic_weight, valid_mask=has_fine
    )
    w_broad_t = torch.tensor(w_broad / domain_factor, dtype=torch.float32, device=device)
    w_fine_t = torch.tensor(w_fine / domain_factor, dtype=torch.float32, device=device)
    y_broad_t = torch.tensor(data.y_broad_tr, dtype=torch.long, device=device)
    y_fine_t = torch.tensor(np.where(has_fine, data.y_fine_tr, 0), dtype=torch.long, device=device)
    clinical_idx, derm_idx = np.flatnonzero(~is_derm), np.flatnonzero(is_derm)
    derm_per_epoch = min(len(derm_idx), max(1, round(args.isic_weight * len(derm_idx)))) if len(derm_idx) else 0

    timm_name, short_name = MODELS[args.model]
    model = SkinViT(timm_name, args.size, args.drop_path, args.head_dropout).to(device)
    log(f"Model: {timm_name} at {args.size}px, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    args.out.mkdir(parents=True, exist_ok=True)
    last_path, best_path = args.out / "last.pt", args.out / "best.pt"
    backbone_lr = args.lr * args.batch_size / 64  # smaller batches give noisier steps, so they take smaller ones
    optimizer = torch.optim.AdamW(parameter_groups(model, backbone_lr, args.layer_decay, args.weight_decay, args.head_lr))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    train_set = PhotoDataset(train_items, args.size, args.mode, train=True)
    steps_per_epoch = max(1, (len(clinical_idx) + derm_per_epoch) // args.batch_size)
    total_epochs = args.head_epochs + args.epochs
    warmup_steps = max(1, round(args.warmup_epochs * steps_per_epoch))
    start_epoch, best, history = 0, {"composite": -1.0, "epoch": -1}, []

    if args.resume and last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best, history = state["epoch"] + 1, state["best"], state["history"]
        log(f"Resumed after epoch {start_epoch} of {total_epochs} (best val composite so far {best['composite']:.4f})")

    for epoch in range(start_epoch, total_epochs):
        heads_only = epoch < args.head_epochs
        for param in model.backbone.parameters():
            param.requires_grad_(not heads_only)
        model.train()
        if heads_only:
            model.backbone.eval()
        rng = np.random.default_rng([SEED, epoch])
        chosen = np.concatenate([clinical_idx, rng.choice(derm_idx, size=derm_per_epoch, replace=False)]) if derm_per_epoch else clinical_idx
        loader = make_loader(
            train_set, args.batch_size, args.workers,
            sampler=torch.utils.data.SubsetRandomSampler(chosen.tolist()), drop_last=len(chosen) > args.batch_size,
        )
        started, running, seen = time.perf_counter(), 0.0, 0
        for step, (pixels, index) in enumerate(loader):
            # Heads: full rate while they train alone, then one cosine decay with no restart. Backbone:
            # a linear warm-up after it is unfrozen, then cosine decay, so the pretrained features are
            # not shaken by the first noisy steps.
            head_factor = body_factor = 1.0
            if not heads_only:
                done = (epoch - args.head_epochs) * steps_per_epoch + step
                total = max(1, args.epochs * steps_per_epoch)
                head_factor = 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, done / total)))
                span = max(1, total - warmup_steps)
                body_factor = (
                    (done + 1) / warmup_steps
                    if done < warmup_steps
                    else 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, (done - warmup_steps) / span)))
                )
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * (head_factor if group["is_head"] else body_factor)
            pixels, index = pixels.to(device, non_blocking=True), index.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits_b, logits_f = model(pixels)
            loss = multitask_loss(
                logits_b, logits_f, y_broad_t[index], y_fine_t[index], w_broad_t[index], w_fine_t[index],
                args.label_smoothing, args.fine_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach()) * len(index)
            seen += len(index)
            if step and step % 100 == 0:
                rate = seen / (time.perf_counter() - started)
                log(f"  epoch {epoch + 1}/{total_epochs} step {step}/{steps_per_epoch} loss {running / seen:.3f} ({rate:.0f} photos/s)")

        val_b, val_f = predict(model, data.val_items, args.size, args.mode, device, args.batch_size, args.workers, amp)
        score = selection_score(val_b, val_f, data.y_broad_va, data.y_fine_va, data.is_derm_va)
        minutes = (time.perf_counter() - started) / 60
        history.append({"epoch": epoch + 1, "heads_only": heads_only, "train_loss": round(running / max(seen, 1), 4), "minutes": round(minutes, 1), **score})
        marker = ""
        if score["composite"] > best["composite"]:
            best = {"composite": score["composite"], "epoch": epoch + 1}
            torch.save(model.state_dict(), best_path)
            marker = "  <- best so far"
        log(
            f"Epoch {epoch + 1}/{total_epochs}{' (heads only)' if heads_only else ''}: loss {running / max(seen, 1):.3f} | val clinical acc "
            f"{score['clinical_broad_acc']:.3f} macro-F1 {score['clinical_broad_macro_f1']:.3f} composite {score['composite']:.4f} | {minutes:.1f} min{marker}"
        )
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "history": history},
            last_path,
        )

    # ------------------------------------------------------------------ evaluate the best epoch once
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    log(f"Best epoch: {best['epoch']} (val composite {best['composite']:.4f}); scoring val and test")
    val_b, val_f = predict(model, data.val_items, args.size, args.mode, device, args.batch_size, args.workers, amp)
    test_b, test_f = predict(model, data.test_items, args.size, args.mode, device, args.batch_size, args.workers, amp)
    temp_broad, temp_fine, threshold = calibrate_on_val(data, val_b, val_f)
    final_model = evaluate_model(data, val_b, test_b, test_f, temp_broad, temp_fine, threshold, MIN_MARGIN)
    val_selection = selection_score(val_b, val_f, data.y_broad_va, data.y_fine_va, data.is_derm_va)
    test_headline = headline(final_model)

    # ------------------------------------------------------------------ export
    backbone_label = f"{short_name}_{args.size}_finetuned"
    xml_path = args.out / "export" / "classifier.xml"
    export_openvino(model, args.size, temp_broad, temp_fine, xml_path, fp16=not args.no_fp16)
    gap = export_gap(model, xml_path, data.test_items[:: max(1, len(data.test_items) // 32)][:32], args.size, args.mode, (temp_broad, temp_fine))
    if gap > 0.15:
        raise RuntimeError(f"The exported graph drifted too far from the trained model ({gap:.3f})")

    metadata = {
        "format": 3,
        "created_at": datetime.now(UTC).isoformat(),
        "backbone": backbone_label,
        "head_type": "finetuned_linear",
        "input_size": args.size,
        "input_layout": "NHWC RGB float32 0-255",
        "preprocessing": args.mode,
        "l2_norm": False,
        "weights_precision": "fp32" if args.no_fp16 else "fp16",
        "output_type": "calibrated_logits",
        "test_time_augmentation": ["orig", "flip"],
        "temperature_broad": float(temp_broad),
        "temperature_fine": float(temp_fine),
        "uncertain_threshold": float(threshold),
        "min_margin": MIN_MARGIN,
        "classes": list(BROAD_CLASSES),
        "display_names": list(BROAD_CLASSES),
        "fine_classes": list(FINE_CLASSES),
        "fine_to_broad": dict(FINE_TO_BROAD),
        "broad_to_fine": {k: list(v) for k, v in BROAD_TO_FINE.items()},
        "trained_with": "backend/finetune_gpu.py",
        "manifest_fingerprint": fingerprint,
        "split_lineage": (manifest.get("audit") or {}).get("split_lineage"),
        "trial_run": trial,  # trained or scored on a subset: not a model to install
    }
    selected = {
        "backbone": backbone_label,
        "timm_checkpoint": timm_name,
        "preprocessing": args.mode,
        "head_type": "finetuned_linear",
        "training": "every layer fine-tuned on a GPU",
        "epochs": args.epochs,
        "head_epochs": args.head_epochs,
        "best_epoch": best["epoch"],
        "batch_size": args.batch_size,
        "lr": backbone_lr,
        "head_lr": args.head_lr,
        "warmup_epochs": args.warmup_epochs,
        "layer_decay": args.layer_decay,
        "weight_decay": args.weight_decay,
        "drop_path": args.drop_path,
        "label_smoothing": args.label_smoothing,
        "isic_weight": args.isic_weight,
        "class_weight": args.class_weight,
        "temperature_broad": float(temp_broad),
        "temperature_fine": float(temp_fine),
        "uncertain_threshold": float(threshold),
        "min_margin": MIN_MARGIN,
    }
    report = {
        "seed": SEED,
        "taxonomy": {"broad_classes": list(BROAD_CLASSES), "fine_classes": list(FINE_CLASSES), "fine_to_broad": dict(FINE_TO_BROAD)},
        "dataset_audit": manifest.get("audit", {}),
        "manifest_fingerprint": fingerprint,
        "selected_configuration": selected,
        "final_model": final_model,
        "training_history": history,
        "comparison": {
            "protocol": (
                "Trained on the train split, epoch chosen and calibrated on the val split, scored once on the "
                "held-out test split. These numbers come from the GPU model; backend/install_bundle.py "
                "re-scores the saved graph on the scanner's own machine before it is installed."
            ),
            "deployed_role": "finetuned",
            "candidates": [
                {
                    "role": "finetuned",
                    "backbone": backbone_label,
                    "preprocessing": args.mode,
                    "head_type": "finetuned_linear",
                    "val_selection": val_selection,
                    "test_headline": test_headline,
                }
            ],
            "previous_deployed_bundle": None,
        },
    }
    bundle = args.out / "classifier.zip"
    with ZipFile(bundle, "w") as archive:
        archive.write(xml_path, "classifier.xml", compress_type=ZIP_DEFLATED)
        archive.write(xml_path.with_suffix(".bin"), "classifier.bin", compress_type=ZIP_STORED)
        archive.writestr("classes.json", json.dumps(list(BROAD_CLASSES), indent=2))
        archive.writestr("fine_classes.json", json.dumps(list(FINE_CLASSES), indent=2))
        archive.writestr("metadata.json", json.dumps(metadata, indent=2))
    (args.out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    result_zip = args.out / "colab_result.zip"
    with ZipFile(result_zip, "w", compression=ZIP_STORED) as archive:
        archive.write(bundle, "classifier.zip")
        archive.write(args.out / "evaluation.json", "evaluation.json")

    log(f"VAL  {val_selection}")
    log(f"TEST {test_headline}")
    current = PROJECT_ROOT / "model" / "evaluation.json"
    if current.is_file():
        try:
            shipped = json.loads(current.read_text(encoding="utf-8"))["comparison"].get("as_shipped") or {}
            if shipped.get("test_headline") and not trial:
                now = shipped["test_headline"]
                log(
                    f"Model on the laptop today: clinical accuracy {now['clinical_broad_acc']:.1%}, macro-F1 {now['clinical_broad_macro_f1']:.3f}, "
                    f"val composite {shipped['val_selection']['composite']:.4f}"
                )
        except (ValueError, KeyError, TypeError):
            pass
    log(
        f"This model: clinical accuracy {test_headline['clinical_broad_acc']:.1%}, macro-F1 {test_headline['clinical_broad_macro_f1']:.3f}, "
        f"val composite {val_selection['composite']:.4f}"
    )
    log(f"Saved {result_zip} ({result_zip.stat().st_size / 1e6:.0f} MB). Download it and run backend\\install_bundle.py on the laptop.")
    if trial:
        log("This was a trial run on a subset of the photos; the model itself is not one to install.")


if __name__ == "__main__":
    main()
