"""
OpenVINO edge inference with aspect-preserving preprocessing, logit-aligned test-time
augmentation, geometry-aligned 3x3 occlusion sensitivity, hierarchical broad + fine-grained
differential prediction, and calibrated uncertainty abstention.

@author: denil gabani
"""
from pathlib import Path
import threading

import cv2
import numpy as np

try:
    from backend.inference import compile_model, load_classifier_bundle_with_meta, resize_rgb
    from backend.preprocess import prepare
except ImportError:
    from inference import compile_model, load_classifier_bundle_with_meta, resize_rgb
    from preprocess import prepare

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "model_tf.xml"
TRAINED_MODEL_PATH = MODEL_PATH.with_name("classifier.zip")
MIN_REPORT_SCORE = 0.65

SKIN_CLASSES = {
    0: "akiec, Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)",
    1: "bcc, Basal Cell Carcinoma",
    2: "bkl, Benign Keratosis",
    3: "df, Dermatofibroma",
    4: "mel, Melanoma",
    5: "nv, Melanocytic Nevi",
    6: "vasc, Vascular skin lesion",
}

uses_trained_model = False
network = None
labels: list[str] = []
fine_labels: list[str] = []
metadata: dict = {}
_model_lock = threading.Lock()
_loaded_signature: tuple | None = None


def _bundle_signature() -> tuple:
    if TRAINED_MODEL_PATH.is_file():
        st = TRAINED_MODEL_PATH.stat()
        return ("trained", st.st_mtime_ns, st.st_size)
    if MODEL_PATH.is_file():
        st = MODEL_PATH.stat()
        return ("legacy", st.st_mtime_ns, st.st_size)
    return ("none", 0, 0)


def reload_model():
    """Load or reload the deployed OpenVINO classifier bundle (supports legacy, v2, and v3 hierarchical)."""
    global uses_trained_model, network, labels, fine_labels, metadata, _loaded_signature
    with _model_lock:
        sig = _bundle_signature()
        uses_trained_model = TRAINED_MODEL_PATH.is_file()
        if uses_trained_model:
            new_network, new_labels, new_metadata = load_classifier_bundle_with_meta(TRAINED_MODEL_PATH)
            new_fine = list(new_metadata.get("fine_classes", []))
        else:
            new_network, new_labels, new_metadata = compile_model(MODEL_PATH), list(SKIN_CLASSES.values()), {}
            new_fine = []
        out_shape = new_network.output(0).partial_shape
        if len(out_shape) != 2 or out_shape[1].get_length() != len(new_labels):
            raise ValueError("The deployed model output does not match the skin lesion class list.")
        if len(new_network.outputs) >= 2 and new_fine:
            fine_shape = new_network.output(1).partial_shape
            if len(fine_shape) != 2 or fine_shape[1].get_length() != len(new_fine):
                raise ValueError("The deployed model fine-grained output does not match fine_classes.")
        network = new_network
        labels = new_labels
        metadata = new_metadata
        fine_labels = new_fine
        _loaded_signature = sig


def _ensure_current_model():
    """Automatically hot-reload the OpenVINO model if classifier.zip was updated on disk."""
    if _loaded_signature != _bundle_signature():
        reload_model()


reload_model()


def _softmax_1d(logits: np.ndarray) -> np.ndarray:
    z = logits - float(np.max(logits))
    e = np.exp(z)
    return e / float(np.sum(e))


def _softmax_2d(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def preprocessing(input_image, height, width):
    resized = cv2.resize(input_image, (width, height))
    # The original 2019 model expects BGR pixels in NCHW layout.
    return np.ascontiguousarray(resized.transpose(2, 0, 1)[None], dtype=np.float32)


def _masked_bgr_views(image_bgr: np.ndarray, size: int, mode: str | None) -> np.ndarray:
    """Build 3x3 occlusion views by masking the ORIGINAL photo before resizing/letterboxing.

    Masking in original image coordinates ensures every (row, col) cell matches the exact 3x3
    CSS overlay rendered in scan.php / uploaded.html even for non-square letterboxed photos.
    """
    height, width = image_bgr.shape[:2]
    rows = np.linspace(0, height, 4, dtype=int)
    columns = np.linspace(0, width, 4, dtype=int)
    fill = np.round(image_bgr.mean(axis=(0, 1))).astype(np.uint8)
    views = []
    for row in range(3):
        for column in range(3):
            masked = image_bgr.copy()
            masked[rows[row] : rows[row + 1], columns[column] : columns[column + 1]] = fill
            if mode:
                views.append(prepare(masked, size, mode=mode))
            else:
                views.append(resize_rgb(masked, size, size).astype(np.float32))
    return np.stack(views, axis=0)


def _best_region_from_scores(base_score: float, masked_scores: np.ndarray):
    drops = base_score - masked_scores
    best_idx = int(np.argmax(drops))
    best_drop = float(drops[best_idx])
    if best_drop < 0.03:
        return None
    return (best_idx // 3, best_idx % 3, best_drop * 100.0)


def _score_region(image, request, output, class_index, original_score):
    masked = image.copy()
    pixels = masked[0] if uses_trained_model else masked[0].transpose(1, 2, 0)
    height, width = pixels.shape[:2]
    rows = np.linspace(0, height, 4, dtype=int)
    columns = np.linspace(0, width, 4, dtype=int)
    fill = pixels.mean(axis=(0, 1))
    best_drop = 0.0
    region = None

    for row in range(3):
        for column in range(3):
            area = pixels[rows[row] : rows[row + 1], columns[column] : columns[column + 1]]
            original = area.copy()
            area[:] = fill
            masked_score = float(request.infer({0: masked})[output][0, class_index])
            area[:] = original
            drop = original_score - masked_score
            if drop > best_drop:
                best_drop = drop
                region = (row, column, drop * 100)
    return region if best_drop >= 0.03 else None


def _is_uncertain(scores: np.ndarray) -> bool:
    index = int(np.argmax(scores))
    top_score = float(scores[index])
    # The saved clinical evaluation reports higher accuracy among results shown at this cutoff.
    threshold = max(MIN_REPORT_SCORE, float(metadata.get("uncertain_threshold", MIN_REPORT_SCORE)))
    min_margin = float(metadata.get("min_margin", 0.035))
    if len(scores) > 1:
        top2 = np.partition(scores, -2)[-2:]
        margin = float(top2[1] - top2[0])
    else:
        margin = 1.0
    return top_score < threshold or margin < min_margin


def _build_differential(
    broad_probs: np.ndarray,
    fine_probs: np.ndarray | None,
    display_labels: list[str],
) -> tuple[list[dict], dict | None, list[dict]]:
    top_k = min(3, len(display_labels))
    top_idxs = np.argsort(-broad_probs)[:top_k]
    top_categories = [
        {"name": display_labels[int(i)], "score": round(float(broad_probs[int(i)]) * 100.0, 1)}
        for i in top_idxs
    ]

    if fine_probs is None or not fine_labels:
        return top_categories, None, []

    fine_to_broad = metadata.get("fine_to_broad", {})
    broad_to_fine = metadata.get("broad_to_fine", {})
    fine_idx_map = {name: i for i, name in enumerate(fine_labels)}

    top_fine_k = min(3, len(fine_labels))
    top_fine_idxs = np.argsort(-fine_probs)[:top_fine_k]
    top_specific = [
        {
            "name": fine_labels[int(i)],
            "parent": fine_to_broad.get(fine_labels[int(i)]),
            "score": round(float(fine_probs[int(i)]) * 100.0, 1),
        }
        for i in top_fine_idxs
    ]

    top_broad_name = display_labels[int(top_idxs[0])]
    children = [c for c in broad_to_fine.get(top_broad_name, []) if c in fine_idx_map]
    specific_condition = None
    if len(children) > 1:
        child_idxs = [fine_idx_map[c] for c in children]
        child_raw = fine_probs[child_idxs]
        child_sum = float(np.sum(child_raw))
        child_cond = child_raw / child_sum if child_sum > 1e-12 else np.full(len(child_idxs), 1.0 / len(child_idxs))
        best_local = int(np.argmax(child_cond))
        best_name = children[best_local]
        specific_condition = {
            "name": best_name,
            "within_category_score": round(float(child_cond[best_local]) * 100.0, 1),
            "global_score": round(float(child_raw[best_local]) * 100.0, 1),
        }
    elif len(children) == 1 and children[0] != top_broad_name:
        idx = fine_idx_map[children[0]]
        specific_condition = {
            "name": children[0],
            "within_category_score": 100.0,
            "global_score": round(float(fine_probs[idx]) * 100.0, 1),
        }

    return top_categories, specific_condition, top_specific


def pred_at_edge_detailed(input_image: np.ndarray) -> dict:
    """Run full edge inference returning primary label, confidence, occlusion region, and hierarchical differential."""
    _ensure_current_model()
    in_shape = network.input().partial_shape
    request = network.create_infer_request()
    out_broad = network.output(0)
    has_fine = len(network.outputs) >= 2 and bool(fine_labels)
    out_fine = network.output(1) if has_fine else None
    display_labels = list(metadata.get("display_names") or labels)
    outputs_logits = metadata.get("output_type") == "calibrated_logits"

    if uses_trained_model:
        height = in_shape[1].get_length()
        width = in_shape[2].get_length()
        mode = metadata.get("preprocessing")
        if mode:
            orig = prepare(input_image, height, mode=mode)
            use_tta = "flip" in metadata.get("test_time_augmentation", ["orig", "flip"])
        else:
            orig = resize_rgb(input_image, height, width).astype(np.float32)
            use_tta = False

        if in_shape[0].is_dynamic:
            masked = _masked_bgr_views(input_image, height, mode)
            if use_tta:
                flip = prepare(cv2.flip(input_image, 1), height, mode=mode)
                batch = np.concatenate([orig[None], flip[None], masked], axis=0)
                res = request.infer({0: batch})
                raw_broad = res[out_broad]
                raw_fine = res[out_fine] if has_fine else None
                if outputs_logits:
                    # Exact logit-space TTA matching training & temperature calibration
                    broad_probs = _softmax_1d(0.5 * (raw_broad[0] + raw_broad[1]))
                    fine_probs = _softmax_1d(0.5 * (raw_fine[0] + raw_fine[1])) if raw_fine is not None else None
                    orig_broad_probs = _softmax_1d(raw_broad[0])
                    masked_broad_probs = _softmax_2d(raw_broad[2:])
                else:
                    broad_probs = 0.5 * (raw_broad[0] + raw_broad[1])
                    fine_probs = 0.5 * (raw_fine[0] + raw_fine[1]) if raw_fine is not None else None
                    orig_broad_probs = raw_broad[0]
                    masked_broad_probs = raw_broad[2:]
            else:
                batch = np.concatenate([orig[None], masked], axis=0)
                res = request.infer({0: batch})
                raw_broad = res[out_broad]
                raw_fine = res[out_fine] if has_fine else None
                if outputs_logits:
                    broad_probs = _softmax_1d(raw_broad[0])
                    fine_probs = _softmax_1d(raw_fine[0]) if raw_fine is not None else None
                    orig_broad_probs = broad_probs
                    masked_broad_probs = _softmax_2d(raw_broad[1:])
                else:
                    broad_probs = raw_broad[0]
                    fine_probs = raw_fine[0] if raw_fine is not None else None
                    orig_broad_probs = raw_broad[0]
                    masked_broad_probs = raw_broad[1:]

            index = int(np.argmax(broad_probs))
            score = float(broad_probs[index])
            region = _best_region_from_scores(float(orig_broad_probs[index]), masked_broad_probs[:, index])
            uncertain = _is_uncertain(broad_probs)
            label = "Uncertain" if uncertain else display_labels[index]
            top_categories, specific_condition, top_specific = _build_differential(
                broad_probs, fine_probs, display_labels
            )
            return {
                "prediction": label,
                "confidence": score,
                "region": region,
                "top_categories": top_categories,
                "specific_condition": specific_condition,
                "top_specific": top_specific,
            }

        image = orig[None]
    else:
        height = in_shape[2].get_length()
        width = in_shape[3].get_length()
        image = preprocessing(input_image, height, width)

    res = request.infer({0: image})
    raw_b = res[out_broad][0]
    broad_probs = _softmax_1d(raw_b) if outputs_logits else raw_b
    fine_probs = None
    if has_fine:
        raw_f = res[out_fine][0]
        fine_probs = _softmax_1d(raw_f) if outputs_logits else raw_f
    index = int(np.argmax(broad_probs))
    score = float(broad_probs[index])
    region = _score_region(image, request, out_broad, index, score) if not outputs_logits else None
    uncertain = _is_uncertain(broad_probs)
    label = "Uncertain" if uncertain else display_labels[index]
    top_categories, specific_condition, top_specific = _build_differential(broad_probs, fine_probs, display_labels)
    return {
        "prediction": label,
        "confidence": score,
        "region": region,
        "top_categories": top_categories,
        "specific_condition": specific_condition,
        "top_specific": top_specific,
    }


def pred_at_edge(input_image):
    """Backward-compatible 3-tuple predictor: (label, confidence, region)."""
    detail = pred_at_edge_detailed(input_image)
    return detail["prediction"], detail["confidence"], detail["region"]
