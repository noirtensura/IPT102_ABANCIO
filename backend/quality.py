"""Image quality gate: rejects blurry, badly exposed, tiny, or non-skin photos before inference.

Thresholds are calibrated from the training images by ``backend/train.py`` and stored in
``model/quality.json``. The defaults below are used when no calibration file exists.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path

import cv2
import numpy as np

QUALITY_PATH = Path(__file__).resolve().parent.parent / "model" / "quality.json"
ANALYSIS_SIDE = 512
FRAMING_MESSAGE = "Could not verify a close-up skin photo. Crop closely around the skin area you want checked and try again."
FULL_FACE_MESSAGE = "This looks like a full-face photo. Upload a close-up of just the skin area you want checked."

# A photo counts as a full-face photo when the YuNet detector finds a confident face that is at
# least a quarter of the photo's shorter side wide, lies almost entirely inside the frame, and has
# both eyes, the nose, and both mouth corners clear of the border. A close-up of a cheek, chin, or
# lips therefore passes, as do hands, limbs, and torsos that a Haar cascade often mistakes for faces.
FACE_MODEL_PATH = QUALITY_PATH.with_name("face_detection_yunet_2023mar.onnx")
FACE_MIN_SCORE = 0.75
FACE_MIN_WIDTH_FRACTION = 0.25
FACE_MIN_INSIDE_FRACTION = 0.90
FACE_LANDMARK_MARGIN = 0.03
_face_model: np.ndarray | None = None

DEFAULT_THRESHOLDS = {
    "min_side": 96,
    "max_blur": 0.72,
    "min_brightness": 22.0,
    "max_brightness": 238.0,
    "min_contrast": 7.0,
    "min_skin_ratio": 0.015,
    "max_clipped_ratio": 0.55,
}

MESSAGES = {
    "too_small": (
        "This photo is too small to analyze. Please upload a larger, clearer photo "
        "(at least {min_side} pixels on each side)."
    ),
    "blurry": (
        "This photo looks blurry. Please retake it with the skin area in focus and the camera held steady."
    ),
    "too_dark": "This photo is too dark. Please retake it in brighter, even lighting.",
    "too_bright": "This photo is overexposed. Please retake it without direct glare or flash.",
    "low_contrast": (
        "This photo has too little detail to analyze. Please retake a clearer photo of the skin area."
    ),
    "no_skin": FRAMING_MESSAGE,
}


@dataclass
class QualityReport:
    ok: bool
    issue: str | None = None
    message: str | None = None
    metrics: dict = field(default_factory=dict)


def load_thresholds(path: Path = QUALITY_PATH) -> dict:
    thresholds = dict(DEFAULT_THRESHOLDS)
    if path.is_file():
        try:
            thresholds.update(json.loads(path.read_text(encoding="utf-8")).get("thresholds", {}))
        except (OSError, ValueError):
            pass
    return thresholds


def _analysis_view(image: np.ndarray) -> np.ndarray:
    """Downscale (never upscale) so metrics are comparable across resolutions."""
    height, width = image.shape[:2]
    scale = ANALYSIS_SIDE / max(height, width)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def _tile_creteroffet(gray: np.ndarray, kernel_len: int) -> float:
    """Crete-Roffet perceptual blur score in [0, 1] on a single 2D patch; higher means blurrier."""
    image = gray.astype(np.float32)
    scores = []
    for axis in (0, 1):
        kernel = (1, kernel_len) if axis == 1 else (kernel_len, 1)
        reblurred = cv2.blur(image, kernel)
        d_image = np.abs(np.diff(image, axis=axis))
        d_blur = np.abs(np.diff(reblurred, axis=axis))
        total = float(d_image.sum())
        if total <= 1e-6:
            return 1.0
        lost = float(np.maximum(0.0, d_image - d_blur).sum())
        scores.append((total - lost) / total)
    return max(scores)


def blur_score(gray: np.ndarray) -> float:
    """Scale-invariant, patch-aware Crete-Roffet blur metric in [0, 1]; higher means blurrier.

    Scaling the re-blur kernel to the image resolution keeps small (e.g. 200 px) and large
    (512 px) photos on the same scale, while evaluating the most textured tiles prevents smooth
    surrounding skin from being mistaken for camera defocus.
    """
    height, width = gray.shape[:2]
    global_blur = _tile_creteroffet(gray, kernel_len=9)
    if min(height, width) < 64:
        return global_blur

    rows = np.linspace(0, height, 5, dtype=int)
    cols = np.linspace(0, width, 5, dtype=int)
    patches = [
        gray[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4],
        gray[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3],
    ]
    for r in range(4):
        for c in range(4):
            patches.append(gray[rows[r] : rows[r + 1], cols[c] : cols[c + 1]])

    tiles = []
    for patch in patches:
        if min(patch.shape[:2]) < 16:
            continue
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        energy = float(np.mean(gx * gx + gy * gy))
        tiles.append((energy, patch))

    tiles.sort(key=lambda item: item[0], reverse=True)
    top_blurs = [_tile_creteroffet(patch, kernel_len=9) for _, patch in tiles[:4]]
    sharpest_region_blur = float(min(top_blurs))
    return min(global_blur, sharpest_region_blur)


def skin_ratio(image: np.ndarray) -> float:
    """Fraction of pixels in a broad skin-color range."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    mask_ycrcb = cv2.inRange(ycrcb, (15, 130, 75), (255, 185, 140))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_hsv1 = cv2.inRange(hsv, (0, 15, 30), (28, 230, 255))
    mask_hsv2 = cv2.inRange(hsv, (160, 15, 30), (180, 230, 255))
    mask = cv2.bitwise_or(mask_ycrcb, cv2.bitwise_or(mask_hsv1, mask_hsv2))
    return float(np.count_nonzero(mask)) / mask.size


def measure(image: np.ndarray) -> dict:
    view = _analysis_view(image)
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    return {
        "min_side": int(min(image.shape[:2])),
        "blur": round(blur_score(gray), 4),
        "brightness": round(float(gray.mean()), 2),
        "contrast": round(float(gray.std()), 2),
        "skin_ratio": round(skin_ratio(view), 4),
        "clipped_ratio": round(float(np.mean((gray <= 3) | (gray >= 252))), 4),
    }


def assess(image: np.ndarray, thresholds: dict | None = None) -> QualityReport:
    t = thresholds or load_thresholds()
    m = measure(image)
    checks = (
        ("too_small", m["min_side"] < t["min_side"]),
        ("too_dark", m["brightness"] < t["min_brightness"]),
        ("too_bright", m["brightness"] > t["max_brightness"] or m["clipped_ratio"] > t["max_clipped_ratio"]),
        ("low_contrast", m["contrast"] < t["min_contrast"]),
        ("blurry", m["blur"] > t["max_blur"]),
        ("no_skin", m["skin_ratio"] < t["min_skin_ratio"]),
    )
    for issue, failed in checks:
        if failed:
            return QualityReport(False, issue, MESSAGES[issue].format(**t), m)
    return QualityReport(True, metrics=m)


def _whole_face_visible(view: np.ndarray) -> bool | None:
    """Whether a whole face is inside the photo; None when the YuNet detector is unavailable."""
    global _face_model
    if _face_model is None:
        if not hasattr(cv2, "FaceDetectorYN") or not FACE_MODEL_PATH.is_file():
            return None
        try:
            _face_model = np.frombuffer(FACE_MODEL_PATH.read_bytes(), dtype=np.uint8)
        except OSError:
            return None
    height, width = view.shape[:2]
    try:
        # A detector per call keeps this safe when the Flask API serves requests on several threads.
        detector = cv2.FaceDetectorYN.create("onnx", _face_model, np.empty(0, np.uint8), (width, height), 0.5, 0.3, 50)
        _, faces = detector.detect(view)
    except cv2.error:
        return None
    for face in faces if faces is not None else ():
        x, y, box_w, box_h = (float(value) for value in face[:4])
        if face[14] < FACE_MIN_SCORE or box_w < FACE_MIN_WIDTH_FRACTION * min(height, width):
            continue
        inside_w = max(0.0, min(x + box_w, width) - max(x, 0.0))
        inside_h = max(0.0, min(y + box_h, height) - max(y, 0.0))
        if inside_w * inside_h < FACE_MIN_INSIDE_FRACTION * box_w * box_h:
            continue
        points = face[4:14].reshape(5, 2)
        margin_x, margin_y = FACE_LANDMARK_MARGIN * width, FACE_LANDMARK_MARGIN * height
        if np.all(
            (points[:, 0] >= margin_x)
            & (points[:, 0] < width - margin_x)
            & (points[:, 1] >= margin_y)
            & (points[:, 1] < height - margin_y)
        ):
            return True
    return False


def framing_error(image: np.ndarray) -> str | None:
    view = _analysis_view(image)
    height, width = view.shape[:2]
    whole_face = _whole_face_visible(view)
    if whole_face is None:
        # Without the YuNet model file, fall back to the Haar cascade (more false alarms on skin texture).
        gray = cv2.equalizeHist(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY))
        detector = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
        min_face = max(28, min(height, width) // 10)
        whole_face = not detector.empty() and len(
            detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face, min_face))
        ) > 0
    if whole_face:
        return FULL_FACE_MESSAGE

    # shortcut: center color coverage screens obvious mismatches, broader coverage needs a validated input model
    center = view[height // 4:3 * height // 4, width // 4:3 * width // 4]
    ycrcb = cv2.cvtColor(center, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 136, 65), (255, 190, 139))
    if np.count_nonzero(mask) < 0.56 * mask.size:
        return FRAMING_MESSAGE
    return None
