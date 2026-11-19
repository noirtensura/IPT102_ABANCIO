"""Image preprocessing shared by training and inference so both see identical pixels."""
import cv2
import numpy as np

PREPROCESSING_MODES = ("squash", "letterbox", "letterbox_sog")


def shades_of_gray(image: np.ndarray, power: int = 6) -> np.ndarray:
    """Shades-of-Gray color constancy: removes lighting color casts while keeping lesion color contrast."""
    pixels = image.astype(np.float32)
    if power == 6:
        p2 = pixels * pixels
        powered = p2 * p2 * p2
    else:
        powered = np.power(pixels, power)
    illuminant = np.power(np.mean(powered, axis=(0, 1)), 1.0 / power)
    illuminant = illuminant / max(float(np.linalg.norm(illuminant)), 1e-6)
    gains = 1.0 / (illuminant * np.sqrt(3.0))
    return np.clip(pixels * gains, 0, 255).astype(np.uint8)


def letterbox(image: np.ndarray, size: int, color_constancy: bool = False) -> np.ndarray:
    """Resize preserving aspect ratio and pad to a square, so no skin area is cropped or distorted."""
    height, width = image.shape[:2]
    scale = size / max(height, width)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    if color_constancy:
        resized = shades_of_gray(resized)
    fill = [float(v) for v in resized.reshape(-1, 3).mean(axis=0)]
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    return cv2.copyMakeBorder(
        resized, top, size - new_h - top, left, size - new_w - left, cv2.BORDER_CONSTANT, value=fill
    )


def lesion_crop(image_bgr: np.ndarray, ratio: float = 0.72) -> np.ndarray:
    """Extract a lesion-focused crop (~72% window) centered on maximum erythema + scale texture."""
    height, width = image_bgr.shape[:2]
    crop_h, crop_w = max(32, round(height * ratio)), max(32, round(width * ratio))
    if crop_h >= height and crop_w >= width:
        return image_bgr

    small_scale = min(1.0, 160.0 / max(height, width))
    small = cv2.resize(
        image_bgr,
        (max(16, round(width * small_scale)), max(16, round(height * small_scale))),
        interpolation=cv2.INTER_AREA,
    )
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_chan, a_chan, b_chan = lab[..., 0], lab[..., 1], lab[..., 2]
    # Erythema (redness above skin median) + pigmentation + flaky scale texture
    erythema = np.maximum(0.0, a_chan - np.median(a_chan))
    color_dev = np.abs(a_chan - np.median(a_chan)) + 0.5 * np.abs(b_chan - np.median(b_chan))
    texture = np.abs(l_chan - cv2.GaussianBlur(l_chan, (9, 9), 2.0))
    saliency = 1.4 * erythema + 0.6 * color_dev + 0.9 * texture

    sh, sw = small.shape[:2]
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    cy, cx = (sh - 1) / 2.0, (sw - 1) / 2.0
    radial = ((yy - cy) / max(cy, 1.0)) ** 2 + ((xx - cx) / max(cx, 1.0)) ** 2
    saliency = cv2.GaussianBlur(saliency, (0, 0), max(2.0, min(sh, sw) * 0.12)) * np.clip(1.0 - 0.25 * radial, 0.4, 1.0)

    win_h = max(4, round(sh * ratio))
    win_w = max(4, round(sw * ratio))
    integral = cv2.integral(saliency)
    sums = (
        integral[win_h:, win_w:]
        - integral[:-win_h, win_w:]
        - integral[win_h:, :-win_w]
        + integral[:-win_h, :-win_w]
    )
    best_y, best_x = np.unravel_index(int(np.argmax(sums)), sums.shape)
    y0 = min(height - crop_h, max(0, round(best_y / small_scale)))
    x0 = min(width - crop_w, max(0, round(best_x / small_scale)))
    return image_bgr[y0 : y0 + crop_h, x0 : x0 + crop_w]


def prepare(image_bgr: np.ndarray, size: int, mode: str = "letterbox_sog") -> np.ndarray:
    """BGR uint8 image of any size -> RGB float32 HWC array in [0, 255] at size x size."""
    if mode not in PREPROCESSING_MODES:
        raise ValueError(f"Unknown preprocessing mode: {mode}")
    image = image_bgr
    if mode == "squash":
        interpolation = cv2.INTER_AREA if max(image.shape[:2]) > size else cv2.INTER_CUBIC
        image = cv2.resize(image, (size, size), interpolation=interpolation)
    else:
        image = letterbox(image, size, color_constancy=(mode == "letterbox_sog"))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)


def augment(image_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Label-preserving training augmentation applied before ``prepare``."""
    height, width = image_bgr.shape[:2]
    angle = rng.uniform(-25, 25)
    scale = rng.uniform(1.0, 1.25)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[:, 2] += rng.uniform(-0.06, 0.06, 2) * (width, height)
    image = cv2.warpAffine(image_bgr, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)
    if rng.random() < 0.5:
        image = cv2.flip(image, 1)
    if rng.random() < 0.3:
        image = cv2.flip(image, 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-4, 4)) % 180
    hsv[..., 1] *= rng.uniform(0.85, 1.15)
    hsv[..., 2] = hsv[..., 2] * rng.uniform(0.8, 1.2) + rng.uniform(-12, 12)
    image = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < 0.2:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.4, 1.0))
    return image
