"""Shared skin image analysis service used by Flask API and CLI runner."""
from base64 import b64encode
from datetime import UTC, datetime
from io import BytesIO
from time import perf_counter
from uuid import uuid4

import cv2
import numpy as np
try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None

try:
    from backend import edge_app
    from backend.quality import assess as assess_image_quality
    from backend.quality import framing_error
    from backend.taxonomy import CONDITION_GUIDES
except ImportError:
    import edge_app
    from quality import assess as assess_image_quality
    from quality import framing_error
    from taxonomy import CONDITION_GUIDES


class AnalysisError(ValueError):
    """Raised when an uploaded photo cannot be validated or decoded."""


def decode_image_bytes(data: bytes) -> np.ndarray | None:
    """Decode any supported image format (JPG, PNG, WebP, BMP, GIF, TIFF, ICO, AVIF, etc.) to BGR uint8."""
    if Image is not None:
        try:
            with Image.open(BytesIO(data)) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
                    rgba = pil_img.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                    rgb = Image.alpha_composite(background, rgba).convert("RGB")
                else:
                    rgb = pil_img.convert("RGB")
                arr = np.asarray(rgb, dtype=np.uint8)
                if arr.ndim == 3 and arr.shape[2] == 3:
                    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    try:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        return None
    return image


def analyze_image_bytes(data: bytes, predict_fn=None, check_quality: bool = True) -> dict:
    """Validate, decode, quality-check, and classify uploaded skin photo bytes."""
    if not data:
        raise AnalysisError("The uploaded image is empty.")

    image = decode_image_bytes(data)
    if image is None:
        raise AnalysisError("Upload a readable image.")

    if check_quality:
        quality_report = assess_image_quality(image)
        if not quality_report.ok:
            raise AnalysisError(quality_report.message or "Please upload a clearer photo of the skin area.")
        if message := framing_error(image):
            raise AnalysisError(message)

    start = perf_counter()
    if predict_fn is None or predict_fn is edge_app.pred_at_edge:
        raw_result = edge_app.pred_at_edge_detailed(image)
    else:
        raw_result = predict_fn(image)
    elapsed = perf_counter() - start

    if isinstance(raw_result, dict):
        prediction = str(raw_result["prediction"])
        confidence = float(raw_result["confidence"])
        region = raw_result.get("region")
        top_categories = list(raw_result.get("top_categories") or [])
        specific_condition = raw_result.get("specific_condition")
        top_specific = list(raw_result.get("top_specific") or [])
    else:
        prediction, confidence, region = raw_result
        prediction = str(prediction)
        confidence = float(confidence)
        top_categories = []
        specific_condition = None
        top_specific = []

    score = round(float(confidence) * 100, 1)
    band = "High" if score >= 80 else "Moderate" if score >= 50 else "Low"
    key = prediction.partition(", ")[0]
    condition, appearance, reference = CONDITION_GUIDES.get(
        prediction,
        CONDITION_GUIDES.get(key, (prediction, None, None)),
    )

    # Enrich appearance/reference with specific sub-condition guide when primary is confident
    if (
        condition != "Uncertain"
        and isinstance(specific_condition, dict)
        and specific_condition.get("name") in CONDITION_GUIDES
        and float(specific_condition.get("within_category_score", 0.0)) >= 35.0
    ):
        _, spec_appearance, spec_reference = CONDITION_GUIDES[specific_condition["name"]]
        if spec_appearance:
            appearance = spec_appearance
            reference = spec_reference

    if condition == "Uncertain":
        # No category is presented as the match. The closest categories stay in the result so the page
        # can list them for reference; the specific-condition hints are withheld.
        region_list = None
        top_categories = top_categories[:3]
        explanation = (
            f"The highest model score was {score:.1f}%. The scores did not meet the criteria for showing a match. "
            + ("The closest categories are listed for reference only. " if len(top_categories) >= 2 else "")
            + "A clinician can evaluate the affected area."
        )
        report_evidence = explanation
        specific_condition = None
        top_specific = []
    elif region:
        region_list = [int(region[0]), int(region[1]), float(region[2])]
        explanation = (
            f"Covering the most influential region of the photo lowered this label\u2019s score by {region_list[2]:.1f} percentage points."
        )
        report_evidence = explanation
    else:
        region_list = None
        explanation = (
            "This label scored highest. Covering individual areas lowered its score by less than 3 percentage points."
        )
        report_evidence = explanation

    created_at = datetime.now(UTC).isoformat()
    report_lines = [
        "Skin photo analysis",
        f"Analyzed at {created_at}",
        "Result: No reliable match" if condition == "Uncertain" else f"Possible condition: {condition}",
        f"{'Highest model score' if condition == 'Uncertain' else 'Model confidence'}: {score:.1f}% of 100% ({band})",
        f"Analysis time: {elapsed:.3f} s",
        "Low below 50%. Moderate 50% to 79.9%. High 80% to 100%.",
    ]
    if isinstance(specific_condition, dict) and specific_condition.get("name"):
        report_lines.append(
            f"Closest specific pattern: {specific_condition['name']} "
            f"({float(specific_condition.get('within_category_score', 0.0)):.1f}% within category)"
        )
    if top_categories:
        report_lines.append("")
        report_lines.append(
            "Closest categories (not a reliable match):" if condition == "Uncertain" else "Top differential categories:"
        )
        for item in top_categories[:3]:
            report_lines.append(f"  - {item['name']}: {float(item['score']):.1f}%")
    if top_specific:
        report_lines.append("")
        report_lines.append("Top specific visual patterns:")
        for item in top_specific[:3]:
            report_lines.append(f"  - {item['name']}: {float(item['score']):.1f}%")
    report_lines.extend(["", report_evidence])
    if appearance:
        report_lines.extend(["", "Typical appearance", appearance, reference])
    report_lines.extend(
        ["", "A diagnosis requires a clinical exam. Ask a qualified healthcare professional for evaluation."]
    )

    record = {
        "id": uuid4().hex,
        "created_at": created_at,
        "prediction": condition,
        "score": score,
        "band": band,
        "explanation": report_evidence,
        "appearance": appearance,
        "report": b64encode("\n".join(report_lines).encode("utf-8")).decode("ascii"),
    }

    encoded, preview = cv2.imencode(".jpg", image)
    if not encoded:
        raise AnalysisError("The uploaded image could not be previewed.")

    return {
        "ok": True,
        "prediction": condition,
        "score": score,
        "band": band,
        "region": region_list,
        "specific_condition": specific_condition,
        "top_categories": top_categories,
        "top_specific": top_specific,
        "appearance": appearance,
        "reference": reference,
        "explanation": explanation,
        "elapsed": round(elapsed, 4),
        "record": record,
        "photo_ratio": image.shape[1] / image.shape[0],
        "image_preview": b64encode(preview.tobytes()).decode("ascii"),
    }
