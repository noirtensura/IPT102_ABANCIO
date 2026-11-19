"""
Flask web and JSON API backend for Skin Disease Detection.

@author: denil gabani
"""
from pathlib import Path
import sys

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.analyzer import AnalysisError, analyze_image_bytes
from backend.edge_app import labels, metadata, pred_at_edge, uses_trained_model

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/results")
def results_history():
    return render_template("results.html")


@app.get("/static/index.html")
@app.get("/static/upload.html")
def legacy_scanner():
    return redirect(url_for("index"))


@app.get("/api/health")
def api_health():
    return jsonify(
        {
            "ok": True,
            "service": "skin-disease-detection",
            "trained_model": uses_trained_model,
            "model_format": metadata.get("format", 1),
            "classes_count": len(labels),
        }
    )


@app.post("/api/predict")
def api_predict():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"ok": False, "error": "Choose an image to upload."}), 400

    try:
        result = analyze_image_bytes(upload.read(), predict_fn=pred_at_edge)
    except AnalysisError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    return (
        jsonify(result),
        200,
        {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
    )


@app.errorhandler(400)
def show_upload_error(error):
    return render_template("index.html", upload_error=error.description), 400


@app.post("/uploaded")
def upload_file():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        abort(400, "Choose an image to upload.")

    try:
        result = analyze_image_bytes(upload.read(), predict_fn=pred_at_edge)
    except AnalysisError as exc:
        abort(400, str(exc))

    return (
        render_template(
            "uploaded.html",
            prediction=result["prediction"],
            score=result["score"],
            band=result["band"],
            region=result["region"],
            specific_condition=result.get("specific_condition"),
            top_categories=result.get("top_categories", []),
            top_specific=result.get("top_specific", []),
            appearance=result["appearance"],
            reference=result["reference"],
            explanation=result["explanation"],
            record=result["record"],
            photo_ratio=result["photo_ratio"],
            image_preview=result["image_preview"],
        ),
        200,
        {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
