# Skin Disease Detection using AI at the Edge

This project uses a **PHP/HTML/CSS/JS frontend** served through XAMPP Apache and a **Python backend** powered by OpenVINO, OpenCV, NumPy, and Flask.

Each result shows the predicted class, a confidence score out of 100%, and how covering parts of the photo affected the model score. The confidence bands describe the model output. The outline marks the tested region with the largest score drop when that drop reaches 3 percentage points.

## Quick start with XAMPP

1. Start **Apache** in the XAMPP Control Panel.
2. Open [http://localhost/IPT_ABANCIO/scan.php](http://localhost/IPT_ABANCIO/scan.php) (or [http://localhost/IPT_ABANCIO/](http://localhost/IPT_ABANCIO/)).
3. Select **Upload Image** or **Open Camera**, frame the skin area in close-up, and select **Check Image**.

Python checks photo quality and framing before it runs the classifier. Full-face portraits and photos that fail the close-up check return a message to crop or retake the photo.

The PHP frontend automatically runs the Python inference backend (`backend/predict.py`) using the project virtual environment (`.venv`).

### Optional: Keep the Python Flask API running for faster repeated scans

You can optionally run `start_backend.bat` (or run the command below) to keep the OpenVINO model loaded in memory on `http://127.0.0.1:5000`:

```powershell
.\.venv\Scripts\python.exe backend\app.py
```

When the Flask API is running, `scan.php` automatically uses `http://127.0.0.1:5000/api/predict`. If it is stopped, `scan.php` automatically falls back to invoking `backend/predict.py` directly.

## Project structure

- `scan.php` — Main scanner frontend (upload/camera UI and analysis result view with top-3 differential and specific pattern hints)
- `results.php` — Saved results history page (keeps up to 20 summaries in the current browser tab)
- `index.php` — Entry point redirecting to `scan.php`
- `includes/` — Shared PHP layout (`layout.php`) and Python backend bridge (`python_client.php`)
- `backend/` — Python inference, taxonomy, quality gate, and training backend (`app.py`, `analyzer.py`, `predict.py`, `edge_app.py`, `inference.py`, `preprocess.py`, `quality.py`, `taxonomy.py`, `train.py`)
- `static/` — CSS, JS, fonts, and example preview image
- `model/` — OpenVINO model files (`model_tf.xml`, `model_tf.bin`, `classifier.zip`, `evaluation.json`, `quality.json`)
- `datasets/` — Kaggle 20 Skin Diseases dataset (`20-skin-diseases.zip` / `20-skin-diseases/`)
- `train/`, `val/`, `test/` — Multi-source 6-class skin disease dataset (automatically audited, deduplicated, and merged by `backend/train.py`)
- `samples/` — Sample images for testing
- `tests/` — Automated unit tests

## Set up Python environment (if reinstalling)

Use 64-bit Python 3.12 in PowerShell from `C:\xampp\htdocs\IPT_ABANCIO`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Train on all workspace datasets

`backend/train.py` automatically ingests and merges **all datasets** in this workspace (`datasets/20-skin-diseases` or `datasets/20-skin-diseases.zip`, plus `train/`, `val/`, and `test/`), removes exact and perceptual duplicates across splits, balances clinical vs. ISIC dermoscopy domains, and trains a dual-output hierarchical classifier (7 broad clinical categories + 20 specific sub-conditions):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe backend\train.py
```

Training exports `model/classifier.zip`, `model/evaluation.json`, and `model/quality.json`. Restart `backend/app.py` (if running) after training so it loads the new bundle.

## Note

This model is for educational and demonstration purposes only. A diagnosis requires a clinical exam by a qualified healthcare professional.

The scanner requires a top category score of at least 65% and a sufficient lead over the next category before showing a possible match. The published model evaluation shows the tradeoff between how often it reports a category and how often those categories are correct on the project’s test set.
