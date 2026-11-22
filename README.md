# Skin Disease Detection using AI at the Edge

This project uses a **PHP/HTML/CSS/JS frontend** served through XAMPP Apache and a **Python backend** powered by OpenVINO, OpenCV, NumPy, and Flask.

Each result shows the predicted class, a confidence score out of 100%, and how covering parts of the photo affected the model score. The confidence bands describe the model output. The outline marks the tested region with the largest score drop when that drop reaches 3 percentage points.

## Quick start with XAMPP

1. Start **Apache** in the XAMPP Control Panel.
2. Open [http://localhost/IPT_ABANCIO/scan.php](http://localhost/IPT_ABANCIO/scan.php) (or [http://localhost/IPT_ABANCIO/](http://localhost/IPT_ABANCIO/)).
3. Select **Upload Image** or **Open Camera**, frame the skin area in close-up, and select **Check Image**.

Python checks photo quality and framing before it runs the classifier. Photos that show a whole face in frame and photos that fail the close-up check return a message to crop or retake the photo; a close-up of part of the face (cheek, chin, lips) is analyzed.

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
- `backend/` — Python inference, taxonomy, quality gate, and training backend (`app.py`, `analyzer.py`, `predict.py`, `edge_app.py`, `inference.py`, `preprocess.py`, `quality.py`, `taxonomy.py`, `train.py`), plus GPU fine-tuning (`finetune_gpu.py`) and the check-and-install step for a model trained elsewhere (`install_bundle.py`)
- `colab/` — Google Colab notebook for GPU fine-tuning (`finetune_colab.ipynb`) and its helpers (`make_pack.py`, `setup_data.py`)
- `static/` — CSS, JS, fonts, and example preview image
- `model/` — OpenVINO model files (`model_tf.xml`, `model_tf.bin`, `classifier.zip`, `evaluation.json`, `quality.json`) and the YuNet face detector used by the framing check (`face_detection_yunet_2023mar.onnx`, MIT license, from OpenCV Zoo)
- `datasets/` — Training images. Not part of the repository; see [Get the training images](#get-the-training-images)
- `build/` — Training caches, logs, candidate models, and backups of replaced models (created by `backend/train.py`)
- `samples/` — Sample images for testing
- `tests/` — Automated unit tests

## Set up Python environment (if reinstalling)

Use 64-bit Python 3.12 or 3.13 in PowerShell from `C:\xampp\htdocs\IPT_ABANCIO`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Get the training images

The images are not stored in the repository. Download the five public Kaggle datasets (no account needed) and extract each archive into its own folder under `datasets/`:

| Folder | Kaggle dataset | Contents |
| --- | --- | --- |
| `datasets/20-skin-diseases/` | `haroonalam16/20-skin-diseases-dataset` | 3,506 clinical photos in 19 class folders |
| `datasets/dermnet/` | `shubhamgoel27/dermnet` | 19,559 clinical photos in 23 class folders (the collection the first set was sampled from) |
| `datasets/skin-diseases-6class/` | `ascanipek/skin-diseases` | 38,761 clinical and dermoscopy images in 6 class folders, pre-split into train/val/test |
| `datasets/skin-disease-dataset/` | `subirbiswas19/skin-disease-dataset` | 1,159 photos of 8 skin infections (673 distinct photos once copies are removed) |
| `datasets/skindiseasedataset/` | `pacificrm/skindiseasedataset` | 15,444 photos in 22 class folders. Most are DermNet photos already in the other sets; 2,189 are new |

```powershell
New-Item -ItemType Directory -Force datasets\20-skin-diseases, datasets\dermnet, datasets\skin-diseases-6class, datasets\skin-disease-dataset, datasets\skindiseasedataset
curl.exe -L -o datasets\20-skin-diseases.zip https://www.kaggle.com/api/v1/datasets/download/haroonalam16/20-skin-diseases-dataset
curl.exe -L -o datasets\dermnet.zip https://www.kaggle.com/api/v1/datasets/download/shubhamgoel27/dermnet
curl.exe -L -o datasets\skin-diseases-6class.zip https://www.kaggle.com/api/v1/datasets/download/ascanipek/skin-diseases
curl.exe -L -o datasets\skin-disease-dataset.zip https://www.kaggle.com/api/v1/datasets/download/subirbiswas19/skin-disease-dataset
curl.exe -L -o datasets\skindiseasedataset.zip https://www.kaggle.com/api/v1/datasets/download/pacificrm/skindiseasedataset
tar -xf datasets\20-skin-diseases.zip -C datasets\20-skin-diseases
tar -xf datasets\dermnet.zip -C datasets\dermnet
tar -xf datasets\skin-diseases-6class.zip -C datasets\skin-diseases-6class
tar -xf datasets\skin-disease-dataset.zip -C datasets\skin-disease-dataset
tar -xf datasets\skindiseasedataset.zip -C datasets\skindiseasedataset
```

The downloads total about 9.6 GB, and the extracted folders take the same again. Once extracted, the `.zip` files can be deleted.

Three folders of `skindiseasedataset` are not used, because the scanner has no category for them: `Unknown_Normal` (random pictures such as animals, objects, and cartoons), `Bullous` (blistering diseases), and the bites and stings in `Infestations_Bites` (its scabies, lice, larva migrans, and similar infestation photos are used as infections).

### Add more photos or another dataset

- **More photos for an existing condition:** copy them into the matching class folder under `datasets/dermnet/train/` and run the training command again. The folder is the label.
- **A new dataset with its own folder names:** extract it into its own folder under `datasets/`, map its class folders to the categories in `backend/taxonomy.py` (see `DATASET_E_FOLDER_TO_LABELS` and `labels_for_dataset_f`), and add a scan block for it in `_iter_raw_records` in `backend/train.py`.

Photos keep the train/val/test split they were first given, and their cached features in `build/` are reused, so a re-run after adding photos only processes the new ones and takes about an hour instead of three. Pass `--fresh-split` to redraw the split for every photo.

## Train on all workspace datasets

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe backend\train.py --require-ac
```

`backend/train.py` reads every dataset under `datasets/` and then:

1. **Audits the data.** It drops exact and near-duplicate images, drops images whose sources disagree on the category, corrects class-folder labels that the filename contradicts (benign moles filed under melanoma, herpes photos filed under warts), and builds a leak-free 75% train / 10% val / 15% test split.
2. **Picks a backbone** by 5-fold grouped cross-validation on a pilot of the train split (`--backbones`, default EfficientNetV2-B2 and DINOv2 ViT-S/14).
3. **Trains the classifier head** on cached backbone features, calibrates it on the val split, and scores it once on the test split.
4. **Re-fits the previously deployed recipe** on the same split, so the old and new recipes are compared like for like. The candidate with the best val score ships.
5. **Exports** `model/classifier.zip`, `model/evaluation.json`, and `model/quality.json`, then scores the saved graph end to end on val/test. The replaced files are copied to `build/model_backups/`.

Restart `backend/app.py` (if running) after training so it loads the new bundle.

Useful options:

- `--require-ac` pauses the run while a laptop is on battery and resumes when the charger is back. The run also pauses when system memory is nearly full. Every stage is checkpointed in `build/`, so an interrupted run continues where it stopped.
- `--no-deploy` writes the bundle to `build/candidate/` instead of replacing `model/`.
- `--fusion` also evaluates one head over both backbones; `--fp16` stores the exported weights in half precision, which halves `classifier.zip`.
- Set the environment variable `SKIN_TRAIN_THREADS` (for example `6`) to cap CPU threads. The default is one per physical core.

On an 8-core laptop CPU without a GPU, a full run takes roughly three hours.

A run never replaces a deployed model that scores higher on the same val photos (for example a model fine-tuned on a GPU); the new model then goes to `build/candidate/`. `--force-deploy` overrides that.

## Fine-tune on a GPU with Google Colab

`backend/train.py` keeps the two backbones frozen and trains only a small head, which is all a laptop CPU can do. `backend/finetune_gpu.py` trains every layer of DINOv2 ViT-S/14 instead. That needs a GPU; a free Colab T4 takes about an hour.

1. On the laptop, build the upload file (2.6 MB: the code and the photo list with its train/val/test split, no photos):

   ```powershell
   .\.venv\Scripts\python.exe colab\make_pack.py
   ```

2. Open [colab.research.google.com](https://colab.research.google.com), choose **Upload**, and pick `colab\finetune_colab.ipynb`.
3. In Colab, choose **Runtime > Change runtime type > T4 GPU**, then **Runtime > Run all**. When step 2 asks, pick `build\colab_pack.zip`. Keep the tab open.
4. The notebook downloads the five Kaggle datasets, keeps exactly the photos in the list (each is checked by SHA-256), trains, and downloads `colab_result.zip` to your computer.
5. Put `colab_result.zip` in the project folder and, with the charger plugged in, run:

   ```powershell
   $env:SKIN_TRAIN_THREADS = '6'
   .\.venv\Scripts\python.exe backend\install_bundle.py colab_result.zip
   ```

   This scores the new model on the laptop exactly as the scanner runs it (about 15 minutes), prints it next to the installed model, and installs it only when it scores higher on the val split. The replaced model is copied to `build/model_backups/`. Add `--dry-run` to compare without installing.

Colab trains, calibrates, and tests on the same photos as the laptop, so the numbers are comparable. Build the pack again whenever the datasets change. `backend\finetune_gpu.py --smoke` runs every step on a handful of photos without a GPU, to check the plumbing after a code change.

## Current model and measured accuracy

The deployed `model/classifier.zip` was trained on 2026-11-22 with:

```powershell
.\.venv\Scripts\python.exe backend\train.py --backbones effnetv2b2_260 dinov2_vits14_224 --modes letterbox_sog --fusion --fp16 --require-ac
```

It is one classifier head over two frozen backbones (EfficientNetV2-B2 and DINOv2 ViT-S/14), saved as a single 65 MB OpenVINO graph. After the audit, 35,189 images remained: 26,394 for training, 3,514 for calibration (val), and 5,281 for the test. The test images (3,068 clinical photos and 2,213 dermoscopy images) were never used for training or tuning.

| Measured on the test images | Model before `skindiseasedataset` was added | Deployed model |
| --- | --- | --- |
| Clinical photos, 7 categories: accuracy | 68.0% (95% CI 66.4 to 69.7) | **71.8%** (95% CI 70.1 to 73.3) |
| Clinical photos, 7 categories: macro-F1 | 0.678 | **0.727** |
| Clinical photos: true category among the top 3 | 90.8% | **93.5%** |
| Clinical photos, 21 specific conditions: accuracy | 61.0% | **64.7%** |
| Dermoscopy images: accuracy | 95.1% | **95.5%** |
| All test images: accuracy | 79.4% | **81.8%** |
| Clinical photos where the scanner shows a match (score of 65% or more) | 61.8% | 61.2% |
| Shown matches that are correct | 82.4% | **87.1%** |
| Clinical photos that end with a correct match on screen | 50.9% | **53.3%** |

Both columns are the same two-backbone recipe; neither model saw any test photo, because photos keep their split when a dataset is added. The gain comes from the test photos of the added dataset:

| Clinical test photos | Photos | Before | Deployed |
| --- | --- | --- | --- |
| Already in the data before `skindiseasedataset` | 2,742 | 72.3% | 71.6% (change of -0.7 points, 95% CI -1.8 to +0.2) |
| New with `skindiseasedataset` | 326 | 32.2% | 73.9% |

Recall per category on the clinical test photos: Acne and Rosacea 85%, Pigmentation Disorders 75%, Benign Skin Lesions 74%, Eczema and Dermatitis 72%, Infectious Skin Conditions 70%, Inflammatory and Systemic Conditions 70%, Malignant and Pre-Malignant Lesions 64%. The full report, including the confusion matrix and the 21 specific conditions, is in `model/evaluation.json`.

How to read these numbers:

- **The test photos come from the same public collections as the training photos.** Photos taken with a phone in ordinary lighting will score lower.
- **The new photos are a different kind of photo.** `skindiseasedataset` mostly repeats DermNet; its 2,189 new photos are small web thumbnails and portrait-style photos (vitiligo, actinic keratosis, drug eruptions, infestations, lupus, lichen). The earlier model handled them poorly; the deployed one handles a wider range of photo styles.
- **On the earlier photos the deployed model is more cautious, not more accurate.** Accuracy there is unchanged within noise. A correct match reaches the screen for 52.6% of them (before: 54.9%) and a wrong match for 7.6% (before: 8.5%).
- **`skin-disease-dataset` was added one run earlier.** Its own 108 test photos went from 28.7% to 99.1%, with no change on the other photos. That 99.1% is not a general estimate: its photos share one look, with rotated and rescaled copies of the same source.
- **The two-backbone recipe is what raised accuracy on the first datasets.** On the 2,649 clinical test photos of the first split, the EfficientNetV2-B2 recipe that shipped originally scored 66.1% when re-fitted on the same training images, and the two-backbone model 71.7%.
- **The original bundle cannot be compared directly.** It was trained on its own split of two of the datasets and reported 64.2% clinical accuracy on its own test images. On the first split's test images it scored 81.0%, because most of them were in its training data. On the 665 of those photos that exist only in DermNet, which it never saw (mostly inflammatory conditions), it scored 57.1% and the two-backbone model 77.7%.
- **The upload check also turns photos away.** Of 210 test photos sent through the full upload path, 19% were rejected by the photo check, 34% got "no reliable match", 40% got a correct match, and 8% got a wrong match (the model before, on the same photos: 38% correct, 8% wrong). Whole-face photos are rejected by design, which removes most portrait-style vitiligo photos.

One scan takes about 2.5 seconds when `scan.php` starts Python itself and about 0.7 seconds when the Flask API is running.

## Note

This model is for educational and demonstration purposes only. A diagnosis requires a clinical exam by a qualified healthcare professional.

The scanner requires a top category score of at least 65% and a sufficient lead over the next category before showing a possible match. Below that, the result reads "No reliable match" and lists the three closest categories with their scores for reference only; on the test photos the true category is among those three for 88% of such results. The published model evaluation shows the tradeoff between how often it reports a category and how often those categories are correct on the project’s test set.
