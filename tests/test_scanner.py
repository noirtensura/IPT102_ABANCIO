from base64 import b64decode
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import sys
from unittest import TestCase
from unittest.mock import patch

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import app


class ScannerTest(TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.sample_path = PROJECT_ROOT / "samples" / "1.jpg"

    def test_home_preview_and_legacy_links(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Add a skin photo", response.data)
        self.assertIn(b"Open Camera", response.data)
        self.assertIn(b"example-skin-photo.png", response.data)
        for route in ("/static/upload.html", "/static/index.html"):
            self.assertEqual(self.client.get(route).location, "/")

    def test_scan_result_uses_the_new_layout(self):
        with patch("backend.app.pred_at_edge", return_value=("Acne and Rosacea", 0.82, (0, 1, 8.2))):
            response = self.client.post(
                "/uploaded",
                data={"file": (BytesIO(self.sample_path.read_bytes()), self.sample_path.name)},
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Analysis Result", response.data)
        self.assertIn(b"Preliminary Analysis", response.data)
        self.assertIn(b"Model confidence", response.data)
        self.assertIn(b'min="0" max="100"', response.data)
        self.assertIn(b"100% max", response.data)
        self.assertIn(b"Why this label", response.data)
        self.assertIn(b"Analyze Another", response.data)
        self.assertIn(b"General Care Tips", response.data)
        self.assertIn(b'id="confidence-dialog"', response.data)
        self.assertIn(b'href="/results"', response.data)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        record = json.loads(re.search(rb'id="analysis-record">(.*?)</script>', response.data).group(1))
        report = b64decode(record["report"]).decode("utf-8")
        self.assertIn("Possible condition: " + record["prediction"], report)
        self.assertIn("of 100%", report)
        self.assertIn("A diagnosis requires a clinical exam.", report)
        self.assertNotIn("data:image", report)
        self.assertNotIn("image_preview", record)

    def test_history_page(self):
        response = self.client.get("/results")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Saved results", response.data)
        self.assertIn(b'id="history-list"', response.data)

    def test_confidence_bands_match_the_displayed_score(self):
        sample = self.sample_path.read_bytes()
        cases = (
            (0.4994, "49.9", "Low"),
            (0.4996, "50.0", "Moderate"),
            (0.7996, "80.0", "High"),
            (1.0, "100.0", "High"),
        )
        for score, displayed, band in cases:
            with self.subTest(score=score), patch(
                "backend.app.pred_at_edge", return_value=("nv, Melanocytic Nevi", score, (0, 1, 8.2))
            ):
                response = self.client.post("/uploaded", data={"file": (BytesIO(sample), "sample.jpg")})
                self.assertEqual(response.status_code, 200)
                page = response.get_data(as_text=True)
                self.assertIn(f'value="{displayed}"', page)
                self.assertIn(f'class="confidence-band">{band}</span>', page)
                self.assertIn("Mole (melanocytic nevus)", page)
                self.assertIn("8.2 percentage points", page)
                self.assertNotIn('class="score-region"', page)
                self.assertIn("About this condition", page)

    def test_unlisted_condition_and_no_highlight(self):
        sample = self.sample_path.read_bytes()
        with patch("backend.app.pred_at_edge", return_value=("Acne and Rosacea Photos", 0.64, None)):
            response = self.client.post("/uploaded", data={"file": (BytesIO(sample), "sample.jpg")})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Acne and Rosacea Photos", response.data)
        self.assertIn(b"less than 3 percentage points", response.data)
        self.assertNotIn(b'class="score-region"', response.data)

    def test_unreadable_image_returns_a_scanner_error(self):
        response = self.client.post("/uploaded", data={"file": (BytesIO(b"invalid photo"), "test.jpg")})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Add a skin photo", response.data)
        self.assertIn(b"Upload a readable image.", response.data)

    def test_blurry_and_low_quality_images_are_rejected(self):
        image = cv2.imread(str(self.sample_path))
        blurred = cv2.GaussianBlur(image, (31, 31), 9.0)
        _, blurred_buf = cv2.imencode(".jpg", blurred)
        response = self.client.post(
            "/api/predict",
            data={"file": (BytesIO(blurred_buf.tobytes()), "blurry.jpg")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("blurry", response.get_json()["error"].lower())

        dark = (image.astype(np.float32) * 0.05).astype(np.uint8)
        _, dark_buf = cv2.imencode(".jpg", dark)
        dark_resp = self.client.post(
            "/api/predict",
            data={"file": (BytesIO(dark_buf.tobytes()), "dark.jpg")},
        )
        self.assertEqual(dark_resp.status_code, 400)
        self.assertIn("too dark", dark_resp.get_json()["error"].lower())

        tiny = cv2.resize(image, (48, 48))
        _, tiny_buf = cv2.imencode(".jpg", tiny)
        tiny_resp = self.client.post(
            "/api/predict",
            data={"file": (BytesIO(tiny_buf.tobytes()), "tiny.jpg")},
        )
        self.assertEqual(tiny_resp.status_code, 400)
        self.assertIn("too small", tiny_resp.get_json()["error"].lower())

    def test_scan_rejects_unsuitable_framing_before_prediction(self):
        from backend.quality import assess

        image = np.full((192, 288, 3), (155, 98, 51), dtype=np.uint8)
        image[:, :48] = (84, 132, 178)
        rows, columns = np.indices(image.shape[:2])
        image[(rows // 6 + columns // 6) % 2 == 0] += 35
        self.assertTrue(assess(image).ok)
        _, buffer = cv2.imencode(".png", image)
        with patch("backend.app.pred_at_edge") as predict:
            scene = self.client.post(
                "/api/predict", data={"file": (BytesIO(buffer.tobytes()), "scene.png")}
            )
            self.assertEqual(scene.status_code, 400)
            self.assertIn("Could not verify a close-up skin photo", scene.get_json()["error"])
            with patch("backend.quality.cv2.CascadeClassifier") as cascade:
                cascade.return_value.empty.return_value = False
                cascade.return_value.detectMultiScale.return_value = [(30, 20, 80, 80)]
                portrait = self.client.post(
                    "/uploaded", data={"file": (BytesIO(self.sample_path.read_bytes()), "portrait.jpg")}
                )
            self.assertEqual(portrait.status_code, 400)
            self.assertIn(b"This looks like a full-face photo", portrait.data)
            predict.assert_not_called()

    def test_uncertain_prediction_is_returned_when_confidence_is_ambiguous(self):
        from backend import edge_app

        with patch.object(edge_app, "metadata", {"uncertain_threshold": 0.36, "min_margin": 0.03}):
            self.assertTrue(edge_app._is_uncertain(np.array([0.609, 0.321, 0.019, 0.051])))
            self.assertFalse(edge_app._is_uncertain(np.array([0.76, 0.19, 0.05])))
        sample = self.sample_path.read_bytes()
        result = {
            "prediction": "Uncertain",
            "confidence": 0.609,
            "region": (1, 1, 18.4),
            "top_categories": [{"name": "Acne and Rosacea", "score": 60.9}],
            "top_specific": [{"name": "Acne and Rosacea", "score": 54.4}],
        }
        with patch("backend.app.pred_at_edge", return_value=result):
            response = self.client.post("/api/predict", data={"file": (BytesIO(sample), "sample.jpg")})
            page = self.client.post("/uploaded", data={"file": (BytesIO(sample), "sample.jpg")})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["prediction"], "Uncertain")
        self.assertAlmostEqual(payload["score"], 60.9)
        self.assertEqual(payload["band"], "Moderate")
        self.assertIn("criteria for showing a match", payload["explanation"])
        self.assertEqual(payload["top_categories"], [])
        self.assertEqual(payload["top_specific"], [])
        self.assertIsNone(payload["region"])
        self.assertIn(b"No reliable match", page.data)
        self.assertIn(b"Highest model score", page.data)
        self.assertNotIn(b"Possible: Acne and Rosacea", page.data)
        self.assertIn("Result: No reliable match", b64decode(payload["record"]["report"]).decode("utf-8"))

    def test_accepts_all_image_file_types(self):
        from PIL import Image

        with Image.open(self.sample_path) as base_img:
            rgb = base_img.convert("RGB")
            rgba = base_img.convert("RGBA")
            for fmt, ext, img in (
                ("PNG", "sample.png", rgba),
                ("WEBP", "sample.webp", rgb),
                ("BMP", "sample.bmp", rgb),
                ("GIF", "sample.gif", rgb),
                ("TIFF", "sample.tiff", rgb),
            ):
                with self.subTest(format=fmt):
                    buf = BytesIO()
                    img.save(buf, format=fmt)
                    buf.seek(0)
                    response = self.client.post("/api/predict", data={"file": (buf, ext)})
                    self.assertEqual(response.status_code, 200, msg=response.get_data(as_text=True))
                    payload = response.get_json()
                    self.assertTrue(payload["ok"])
                    self.assertIn("prediction", payload)

    def test_api_health_and_predict_endpoints(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.get_json()["ok"])

        response = self.client.post(
            "/api/predict",
            data={"file": (BytesIO(self.sample_path.read_bytes()), self.sample_path.name)},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertIn("prediction", payload)
        self.assertIn("record", payload)
        self.assertIn("image_preview", payload)

        bad = self.client.post("/api/predict", data={"file": (BytesIO(b"not an image"), "bad.jpg")})
        self.assertEqual(bad.status_code, 400)
        self.assertFalse(bad.get_json()["ok"])
        self.assertEqual(bad.get_json()["error"], "Upload a readable image.")

    def test_cli_predict_script(self):
        proc = subprocess.run(
            [sys.executable, "-B", str(PROJECT_ROOT / "backend" / "predict.py"), str(self.sample_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload["ok"])
        self.assertIn("prediction", payload)
        self.assertIn("record", payload)
        self.assertIn("top_categories", payload)

    def test_hierarchical_differential_renders_in_html_api_and_report(self):
        sample = self.sample_path.read_bytes()
        detailed = {
            "prediction": "Inflammatory and Systemic Conditions",
            "confidence": 0.784,
            "region": (1, 1, 12.4),
            "top_categories": [
                {"name": "Inflammatory and Systemic Conditions", "score": 78.4},
                {"name": "Eczema and Dermatitis", "score": 14.2},
                {"name": "Infectious Skin Conditions", "score": 4.9},
            ],
            "specific_condition": {
                "name": "Psoriasis, Lichen Planus and Related Conditions",
                "within_category_score": 81.5,
                "global_score": 64.0,
            },
            "top_specific": [
                {
                    "name": "Psoriasis, Lichen Planus and Related Conditions",
                    "parent": "Inflammatory and Systemic Conditions",
                    "score": 64.0,
                },
                {"name": "Eczema", "parent": "Eczema and Dermatitis", "score": 15.1},
                {"name": "Tinea, Ringworm and Fungal Infections", "parent": "Infectious Skin Conditions", "score": 7.3},
            ],
        }
        with patch("backend.app.pred_at_edge", return_value=detailed):
            api_resp = self.client.post("/api/predict", data={"file": (BytesIO(sample), "sample.jpg")})
            self.assertEqual(api_resp.status_code, 200)
            payload = api_resp.get_json()
            self.assertEqual(payload["prediction"], "Inflammatory and Systemic Conditions")
            self.assertEqual(
                payload["specific_condition"]["name"],
                "Psoriasis, Lichen Planus and Related Conditions",
            )
            self.assertEqual(len(payload["top_categories"]), 3)
            self.assertIn("silvery-white scales", payload["appearance"])
            report_text = b64decode(payload["record"]["report"]).decode("utf-8")
            self.assertIn("Closest specific pattern: Psoriasis, Lichen Planus and Related Conditions", report_text)
            self.assertIn("Top differential categories:", report_text)

            html_resp = self.client.post("/uploaded", data={"file": (BytesIO(sample), "sample.jpg")})
            self.assertEqual(html_resp.status_code, 200)
            page = html_resp.get_data(as_text=True)
            self.assertIn("Closest specific pattern:", page)
            self.assertIn("Top differential possibilities", page)
            self.assertIn("Psoriasis, Lichen Planus and Related Conditions", page)

    def test_taxonomy_completeness_and_letterbox_occlusion_alignment(self):
        from backend import edge_app
        from backend.taxonomy import BROAD_CLASSES, BROAD_TO_FINE, CONDITION_GUIDES, FINE_CLASSES, FINE_TO_BROAD

        for broad in BROAD_CLASSES:
            self.assertIn(broad, CONDITION_GUIDES)
            self.assertGreaterEqual(len(BROAD_TO_FINE[broad]), 1)
        for fine in FINE_CLASSES:
            self.assertIn(fine, CONDITION_GUIDES)
            self.assertIn(FINE_TO_BROAD[fine], BROAD_CLASSES)

        # Verify pre-letterbox 3x3 occlusion views align to original non-square image geometry
        widescreen = np.full((180, 360, 3), 200, dtype=np.uint8)
        widescreen[60:120, 120:240] = (30, 50, 210)  # colored center cell (row=1, col=1)
        views = edge_app._masked_bgr_views(widescreen, 128, mode="letterbox")
        self.assertEqual(views.shape, (9, 128, 128, 3))
        # Masking center cell (idx=4) should remove the red patch, producing the largest difference vs unmasked
        base = edge_app.prepare(widescreen, 128, mode="letterbox")
        diffs = np.mean(np.abs(views - base[None]), axis=(1, 2, 3))
        self.assertEqual(int(np.argmax(diffs)), 4)

    def test_hot_reload_only_fires_when_model_signature_changes(self):
        from backend import edge_app

        with patch.object(edge_app, "reload_model") as mock_reload:
            edge_app._ensure_current_model()
            mock_reload.assert_not_called()
            with patch.object(edge_app, "_bundle_signature", return_value=("trained", 123, 456)):
                edge_app._ensure_current_model()
                mock_reload.assert_called_once()
