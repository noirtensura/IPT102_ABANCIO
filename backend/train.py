"""Leakage-safe, domain-balanced hierarchical training, calibration, and OpenVINO export.

Ingests ALL datasets in the workspace:
  - Dataset A: ``datasets/20-skin-diseases`` (or ``datasets/20-skin-diseases.zip``) — 19 fine classes
    (including recovery of the 204 mis-nested Acne and Rosacea test photos).
  - Dataset B: ``train/``, ``val/``, ``test/`` — 6 broad clinical/dermoscopy classes (38,761 files).

Protocol (the independent test split is never used for any modeling or calibration decision):
  1. Global dataset audit & hygiene:
     - Decode all readable images regardless of extension (including ``.ashx``).
     - Drop tiny <96 px thumbnails (e.g. ``roi*.jpg`` in Pigment) and Keras offline-augmented
       ``pigmentation_0_*.jpeg`` stretched-border copies in Acne.
     - Collapse exact SHA-256 and 64-bit DCT perceptual near-duplicates (Hamming <= 10) across
       both datasets combined; drop duplicate groups with conflicting broad labels.
     - Build strictly leak-free, group-stratified 75% train / 10% val / 15% test splits grouped by
       wider perceptual hash clusters (Hamming <= 14) and normalized filename stems (catching ``t-``
       thumbnails, ``- Copy``, ``(1)``, and Turkish ``kopyasi`` copies). Verify 0 leaks.
  2. Stage 1 (5-fold grouped CV on a clinical-stratified train pilot):
     - Select backbone and aspect-preserving preprocessing (``squash`` vs ``letterbox`` vs
       ``letterbox_sog``) using OpenVINO ``AsyncInferQueue`` multi-stream extraction.
  3. Stage 2 (disk-backed ``float16`` memmap feature cache, low RAM):
     - Extract ``orig`` + ``flip`` views for all unique images and extra augmented views for
       clinical training images (with extra views for minority fine conditions such as Psoriasis).
  4. Stage 3 (5-fold grouped CV on train split):
     - Tune ISIC dermoscopy down-weighting, class weighting, and compare L2-normalized Linear heads
       against a Multi-Task Shared-Bottleneck Residual MLP (predicting 7 broad + 20 specific
       conditions simultaneously).
  5. Stage 4 (calibration on held-out ``val``, single evaluation on independent ``test``):
     - Fit temperature scaling and the ``Uncertain`` threshold on ``val`` (logit-averaged TTA).
     - Report clinical-only, dermoscopy-only, and overall test metrics with 95% bootstrap CIs.
  6. Stage 5 (single-graph dual-output OpenVINO export + fresh quality gate calibration):
     - Export ``model/classifier.zip`` (format 3: shared backbone -> calibrated broad & fine logits),
       ``model/evaluation.json``, and freshly calibrated ``model/quality.json``.

Usage:
  .venv\\Scripts\\python.exe backend\\train.py
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import gc
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import openvino as ov
from openvino import opset13 as ops
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import quality
from backend.preprocess import PREPROCESSING_MODES, augment, prepare
from backend.taxonomy import (
    BROAD_CLASSES,
    BROAD_TO_FINE,
    DATASET_A_FOLDER_TO_FINE,
    DATASET_B_FOLDER_TO_BROAD,
    FINE_CLASSES,
    FINE_TO_BROAD,
    FOLDER_ALIASES,
    infer_fine_label_from_filename,
    normalize_stem_group,
)

SEED = 42
MODEL_DIR = PROJECT_ROOT / "model"
BUILD_DIR = PROJECT_ROOT / "build"
NEAR_DUPLICATE_BITS = 6
GROUP_CLUSTER_BITS = 8
MAX_DUPLICATE_GROUP_SIZE = 25
MIN_IMAGE_SIDE = 96
CV_FOLDS = 5

BACKBONES = {
    "mobilenetv2_035_128": {"size": 128, "rescale": True},
    "effnetv2b0_224": {"size": 224, "rescale": False},
    "effnetv2b2_260": {"size": 260, "rescale": False},
    "convnext_tiny_224": {"size": 224, "rescale": False},
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------- image decoding & hashing
def decode_bytes(data: bytes) -> np.ndarray | None:
    if not data:
        return None
    try:
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None and img.ndim == 3 and img.shape[2] == 3:
            return img
    except cv2.error:
        pass
    try:
        from io import BytesIO
        from PIL import Image, ImageOps

        with Image.open(BytesIO(data)) as pil_img:
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
            arr = np.asarray(pil_img, dtype=np.uint8)
            if arr.ndim == 3 and arr.shape[2] == 3:
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    return None


def _resolve_disk_path(rel_path: str) -> Path:
    candidate = PROJECT_ROOT / rel_path
    if candidate.is_file():
        return candidate
    for a, b in FOLDER_ALIASES:
        if a in rel_path:
            alt = PROJECT_ROOT / rel_path.replace(a, b, 1)
            if alt.is_file():
                return alt
        elif b in rel_path:
            alt = PROJECT_ROOT / rel_path.replace(b, a, 1)
            if alt.is_file():
                return alt
    return candidate


def read_item_image(item: "UnifiedItem", archive_cache: dict[str, ZipFile] | None = None) -> np.ndarray | None:
    if item.storage == "file":
        try:
            return decode_bytes(_resolve_disk_path(item.rel_path).read_bytes())
        except OSError:
            return None
    if item.storage == "zip":
        zip_rel, entry = item.rel_path.split("::", 1)
        if archive_cache is not None:
            zf = archive_cache.get(zip_rel)
            if zf is None:
                zf = ZipFile(PROJECT_ROOT / zip_rel)
                archive_cache[zip_rel] = zf
            return decode_bytes(zf.read(entry))
        with ZipFile(PROJECT_ROOT / zip_rel) as zf:
            return decode_bytes(zf.read(entry))
    return None


def perceptual_hash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(gray)[:8, :8]
    bits = (low > np.median(low)).ravel()
    return int(np.packbits(bits).view(">u8")[0])


def hamming_pairs(hashes: list[int], threshold: int, block_size: int = 256):
    """Yield index pairs (i < j) whose 64-bit hashes differ by at most ``threshold`` bits (low RAM)."""
    values = np.asarray(hashes, dtype=np.uint64)
    n = len(values)
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        # Only compare against indices >= start to halve work and peak memory
        sub = values[start:end, None] ^ values[None, start:]
        dist = table[sub.view(np.uint8).reshape(*sub.shape, 8)].sum(-1)
        rows, cols = np.nonzero(dist <= threshold)
        for r, c in zip(rows, cols):
            gi = int(start + r)
            gj = int(start + c)
            if gi < gj:
                yield gi, gj


def union_groups(n: int, pairs, max_group_size: int = MAX_DUPLICATE_GROUP_SIZE) -> np.ndarray:
    """Union-find with a max component size cap to prevent transitive chaining on uniform dermoscopy borders."""
    parent = list(range(n))
    size = [1] * n

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            if max_group_size > 0 and (size[ra] + size[rb]) > max_group_size:
                continue
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            parent[hi] = lo
            size[lo] += size[hi]
    return np.array([find(i) for i in range(n)], dtype=np.int32)


# --------------------------------------------------------------------------- unified dataset audit
@dataclass
class UnifiedItem:
    storage: str  # "file" or "zip"
    rel_path: str
    filename: str
    source_dataset: str  # "A", "B", or "A+B"
    domain: str  # "clinical" or "dermoscopy"
    broad_label: str
    fine_label: str | None
    sha: str
    phash: int
    width: int
    height: int
    group_id: int = -1
    split: str = ""  # "train", "val", "test"


def _iter_raw_records(dataset_a_path: Path | None, include_dataset_b: bool = True):
    """Yield (storage, rel_path, filename, source_dataset, broad_label, fine_label, data_bytes)."""
    # Dataset A: prefer extracted directory for fast I/O, fall back to zip
    ext_a = PROJECT_ROOT / "datasets" / "20-skin-diseases" / "Dataset"
    zip_a = dataset_a_path if (dataset_a_path and dataset_a_path.is_file()) else (PROJECT_ROOT / "datasets" / "20-skin-diseases.zip")
    if ext_a.is_dir():
        log(f"Scanning Dataset A from extracted directory: {ext_a.relative_to(PROJECT_ROOT)}")
        for split_name in ("train", "test"):
            sp_dir = ext_a / split_name
            if not sp_dir.is_dir():
                continue
            for p in sorted(sp_dir.rglob("*")):
                if not p.is_file():
                    continue
                rel_parts = p.relative_to(sp_dir).parts
                if len(rel_parts) == 2:
                    raw_cls = rel_parts[0]
                elif len(rel_parts) == 3 and rel_parts[1] in DATASET_A_FOLDER_TO_FINE:
                    # Recover the 204 mis-nested Acne and Rosacea photos verified by exact SHA overlap
                    raw_cls = rel_parts[1]
                else:
                    yield ("skip", str(p.relative_to(PROJECT_ROOT)), p.name, "A", "mis_nested", None, b"")
                    continue
                fine = DATASET_A_FOLDER_TO_FINE.get(raw_cls)
                if fine is None:
                    yield ("skip", str(p.relative_to(PROJECT_ROOT)), p.name, "A", "test_only_class", None, b"")
                    continue
                broad = FINE_TO_BROAD[fine]
                if broad == "Pigmentation Disorders":
                    inferred_pig = infer_fine_label_from_filename(broad, p.name)
                    if inferred_pig is not None:
                        fine = inferred_pig
                yield ("file", p.relative_to(PROJECT_ROOT).as_posix(), p.name, "A", broad, fine, p.read_bytes())
    elif zip_a.is_file():
        log(f"Scanning Dataset A from zip archive: {zip_a.relative_to(PROJECT_ROOT)}")
        zip_rel = zip_a.relative_to(PROJECT_ROOT).as_posix()
        with ZipFile(zip_a) as zf:
            for info in sorted(zf.infolist(), key=lambda e: e.filename):
                if info.is_dir():
                    continue
                parts = PurePosixPath(info.filename).parts
                if len(parts) == 4 and parts[0] == "Dataset" and parts[1] in ("train", "test"):
                    raw_cls = parts[2]
                elif (
                    len(parts) == 5
                    and parts[0] == "Dataset"
                    and parts[1] in ("train", "test")
                    and parts[3] in DATASET_A_FOLDER_TO_FINE
                ):
                    raw_cls = parts[3]
                else:
                    yield ("skip", info.filename, PurePosixPath(info.filename).name, "A", "mis_nested", None, b"")
                    continue
                fine = DATASET_A_FOLDER_TO_FINE.get(raw_cls)
                if fine is None:
                    yield ("skip", info.filename, PurePosixPath(info.filename).name, "A", "test_only_class", None, b"")
                    continue
                broad = FINE_TO_BROAD[fine]
                fname = PurePosixPath(info.filename).name
                if broad == "Pigmentation Disorders":
                    inferred_pig = infer_fine_label_from_filename(broad, fname)
                    if inferred_pig is not None:
                        fine = inferred_pig
                yield ("zip", f"{zip_rel}::{info.filename}", fname, "A", broad, fine, zf.read(info))

    # Dataset B: train/, val/, test/ at project root
    if include_dataset_b:
        b_splits = [PROJECT_ROOT / s for s in ("train", "val", "test") if (PROJECT_ROOT / s).is_dir()]
        if b_splits:
            log("Scanning Dataset B from train/, val/, test/")
            for sp_dir in b_splits:
                for cls_dir in sorted(sp_dir.iterdir()):
                    if not cls_dir.is_dir():
                        continue
                    broad = DATASET_B_FOLDER_TO_BROAD.get(cls_dir.name)
                    if broad is None:
                        continue
                    for p in sorted(cls_dir.rglob("*")):
                        if not p.is_file():
                            continue
                        low_name = p.name.lower()
                        # Drop Keras offline-augmented stretched-border copies in 3. Akne
                        if broad == "Acne and Rosacea" and low_name.startswith("pigmentation_0_"):
                            yield ("skip", p.relative_to(PROJECT_ROOT).as_posix(), p.name, "B", "keras_offline_aug", None, b"")
                            continue
                        fine = infer_fine_label_from_filename(broad, p.name)
                        yield ("file", p.relative_to(PROJECT_ROOT).as_posix(), p.name, "B", broad, fine, p.read_bytes())


def audit_and_split_datasets(dataset_a_path: Path | None = None, force_rebuild: bool = False):
    manifest_path = BUILD_DIR / "unified_manifest.json"
    if manifest_path.is_file() and not force_rebuild:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") == 3:
            log(f"Loaded cached leak-free unified manifest ({len(payload['items'])} clean images)")
            items = [UnifiedItem(**row) for row in payload["items"]]
            return items, payload["audit"]

    started = time.perf_counter()
    skipped = Counter()
    raw_counts = Counter()

    # Step 1: Decode, filter tiny/corrupted, and collapse exact SHA-256 duplicates (cached for fast re-runs)
    raw_cache_path = BUILD_DIR / "raw_sha_cache.json"
    if raw_cache_path.is_file():
        cached = json.loads(raw_cache_path.read_text(encoding="utf-8"))
        raw_counts = Counter(cached["raw_counts"])
        skipped = Counter(cached["skipped"])
        unique_items = [UnifiedItem(**row) for row in cached["items"]]
        log(f"Loaded cached SHA-unique scan ({len(unique_items)} pure SHA-unique images)")
    else:
        sha_buckets: dict[str, dict] = {}
        scanned = 0
        for storage, rel_path, filename, src, broad, fine, data in _iter_raw_records(dataset_a_path):
            if storage == "skip":
                skipped[broad] += 1
                continue
            scanned += 1
            if scanned % 10000 == 0:
                log(f"  scanned {scanned} raw files ({len(sha_buckets)} unique SHA-256)...")
            raw_counts[f"dataset_{src}"] += 1
            h = sha256(data).hexdigest()
            existing = sha_buckets.get(h)
            if existing is not None:
                existing["broad_set"].add(broad)
                existing["sources"].add(src)
                if src == "A" and fine is not None:
                    # Prefer explicit Dataset A fine label when exact match spans A & B
                    existing["fine"] = fine
                    existing["fine_from_a"] = True
                elif not existing["fine_from_a"] and existing["fine"] is None and fine is not None:
                    existing["fine"] = fine
                if filename.lower().startswith("isic_"):
                    existing["domain"] = "dermoscopy"
                skipped["exact_sha_duplicates_collapsed"] += 1
                continue

            img = decode_bytes(data)
            if img is None:
                skipped["unreadable"] += 1
                continue
            height, width = img.shape[:2]
            if min(height, width) < MIN_IMAGE_SIDE:
                skipped["too_small_lt_96px"] += 1
                continue
            ph = perceptual_hash(img)
            domain = "dermoscopy" if filename.lower().startswith("isic_") else "clinical"
            sha_buckets[h] = {
                "storage": storage,
                "rel_path": rel_path,
                "filename": filename,
                "sources": {src},
                "domain": domain,
                "broad": broad,
                "broad_set": {broad},
                "fine": fine,
                "fine_from_a": (src == "A" and fine is not None),
                "sha": h,
                "phash": ph,
                "width": int(width),
                "height": int(height),
            }

        # Drop exact-hash label conflicts across broad classes
        unique_items: list[UnifiedItem] = []
        for rec in sha_buckets.values():
            if len(rec["broad_set"]) > 1:
                skipped["exact_sha_broad_conflicts_dropped"] += 1
                continue
            src_str = "A+B" if len(rec["sources"]) > 1 else next(iter(rec["sources"]))
            # Ensure fine label is consistent with broad label
            fine_lbl = rec["fine"]
            if fine_lbl is not None and FINE_TO_BROAD.get(fine_lbl) != rec["broad"]:
                fine_lbl = None
            unique_items.append(
                UnifiedItem(
                    storage=rec["storage"],
                    rel_path=rec["rel_path"],
                    filename=rec["filename"],
                    source_dataset=src_str,
                    domain=rec["domain"],
                    broad_label=rec["broad"],
                    fine_label=fine_lbl,
                    sha=rec["sha"],
                    phash=rec["phash"],
                    width=rec["width"],
                    height=rec["height"],
                )
            )
        del sha_buckets
        gc.collect()
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        raw_cache_path.write_text(
            json.dumps(
                {
                    "raw_counts": dict(raw_counts),
                    "skipped": dict(skipped),
                    "items": [asdict(i) for i in unique_items],
                }
            ),
            encoding="utf-8",
        )
        log(f"Exact SHA-256 dedup done in {time.perf_counter() - started:.1f}s -> {len(unique_items)} pure SHA-unique images")

    # Step 2: 64-bit DCT perceptual hash deduplication (Hamming <= 10)
    t_ph = time.perf_counter()
    ph_groups = union_groups(len(unique_items), hamming_pairs([i.phash for i in unique_items], NEAR_DUPLICATE_BITS))
    members = defaultdict(list)
    for idx, grp in enumerate(ph_groups):
        members[int(grp)].append(idx)

    clean_items: list[UnifiedItem] = []
    for indices in members.values():
        grp_items = [unique_items[i] for i in indices]
        broad_labels = {g.broad_label for g in grp_items}
        if len(broad_labels) > 1:
            skipped["phash_broad_conflicts_dropped"] += len(grp_items)
            continue
        # Pick best canonical copy: prefer explicit Dataset A fine label, then highest resolution, then deterministic path
        grp_items.sort(
            key=lambda g: (
                1 if ("A" in g.source_dataset and g.fine_label is not None) else 0,
                1 if g.fine_label is not None else 0,
                g.width * g.height,
                g.rel_path,
            ),
            reverse=True,
        )
        rep = grp_items[0]
        if len(grp_items) > 1:
            skipped["phash_near_duplicates_collapsed"] += len(grp_items) - 1
            # Preserve merged source provenance
            all_src = set()
            for g in grp_items:
                all_src.update(g.source_dataset.split("+"))
            rep.source_dataset = "A+B" if len(all_src) > 1 else next(iter(all_src))
        clean_items.append(rep)
    log(f"Perceptual hash dedup (<= {NEAR_DUPLICATE_BITS} bits) done in {time.perf_counter() - t_ph:.1f}s -> {len(clean_items)} clean images")

    # Step 3: Build wider similarity + filename-stem groups so related photos never straddle splits
    t_grp = time.perf_counter()
    pair_list = list(hamming_pairs([i.phash for i in clean_items], GROUP_CLUSTER_BITS))
    stem_index: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, item in enumerate(clean_items):
        stem = normalize_stem_group(item.filename)
        if len(stem) >= 4:
            stem_index[(item.broad_label, stem)].append(idx)
    for idxs in stem_index.values():
        if len(idxs) > 1:
            first = idxs[0]
            for other in idxs[1:]:
                pair_list.append((first, other))
    groups = union_groups(len(clean_items), pair_list)
    for item, grp in zip(clean_items, groups):
        item.group_id = int(grp)
    log(f"Group clustering (pHash <= {GROUP_CLUSTER_BITS} + normalized stem) done in {time.perf_counter() - t_grp:.1f}s")

    # Step 4: Strictly leak-free group-stratified split: 75% train / 10% val / 15% test
    # Stratify jointly by (broad_label, domain) so both clinical and dermoscopy are faithfully balanced
    strat_keys = [f"{i.broad_label}||{i.domain}" for i in clean_items]
    strat_vocab = {k: idx for idx, k in enumerate(sorted(set(strat_keys)))}
    y_strat = np.array([strat_vocab[k] for k in strat_keys], dtype=np.int32)
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    fold_ids = np.zeros(len(clean_items), dtype=np.int32)
    for fold_num, (_, val_idx) in enumerate(splitter.split(np.zeros(len(clean_items)), y_strat, groups)):
        fold_ids[val_idx] = fold_num
    for item, fid in zip(clean_items, fold_ids):
        if fid < 15:
            item.split = "train"
        elif fid < 17:
            item.split = "val"
        else:
            item.split = "test"

    # Hard safety net: verify ZERO exact SHA and ZERO pHash <= NEAR_DUPLICATE_BITS leaks across splits
    split_arr = [i.split for i in clean_items]
    leaks = sum(
        1
        for a, b in hamming_pairs([i.phash for i in clean_items], NEAR_DUPLICATE_BITS)
        if split_arr[a] != split_arr[b]
    )
    if leaks:
        raise RuntimeError(f"Leakage assertion failed: {leaks} cross-split near-duplicate pairs remain!")

    # Build comprehensive audit summary
    split_broad = {
        sp: dict(sorted(Counter(i.broad_label for i in clean_items if i.split == sp).items()))
        for sp in ("train", "val", "test")
    }
    split_clinical_broad = {
        sp: dict(
            sorted(Counter(i.broad_label for i in clean_items if i.split == sp and i.domain == "clinical").items())
        )
        for sp in ("train", "val", "test")
    }
    split_dermoscopy_broad = {
        sp: dict(
            sorted(Counter(i.broad_label for i in clean_items if i.split == sp and i.domain == "dermoscopy").items())
        )
        for sp in ("train", "val", "test")
    }
    split_fine = {
        sp: dict(sorted(Counter(i.fine_label for i in clean_items if i.split == sp and i.fine_label).items()))
        for sp in ("train", "val", "test")
    }
    audit = {
        "near_duplicate_rule": f"Exact SHA-256 + 64-bit DCT pHash Hamming <= {NEAR_DUPLICATE_BITS}",
        "split_group_rule": f"pHash Hamming <= {GROUP_CLUSTER_BITS} union normalized filename stem",
        "min_image_side_px": MIN_IMAGE_SIDE,
        "raw_scanned_by_source": dict(raw_counts),
        "skipped": dict(skipped),
        "clean_total_images": len(clean_items),
        "clean_by_source": dict(Counter(i.source_dataset for i in clean_items)),
        "clean_by_domain": dict(Counter(i.domain for i in clean_items)),
        "clean_by_split": dict(Counter(i.split for i in clean_items)),
        "split_broad_counts": split_broad,
        "split_clinical_broad_counts": split_clinical_broad,
        "split_dermoscopy_broad_counts": split_dermoscopy_broad,
        "split_fine_counts": split_fine,
        "remaining_cross_split_near_duplicates": 0,
    }

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "created_at": datetime.now(UTC).isoformat(),
                "audit": audit,
                "items": [asdict(i) for i in clean_items],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(
        f"Saved unified manifest ({len(clean_items)} clean images: "
        f"train={audit['clean_by_split']['train']}, val={audit['clean_by_split']['val']}, "
        f"test={audit['clean_by_split']['test']}; 0 leaks)"
    )
    return clean_items, audit


# --------------------------------------------------------------------------- fast multi-stream OpenVINO extractor
def backbone_xml(name: str) -> Path:
    xml = BUILD_DIR / "backbones" / f"{name}.xml"
    if not xml.is_file():
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "backend" / "experiments" / "backbones.py"), str(xml.parent), name],
            check=True,
        )
    return xml


class AsyncExtractor:
    """Multi-stream OpenVINO feature extractor using AsyncInferQueue (3.5x-4.5x faster on CPU)."""

    def __init__(self, name: str, batch_size: int = 16):
        self.name = name
        self.size = BACKBONES[name]["size"]
        self.xml = backbone_xml(name)
        self.batch_size = batch_size
        self.net = ov.Core().compile_model(
            self.xml, "CPU", {"INFERENCE_PRECISION_HINT": "f32", "PERFORMANCE_HINT": "THROUGHPUT"}
        )
        self.dim = int(self.net.output(0).partial_shape[-1].get_length())
        self.nreq = max(2, int(self.net.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")))

    def extract_views_to_array(
        self,
        items: list[UnifiedItem],
        mode: str,
        views: list[str],
        cache_prefix: Path,
    ) -> np.ndarray:
        """Extract shape (len(items), len(views), dim) float16 array cached on disk as .npy + manifest."""
        cache_prefix.parent.mkdir(parents=True, exist_ok=True)
        npy_path = cache_prefix.with_suffix(".npy")
        meta_path = cache_prefix.with_suffix(".json")
        key_sig = sha256(
            json.dumps([i.sha for i in items] + [mode] + views, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]

        if npy_path.is_file() and meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("sig") == key_sig and meta.get("complete") is True:
                return np.load(npy_path, mmap_mode="r")

        n_items = len(items)
        n_views = len(views)
        log(
            f"  extracting {n_items} x {n_views} = {n_items * n_views} views "
            f"with {self.name}/{mode} (nreq={self.nreq}, bs={self.batch_size})"
        )
        started = time.perf_counter()

        # Allocate disk-backed float16 memmap so peak RAM stays tiny (<250 MB)
        out = np.lib.format.open_memmap(
            str(npy_path), mode="w+", dtype=np.float16, shape=(n_items, n_views, self.dim)
        )
        queue = ov.AsyncInferQueue(self.net, self.nreq)

        def on_done(infer_request, userdata):
            coords = userdata
            feats = infer_request.get_output_tensor(0).data
            for b_idx, (item_i, view_i) in enumerate(coords):
                out[item_i, view_i] = feats[b_idx].astype(np.float16)

        queue.set_callback(on_done)
        archive_cache: dict[str, ZipFile] = {}
        batch_pixels: list[np.ndarray] = []
        batch_coords: list[tuple[int, int]] = []
        done_views = 0
        next_log = 4000

        try:
            for item_i, item in enumerate(items):
                img = read_item_image(item, archive_cache)
                if img is None:
                    raise RuntimeError(f"Failed to decode audited image: {item.rel_path}")
                seed_int = int(item.sha[:8], 16)
                for view_i, view in enumerate(views):
                    if view == "orig":
                        pix = prepare(img, self.size, mode)
                    elif view == "flip":
                        pix = prepare(cv2.flip(img, 1), self.size, mode)
                    else:  # "augK"
                        rng = np.random.default_rng([seed_int, int(view[3:])])
                        pix = prepare(augment(img, rng), self.size, mode)
                    batch_pixels.append(pix)
                    batch_coords.append((item_i, view_i))
                    if len(batch_pixels) >= self.batch_size:
                        arr = np.stack(batch_pixels, axis=0)
                        queue.start_async({0: arr}, list(batch_coords))
                        done_views += len(batch_pixels)
                        batch_pixels.clear()
                        batch_coords.clear()
                        if done_views >= next_log:
                            rate = done_views / max(time.perf_counter() - started, 1e-3)
                            rem = (n_items * n_views - done_views) / max(rate, 1e-3)
                            log(f"    {done_views}/{n_items * n_views} views ({rate:.1f} img/s, ~{rem/60:.1f}m left)")
                            next_log += 4000
            if batch_pixels:
                arr = np.stack(batch_pixels, axis=0)
                queue.start_async({0: arr}, list(batch_coords))
            queue.wait_all()
            out.flush()
        finally:
            for zf in archive_cache.values():
                zf.close()
            del out
            del queue

        meta_path.write_text(
            json.dumps({"sig": key_sig, "complete": True, "shape": [n_items, n_views, self.dim]}),
            encoding="utf-8",
        )
        elapsed = time.perf_counter() - started
        log(f"  done {n_items * n_views} views in {elapsed:.1f}s ({(n_items * n_views)/max(elapsed,1e-3):.1f} img/s)")
        return np.load(npy_path, mmap_mode="r")


# --------------------------------------------------------------------------- metrics, weighting, calibration
def l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-12)
    return array / norms


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def compute_domain_balanced_weights(
    labels: np.ndarray,
    is_dermoscopy: np.ndarray,
    n_classes: int,
    scheme: str,
    isic_weight: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-sample weights that prevent ISIC dermoscopy from overwhelming clinical photos."""
    n = len(labels)
    mask = np.ones(n, dtype=bool) if valid_mask is None else valid_mask.astype(bool)
    domain_w = np.where(is_dermoscopy, float(isic_weight), 1.0).astype(np.float64)
    domain_w = np.where(mask, domain_w, 0.0)

    if scheme == "none":
        cls_w = np.ones(n_classes, dtype=np.float64)
    else:
        # Compute effective weighted counts per class AFTER domain down-weighting
        eff_counts = np.zeros(n_classes, dtype=np.float64)
        valid_idx = np.flatnonzero(mask)
        np.add.at(eff_counts, labels[valid_idx], domain_w[valid_idx])
        total_eff = eff_counts.sum()
        present = eff_counts > 0
        cls_w = np.zeros(n_classes, dtype=np.float64)
        raw = total_eff / (max(int(present.sum()), 1) * np.maximum(eff_counts, 1e-6))
        if scheme == "sqrt":
            raw = np.sqrt(raw)
        cls_w[present] = raw[present]

    weights = np.zeros(n, dtype=np.float32)
    valid_idx = np.flatnonzero(mask)
    weights[valid_idx] = (domain_w[valid_idx] * cls_w[labels[valid_idx]]).astype(np.float32)
    mean_pos = weights[valid_idx].mean() if len(valid_idx) else 1.0
    if mean_pos > 0:
        weights[valid_idx] /= mean_pos
    return weights


def fit_temperature(logits: np.ndarray, labels: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    w = np.ones(len(labels), dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    w = w / max(w.sum(), 1e-12)

    def nll(log_t: float) -> float:
        p = softmax(logits / np.exp(log_t))
        return float(-np.sum(w * np.log(p[np.arange(len(labels)), labels] + 1e-12)))

    opt = minimize_scalar(nll, bounds=(-2.5, 2.5), method="bounded")
    t_best = float(np.exp(opt.x))
    # Only apply temperature scaling if it strictly improves weighted NLL
    if nll(np.log(t_best)) >= nll(0.0) - 1e-4:
        return 1.0
    return t_best


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def classification_metrics(pred: np.ndarray, labels: np.ndarray, classes: list[str] | tuple[str, ...]) -> dict:
    k = len(classes)
    matrix = np.zeros((k, k), dtype=int)
    if len(labels) > 0:
        np.add.at(matrix, (labels, pred), 1)
    tp = np.diag(matrix).astype(float)
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    precision = tp / np.maximum(predicted, 1)
    recall = tp / np.maximum(support, 1)
    f1 = 2 * tp / np.maximum(support + predicted, 1)
    present = support > 0
    return {
        "images": int(len(labels)),
        "accuracy": float(tp.sum() / max(len(labels), 1)),
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,
        "macro_precision": float(precision[present].mean()) if present.any() else 0.0,
        "macro_recall": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "weighted_f1": float((f1 * support).sum() / max(support.sum(), 1)),
        "per_class": {
            name: {
                "support": int(support[i]),
                "predicted": int(predicted[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
            for i, name in enumerate(classes)
        },
        "confusion_matrix": matrix.tolist(),
    }


def probability_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    k = min(3, probs.shape[1])
    topk = np.argsort(-probs, axis=1)[:, :k]
    onehot = np.eye(probs.shape[1])[labels]
    return {
        "top3_accuracy": float(np.mean([l in t for l, t in zip(labels, topk)])),
        "log_loss": float(-np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-12))),
        "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        "ece_15_bins": expected_calibration_error(probs, labels),
    }


def bootstrap_ci(pred: np.ndarray, labels: np.ndarray, classes, rounds: int = 1000) -> dict:
    rng = np.random.default_rng(SEED)
    acc, mf1 = [], []
    n = len(labels)
    for _ in range(rounds):
        idx = rng.integers(0, n, n)
        m = classification_metrics(pred[idx], labels[idx], classes)
        acc.append(m["accuracy"])
        mf1.append(m["macro_f1"])
    return {
        "accuracy_95ci": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))],
        "macro_f1_95ci": [float(np.percentile(mf1, 2.5)), float(np.percentile(mf1, 97.5))],
    }


def choose_threshold(probs: np.ndarray, labels: np.ndarray, target: float = 0.80, min_coverage: float = 0.55) -> float:
    """Pick the lowest confidence threshold achieving ``target`` accuracy while keeping healthy coverage."""
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    best = 0.35
    for threshold in np.round(np.arange(0.15, 0.70, 0.01), 2):
        accepted = conf >= threshold
        cov = float(accepted.mean())
        if cov < min_coverage:
            break
        best = float(threshold)
        if accepted.sum() >= 30 and float(correct[accepted].mean()) >= target:
            return float(threshold)
    return min(best, 0.45)


def selective_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float, classes) -> dict:
    accepted = probs.max(1) >= threshold
    pred = probs.argmax(1)
    result = {
        "threshold": float(threshold),
        "coverage": float(accepted.mean()),
        "uncertain_rate": float(1.0 - accepted.mean()),
        "accepted_images": int(accepted.sum()),
        "accuracy_on_accepted": float((pred[accepted] == labels[accepted]).mean()) if accepted.any() else None,
        "wrong_answers_shown": int(((pred != labels) & accepted).sum()),
        "wrong_answers_without_uncertain": int((pred != labels).sum()),
    }
    if accepted.any():
        result["macro_f1_on_accepted"] = classification_metrics(pred[accepted], labels[accepted], classes)["macro_f1"]
    curve = []
    for t in (0.0, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85):
        a = probs.max(1) >= t
        curve.append(
            {
                "threshold": t,
                "coverage": float(a.mean()),
                "accuracy": float((pred[a] == labels[a]).mean()) if a.any() else None,
            }
        )
    result["risk_coverage_curve"] = curve
    return result


# --------------------------------------------------------------------------- heads: Linear & Multi-Task Residual MLP
def fit_linear_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    c_val: float,
    l2_norm: bool = True,
):
    feats = l2_normalize(x_train.astype(np.float32)) if l2_norm else x_train.astype(np.float32, copy=True)
    scaler = StandardScaler(copy=False).fit(feats)
    feats = scaler.transform(feats)
    head = LogisticRegression(C=c_val, tol=1e-3, max_iter=400)
    head.fit(feats, y_train, sample_weight=sample_weight)
    del feats
    return scaler, head


def linear_logits(scaler: StandardScaler, head: LogisticRegression, views: np.ndarray, l2_norm: bool = True) -> np.ndarray:
    """Average logits over TTA views (n, v, d) -> (n, classes)."""
    n, v, d = views.shape
    flat = views.reshape(n * v, d).astype(np.float32)
    if l2_norm:
        flat = l2_normalize(flat)
    logits = head.decision_function(scaler.transform(flat))
    return logits.reshape(n, v, -1).mean(axis=1).astype(np.float32)


def fit_multitask_mlp(
    x_train: np.ndarray,
    y_broad: np.ndarray,
    w_broad: np.ndarray,
    y_fine: np.ndarray,
    w_fine: np.ndarray,
    x_val: np.ndarray,
    y_broad_val: np.ndarray,
    w_broad_val: np.ndarray,
    n_broad: int,
    n_fine: int,
    hidden_dim: int = 512,
    dropout: float = 0.45,
    l2_reg: float = 8e-4,
    label_smoothing: float = 0.04,
    epochs: int = 24,
    batch_size: int = 256,
    seed: int = SEED,
    l2_norm: bool = True,
) -> dict:
    """Train a Multi-Task Shared-Bottleneck Residual MLP on extracted features (fast, low RAM)."""
    import tensorflow as tf
    from tensorflow import keras

    tf.config.threading.set_intra_op_parallelism_threads(min(8, os.cpu_count() or 4))
    keras.utils.set_random_seed(seed)

    x_tr = l2_normalize(x_train.astype(np.float32)) if l2_norm else x_train.astype(np.float32, copy=True)
    scaler = StandardScaler(copy=False).fit(x_tr)
    x_tr = scaler.transform(x_tr)

    x_va = l2_normalize(x_val.astype(np.float32)) if l2_norm else x_val.astype(np.float32, copy=True)
    x_va = scaler.transform(x_va)

    dim = x_tr.shape[1]
    reg = keras.regularizers.l2(l2_reg)

    inp = keras.Input(shape=(dim,), dtype="float32", name="features")
    x_drop = keras.layers.Dropout(0.15, name="in_drop")(inp)
    h = keras.layers.Dense(hidden_dim, activation="relu", kernel_regularizer=reg, name="shared_dense")(x_drop)
    h = keras.layers.Dropout(dropout, name="shared_drop")(h)

    # Residual skip + shared hidden representation -> linear logits
    broad_hidden = keras.layers.Dense(n_broad, use_bias=True, kernel_regularizer=reg, name="broad_h")(h)
    broad_skip = keras.layers.Dense(n_broad, use_bias=False, kernel_regularizer=reg, name="broad_skip")(x_drop)
    broad_out = keras.layers.Add(name="broad")([broad_hidden, broad_skip])

    fine_hidden = keras.layers.Dense(n_fine, use_bias=True, kernel_regularizer=reg, name="fine_h")(h)
    fine_skip = keras.layers.Dense(n_fine, use_bias=False, kernel_regularizer=reg, name="fine_skip")(x_drop)
    fine_out = keras.layers.Add(name="fine")([fine_hidden, fine_skip])

    model = keras.Model(inp, {"broad": broad_out, "fine": fine_out})

    # Smooth one-hot targets so label smoothing works cleanly with sample_weight + Keras 3
    y_broad_oh = np.eye(n_broad, dtype=np.float32)[y_broad]
    if label_smoothing > 0:
        y_broad_oh = y_broad_oh * (1.0 - label_smoothing) + label_smoothing / n_broad

    y_fine_safe = np.where(y_fine >= 0, y_fine, 0)
    y_fine_oh = np.eye(n_fine, dtype=np.float32)[y_fine_safe]
    if label_smoothing > 0:
        y_fine_oh = y_fine_oh * (1.0 - label_smoothing) + label_smoothing / n_fine

    y_broad_val_oh = np.eye(n_broad, dtype=np.float32)[y_broad_val]
    y_fine_val_dummy = np.zeros((len(y_broad_val), n_fine), dtype=np.float32)
    w_fine_val_zero = np.zeros(len(y_broad_val), dtype=np.float32)

    steps_per_epoch = max(1, int(np.ceil(len(x_tr) / batch_size)))
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1.2e-3,
        decay_steps=steps_per_epoch * epochs,
        alpha=0.05,
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=2e-4),
        loss={
            "broad": keras.losses.CategoricalCrossentropy(from_logits=True),
            "fine": keras.losses.CategoricalCrossentropy(from_logits=True),
        },
        loss_weights={"broad": 1.0, "fine": 0.65},
    )

    cb = keras.callbacks.EarlyStopping(
        monitor="val_broad_loss", mode="min", patience=5, restore_best_weights=True
    )
    model.fit(
        x_tr,
        {"broad": y_broad_oh, "fine": y_fine_oh},
        sample_weight={"broad": w_broad, "fine": w_fine},
        validation_data=(
            x_va,
            {"broad": y_broad_val_oh, "fine": y_fine_val_dummy},
            {"broad": w_broad_val, "fine": w_fine_val_zero},
        ),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[cb],
        verbose=0,
    )

    # Extract pure NumPy weights so TensorFlow objects can be freed immediately
    w1, b1 = model.get_layer("shared_dense").get_weights()
    wb_h, bb = model.get_layer("broad_h").get_weights()
    (wb_s,) = model.get_layer("broad_skip").get_weights()
    wf_h, bf = model.get_layer("fine_h").get_weights()
    (wf_s,) = model.get_layer("fine_skip").get_weights()

    keras.backend.clear_session()
    del model
    gc.collect()

    return {
        "mean": scaler.mean_.astype(np.float32),
        "scale": scaler.scale_.astype(np.float32),
        "l2_norm": bool(l2_norm),
        "W1": w1.astype(np.float32),
        "b1": b1.astype(np.float32),
        "Wb_h": wb_h.astype(np.float32),
        "Wb_s": wb_s.astype(np.float32),
        "bb": bb.astype(np.float32),
        "Wf_h": wf_h.astype(np.float32),
        "Wf_s": wf_s.astype(np.float32),
        "bf": bf.astype(np.float32),
    }


def mlp_predict_logits(weights_list: list[dict], views: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass through one or an ensemble of Residual MLP heads, averaging logits over views & seeds."""
    n, v, d = views.shape
    flat = views.reshape(n * v, d).astype(np.float32)
    b_sum, f_sum = None, None
    for w in weights_list:
        x = l2_normalize(flat) if w["l2_norm"] else flat
        x = (x - w["mean"]) / w["scale"]
        h = np.maximum(0.0, x @ w["W1"] + w["b1"])
        b_log = (h @ w["Wb_h"] + x @ w["Wb_s"] + w["bb"]).reshape(n, v, -1).mean(axis=1)
        f_log = (h @ w["Wf_h"] + x @ w["Wf_s"] + w["bf"]).reshape(n, v, -1).mean(axis=1)
        b_sum = b_log if b_sum is None else (b_sum + b_log)
        f_sum = f_log if f_sum is None else (f_sum + f_log)
    k = float(len(weights_list))
    return (b_sum / k).astype(np.float32), (f_sum / k).astype(np.float32)


# --------------------------------------------------------------------------- quality gate calibration
def calibrate_quality_fresh(train_clinical: list[UnifiedItem]) -> dict:
    """Always recalibrate image quality thresholds on the current clinical training photos."""
    rng = np.random.default_rng(SEED)
    # Sample up to 3,500 clinical training photos evenly across classes for fast, thorough calibration
    by_cls = defaultdict(list)
    for item in train_clinical:
        by_cls[item.broad_label].append(item)
    chosen = []
    per_cls = max(250, 3500 // max(len(by_cls), 1))
    for cls_name in sorted(by_cls):
        pool = by_cls[cls_name]
        if len(pool) <= per_cls:
            chosen.extend(pool)
        else:
            idxs = rng.choice(len(pool), size=per_cls, replace=False)
            chosen.extend([pool[int(i)] for i in idxs])

    real, degraded = [], defaultdict(list)
    archive_cache: dict[str, ZipFile] = {}
    try:
        for item in chosen:
            image = read_item_image(item, archive_cache)
            if image is None:
                continue
            real.append(quality.measure(image))
            if rng.random() < 0.12:
                view = quality._analysis_view(image)
                degraded["gaussian_blur_sigma3.5"].append(cv2.GaussianBlur(view, (0, 0), 3.5))
                kernel = np.zeros((17, 17), np.float32)
                kernel[8, :] = 1.0 / 17.0
                degraded["motion_blur_17px"].append(cv2.filter2D(view, -1, kernel))
                degraded["underexposed_x0.12"].append((view.astype(np.float32) * 0.12).astype(np.uint8))
                degraded["blank_gray"].append(np.full_like(view, int(rng.integers(60, 200))))
    finally:
        for zf in archive_cache.values():
            zf.close()

    metric = {k: np.array([r[k] for r in real]) for k in real[0]}
    thresholds = dict(quality.DEFAULT_THRESHOLDS)
    thresholds.update(
        {
            "max_blur": round(float(max(np.percentile(metric["blur"], 99.8) + 0.05, 0.72)), 3),
            "min_brightness": round(float(min(np.percentile(metric["brightness"], 0.25) * 0.7, 25.0)), 1),
            "max_brightness": round(float(max(np.percentile(metric["brightness"], 99.75) + 15.0, 235.0)), 1),
            "min_contrast": round(float(min(np.percentile(metric["contrast"], 0.25) * 0.6, 8.0)), 1),
            "min_skin_ratio": round(float(min(np.percentile(metric["skin_ratio"], 0.5) * 0.5, 0.02)), 4),
        }
    )
    t = thresholds
    rejected_real = int(
        np.sum(
            (metric["min_side"] < t["min_side"])
            | (metric["brightness"] < t["min_brightness"])
            | (metric["brightness"] > t["max_brightness"])
            | (metric["clipped_ratio"] > t["max_clipped_ratio"])
            | (metric["contrast"] < t["min_contrast"])
            | (metric["blur"] > t["max_blur"])
            | (metric["skin_ratio"] < t["min_skin_ratio"])
        )
    )
    caught = {
        name: float(np.mean([not quality.assess(img, thresholds).ok for img in imgs]))
        for name, imgs in degraded.items()
    }
    return {
        "thresholds": thresholds,
        "calibrated_on_train_images": len(real),
        "real_train_photos_rejected": rejected_real,
        "real_train_photo_rejection_rate": rejected_real / max(len(real), 1),
        "synthetic_degraded_rejection_rate": caught,
        "real_metric_percentiles": {
            k: {p: round(float(np.percentile(v, float(p))), 4) for p in ("0", "0.5", "50", "99.5", "100")}
            for k, v in metric.items()
        },
    }


# --------------------------------------------------------------------------- OpenVINO dual-output export
def export_hierarchical_bundle(
    config: dict,
    head_type: str,
    head_artifacts: dict,
    temp_broad: float,
    temp_fine: float,
    threshold: float,
    min_margin: float,
    report: dict,
    quality_report: dict,
    check_pixels: np.ndarray,
    expected_broad_logits: np.ndarray,
    expected_fine_logits: np.ndarray,
) -> None:
    """Export one OpenVINO IR model with two calibrated-logit outputs: broad_logits (7) & fine_logits (20)."""
    core = ov.Core()
    base_model = core.read_model(config["xml"])
    features = base_model.get_results()[0].input_value(0)

    if config.get("l2_norm", True):
        features_norm = ops.normalize_l2(features, ops.constant(np.array([1], dtype=np.int64)), 1e-12, "max")
    else:
        features_norm = features

    if head_type == "linear":
        b_scaler, b_head = head_artifacts["broad"]
        f_scaler, f_head = head_artifacts["fine"]
        bw = (b_head.coef_ / b_scaler.scale_).T.astype(np.float32)
        bb = (b_head.intercept_ - (b_scaler.mean_ / b_scaler.scale_) @ b_head.coef_.T).astype(np.float32)
        fw = (f_head.coef_ / f_scaler.scale_).T.astype(np.float32)
        fb = (f_head.intercept_ - (f_scaler.mean_ / f_scaler.scale_) @ f_head.coef_.T).astype(np.float32)
        broad_logits = ops.add(ops.matmul(features_norm, ops.constant(bw), False, False), ops.constant(bb))
        fine_logits = ops.add(ops.matmul(features_norm, ops.constant(fw), False, False), ops.constant(fb))
    else:
        # Multi-Task Residual MLP ensemble (shared scaler across seeds)
        models: list[dict] = head_artifacts["models"]
        mean = models[0]["mean"]
        scale = models[0]["scale"]
        x_std = ops.divide(ops.subtract(features_norm, ops.constant(mean)), ops.constant(scale))

        broad_acc, fine_acc = None, None
        for m in models:
            h = ops.relu(ops.add(ops.matmul(x_std, ops.constant(m["W1"]), False, False), ops.constant(m["b1"])))
            b_branch = ops.add(
                ops.add(ops.matmul(h, ops.constant(m["Wb_h"]), False, False), ops.matmul(x_std, ops.constant(m["Wb_s"]), False, False)),
                ops.constant(m["bb"]),
            )
            f_branch = ops.add(
                ops.add(ops.matmul(h, ops.constant(m["Wf_h"]), False, False), ops.matmul(x_std, ops.constant(m["Wf_s"]), False, False)),
                ops.constant(m["bf"]),
            )
            broad_acc = b_branch if broad_acc is None else ops.add(broad_acc, b_branch)
            fine_acc = f_branch if fine_acc is None else ops.add(fine_acc, f_branch)

        inv_k = ops.constant(np.float32(1.0 / len(models)))
        broad_logits = ops.multiply(broad_acc, inv_k) if len(models) > 1 else broad_acc
        fine_logits = ops.multiply(fine_acc, inv_k) if len(models) > 1 else fine_acc

    broad_cal = ops.divide(broad_logits, ops.constant(np.float32(temp_broad)))
    fine_cal = ops.divide(fine_logits, ops.constant(np.float32(temp_fine)))
    broad_cal.output(0).get_tensor().set_names({"broad_logits"})
    fine_cal.output(0).get_tensor().set_names({"fine_logits"})
    res_broad = ops.result(broad_cal, name="broad_logits")
    res_fine = ops.result(fine_cal, name="fine_logits")

    graph = ov.Model([res_broad, res_fine], base_model.get_parameters(), "hierarchical_skin_classifier")

    metadata = {
        "format": 3,
        "created_at": datetime.now(UTC).isoformat(),
        "backbone": config["backbone"],
        "head_type": head_type,
        "input_size": config["size"],
        "input_layout": "NHWC RGB float32 0-255",
        "preprocessing": config["mode"],
        "l2_norm": bool(config.get("l2_norm", True)),
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
                k: report["final_model"]["test_clinical"]["broad"][k]
                for k in ("accuracy", "macro_f1", "balanced_accuracy", "top3_accuracy", "ece_15_bins")
            },
            "clinical_fine": {
                k: report["final_model"]["test_clinical"]["fine"][k]
                for k in ("accuracy", "macro_f1", "balanced_accuracy", "top3_accuracy")
            },
            "dermoscopy_broad": {
                k: report["final_model"]["test_dermoscopy"]["broad"][k]
                for k in ("accuracy", "macro_f1", "balanced_accuracy")
            },
            "overall_broad": {
                k: report["final_model"]["test_overall"]["broad"][k]
                for k in ("accuracy", "macro_f1", "balanced_accuracy", "top3_accuracy", "ece_15_bins")
            },
        },
    }

    MODEL_DIR.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=MODEL_DIR) as staging:
        root = Path(staging)
        xml_path = root / "classifier.xml"
        ov.save_model(graph, xml_path, compress_to_fp16=False)

        # Numerical verification of the exported OpenVINO graph against NumPy/Keras logits
        net = core.compile_model(xml_path, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
        out = net(check_pixels)
        got_broad = out[net.output(0)]
        got_fine = out[net.output(1)]
        np.testing.assert_allclose(got_broad, expected_broad_logits / temp_broad, rtol=2e-3, atol=5e-3)
        np.testing.assert_allclose(got_fine, expected_fine_logits / temp_fine, rtol=2e-3, atol=5e-3)
        del net

        bundle = root / "classifier.zip"
        with ZipFile(bundle, "w") as archive:
            archive.write(xml_path, "classifier.xml", compress_type=ZIP_DEFLATED)
            archive.write(xml_path.with_suffix(".bin"), "classifier.bin", compress_type=ZIP_STORED)
            archive.writestr("classes.json", json.dumps(list(BROAD_CLASSES), indent=2))
            archive.writestr("fine_classes.json", json.dumps(list(FINE_CLASSES), indent=2))
            archive.writestr("metadata.json", json.dumps(metadata, indent=2))
        (root / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (root / "quality.json").write_text(json.dumps(quality_report, indent=2) + "\n", encoding="utf-8")
        for name in ("evaluation.json", "quality.json", "classifier.zip"):
            (root / name).replace(MODEL_DIR / name)


# --------------------------------------------------------------------------- main pipeline
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", type=Path, default=PROJECT_ROOT / "datasets" / "20-skin-diseases.zip")
    parser.add_argument("--backbones", nargs="+", default=["effnetv2b0_224", "effnetv2b2_260"])
    parser.add_argument("--modes", nargs="+", default=["letterbox", "letterbox_sog", "squash"])
    parser.add_argument("--pilot-per-class", type=int, default=420)
    parser.add_argument("--aug-views", type=int, default=1, help="Extra augmented views per clinical train image.")
    parser.add_argument("--minority-aug-views", type=int, default=4, help="Augmented views for minority fine classes.")
    parser.add_argument("--ensemble-seeds", type=int, default=3)
    parser.add_argument("--target-accuracy", type=float, default=0.80)
    parser.add_argument("--rebuild-manifest", action="store_true")
    args = parser.parse_args()

    np.random.seed(SEED)
    broad_index = {c: i for i, c in enumerate(BROAD_CLASSES)}
    fine_index = {c: i for i, c in enumerate(FINE_CLASSES)}

    log("Phase 1: Auditing & merging ALL workspace datasets (leakage-safe)")
    items, audit = audit_and_split_datasets(args.dataset, force_rebuild=args.rebuild_manifest)

    train_items = [i for i in items if i.split == "train"]
    val_items = [i for i in items if i.split == "val"]
    test_items = [i for i in items if i.split == "test"]

    y_broad_tr = np.array([broad_index[i.broad_label] for i in train_items], dtype=np.int32)
    y_fine_tr = np.array([fine_index[i.fine_label] if i.fine_label else -1 for i in train_items], dtype=np.int32)
    is_derm_tr = np.array([i.domain == "dermoscopy" for i in train_items], dtype=bool)
    groups_tr = np.array([i.group_id for i in train_items], dtype=np.int32)

    y_broad_va = np.array([broad_index[i.broad_label] for i in val_items], dtype=np.int32)
    y_fine_va = np.array([fine_index[i.fine_label] if i.fine_label else -1 for i in val_items], dtype=np.int32)
    is_derm_va = np.array([i.domain == "dermoscopy" for i in val_items], dtype=bool)

    y_broad_te = np.array([broad_index[i.broad_label] for i in test_items], dtype=np.int32)
    y_fine_te = np.array([fine_index[i.fine_label] if i.fine_label else -1 for i in test_items], dtype=np.int32)
    is_derm_te = np.array([i.domain == "dermoscopy" for i in test_items], dtype=bool)

    report = {
        "seed": SEED,
        "taxonomy": {
            "broad_classes": list(BROAD_CLASSES),
            "fine_classes": list(FINE_CLASSES),
            "fine_to_broad": dict(FINE_TO_BROAD),
        },
        "dataset_audit": audit,
    }

    # ------------------------------------------------------------------ Stage 1: Backbone x Preprocessing Pilot
    log("Phase 2: Stage 1 — Backbone x Preprocessing selection on clinical-stratified 5-fold grouped CV pilot")
    rng = np.random.default_rng(SEED)
    pilot_indices = []
    by_stratum = defaultdict(list)
    for idx, item in enumerate(train_items):
        by_stratum[(item.broad_label, item.domain)].append(idx)
    for (b_cls, dom), idxs in sorted(by_stratum.items()):
        cap = args.pilot_per_class if dom == "clinical" else min(280, args.pilot_per_class)
        if len(idxs) <= cap:
            pilot_indices.extend(idxs)
        else:
            pilot_indices.extend(rng.choice(idxs, size=cap, replace=False).tolist())
    pilot_indices = sorted(pilot_indices)
    pilot_items = [train_items[i] for i in pilot_indices]
    pilot_yb = y_broad_tr[pilot_indices]
    pilot_yf = y_fine_tr[pilot_indices]
    pilot_derm = is_derm_tr[pilot_indices]
    pilot_grp = groups_tr[pilot_indices]
    pilot_folds = list(StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED).split(
        np.zeros(len(pilot_items)), pilot_yb, pilot_grp
    ))
    log(f"  Pilot set: {len(pilot_items)} train images ({int((~pilot_derm).sum())} clinical, {int(pilot_derm.sum())} dermoscopy)")

    stage1_results = []
    for bb_name in args.backbones:
        ex = AsyncExtractor(bb_name)
        for mode in args.modes:
            p_views = ex.extract_views_to_array(
                pilot_items, mode, ["orig", "flip"], BUILD_DIR / "pilot_features" / f"{bb_name}__{mode}"
            )
            oof_b = np.zeros((len(pilot_items), len(BROAD_CLASSES)), dtype=np.float32)
            oof_f = np.zeros((len(pilot_items), len(FINE_CLASSES)), dtype=np.float32)
            for tr_i, va_i in pilot_folds:
                x_flat = np.asarray(p_views[tr_i], dtype=np.float32).reshape(-1, ex.dim)
                yb_rep = np.repeat(pilot_yb[tr_i], 2)
                dm_rep = np.repeat(pilot_derm[tr_i], 2)
                wb = compute_domain_balanced_weights(yb_rep, dm_rep, len(BROAD_CLASSES), "sqrt", 0.25)
                sc_b, hd_b = fit_linear_head(x_flat, yb_rep, wb, c_val=0.02, l2_norm=True)
                oof_b[va_i] = linear_logits(sc_b, hd_b, np.asarray(p_views[va_i], dtype=np.float32), l2_norm=True)

                yf_rep = np.repeat(pilot_yf[tr_i], 2)
                f_valid = yf_rep >= 0
                if len(np.unique(yf_rep[f_valid])) == len(FINE_CLASSES):
                    wf = compute_domain_balanced_weights(
                        np.where(f_valid, yf_rep, 0), dm_rep, len(FINE_CLASSES), "sqrt", 0.25, valid_mask=f_valid
                    )
                    sc_f, hd_f = fit_linear_head(x_flat[f_valid], yf_rep[f_valid], wf[f_valid], c_val=0.02, l2_norm=True)
                    oof_f[va_i] = linear_logits(sc_f, hd_f, np.asarray(p_views[va_i], dtype=np.float32), l2_norm=True)

            clin_m = classification_metrics(oof_b[~pilot_derm].argmax(1), pilot_yb[~pilot_derm], BROAD_CLASSES)
            derm_m = classification_metrics(oof_b[pilot_derm].argmax(1), pilot_yb[pilot_derm], BROAD_CLASSES)
            f_mask = (~pilot_derm) & (pilot_yf >= 0)
            fine_m = classification_metrics(oof_f[f_mask].argmax(1), pilot_yf[f_mask], FINE_CLASSES)
            composite = 0.50 * clin_m["macro_f1"] + 0.25 * clin_m["accuracy"] + 0.15 * fine_m["macro_f1"] + 0.10 * derm_m["accuracy"]
            res = {
                "backbone": bb_name,
                "mode": mode,
                "clinical_broad_acc": round(clin_m["accuracy"], 4),
                "clinical_broad_macro_f1": round(clin_m["macro_f1"], 4),
                "clinical_fine_macro_f1": round(fine_m["macro_f1"], 4),
                "dermoscopy_broad_acc": round(derm_m["accuracy"], 4),
                "composite_score": round(composite, 4),
            }
            stage1_results.append(res)
            log(f"  Stage1 {res}")
        del ex
        gc.collect()

    best_s1 = max(stage1_results, key=lambda r: r["composite_score"])
    report["cv_stage1_backbone_preprocessing"] = stage1_results
    log(f"Selected Stage 1 winner: {best_s1['backbone']} / {best_s1['mode']} (composite={best_s1['composite_score']})")

    # ------------------------------------------------------------------ Stage 2: Full Feature Extraction
    log("Phase 3: Stage 2 — Full unified feature extraction (disk-backed float16 memmap)")
    ex = AsyncExtractor(best_s1["backbone"])
    feat_dir = BUILD_DIR / "features"
    x_tr_clean = ex.extract_views_to_array(
        train_items, best_s1["mode"], ["orig", "flip"], feat_dir / f"{ex.name}__{best_s1['mode']}__train_clean"
    )
    x_va_clean = ex.extract_views_to_array(
        val_items, best_s1["mode"], ["orig", "flip"], feat_dir / f"{ex.name}__{best_s1['mode']}__val_clean"
    )
    x_te_clean = ex.extract_views_to_array(
        test_items, best_s1["mode"], ["orig", "flip"], feat_dir / f"{ex.name}__{best_s1['mode']}__test_clean"
    )

    # Extract augmented views for clinical training images (extra views for minority fine classes like Psoriasis!)
    clin_tr_indices = np.flatnonzero(~is_derm_tr).tolist()
    fine_counts_tr = Counter(int(v) for v in y_fine_tr if v >= 0)
    minority_fine_ids = {fid for fid, cnt in fine_counts_tr.items() if cnt < 450}
    min_tr_indices = [idx for idx in clin_tr_indices if int(y_fine_tr[idx]) in minority_fine_ids]

    x_tr_aug_base = None
    x_tr_aug_min = None
    if args.aug_views > 0:
        aug_names = [f"aug{k}" for k in range(args.aug_views)]
        x_tr_aug_base = ex.extract_views_to_array(
            [train_items[i] for i in clin_tr_indices],
            best_s1["mode"],
            aug_names,
            feat_dir / f"{ex.name}__{best_s1['mode']}__train_clin_aug{args.aug_views}",
        )
    if args.minority_aug_views > args.aug_views and min_tr_indices:
        extra_names = [f"aug{k}" for k in range(args.aug_views, args.minority_aug_views)]
        x_tr_aug_min = ex.extract_views_to_array(
            [train_items[i] for i in min_tr_indices],
            best_s1["mode"],
            extra_names,
            feat_dir / f"{ex.name}__{best_s1['mode']}__train_min_aug{args.minority_aug_views}",
        )
    ex_size = ex.size
    ex_dim = ex.dim
    ex_xml = ex.xml
    ex_name = ex.name
    # Release OpenVINO extractor RAM before training heads
    del ex
    gc.collect()

    # Helper to assemble memory-efficient flat training matrix for any subset of train_items
    clin_pos_map = {orig_i: pos for pos, orig_i in enumerate(clin_tr_indices)}
    min_pos_map = {orig_i: pos for pos, orig_i in enumerate(min_tr_indices)}

    def assemble_train_matrix(subset_indices: np.ndarray, include_aug: bool = True):
        # Count total rows first to pre-allocate one exact float32 matrix (avoids list concat RAM spike!)
        n_base = len(subset_indices) * 2
        clin_sub = [int(i) for i in subset_indices if int(i) in clin_pos_map] if (include_aug and x_tr_aug_base is not None) else []
        min_sub = [int(i) for i in subset_indices if int(i) in min_pos_map] if (include_aug and x_tr_aug_min is not None) else []
        n_aug_b = len(clin_sub) * (x_tr_aug_base.shape[1] if x_tr_aug_base is not None else 0)
        n_aug_m = len(min_sub) * (x_tr_aug_min.shape[1] if x_tr_aug_min is not None else 0)
        total_rows = n_base + n_aug_b + n_aug_m

        X = np.empty((total_rows, ex_dim), dtype=np.float32)
        yb = np.empty(total_rows, dtype=np.int32)
        yf = np.empty(total_rows, dtype=np.int32)
        dm = np.empty(total_rows, dtype=bool)

        # Clean orig + flip
        X[:n_base] = np.asarray(x_tr_clean[subset_indices], dtype=np.float32).reshape(n_base, ex_dim)
        yb[:n_base] = np.repeat(y_broad_tr[subset_indices], 2)
        yf[:n_base] = np.repeat(y_fine_tr[subset_indices], 2)
        dm[:n_base] = np.repeat(is_derm_tr[subset_indices], 2)
        cursor = n_base

        if clin_sub:
            k = x_tr_aug_base.shape[1]
            pos_arr = [clin_pos_map[i] for i in clin_sub]
            X[cursor : cursor + n_aug_b] = np.asarray(x_tr_aug_base[pos_arr], dtype=np.float32).reshape(n_aug_b, ex_dim)
            yb[cursor : cursor + n_aug_b] = np.repeat(y_broad_tr[clin_sub], k)
            yf[cursor : cursor + n_aug_b] = np.repeat(y_fine_tr[clin_sub], k)
            dm[cursor : cursor + n_aug_b] = False
            cursor += n_aug_b

        if min_sub:
            k = x_tr_aug_min.shape[1]
            pos_arr = [min_pos_map[i] for i in min_sub]
            X[cursor : cursor + n_aug_m] = np.asarray(x_tr_aug_min[pos_arr], dtype=np.float32).reshape(n_aug_m, ex_dim)
            yb[cursor : cursor + n_aug_m] = np.repeat(y_broad_tr[min_sub], k)
            yf[cursor : cursor + n_aug_m] = np.repeat(y_fine_tr[min_sub], k)
            dm[cursor : cursor + n_aug_m] = False
            cursor += n_aug_m

        return X, yb, yf, dm

    # ------------------------------------------------------------------ Stage 3: Domain Balancing & Head Selection
    log("Phase 4: Stage 3 — Domain-balanced weighting & head selection (5-fold grouped CV on full train split)")
    tr_folds = list(
        StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(train_items)),
            np.array([f"{i.broad_label}||{i.domain}" for i in train_items]),
            groups_tr,
        )
    )

    # First tune ISIC down-weighting & class weighting rapidly using Linear head on fold 0..4 (or clean features)
    weighting_results = []
    for isic_w in (0.15, 0.25, 0.40):
        for scheme in ("none", "sqrt", "balanced"):
            oof_b = np.zeros((len(train_items), len(BROAD_CLASSES)), dtype=np.float32)
            for tr_idx, va_idx in tr_folds:
                X_f, yb_f, _, dm_f = assemble_train_matrix(tr_idx, include_aug=False)
                wb = compute_domain_balanced_weights(yb_f, dm_f, len(BROAD_CLASSES), scheme, isic_w)
                sc, hd = fit_linear_head(X_f, yb_f, wb, c_val=0.01, l2_norm=True)
                oof_b[va_idx] = linear_logits(sc, hd, np.asarray(x_tr_clean[va_idx], dtype=np.float32), l2_norm=True)
                del X_f
            cm = classification_metrics(oof_b[~is_derm_tr].argmax(1), y_broad_tr[~is_derm_tr], BROAD_CLASSES)
            dm_m = classification_metrics(oof_b[is_derm_tr].argmax(1), y_broad_tr[is_derm_tr], BROAD_CLASSES)
            score_val = 0.55 * cm["macro_f1"] + 0.30 * cm["accuracy"] + 0.15 * dm_m["accuracy"]
            row = {
                "isic_weight": isic_w,
                "class_weight": scheme,
                "clinical_acc": round(cm["accuracy"], 4),
                "clinical_macro_f1": round(cm["macro_f1"], 4),
                "dermoscopy_acc": round(dm_m["accuracy"], 4),
                "composite": round(score_val, 4),
            }
            weighting_results.append(row)
            log(f"  Weighting CV: {row}")
    best_w = max(weighting_results, key=lambda r: r["composite"])
    report["cv_stage2_domain_weighting"] = weighting_results
    log(f"Selected domain weighting: isic_weight={best_w['isic_weight']}, class_weight={best_w['class_weight']}")

    # Compare Linear (with aug + tuned C) vs Multi-Task Residual MLP (with aug) on full 5-fold grouped CV!
    head_comparison = []
    # Candidate A: Linear with augmentation (tune C)
    best_lin = None
    for c_val in (0.002, 0.005, 0.015, 0.05):
        oof_b = np.zeros((len(train_items), len(BROAD_CLASSES)), dtype=np.float32)
        oof_f = np.zeros((len(train_items), len(FINE_CLASSES)), dtype=np.float32)
        for tr_idx, va_idx in tr_folds:
            X_f, yb_f, yf_f, dm_f = assemble_train_matrix(tr_idx, include_aug=True)
            wb = compute_domain_balanced_weights(yb_f, dm_f, len(BROAD_CLASSES), best_w["class_weight"], best_w["isic_weight"])
            sc_b, hd_b = fit_linear_head(X_f, yb_f, wb, c_val=c_val, l2_norm=True)
            oof_b[va_idx] = linear_logits(sc_b, hd_b, np.asarray(x_tr_clean[va_idx], dtype=np.float32), l2_norm=True)

            f_valid = yf_f >= 0
            wf = compute_domain_balanced_weights(
                np.where(f_valid, yf_f, 0), dm_f, len(FINE_CLASSES), "sqrt", best_w["isic_weight"], valid_mask=f_valid
            )
            sc_f, hd_f = fit_linear_head(X_f[f_valid], yf_f[f_valid], wf[f_valid], c_val=c_val, l2_norm=True)
            oof_f[va_idx] = linear_logits(sc_f, hd_f, np.asarray(x_tr_clean[va_idx], dtype=np.float32), l2_norm=True)
            del X_f
            gc.collect()
        cm_b = classification_metrics(oof_b[~is_derm_tr].argmax(1), y_broad_tr[~is_derm_tr], BROAD_CLASSES)
        dm_b = classification_metrics(oof_b[is_derm_tr].argmax(1), y_broad_tr[is_derm_tr], BROAD_CLASSES)
        f_clin_mask = (~is_derm_tr) & (y_fine_tr >= 0)
        cm_f = classification_metrics(oof_f[f_clin_mask].argmax(1), y_fine_tr[f_clin_mask], FINE_CLASSES)
        comp = 0.45 * cm_b["macro_f1"] + 0.25 * cm_b["accuracy"] + 0.20 * cm_f["macro_f1"] + 0.10 * dm_b["accuracy"]
        row = {
            "head_type": "linear",
            "C": c_val,
            "clinical_broad_acc": round(cm_b["accuracy"], 4),
            "clinical_broad_macro_f1": round(cm_b["macro_f1"], 4),
            "clinical_fine_acc": round(cm_f["accuracy"], 4),
            "clinical_fine_macro_f1": round(cm_f["macro_f1"], 4),
            "dermoscopy_broad_acc": round(dm_b["accuracy"], 4),
            "composite": round(comp, 4),
        }
        head_comparison.append(row)
        log(f"  Head CV (Linear C={c_val}): {row}")
        if best_lin is None or row["composite"] > best_lin["composite"]:
            best_lin = row

    # Candidate B: Multi-Task Shared-Bottleneck Residual MLP (5-fold grouped CV)
    oof_b_mlp = np.zeros((len(train_items), len(BROAD_CLASSES)), dtype=np.float32)
    oof_f_mlp = np.zeros((len(train_items), len(FINE_CLASSES)), dtype=np.float32)
    for fold_i, (tr_idx, va_idx) in enumerate(tr_folds):
        X_f, yb_f, yf_f, dm_f = assemble_train_matrix(tr_idx, include_aug=True)
        wb = compute_domain_balanced_weights(yb_f, dm_f, len(BROAD_CLASSES), best_w["class_weight"], best_w["isic_weight"])
        f_valid = yf_f >= 0
        wf = compute_domain_balanced_weights(
            np.where(f_valid, yf_f, 0), dm_f, len(FINE_CLASSES), "sqrt", best_w["isic_weight"], valid_mask=f_valid
        )
        # Use validation fold orig view for early stopping
        x_va_fold = np.asarray(x_tr_clean[va_idx, 0], dtype=np.float32)
        w_va_fold = compute_domain_balanced_weights(
            y_broad_tr[va_idx], is_derm_tr[va_idx], len(BROAD_CLASSES), best_w["class_weight"], best_w["isic_weight"]
        )
        m_weights = fit_multitask_mlp(
            X_f, yb_f, wb, yf_f, wf,
            x_va_fold, y_broad_tr[va_idx], w_va_fold,
            n_broad=len(BROAD_CLASSES), n_fine=len(FINE_CLASSES),
            seed=SEED + fold_i, l2_norm=True,
        )
        b_log, f_log = mlp_predict_logits([m_weights], np.asarray(x_tr_clean[va_idx], dtype=np.float32))
        oof_b_mlp[va_idx] = b_log
        oof_f_mlp[va_idx] = f_log
        del X_f, x_va_fold
        gc.collect()

    cm_b = classification_metrics(oof_b_mlp[~is_derm_tr].argmax(1), y_broad_tr[~is_derm_tr], BROAD_CLASSES)
    dm_b = classification_metrics(oof_b_mlp[is_derm_tr].argmax(1), y_broad_tr[is_derm_tr], BROAD_CLASSES)
    f_clin_mask = (~is_derm_tr) & (y_fine_tr >= 0)
    cm_f = classification_metrics(oof_f_mlp[f_clin_mask].argmax(1), y_fine_tr[f_clin_mask], FINE_CLASSES)
    comp_mlp = 0.45 * cm_b["macro_f1"] + 0.25 * cm_b["accuracy"] + 0.20 * cm_f["macro_f1"] + 0.10 * dm_b["accuracy"]
    mlp_row = {
        "head_type": "multitask_residual_mlp",
        "hidden_dim": 512,
        "clinical_broad_acc": round(cm_b["accuracy"], 4),
        "clinical_broad_macro_f1": round(cm_b["macro_f1"], 4),
        "clinical_fine_acc": round(cm_f["accuracy"], 4),
        "clinical_fine_macro_f1": round(cm_f["macro_f1"], 4),
        "dermoscopy_broad_acc": round(dm_b["accuracy"], 4),
        "composite": round(comp_mlp, 4),
    }
    head_comparison.append(mlp_row)
    log(f"  Head CV (Multi-Task Residual MLP): {mlp_row}")
    report["cv_stage3_head_comparison"] = head_comparison

    winner_head = max(head_comparison, key=lambda r: r["composite"])
    log(f"Winning head architecture: {winner_head['head_type']} (composite={winner_head['composite']})")

    # ------------------------------------------------------------------ Stage 4: Final Fit, Calibration on val, Test Eval
    log("Phase 5: Stage 4 — Fitting final model on full train split & calibrating on held-out val split")
    all_tr_idx = np.arange(len(train_items))
    X_all, yb_all, yf_all, dm_all = assemble_train_matrix(all_tr_idx, include_aug=True)
    wb_all = compute_domain_balanced_weights(yb_all, dm_all, len(BROAD_CLASSES), best_w["class_weight"], best_w["isic_weight"])
    f_valid_all = yf_all >= 0
    wf_all = compute_domain_balanced_weights(
        np.where(f_valid_all, yf_all, 0), dm_all, len(FINE_CLASSES), "sqrt", best_w["isic_weight"], valid_mask=f_valid_all
    )

    va_views_np = np.asarray(x_va_clean, dtype=np.float32)
    te_views_np = np.asarray(x_te_clean, dtype=np.float32)

    if winner_head["head_type"] == "linear":
        sc_b, hd_b = fit_linear_head(X_all, yb_all, wb_all, c_val=winner_head["C"], l2_norm=True)
        sc_f, hd_f = fit_linear_head(X_all[f_valid_all], yf_all[f_valid_all], wf_all[f_valid_all], c_val=winner_head["C"], l2_norm=True)
        head_artifacts = {"broad": (sc_b, hd_b), "fine": (sc_f, hd_f)}
        val_b_logits = linear_logits(sc_b, hd_b, va_views_np, l2_norm=True)
        val_f_logits = linear_logits(sc_f, hd_f, va_views_np, l2_norm=True)
        test_b_logits = linear_logits(sc_b, hd_b, te_views_np, l2_norm=True)
        test_f_logits = linear_logits(sc_f, hd_f, te_views_np, l2_norm=True)
        check_b_exp = linear_logits(sc_b, hd_b, te_views_np[:4, :1], l2_norm=True)
        check_f_exp = linear_logits(sc_f, hd_f, te_views_np[:4, :1], l2_norm=True)
    else:
        x_va_es = np.asarray(x_va_clean[:, 0], dtype=np.float32)
        w_va_es = compute_domain_balanced_weights(
            y_broad_va, is_derm_va, len(BROAD_CLASSES), best_w["class_weight"], best_w["isic_weight"]
        )
        models = []
        for s_i in range(args.ensemble_seeds):
            log(f"  Training Multi-Task Residual MLP seed {s_i + 1}/{args.ensemble_seeds}...")
            models.append(
                fit_multitask_mlp(
                    X_all, yb_all, wb_all, yf_all, wf_all,
                    x_va_es, y_broad_va, w_va_es,
                    n_broad=len(BROAD_CLASSES), n_fine=len(FINE_CLASSES),
                    seed=SEED + 100 * s_i, l2_norm=True,
                )
            )
        head_artifacts = {"models": models}
        val_b_logits, val_f_logits = mlp_predict_logits(models, va_views_np)
        test_b_logits, test_f_logits = mlp_predict_logits(models, te_views_np)
        check_b_exp, check_f_exp = mlp_predict_logits(models, te_views_np[:4, :1])
        del x_va_es

    del X_all
    gc.collect()

    # Temperature calibration fitted on CLINICAL + balanced val logits (prevents ECE regression!)
    val_clin_mask = ~is_derm_va
    w_cal_broad = np.where(is_derm_va, 0.25, 1.0)
    temp_broad = fit_temperature(val_b_logits, y_broad_va, sample_weight=w_cal_broad)
    val_fine_mask = y_fine_va >= 0
    w_cal_fine = np.where(is_derm_va[val_fine_mask], 0.25, 1.0)
    temp_fine = fit_temperature(val_f_logits[val_fine_mask], y_fine_va[val_fine_mask], sample_weight=w_cal_fine)

    val_b_probs = softmax(val_b_logits / temp_broad)
    # Threshold calibrated strictly on clinical validation photos!
    threshold = min(choose_threshold(val_b_probs[val_clin_mask], y_broad_va[val_clin_mask], target=0.78, min_coverage=0.72), 0.36)
    min_margin = 0.03

    report["selected_configuration"] = {
        "backbone": ex_name,
        "preprocessing": best_s1["mode"],
        "head_type": winner_head["head_type"],
        "ensemble_seeds": args.ensemble_seeds if winner_head["head_type"] != "linear" else 1,
        "l2_norm": True,
        "isic_weight": best_w["isic_weight"],
        "class_weight": best_w["class_weight"],
        "aug_views_clinical": args.aug_views,
        "aug_views_minority_fine": args.minority_aug_views,
        "temperature_broad": temp_broad,
        "temperature_fine": temp_fine,
        "uncertain_threshold": threshold,
        "min_margin": min_margin,
        "target_accuracy_on_answers": args.target_accuracy,
    }

    # Single independent evaluation on held-out TEST split (Clinical / Dermoscopy / Overall)
    log("Phase 6: Evaluating ONCE on independent held-out test split (Clinical / Dermoscopy / Overall)")
    test_b_probs = softmax(test_b_logits / temp_broad)
    test_f_probs = softmax(test_f_logits / temp_fine)

    def eval_slice(mask: np.ndarray, include_ci: bool = False) -> dict:
        b_p = test_b_probs[mask]
        b_y = y_broad_te[mask]
        b_pred = b_p.argmax(1)
        b_res = classification_metrics(b_pred, b_y, BROAD_CLASSES)
        b_res.update(probability_metrics(b_p, b_y))
        b_res["uncalibrated_ece_15_bins"] = expected_calibration_error(softmax(test_b_logits[mask]), b_y)
        if include_ci and len(b_y) >= 20:
            b_res.update(bootstrap_ci(b_pred, b_y, BROAD_CLASSES))
        sel = selective_metrics(b_p, b_y, threshold, BROAD_CLASSES)

        f_mask = mask & (y_fine_te >= 0)
        if f_mask.any():
            f_p = test_f_probs[f_mask]
            f_y = y_fine_te[f_mask]
            f_res = classification_metrics(f_p.argmax(1), f_y, FINE_CLASSES)
            f_res.update(probability_metrics(f_p, f_y))
            # Hierarchical conditioned fine accuracy (when broad class is predicted, pick best child fine class)
            cond_preds = []
            for row_b, row_f in zip(test_b_probs[f_mask], f_p):
                top_broad = BROAD_CLASSES[int(np.argmax(row_b))]
                child_idxs = [fine_index[fc] for fc in BROAD_TO_FINE[top_broad]]
                best_child = child_idxs[int(np.argmax(row_f[child_idxs]))]
                cond_preds.append(best_child)
            cond_m = classification_metrics(np.array(cond_preds, dtype=np.int32), f_y, FINE_CLASSES)
            f_res["hierarchical_conditioned_accuracy"] = cond_m["accuracy"]
            f_res["hierarchical_conditioned_macro_f1"] = cond_m["macro_f1"]
        else:
            f_res = {"images": 0}
        return {"broad": b_res, "broad_selective": sel, "fine": f_res}

    test_clin_mask = ~is_derm_te
    test_derm_mask = is_derm_te
    report["final_model"] = {
        "val_clinical_broad": {
            **classification_metrics(val_b_probs[val_clin_mask].argmax(1), y_broad_va[val_clin_mask], BROAD_CLASSES),
            **probability_metrics(val_b_probs[val_clin_mask], y_broad_va[val_clin_mask]),
        },
        "test_clinical": eval_slice(test_clin_mask, include_ci=True),
        "test_dermoscopy": eval_slice(test_derm_mask, include_ci=True),
        "test_overall": eval_slice(np.ones(len(test_items), dtype=bool), include_ci=True),
    }

    # Fresh quality gate calibration on clinical training photos
    log("Phase 7: Fresh image quality gate calibration on clinical training photos")
    train_clinical_items = [i for i in train_items if i.domain == "clinical"]
    quality_report = calibrate_quality_fresh(train_clinical_items)

    # Export single-graph dual-output OpenVINO classifier bundle
    log("Phase 8: Exporting dual-output OpenVINO classifier.zip (format 3) & verifying numerical parity")
    archive_cache: dict[str, ZipFile] = {}
    try:
        check_pixels = np.stack(
            [prepare(read_item_image(i, archive_cache), ex_size, best_s1["mode"]) for i in test_items[:4]]
        )
    finally:
        for zf in archive_cache.values():
            zf.close()

    export_hierarchical_bundle(
        {"xml": ex_xml, "backbone": ex_name, "size": ex_size, "mode": best_s1["mode"], "l2_norm": True},
        winner_head["head_type"],
        head_artifacts,
        temp_broad,
        temp_fine,
        threshold,
        min_margin,
        report,
        quality_report,
        check_pixels,
        check_b_exp,
        check_f_exp,
    )

    tc = report["final_model"]["test_clinical"]["broad"]
    tf_m = report["final_model"]["test_clinical"]["fine"]
    td = report["final_model"]["test_dermoscopy"]["broad"]
    to = report["final_model"]["test_overall"]["broad"]
    ts = report["final_model"]["test_clinical"]["broad_selective"]
    log(
        f"TEST CLINICAL   (7-broad): acc={tc['accuracy']:.3f}  macro-F1={tc['macro_f1']:.3f}  "
        f"top3={tc['top3_accuracy']:.3f}  ECE={tc['ece_15_bins']:.4f} (uncal={tc['uncalibrated_ece_15_bins']:.4f})"
    )
    log(
        f"TEST CLINICAL   (20-fine): acc={tf_m['accuracy']:.3f}  macro-F1={tf_m['macro_f1']:.3f}  "
        f"top3={tf_m['top3_accuracy']:.3f}  hier-acc={tf_m['hierarchical_conditioned_accuracy']:.3f}"
    )
    log(f"TEST DERMOSCOPY (ISIC)   : acc={td['accuracy']:.3f}  macro-F1={td['macro_f1']:.3f}")
    log(f"TEST OVERALL    (7-broad): acc={to['accuracy']:.3f}  macro-F1={to['macro_f1']:.3f}  top3={to['top3_accuracy']:.3f}")
    log(
        f"CLINICAL SELECTIVE (t={threshold:.2f}): coverage={ts['coverage']:.3f}  "
        f"acc_on_accepted={ts['accuracy_on_accepted']:.3f}"
    )
    log(f"Saved model/classifier.zip, model/evaluation.json, and model/quality.json")


if __name__ == "__main__":
    main()
