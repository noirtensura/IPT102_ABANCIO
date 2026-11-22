"""Leakage-safe, domain-balanced hierarchical training, calibration, and OpenVINO export.

Ingests ALL datasets in the workspace:
  - Dataset A: ``datasets/20-skin-diseases`` (or ``datasets/20-skin-diseases.zip``) — 19 fine classes
    (including recovery of the 204 mis-nested Acne and Rosacea test photos).
  - Dataset B: ``train/``, ``val/``, ``test/`` — 6 broad clinical/dermoscopy classes (38,761 files).
  - Dataset D: ``datasets/dermnet`` — the full DermNet collection Dataset A was sampled from.
  - Dataset E: ``datasets/skin-disease-dataset`` — 8 infection classes (1,159 files).
  - Dataset F: ``datasets/skindiseasedataset`` — 22 class folders (15,444 files), mostly DermNet photos.

Photos keep the split an earlier audit gave them, so adding a dataset only assigns the new photos and
only their features are extracted; everything else is reused from ``build/``.

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
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Worker threads for decoding, feature extraction, and head fitting. The default is one thread per
# physical core; set SKIN_TRAIN_THREADS lower to keep a laptop cool and responsive during long runs.
CPU_THREADS = max(1, int(os.environ.get("SKIN_TRAIN_THREADS") or max(2, (os.cpu_count() or 4) // 2)))
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_thread_var, str(CPU_THREADS))

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
    LABEL_RULES_VERSION,
    dataset_e_source_key,
    fine_label_for_class_folder,
    labels_for_dataset_b,
    labels_for_dataset_e,
    labels_for_dataset_f,
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
    # ImageNet-supervised CNNs exported from Keras (backend/experiments/backbones.py)
    "mobilenetv2_035_128": {"size": 128, "rescale": True, "source": "keras"},
    "effnetv2b0_224": {"size": 224, "rescale": False, "source": "keras"},
    "effnetv2b2_260": {"size": 260, "rescale": False, "source": "keras"},
    "convnext_tiny_224": {"size": 224, "rescale": False, "source": "keras"},
    # Self-/language-supervised Vision Transformers exported from timm (backend/experiments/vit_backbones.py)
    "dinov2_vits14_224": {"size": 224, "rescale": False, "source": "timm"},
    "dinov2_vitb14_224": {"size": 224, "rescale": False, "source": "timm"},
    "siglip2_vitb16_224": {"size": 224, "rescale": False, "source": "timm"},
}

# Sources whose class folder names give an explicit specific-condition label (A = Kaggle 20 Skin
# Diseases, D = the full DermNet collection it was sampled from). Dataset B only has broad folders.
EXPLICIT_LABEL_SOURCES = ("A", "D")
MANIFEST_SCHEMA_VERSION = 4


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# --------------------------------------------------------------------------- laptop safety pauses
REQUIRE_AC = False  # set by --require-ac
# Free commit memory below which a stage waits. Windows grows a system-managed page file on demand, so
# this is a brake that lets other apps settle, not a wall: after MAX_MEMORY_WAIT_S the stage goes ahead.
MIN_FREE_COMMIT_GB = float(os.environ.get("SKIN_TRAIN_MIN_FREE_GB") or 1.0)
MAX_MEMORY_WAIT_S = 300.0


def _system_status() -> tuple[bool, float]:
    """(running on battery, free commit memory in GB). Reports (False, inf) where it cannot be queried."""
    if os.name != "nt":
        return False, float("inf")
    import ctypes
    from ctypes import wintypes

    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_ubyte),
            ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte),
            ("SystemStatusFlag", ctypes.c_ubyte),
            ("BatteryLifeTime", wintypes.DWORD),
            ("BatteryFullLifeTime", wintypes.DWORD),
        ]

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    kernel32 = ctypes.windll.kernel32
    power = SystemPowerStatus()
    on_battery = bool(kernel32.GetSystemPowerStatus(ctypes.byref(power))) and power.ACLineStatus == 0
    memory = MemoryStatusEx()
    memory.dwLength = ctypes.sizeof(memory)
    free_commit = memory.ullAvailPageFile / 2**30 if kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)) else float("inf")
    return on_battery, free_commit


def wait_until_safe(stage: str = "") -> None:
    """Pause while the machine is on battery (with --require-ac) or is very short of memory.

    Hours of full CPU load on a low battery can freeze or reset a laptop, so that pause has no time
    limit. The memory pause is bounded (see MIN_FREE_COMMIT_GB). Every stage is checkpointed, so
    waiting here costs nothing but time.
    """
    announced = 0.0
    memory_wait_started = time.time()
    while True:
        on_battery, free_commit = _system_status()
        reasons = []
        if REQUIRE_AC and on_battery:
            reasons.append("running on battery, waiting for the charger")
            memory_wait_started = time.time()  # the memory allowance starts once the charger is back
        elif free_commit < MIN_FREE_COMMIT_GB:
            if time.time() - memory_wait_started > MAX_MEMORY_WAIT_S:
                log(f"Low memory ({free_commit:.1f} GB free) did not clear; continuing{' with ' + stage if stage else ''}")
                return
            reasons.append(f"only {free_commit:.1f} GB of system memory left, waiting for other apps to free some")
        if not reasons:
            if announced:
                log("Resuming")
            return
        if time.time() - announced > 300:
            log(f"PAUSED{' before ' + stage if stage else ''}: " + "; ".join(reasons))
            announced = time.time()
        time.sleep(20)


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
    source_dataset: str  # the datasets holding this photo, e.g. "A", "B+D", or "E"
    domain: str  # "clinical" or "dermoscopy"
    broad_label: str
    fine_label: str | None
    sha: str
    phash: int
    width: int
    height: int
    group_id: int = -1
    split: str = ""  # "train", "val", "test"


DERMNET_DIR = PROJECT_ROOT / "datasets" / "dermnet"
DATASET_B_DIR = PROJECT_ROOT / "datasets" / "skin-diseases-6class"
DATASET_E_DIR = PROJECT_ROOT / "datasets" / "skin-disease-dataset"
DATASET_F_DIR = PROJECT_ROOT / "datasets" / "skindiseasedataset"


def dataset_e_key_for_path(rel_path: str) -> str | None:
    """Source-photo key of a Dataset E file (see ``dataset_e_source_key``); None for every other dataset."""
    path = PurePosixPath(rel_path)
    if path.parts[:2] != ("datasets", DATASET_E_DIR.name):
        return None
    return dataset_e_source_key(path.parent.name.replace(" ", ""), path.name)


def dataset_b_split_dirs() -> list[Path]:
    """Dataset B split folders that hold at least one known class folder.

    Looks in the extracted archive under ``datasets/skin-diseases-6class`` (directly or one folder
    down, as archives often wrap their content), then at the project root (the original layout).
    """
    bases = [DATASET_B_DIR]
    if DATASET_B_DIR.is_dir():
        bases += sorted(d for d in DATASET_B_DIR.iterdir() if d.is_dir())
    bases.append(PROJECT_ROOT)
    for base in bases:
        found = [
            base / split
            for split in ("train", "val", "test")
            if (base / split).is_dir() and any((base / split / name).is_dir() for name in DATASET_B_FOLDER_TO_BROAD)
        ]
        if found:
            return found
    return []


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
                fine = fine_label_for_class_folder(raw_cls, p.name)
                if fine is None:
                    yield ("skip", str(p.relative_to(PROJECT_ROOT)), p.name, "A", "test_only_class", None, b"")
                    continue
                broad = FINE_TO_BROAD[fine]
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
                fname = PurePosixPath(info.filename).name
                fine = fine_label_for_class_folder(raw_cls, fname)
                if fine is None:
                    yield ("skip", info.filename, fname, "A", "test_only_class", None, b"")
                    continue
                broad = FINE_TO_BROAD[fine]
                yield ("zip", f"{zip_rel}::{info.filename}", fname, "A", broad, fine, zf.read(info))

    # Dataset D (optional): the full DermNet collection. Dataset A is a sample of it with identical
    # class folders, so the same folder -> specific-condition mapping applies. Class folders outside
    # the taxonomy (bullous, hair, nail, and infestation photos) are skipped.
    zip_d = PROJECT_ROOT / "datasets" / "dermnet.zip"
    if any((DERMNET_DIR / s).is_dir() for s in ("train", "test")):
        log(f"Scanning Dataset D (DermNet) from extracted directory: {DERMNET_DIR.relative_to(PROJECT_ROOT)}")
        for split_name in ("train", "test"):
            sp_dir = DERMNET_DIR / split_name
            if not sp_dir.is_dir():
                continue
            for p in sorted(sp_dir.rglob("*")):
                if not p.is_file():
                    continue
                rel_parts = p.relative_to(sp_dir).parts
                rel_path = p.relative_to(PROJECT_ROOT).as_posix()
                if len(rel_parts) != 2:
                    yield ("skip", rel_path, p.name, "D", "dermnet_unexpected_path", None, b"")
                    continue
                fine = fine_label_for_class_folder(rel_parts[0], p.name)
                if fine is None:
                    yield ("skip", rel_path, p.name, "D", "dermnet_unmapped_class", None, b"")
                    continue
                yield ("file", rel_path, p.name, "D", FINE_TO_BROAD[fine], fine, p.read_bytes())
    elif zip_d.is_file():
        log(f"Scanning Dataset D (DermNet) from zip archive: {zip_d.relative_to(PROJECT_ROOT)}")
        zip_rel = zip_d.relative_to(PROJECT_ROOT).as_posix()
        with ZipFile(zip_d) as zf:
            for info in sorted(zf.infolist(), key=lambda e: e.filename):
                if info.is_dir():
                    continue
                parts = PurePosixPath(info.filename).parts
                fname = parts[-1]
                split_at = next((k for k, part in enumerate(parts[:-2]) if part in ("train", "test")), None)
                if split_at is None or len(parts) - split_at != 3:
                    yield ("skip", info.filename, fname, "D", "dermnet_unexpected_path", None, b"")
                    continue
                fine = fine_label_for_class_folder(parts[split_at + 1], fname)
                if fine is None:
                    yield ("skip", info.filename, fname, "D", "dermnet_unmapped_class", None, b"")
                    continue
                yield ("zip", f"{zip_rel}::{info.filename}", fname, "D", FINE_TO_BROAD[fine], fine, zf.read(info))

    # Dataset B: train/, val/, test/ split folders (extracted under datasets/ or at the project root)
    if include_dataset_b:
        b_splits = dataset_b_split_dirs()
        if b_splits:
            log(f"Scanning Dataset B from {', '.join(d.relative_to(PROJECT_ROOT).as_posix() + '/' for d in b_splits)}")
            for sp_dir in b_splits:
                for cls_dir in sorted(sp_dir.iterdir()):
                    if not cls_dir.is_dir():
                        continue
                    folder_broad = DATASET_B_FOLDER_TO_BROAD.get(cls_dir.name)
                    if folder_broad is None:
                        continue
                    for p in sorted(cls_dir.rglob("*")):
                        if not p.is_file():
                            continue
                        low_name = p.name.lower()
                        # Drop Keras offline-augmented stretched-border copies in 3. Akne
                        if folder_broad == "Acne and Rosacea" and low_name.startswith("pigmentation_0_"):
                            yield ("skip", p.relative_to(PROJECT_ROOT).as_posix(), p.name, "B", "keras_offline_aug", None, b"")
                            continue
                        broad, fine = labels_for_dataset_b(cls_dir.name, p.name)
                        yield ("file", p.relative_to(PROJECT_ROOT).as_posix(), p.name, "B", broad, fine, p.read_bytes())

    # Dataset E (optional): eight infection class folders, at any depth below the dataset folder.
    if DATASET_E_DIR.is_dir():
        log(f"Scanning Dataset E from extracted directory: {DATASET_E_DIR.relative_to(PROJECT_ROOT)}")
        for p in sorted(DATASET_E_DIR.rglob("*")):
            if not p.is_file():
                continue
            rel_path = p.relative_to(PROJECT_ROOT).as_posix()
            labels = labels_for_dataset_e(p.parent.name)
            if labels is None:
                yield ("skip", rel_path, p.name, "E", "dataset_e_unmapped_class", None, b"")
                continue
            yield ("file", rel_path, p.name, "E", labels[0], labels[1], p.read_bytes())

    # Dataset F (optional): 22 class folders, most of them DermNet photos under shorter folder names.
    if DATASET_F_DIR.is_dir():
        log(f"Scanning Dataset F from extracted directory: {DATASET_F_DIR.relative_to(PROJECT_ROOT)}")
        for p in sorted(DATASET_F_DIR.rglob("*")):
            if not p.is_file() or p.suffix.lower() == ".md":
                continue
            rel_path = p.relative_to(PROJECT_ROOT).as_posix()
            labels = labels_for_dataset_f(p.parent.name, p.name)
            if labels is None:
                yield ("skip", rel_path, p.name, "F", "dataset_f_unmapped_class", None, b"")
                continue
            yield ("file", rel_path, p.name, "F", labels[0], labels[1], p.read_bytes())


def dataset_source_signature(dataset_a_path: Path | None = None) -> str:
    """Fingerprint of the dataset files and label rules, so cached audits are rebuilt when either changes."""
    entries = [f"label-rules:{LABEL_RULES_VERSION}"]
    zip_a = dataset_a_path if (dataset_a_path and dataset_a_path.is_file()) else (PROJECT_ROOT / "datasets" / "20-skin-diseases.zip")
    for archive in (zip_a, PROJECT_ROOT / "datasets" / "dermnet.zip"):
        if archive.is_file():
            entries.append(f"{archive.name}:{archive.stat().st_size}")
    folders = [PROJECT_ROOT / "datasets" / "20-skin-diseases" / "Dataset", DERMNET_DIR / "train", DERMNET_DIR / "test"]
    folders += dataset_b_split_dirs()
    folders += [DATASET_E_DIR, DATASET_F_DIR]
    for folder in folders:
        if not folder.is_dir():
            continue
        count = 0
        total = 0
        for walk_root, _, files in os.walk(folder):
            for name in files:
                count += 1
                try:
                    total += os.path.getsize(os.path.join(walk_root, name))
                except OSError:
                    pass
        entries.append(f"{folder.relative_to(PROJECT_ROOT).as_posix()}:{count}:{total}")
    return sha256("|".join(entries).encode("utf-8")).hexdigest()[:16]


def _probe_image(data: bytes) -> tuple[int, int, int | None] | None:
    """Decode once and return (width, height, pHash); pHash is None for images below MIN_IMAGE_SIDE."""
    img = decode_bytes(data)
    if img is None:
        return None
    height, width = img.shape[:2]
    if min(height, width) < MIN_IMAGE_SIDE:
        return int(width), int(height), None
    return int(width), int(height), perceptual_hash(img)


def previous_split_assignments(manifest_path: Path) -> tuple[dict[str, str], str | None]:
    """(photo hash -> split, split lineage id) from the manifest of an earlier audit; ({}, None) when there is none."""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        assignments = {row["sha"]: row["split"] for row in payload["items"] if row.get("split") in ("train", "val", "test")}
        return assignments, (payload.get("audit") or {}).get("split_lineage")
    except (OSError, ValueError, KeyError, TypeError):
        return {}, None


def audit_and_split_datasets(
    dataset_a_path: Path | None = None, force_rebuild: bool = False, fresh_split: bool = False
):
    manifest_path = BUILD_DIR / "unified_manifest.json"
    signature = dataset_source_signature(dataset_a_path)
    if manifest_path.is_file() and not force_rebuild and not fresh_split:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") == MANIFEST_SCHEMA_VERSION and payload.get("source_signature") == signature:
            log(f"Loaded cached leak-free unified manifest ({len(payload['items'])} clean images)")
            items = [UnifiedItem(**row) for row in payload["items"]]
            return items, payload["audit"]

    started = time.perf_counter()
    skipped = Counter()
    raw_counts = Counter()

    # Step 1: Decode, filter tiny/corrupted, and collapse exact SHA-256 duplicates (cached for fast re-runs)
    raw_cache_path = BUILD_DIR / "raw_sha_cache.json"
    cached = None
    if raw_cache_path.is_file():
        cached = json.loads(raw_cache_path.read_text(encoding="utf-8"))
        if cached.get("source_signature") != signature:
            log("Dataset files changed since the cached scan; rescanning")
            cached = None
    if cached is not None:
        raw_counts = Counter(cached["raw_counts"])
        skipped = Counter(cached["skipped"])
        unique_items = [UnifiedItem(**row) for row in cached["items"]]
        log(f"Loaded cached SHA-unique scan ({len(unique_items)} pure SHA-unique images)")
    else:
        from concurrent.futures import ThreadPoolExecutor

        sha_buckets: dict[str, dict] = {}
        scanned = 0

        def consume(chunk: list[tuple], pool: ThreadPoolExecutor) -> None:
            """Apply the sequential dedup rules to one chunk; only the image decoding runs in parallel."""
            nonlocal scanned
            hashes = [sha256(rec[6]).hexdigest() if rec[0] != "skip" else "" for rec in chunk]
            pending: dict[str, bytes] = {}
            for rec, h in zip(chunk, hashes):
                if rec[0] != "skip" and h not in sha_buckets and h not in pending:
                    pending[h] = rec[6]
            probes = dict(zip(pending.keys(), pool.map(_probe_image, pending.values())))
            for (storage, rel_path, filename, src, broad, fine, _data), h in zip(chunk, hashes):
                if storage == "skip":
                    skipped[broad] += 1
                    continue
                scanned += 1
                if scanned % 10000 == 0:
                    log(f"  scanned {scanned} raw files ({len(sha_buckets)} unique SHA-256)...")
                raw_counts[f"dataset_{src}"] += 1
                explicit = src in EXPLICIT_LABEL_SOURCES and fine is not None
                existing = sha_buckets.get(h)
                if existing is not None:
                    existing["broad_set"].add(broad)
                    existing["sources"].add(src)
                    if explicit:
                        # Prefer an explicit class-folder fine label when an exact match spans sources
                        existing["fine"] = fine
                        existing["fine_from_a"] = True
                    elif not existing["fine_from_a"] and existing["fine"] is None and fine is not None:
                        existing["fine"] = fine
                    if filename.lower().startswith("isic_"):
                        existing["domain"] = "dermoscopy"
                    skipped["exact_sha_duplicates_collapsed"] += 1
                    continue

                probe = probes[h]
                if probe is None:
                    skipped["unreadable"] += 1
                    continue
                width, height, ph = probe
                if ph is None:
                    skipped["too_small_lt_96px"] += 1
                    continue
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
                    "fine_from_a": explicit,
                    "sha": h,
                    "phash": ph,
                    "width": int(width),
                    "height": int(height),
                }

        with ThreadPoolExecutor(max_workers=max(2, min(8, CPU_THREADS))) as pool:
            chunk: list[tuple] = []
            for record in _iter_raw_records(dataset_a_path):
                chunk.append(record)
                if len(chunk) >= 256:
                    wait_until_safe("the dataset scan")
                    consume(chunk, pool)
                    chunk = []
            if chunk:
                consume(chunk, pool)

        # Drop exact-hash label conflicts across broad classes
        unique_items: list[UnifiedItem] = []
        for rec in sha_buckets.values():
            if len(rec["broad_set"]) > 1:
                skipped["exact_sha_broad_conflicts_dropped"] += 1
                continue
            src_str = "+".join(sorted(rec["sources"]))
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
                    "source_signature": signature,
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
    merged_shas: dict[str, list[str]] = {}  # kept photo -> every photo collapsed into it (itself included)
    e_source_keys: dict[str, set[str]] = {}  # kept photo -> Dataset E source photos among those
    for indices in members.values():
        grp_items = [unique_items[i] for i in indices]
        broad_labels = {g.broad_label for g in grp_items}
        if len(broad_labels) > 1:
            skipped["phash_broad_conflicts_dropped"] += len(grp_items)
            continue
        # Pick best canonical copy: prefer explicit Dataset A fine label, then highest resolution, then deterministic path
        grp_items.sort(
            key=lambda g: (
                1
                if (
                    g.fine_label is not None
                    and any(s in EXPLICIT_LABEL_SOURCES for s in g.source_dataset.split("+"))
                )
                else 0,
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
            rep.source_dataset = "+".join(sorted(all_src))
        merged_shas[rep.sha] = [g.sha for g in grp_items]
        keys = {key for g in grp_items if (key := dataset_e_key_for_path(g.rel_path)) is not None}
        if keys:
            e_source_keys[rep.sha] = keys
        clean_items.append(rep)
    log(f"Perceptual hash dedup (<= {NEAR_DUPLICATE_BITS} bits) done in {time.perf_counter() - t_ph:.1f}s -> {len(clean_items)} clean images")

    # Step 3: Build wider similarity + filename-stem groups so related photos never straddle splits
    t_grp = time.perf_counter()
    pair_list = list(hamming_pairs([i.phash for i in clean_items], GROUP_CLUSTER_BITS))
    stem_index: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, item in enumerate(clean_items):
        # Dataset E names carry the class and a copy number, not a photo id, so its files are grouped
        # by source photo instead of by filename stem.
        if dataset_e_key_for_path(item.rel_path) is None:
            stem = normalize_stem_group(item.filename)
            if len(stem) >= 4:
                stem_index[(item.broad_label, stem)].append(idx)
        for key in e_source_keys.get(item.sha, ()):
            stem_index[(item.broad_label, f"dataset-e:{key}")].append(idx)
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

    # A photo that an earlier audit already placed keeps its split, and so does everything grouped with
    # it. Adding a dataset therefore never moves an earlier training photo into val/test (which would
    # flatter the model that was trained on it) and leaves the cached features of earlier photos valid.
    # A group that now joins photos from different splits goes to the first of train, val, test it touches.
    prior, lineage = ({}, None) if fresh_split else previous_split_assignments(manifest_path)
    # The lineage id names one unbroken history of this split. A model that records the same id was
    # never trained on a photo that is in val or test today, so it can be compared fairly with new ones.
    lineage = (lineage if prior else None) or uuid4().hex
    group_members: dict[int, list[int]] = defaultdict(list)
    for idx, grp in enumerate(groups):
        group_members[int(grp)].append(idx)
    unassigned: list[int] = []
    for indices in group_members.values():
        seen = {prior[sha] for i in indices for sha in merged_shas[clean_items[i].sha] if sha in prior}
        if seen:
            kept_split = next(sp for sp in ("train", "val", "test") if sp in seen)
            for i in indices:
                clean_items[i].split = kept_split
        else:
            unassigned.extend(indices)
    carried_over = len(clean_items) - len(unassigned)

    if unassigned:
        sub = np.array(sorted(unassigned), dtype=np.int64)
        sub_groups = np.asarray(groups)[sub]
        y_sub = y_strat[sub]
        if prior:
            # Photos added to an existing split are few, so they are balanced per class folder (the
            # finest label a dataset gives) to keep every added class present in train, val, and test.
            folder_keys = [
                f"{strat_keys[i]}||{PurePosixPath(clean_items[i].rel_path.split('::')[-1]).parent.name}" for i in sub
            ]
            folder_vocab = {k: n for n, k in enumerate(sorted(set(folder_keys)))}
            y_sub = np.array([folder_vocab[k] for k in folder_keys], dtype=np.int32)
        fold_ids = np.zeros(len(sub), dtype=np.int32)
        try:
            splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
            for fold_num, (_, val_idx) in enumerate(splitter.split(np.zeros(len(sub)), y_sub, sub_groups)):
                fold_ids[val_idx] = fold_num
        except ValueError:
            # Too few new groups for 20 stratified folds: place each group by a hash of its first photo.
            first_sha: dict[int, str] = {}
            for pos, idx in enumerate(sub):
                grp = int(sub_groups[pos])
                first_sha[grp] = min(first_sha.get(grp, clean_items[idx].sha), clean_items[idx].sha)
            for pos in range(len(sub)):
                fold_ids[pos] = int(first_sha[int(sub_groups[pos])][:8], 16) % 20
        for idx, fid in zip(sub, fold_ids):
            clean_items[int(idx)].split = "train" if fid < 15 else "val" if fid < 17 else "test"
    if prior:
        log(f"Kept the earlier split for {carried_over} images; assigned {len(unassigned)} new images")

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
        "split_kept_from_earlier_audit": carried_over if prior else 0,
        "split_newly_assigned": len(unassigned),
        "split_lineage": lineage,
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
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "source_signature": signature,
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
        script = "vit_backbones.py" if BACKBONES[name].get("source") == "timm" else "backbones.py"
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "backend" / "experiments" / script), str(xml.parent), name],
            check=True,
        )
    return xml


def fused_backbone_xml(names: list[str]) -> Path:
    """One feature extractor made of several backbones that share a single image input.

    The input takes the size of the largest backbone; a smaller backbone gets a bilinear resize inside
    the graph. Each backbone's features are L2-normalised before they are concatenated, so no single
    backbone dominates by scale.
    """
    xml = BUILD_DIR / "backbones" / f"{'+'.join(names)}.xml"
    if xml.is_file():
        return xml
    core = ov.Core()
    size = max(BACKBONES[name]["size"] for name in names)
    shared = ops.parameter((-1, size, size, 3), np.float32, name="image")
    parts = []
    for name in names:
        model = core.read_model(backbone_xml(name))
        for node in model.get_ordered_ops():
            if node.get_type_name() not in ("Parameter", "Result"):
                node.set_friendly_name(f"{name}/{node.get_friendly_name()}")
        branch_size = BACKBONES[name]["size"]
        feed = shared
        if branch_size != size:
            feed = ops.interpolate(
                shared,
                ops.constant(np.array([branch_size, branch_size], dtype=np.int64)),
                mode="linear",
                shape_calculation_mode="sizes",
                axes=ops.constant(np.array([1, 2], dtype=np.int64)),
            )
        model.get_parameters()[0].output(0).replace(feed.output(0))
        features = model.get_results()[0].input_value(0)
        parts.append(ops.normalize_l2(features, ops.constant(np.array([1], dtype=np.int64)), 1e-12, "max"))
    fused = ov.Model([ops.result(ops.concat(parts, 1))], [shared], "fused_feature_extractor")
    tmp_xml = xml.with_name(xml.stem + ".partial.xml")
    ov.save_model(fused, tmp_xml, compress_to_fp16=False)
    tmp_xml.with_suffix(".bin").replace(xml.with_suffix(".bin"))
    tmp_xml.replace(xml)
    return xml


def _read_json(path: Path) -> dict:
    """Parsed JSON object from ``path``; an empty dict when the file is missing or unreadable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


class AsyncExtractor:
    """Multi-stream OpenVINO feature extractor using AsyncInferQueue (3.5x-4.5x faster on CPU)."""

    def __init__(self, name: str, batch_size: int = 16):
        self.name = name
        self.size = BACKBONES[name]["size"]
        self.xml = backbone_xml(name)
        self.batch_size = batch_size
        self.net = ov.Core().compile_model(
            self.xml,
            "CPU",
            {"INFERENCE_PRECISION_HINT": "f32", "PERFORMANCE_HINT": "THROUGHPUT", "INFERENCE_NUM_THREADS": CPU_THREADS},
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
        """Extract shape (len(items), len(views), dim) float16 array cached on disk as .npy + manifest.

        A view is a deterministic function of the photo, so rows are reused by photo hash from every
        finished cache of the same backbone, preprocessing, and views, including the previous version
        of this cache. When the photo list changes, only the photos that are new get extracted.
        """
        cache_prefix.parent.mkdir(parents=True, exist_ok=True)
        npy_path = cache_prefix.with_suffix(".npy")
        meta_path = cache_prefix.with_suffix(".json")
        shas_path = cache_prefix.with_suffix(".shas.json")
        prev_npy_path = cache_prefix.with_suffix(".prev.npy")
        prev_shas_path = cache_prefix.with_suffix(".prev.shas.json")
        shas = [i.sha for i in items]
        key_sig = sha256(json.dumps(shas + [mode] + views, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
        sidecar = json.dumps({"sig": key_sig, "backbone": self.name, "mode": mode, "views": views, "shas": shas})

        n_items = len(items)
        n_views = len(views)
        resume_from = 0
        if npy_path.is_file() and meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("sig") == key_sig:
                if meta.get("complete") is True:
                    if not shas_path.is_file():
                        shas_path.write_text(sidecar, encoding="utf-8")
                    return np.load(npy_path, mmap_mode="r")
                resume_from = min(int(meta.get("done_items", 0)), n_items)
            elif meta.get("complete") is True and _read_json(shas_path).get("sig") == meta.get("sig"):
                # The photo list changed: set the finished cache aside as a source of rows for the new one.
                npy_path.replace(prev_npy_path)
                shas_path.replace(prev_shas_path)
                meta_path.unlink()

        # Allocate disk-backed float16 memmap so peak RAM stays tiny (<250 MB). An interrupted
        # extraction resumes from its last checkpoint instead of starting over.
        out = None
        if resume_from > 0:
            out = np.lib.format.open_memmap(str(npy_path), mode="r+")
            if out.shape != (n_items, n_views, self.dim) or out.dtype != np.float16:
                del out
                out, resume_from = None, 0
        if out is None:
            shas_path.unlink(missing_ok=True)  # it described the array that is about to be overwritten
            out = np.lib.format.open_memmap(
                str(npy_path), mode="w+", dtype=np.float16, shape=(n_items, n_views, self.dim)
            )
            meta_path.write_text(json.dumps({"sig": key_sig, "complete": False, "done_items": 0}), encoding="utf-8")

        # Copy every row that an existing cache already holds; only the rest is run through the backbone.
        row_source = self._reusable_rows(cache_prefix, mode, views, set(shas[resume_from:]))
        pending = [i for i in range(resume_from, n_items) if shas[i] not in row_source]
        reused = n_items - resume_from - len(pending)
        if reused:
            by_donor: dict[Path, list[tuple[int, int]]] = defaultdict(list)
            for item_i in range(resume_from, n_items):
                source = row_source.get(shas[item_i])
                if source is not None:
                    by_donor[source[0]].append((item_i, source[1]))
            for donor_path, pairs in by_donor.items():
                donor = np.load(donor_path, mmap_mode="r")
                for start in range(0, len(pairs), 4096):
                    chunk = pairs[start : start + 4096]
                    out[[target for target, _ in chunk]] = donor[[row for _, row in chunk]]
                del donor
            out.flush()
        n_new_views = len(pending) * n_views
        log(
            f"  extracting {len(pending)} x {n_views} = {n_new_views} views "
            f"with {self.name}/{mode} (nreq={self.nreq}, bs={self.batch_size})"
            + (f", {reused} images reused from cached features" if reused else "")
            + (f", resuming at image {resume_from}" if resume_from else "")
        )
        if pending:
            wait_until_safe("feature extraction")
        started = time.perf_counter()
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
            for n_started, item_i in enumerate(pending):
                item = items[item_i]
                if n_started and n_started % 2000 == 0:
                    # Checkpoint: drain in-flight requests so every image before item_i is on disk
                    if batch_pixels:
                        queue.start_async({0: np.stack(batch_pixels, axis=0)}, list(batch_coords))
                        done_views += len(batch_pixels)
                        batch_pixels.clear()
                        batch_coords.clear()
                    queue.wait_all()
                    out.flush()
                    meta_path.write_text(
                        json.dumps({"sig": key_sig, "complete": False, "done_items": item_i}), encoding="utf-8"
                    )
                    wait_until_safe("the next feature-extraction chunk")
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
                            rem = (n_new_views - done_views) / max(rate, 1e-3)
                            log(f"    {done_views}/{n_new_views} views ({rate:.1f} img/s, ~{rem/60:.1f}m left)")
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

        shas_path.write_text(sidecar, encoding="utf-8")
        meta_path.write_text(
            json.dumps({"sig": key_sig, "complete": True, "shape": [n_items, n_views, self.dim]}),
            encoding="utf-8",
        )
        gc.collect()
        for leftover in (prev_npy_path, prev_shas_path):
            try:
                leftover.unlink(missing_ok=True)
            except OSError:
                pass  # still mapped somewhere; it is only disk space and the next rebuild replaces it
        elapsed = time.perf_counter() - started
        log(f"  done {n_new_views} views in {elapsed:.1f}s ({n_new_views / max(elapsed, 1e-3):.1f} img/s)")
        return np.load(npy_path, mmap_mode="r")

    def _reusable_rows(
        self, cache_prefix: Path, mode: str, views: list[str], wanted: set[str]
    ) -> dict[str, tuple[Path, int]]:
        """Photo hash -> (cache file, row) for wanted photos that a finished cache already holds."""
        found: dict[str, tuple[Path, int]] = {}
        if not wanted:
            return found
        target_npy = cache_prefix.with_suffix(".npy")
        own_previous = cache_prefix.with_suffix(".prev.shas.json")
        folders = {cache_prefix.parent, BUILD_DIR / "features", BUILD_DIR / "pilot_features"}
        candidates = {p for folder in folders if folder.is_dir() for p in folder.glob("*.shas.json")}
        # The previous version of this cache goes first: it holds most of the wanted rows.
        for sidecar_path in sorted(candidates, key=lambda p: (p != own_previous, str(p))):
            base = sidecar_path.name[: -len(".shas.json")]
            donor_npy = sidecar_path.with_name(base + ".npy")
            if donor_npy == target_npy or not donor_npy.is_file():
                continue
            info = _read_json(sidecar_path)
            if info.get("backbone") != self.name or info.get("mode") != mode or info.get("views") != views:
                continue
            if not base.endswith(".prev"):
                # A live cache counts only when it is finished and its hash list describes that array.
                meta = _read_json(sidecar_path.with_name(base + ".json"))
                if meta.get("complete") is not True or meta.get("sig") != info.get("sig"):
                    continue
            donor_shas = info.get("shas") or []
            if np.load(donor_npy, mmap_mode="r").shape != (len(donor_shas), len(views), self.dim):
                continue
            for row, sha in enumerate(donor_shas):
                if sha in wanted and sha not in found:
                    found[sha] = (donor_npy, row)
            if len(found) == len(wanted):
                break
        return found


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
    wait_until_safe("fitting a classifier head")
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

    wait_until_safe("fitting a classifier head")
    tf.config.threading.set_intra_op_parallelism_threads(min(8, CPU_THREADS))
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
def build_classifier_graph(
    config: dict, head_type: str, head_artifacts: dict, temp_broad: float, temp_fine: float
) -> ov.Model:
    """Backbone and head(s) as one OpenVINO graph with two calibrated-logit outputs: broad (7) and fine (21)."""
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

    return ov.Model([res_broad, res_fine], base_model.get_parameters(), "hierarchical_skin_classifier")


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
    expected_broad_logits: np.ndarray | None,
    expected_fine_logits: np.ndarray | None,
    out_dir: Path = MODEL_DIR,
    compress_fp16: bool = False,
    verify=None,
) -> None:
    """Write ``classifier.zip``, ``evaluation.json``, and ``quality.json`` into ``out_dir``.

    ``verify(xml_path)`` may score the saved graph end to end; its ``final_model`` then replaces the
    one in ``report`` so that the published numbers describe the exact file that ships.
    """
    core = ov.Core()
    graph = build_classifier_graph(config, head_type, head_artifacts, temp_broad, temp_fine)

    out_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out_dir) as staging:
        root = Path(staging)
        xml_path = root / "classifier.xml"
        ov.save_model(graph, xml_path, compress_to_fp16=compress_fp16)

        net = core.compile_model(xml_path, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
        out = net(check_pixels)
        got_broad = out[net.output(0)]
        got_fine = out[net.output(1)]
        del net

        # Check 1: the exported graph must reproduce the NumPy head applied to float32 features taken
        # straight from the backbone for the same pixels. Half-precision weights round the backbone
        # slightly, so that export gets a wider tolerance (and is scored end to end by ``verify``).
        feature_net = core.compile_model(config["xml"], "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
        fresh_views = np.asarray(feature_net(check_pixels)[0], dtype=np.float32)[:, None, :]
        del feature_net
        if head_type == "linear":
            fresh_broad = linear_logits(*head_artifacts["broad"], fresh_views, l2_norm=config.get("l2_norm", True))
            fresh_fine = linear_logits(*head_artifacts["fine"], fresh_views, l2_norm=config.get("l2_norm", True))
        else:
            fresh_broad, fresh_fine = mlp_predict_logits(head_artifacts["models"], fresh_views)
        head_gap = max(
            float(np.abs(got_broad - fresh_broad / temp_broad).max()),
            float(np.abs(got_fine - fresh_fine / temp_fine).max()),
        )
        log(f"  export check: saved graph vs NumPy head on fresh features differ by at most {head_gap:.4f}")
        if compress_fp16:
            if head_gap > 0.15:
                raise RuntimeError(f"Half-precision export drifted too far from the trained head ({head_gap:.3f})")
        else:
            np.testing.assert_allclose(got_broad, fresh_broad / temp_broad, rtol=2e-3, atol=5e-3)
            np.testing.assert_allclose(got_fine, fresh_fine / temp_fine, rtol=2e-3, atol=5e-3)

        # Check 2 (loose): the cached float16 training features must describe the same pixels the
        # deployed graph sees, i.e. training and inference preprocessing agree end to end.
        if expected_broad_logits is not None and expected_fine_logits is not None:
            drift = max(
                float(np.abs(got_broad - expected_broad_logits / temp_broad).max()),
                float(np.abs(got_fine - expected_fine_logits / temp_fine).max()),
            )
            log(f"  export check: deployed graph vs cached-feature logits differ by at most {drift:.4f}")
            if drift > 0.25:
                raise RuntimeError(f"Exported graph disagrees with cached training features (max logit diff {drift:.3f})")

        if verify is not None:
            shipped = verify(xml_path)
            report["comparison"]["as_shipped"] = {
                "note": "The saved classifier graph scored end to end on val/test (orig + flip), exactly as the scanner runs it.",
                "val_selection": shipped["val_selection"],
                "test_headline": shipped["test_headline"],
            }
            report["final_model"] = shipped["final_model"]

        final_model = report["final_model"]
        metadata = {
            "format": 3,
            "created_at": datetime.now(UTC).isoformat(),
            "backbone": config["backbone"],
            "head_type": head_type,
            "input_size": config["size"],
            "input_layout": "NHWC RGB float32 0-255",
            "preprocessing": config["mode"],
            "l2_norm": bool(config.get("l2_norm", True)),
            "weights_precision": "fp16" if compress_fp16 else "fp32",
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
            "split_lineage": (report.get("dataset_audit") or {}).get("split_lineage"),
            "test_summary": {
                "clinical_broad": {
                    k: final_model["test_clinical"]["broad"][k]
                    for k in ("accuracy", "macro_f1", "balanced_accuracy", "top3_accuracy", "ece_15_bins")
                },
                "clinical_fine": {
                    k: final_model["test_clinical"]["fine"][k]
                    for k in ("accuracy", "macro_f1", "balanced_accuracy", "top3_accuracy")
                },
                "dermoscopy_broad": {
                    k: final_model["test_dermoscopy"]["broad"][k]
                    for k in ("accuracy", "macro_f1", "balanced_accuracy")
                },
                "overall_broad": {
                    k: final_model["test_overall"]["broad"][k]
                    for k in ("accuracy", "macro_f1", "balanced_accuracy", "top3_accuracy", "ece_15_bins")
                },
            },
        }

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
            (root / name).replace(out_dir / name)


# --------------------------------------------------------------------------- recipe training & evaluation
# The scanner (backend/edge_app.py) only shows a category when the top score reaches this value.
APP_MIN_REPORT_SCORE = 0.65


@dataclass
class SplitArrays:
    train_items: list
    val_items: list
    test_items: list
    y_broad_tr: np.ndarray
    y_fine_tr: np.ndarray
    is_derm_tr: np.ndarray
    groups_tr: np.ndarray
    y_broad_va: np.ndarray
    y_fine_va: np.ndarray
    is_derm_va: np.ndarray
    y_broad_te: np.ndarray
    y_fine_te: np.ndarray
    is_derm_te: np.ndarray


def manifest_fingerprint(items: list[UnifiedItem]) -> str:
    """Short hash of which photo sits in which split with which labels.

    Two machines with the same fingerprint train, calibrate, and test on exactly the same photos.
    """
    rows = sorted(f"{i.sha}:{i.split}:{i.broad_label}:{i.fine_label}" for i in items)
    return sha256("\n".join(rows).encode("utf-8")).hexdigest()[:16]


def build_split_arrays(items: list[UnifiedItem]) -> SplitArrays:
    broad_index = {c: i for i, c in enumerate(BROAD_CLASSES)}
    fine_index = {c: i for i, c in enumerate(FINE_CLASSES)}

    def labels(subset: list[UnifiedItem]):
        return (
            np.array([broad_index[i.broad_label] for i in subset], dtype=np.int32),
            np.array([fine_index[i.fine_label] if i.fine_label else -1 for i in subset], dtype=np.int32),
            np.array([i.domain == "dermoscopy" for i in subset], dtype=bool),
        )

    train_items = [i for i in items if i.split == "train"]
    val_items = [i for i in items if i.split == "val"]
    test_items = [i for i in items if i.split == "test"]
    yb_tr, yf_tr, dm_tr = labels(train_items)
    yb_va, yf_va, dm_va = labels(val_items)
    yb_te, yf_te, dm_te = labels(test_items)
    return SplitArrays(
        train_items, val_items, test_items,
        yb_tr, yf_tr, dm_tr, np.array([i.group_id for i in train_items], dtype=np.int32),
        yb_va, yf_va, dm_va,
        yb_te, yf_te, dm_te,
    )


def selection_score(
    b_logits: np.ndarray, f_logits: np.ndarray, y_broad: np.ndarray, y_fine: np.ndarray, is_derm: np.ndarray
) -> dict:
    """Composite used for every model-selection decision (computed on train CV or val, never on test)."""
    clin = ~is_derm
    cm_b = classification_metrics(b_logits[clin].argmax(1), y_broad[clin], BROAD_CLASSES)
    dm_b = classification_metrics(b_logits[is_derm].argmax(1), y_broad[is_derm], BROAD_CLASSES)
    f_mask = clin & (y_fine >= 0)
    cm_f = classification_metrics(f_logits[f_mask].argmax(1), y_fine[f_mask], FINE_CLASSES)
    composite = 0.45 * cm_b["macro_f1"] + 0.25 * cm_b["accuracy"] + 0.20 * cm_f["macro_f1"] + 0.10 * dm_b["accuracy"]
    return {
        "clinical_broad_acc": round(cm_b["accuracy"], 4),
        "clinical_broad_macro_f1": round(cm_b["macro_f1"], 4),
        "clinical_fine_acc": round(cm_f["accuracy"], 4),
        "clinical_fine_macro_f1": round(cm_f["macro_f1"], 4),
        "dermoscopy_broad_acc": round(dm_b["accuracy"], 4),
        "composite": round(composite, 4),
    }


def app_rule_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float, min_margin: float) -> dict:
    """What the scanner shows: an answer only when the top score and its lead pass the report rule."""
    report_threshold = max(APP_MIN_REPORT_SCORE, float(threshold))
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    shown = (top2[:, 1] >= report_threshold) & ((top2[:, 1] - top2[:, 0]) >= min_margin)
    pred = probs.argmax(1)
    return {
        "report_threshold": report_threshold,
        "min_margin": float(min_margin),
        "answered_rate": float(shown.mean()) if len(labels) else 0.0,
        "accuracy_when_answered": float((pred[shown] == labels[shown]).mean()) if shown.any() else None,
        "correct_answers_shown_rate": float(((pred == labels) & shown).mean()) if len(labels) else 0.0,
        "wrong_answers_shown": int(((pred != labels) & shown).sum()),
    }


def calibrate_on_val(data: SplitArrays, val_b_logits: np.ndarray, val_f_logits: np.ndarray):
    """Temperature scaling and the Uncertain threshold, fitted on the held-out val split only."""
    w_cal_broad = np.where(data.is_derm_va, 0.25, 1.0)
    temp_broad = fit_temperature(val_b_logits, data.y_broad_va, sample_weight=w_cal_broad)
    val_fine_mask = data.y_fine_va >= 0
    w_cal_fine = np.where(data.is_derm_va[val_fine_mask], 0.25, 1.0)
    temp_fine = fit_temperature(val_f_logits[val_fine_mask], data.y_fine_va[val_fine_mask], sample_weight=w_cal_fine)
    val_b_probs = softmax(val_b_logits / temp_broad)
    clin = ~data.is_derm_va
    threshold = min(choose_threshold(val_b_probs[clin], data.y_broad_va[clin], target=0.78, min_coverage=0.72), 0.36)
    return temp_broad, temp_fine, threshold


def evaluate_model(
    data: SplitArrays,
    val_b_logits: np.ndarray,
    test_b_logits: np.ndarray,
    test_f_logits: np.ndarray,
    temp_broad: float,
    temp_fine: float,
    threshold: float,
    min_margin: float,
) -> dict:
    """Clinical / dermoscopy / overall metrics on the held-out test split (plus clinical val for reference)."""
    fine_index = {c: i for i, c in enumerate(FINE_CLASSES)}
    val_b_probs = softmax(val_b_logits / temp_broad)
    test_b_probs = softmax(test_b_logits / temp_broad)
    test_f_probs = softmax(test_f_logits / temp_fine)

    def eval_slice(mask: np.ndarray) -> dict:
        b_p = test_b_probs[mask]
        b_y = data.y_broad_te[mask]
        b_pred = b_p.argmax(1)
        b_res = classification_metrics(b_pred, b_y, BROAD_CLASSES)
        b_res.update(probability_metrics(b_p, b_y))
        b_res["uncalibrated_ece_15_bins"] = expected_calibration_error(softmax(test_b_logits[mask]), b_y)
        if len(b_y) >= 20:
            b_res.update(bootstrap_ci(b_pred, b_y, BROAD_CLASSES))
        sel = selective_metrics(b_p, b_y, threshold, BROAD_CLASSES)

        f_mask = mask & (data.y_fine_te >= 0)
        if f_mask.any():
            f_p = test_f_probs[f_mask]
            f_y = data.y_fine_te[f_mask]
            f_res = classification_metrics(f_p.argmax(1), f_y, FINE_CLASSES)
            f_res.update(probability_metrics(f_p, f_y))
            # Hierarchical conditioned fine accuracy (when broad class is predicted, pick best child fine class)
            cond_preds = []
            for row_b, row_f in zip(test_b_probs[f_mask], f_p):
                top_broad = BROAD_CLASSES[int(np.argmax(row_b))]
                child_idxs = [fine_index[fc] for fc in BROAD_TO_FINE[top_broad]]
                cond_preds.append(child_idxs[int(np.argmax(row_f[child_idxs]))])
            cond_m = classification_metrics(np.array(cond_preds, dtype=np.int32), f_y, FINE_CLASSES)
            f_res["hierarchical_conditioned_accuracy"] = cond_m["accuracy"]
            f_res["hierarchical_conditioned_macro_f1"] = cond_m["macro_f1"]
        else:
            f_res = {"images": 0}
        return {
            "broad": b_res,
            "broad_selective": sel,
            "scanner_report_rule": app_rule_metrics(b_p, b_y, threshold, min_margin),
            "fine": f_res,
        }

    val_clin = ~data.is_derm_va
    return {
        "val_clinical_broad": {
            **classification_metrics(val_b_probs[val_clin].argmax(1), data.y_broad_va[val_clin], BROAD_CLASSES),
            **probability_metrics(val_b_probs[val_clin], data.y_broad_va[val_clin]),
        },
        "test_clinical": eval_slice(~data.is_derm_te),
        "test_dermoscopy": eval_slice(data.is_derm_te),
        "test_overall": eval_slice(np.ones(len(data.test_items), dtype=bool)),
    }


def headline(final_model: dict) -> dict:
    """Compact test-set summary used for logs and the old-vs-new comparison table."""
    tc = final_model["test_clinical"]
    rule = tc["scanner_report_rule"]
    return {
        "clinical_broad_acc": round(tc["broad"]["accuracy"], 4),
        "clinical_broad_acc_95ci": [round(v, 4) for v in tc["broad"].get("accuracy_95ci", [])],
        "clinical_broad_macro_f1": round(tc["broad"]["macro_f1"], 4),
        "clinical_broad_top3_acc": round(tc["broad"]["top3_accuracy"], 4),
        "clinical_ece": round(tc["broad"]["ece_15_bins"], 4),
        "clinical_fine_acc": round(tc["fine"].get("accuracy", 0.0), 4),
        "clinical_fine_macro_f1": round(tc["fine"].get("macro_f1", 0.0), 4),
        "dermoscopy_broad_acc": round(final_model["test_dermoscopy"]["broad"]["accuracy"], 4),
        "overall_broad_acc": round(final_model["test_overall"]["broad"]["accuracy"], 4),
        "scanner_answered_rate_clinical": round(rule["answered_rate"], 4),
        "scanner_accuracy_when_answered_clinical": (
            round(rule["accuracy_when_answered"], 4) if rule["accuracy_when_answered"] is not None else None
        ),
        "scanner_correct_answers_shown_rate_clinical": round(rule["correct_answers_shown_rate"], 4),
    }


def predict_bundle_logits(compiled, items: list[UnifiedItem], size: int, mode: str, batch_size: int = 16):
    """Run a dual-output classifier graph over items with orig + flip logit averaging."""
    nreq = max(2, int(compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")))
    n = len(items)
    b_out = np.zeros((n, 2, len(BROAD_CLASSES)), dtype=np.float32)
    f_out = np.zeros((n, 2, len(FINE_CLASSES)), dtype=np.float32)
    queue = ov.AsyncInferQueue(compiled, nreq)

    def on_done(infer_request, coords):
        rb = infer_request.get_output_tensor(0).data
        rf = infer_request.get_output_tensor(1).data
        for k, (item_i, view_i) in enumerate(coords):
            b_out[item_i, view_i] = rb[k]
            f_out[item_i, view_i] = rf[k]

    queue.set_callback(on_done)
    archive_cache: dict[str, ZipFile] = {}
    batch_pixels: list[np.ndarray] = []
    batch_coords: list[tuple[int, int]] = []
    try:
        for item_i, item in enumerate(items):
            img = read_item_image(item, archive_cache)
            if img is None:
                raise RuntimeError(f"Failed to decode audited image: {item.rel_path}")
            for view_i, view in enumerate((img, cv2.flip(img, 1))):
                batch_pixels.append(prepare(view, size, mode))
                batch_coords.append((item_i, view_i))
                if len(batch_pixels) >= batch_size:
                    queue.start_async({0: np.stack(batch_pixels, axis=0)}, list(batch_coords))
                    batch_pixels.clear()
                    batch_coords.clear()
        if batch_pixels:
            queue.start_async({0: np.stack(batch_pixels, axis=0)}, list(batch_coords))
        queue.wait_all()
    finally:
        for zf in archive_cache.values():
            zf.close()
    del queue
    return b_out.mean(axis=1), f_out.mean(axis=1)


def graph_val_test_logits(model: ov.Model, weights_digest: str, data: SplitArrays, size: int, mode: str):
    """Orig + flip logits of a dual-output classifier graph on the val and test splits (cached on disk)."""
    cache_key = sha256(
        (f"{weights_digest}|{size}|{mode}|" + "|".join(i.sha for i in data.val_items + data.test_items)).encode("utf-8")
    ).hexdigest()[:16]
    cache_path = BUILD_DIR / "graph_eval" / f"{cache_key}.npz"
    if cache_path.is_file():
        arrays = np.load(cache_path)
        return arrays["val_b"], arrays["val_f"], arrays["test_b"], arrays["test_f"]
    wait_until_safe("scoring a classifier graph on val/test")
    compiled = ov.Core().compile_model(
        model, "CPU",
        {"INFERENCE_PRECISION_HINT": "f32", "PERFORMANCE_HINT": "THROUGHPUT", "INFERENCE_NUM_THREADS": CPU_THREADS},
    )
    val_b, val_f = predict_bundle_logits(compiled, data.val_items, size, mode)
    test_b, test_f = predict_bundle_logits(compiled, data.test_items, size, mode)
    del compiled
    gc.collect()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, val_b=val_b, val_f=val_f, test_b=test_b, test_f=test_f)
    return val_b, val_f, test_b, test_f


def as_shipped_evaluation(
    xml_path: Path, data: SplitArrays, config: dict, temp_broad: float, temp_fine: float, threshold: float, min_margin: float
) -> dict:
    """Score a saved classifier graph end to end on val/test, exactly as the scanner runs it."""
    digest = sha256(xml_path.with_suffix(".bin").read_bytes()).hexdigest()
    log(f"Scoring the saved {config['backbone']} graph end to end on the val/test splits")
    val_b, val_f, test_b, test_f = graph_val_test_logits(
        ov.Core().read_model(xml_path), digest, data, config["size"], config["mode"]
    )
    # The graph already divides by its temperatures; undo that so the scorer applies them itself.
    val_b, test_b = val_b * temp_broad, test_b * temp_broad
    val_f, test_f = val_f * temp_fine, test_f * temp_fine
    final_model = evaluate_model(data, val_b, test_b, test_f, temp_broad, temp_fine, threshold, min_margin)
    return {
        "val_selection": selection_score(val_b, val_f, data.y_broad_va, data.y_fine_va, data.is_derm_va),
        "test_headline": headline(final_model),
        "final_model": final_model,
        "raw_val_logits": (val_b, val_f),
    }


def evaluate_previous_bundle(bundle_path: Path, data: SplitArrays, lineage: str | None = None) -> dict | None:
    """Score a saved bundle (normally the deployed one) on the val/test splits, exactly as shipped.

    A bundle that records today's split ``lineage`` was never trained on a photo that is in val or test
    now, so its numbers are a fair comparison. Any other bundle may have seen these photos in training,
    and its numbers can only be optimistic for it.
    """
    if not bundle_path.is_file():
        return None
    with ZipFile(bundle_path) as zf:
        if "metadata.json" not in zf.namelist():
            return None
        meta = json.loads(zf.read("metadata.json"))
        compatible = (
            meta.get("format") == 3
            and meta.get("output_type") == "calibrated_logits"
            and list(meta.get("classes", [])) == list(BROAD_CLASSES)
            and list(meta.get("fine_classes", [])) == list(FINE_CLASSES)
            and meta.get("preprocessing") in PREPROCESSING_MODES
        )
        if not compatible:
            log("Deployed bundle uses a different format/taxonomy; skipping its evaluation")
            return None
        xml_bytes, bin_bytes = zf.read("classifier.xml"), zf.read("classifier.bin")

    log(f"Scoring the saved bundle {bundle_path.parent.name}/{bundle_path.name} ({meta.get('backbone')}) on the val/test splits")
    val_b, val_f, test_b, test_f = graph_val_test_logits(
        ov.Core().read_model(xml_bytes, bin_bytes),
        sha256(bin_bytes).hexdigest(),
        data,
        int(meta["input_size"]),
        meta["preprocessing"],
    )

    threshold = float(meta.get("uncertain_threshold", 0.36))
    min_margin = float(meta.get("min_margin", 0.03))
    # The bundle already divides by its own temperatures, so no further scaling is applied here.
    final_model = evaluate_model(data, val_b, test_b, test_f, 1.0, 1.0, threshold, min_margin)
    same_lineage = bool(lineage) and meta.get("split_lineage") == lineage
    return {
        "backbone": meta.get("backbone"),
        "preprocessing": meta.get("preprocessing"),
        "head_type": meta.get("head_type"),
        "created_at": meta.get("created_at"),
        "same_split_lineage": same_lineage,
        "caveat": (
            "Trained on the same split lineage: none of these val/test images were in its training data."
            if same_lineage
            else "May have been trained on an earlier split, so some of these val/test images may have been in its "
            "training data; these numbers can only be optimistic for it."
        ),
        "val_selection": selection_score(val_b, val_f, data.y_broad_va, data.y_fine_va, data.is_derm_va),
        "test_headline": headline(final_model),
        "final_model": final_model,
    }


def run_recipe(data: SplitArrays, bb_name: str, mode: str, args, fixed: dict | None = None) -> dict:
    """Full feature extraction, head selection and fit, val calibration, and one test evaluation.

    ``fixed`` pins the weighting/head choices (used to re-fit the previous recipe on the new split);
    otherwise they are selected by 5-fold grouped CV on the train split.
    """
    n_broad, n_fine = len(BROAD_CLASSES), len(FINE_CLASSES)
    y_broad_tr, y_fine_tr, is_derm_tr = data.y_broad_tr, data.y_fine_tr, data.is_derm_tr
    train_items = data.train_items

    log(f"[{bb_name}/{mode}] Full feature extraction (disk-backed float16 memmap)")
    ex = AsyncExtractor(bb_name)
    feat_dir = BUILD_DIR / "features"
    prefix = f"{ex.name}__{mode}"
    x_tr_clean = ex.extract_views_to_array(train_items, mode, ["orig", "flip"], feat_dir / f"{prefix}__train_clean")
    x_va_clean = ex.extract_views_to_array(data.val_items, mode, ["orig", "flip"], feat_dir / f"{prefix}__val_clean")
    x_te_clean = ex.extract_views_to_array(data.test_items, mode, ["orig", "flip"], feat_dir / f"{prefix}__test_clean")

    # Augmented views for clinical training images (extra views for minority fine classes)
    clin_tr_indices = np.flatnonzero(~is_derm_tr).tolist()
    fine_counts_tr = Counter(int(v) for v in y_fine_tr if v >= 0)
    minority_fine_ids = {fid for fid, cnt in fine_counts_tr.items() if cnt < 450}
    min_tr_indices = [idx for idx in clin_tr_indices if int(y_fine_tr[idx]) in minority_fine_ids]

    x_tr_aug_base = None
    x_tr_aug_min = None
    if args.aug_views > 0:
        x_tr_aug_base = ex.extract_views_to_array(
            [train_items[i] for i in clin_tr_indices],
            mode,
            [f"aug{k}" for k in range(args.aug_views)],
            feat_dir / f"{prefix}__train_clin_aug{args.aug_views}",
        )
    if args.minority_aug_views > args.aug_views and min_tr_indices:
        x_tr_aug_min = ex.extract_views_to_array(
            [train_items[i] for i in min_tr_indices],
            mode,
            [f"aug{k}" for k in range(args.aug_views, args.minority_aug_views)],
            feat_dir / f"{prefix}__train_min_aug{args.minority_aug_views}",
        )
    ex_size, ex_dim, ex_xml, ex_name = ex.size, ex.dim, ex.xml, ex.name
    # Release OpenVINO extractor RAM before training heads
    del ex
    gc.collect()

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

    tr_folds = list(
        StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED).split(
            np.zeros(len(train_items)),
            np.array([f"{i.broad_label}||{i.domain}" for i in train_items]),
            data.groups_tr,
        )
    )

    def cv_linear(isic_w: float, scheme: str, c_val: float, include_aug: bool, with_fine: bool) -> dict:
        oof_b = np.zeros((len(train_items), n_broad), dtype=np.float32)
        oof_f = np.zeros((len(train_items), n_fine), dtype=np.float32)
        for tr_idx, va_idx in tr_folds:
            X_f, yb_f, yf_f, dm_f = assemble_train_matrix(tr_idx, include_aug=include_aug)
            wb = compute_domain_balanced_weights(yb_f, dm_f, n_broad, scheme, isic_w)
            sc_b, hd_b = fit_linear_head(X_f, yb_f, wb, c_val=c_val, l2_norm=True)
            va_views = np.asarray(x_tr_clean[va_idx], dtype=np.float32)
            oof_b[va_idx] = linear_logits(sc_b, hd_b, va_views, l2_norm=True)
            if with_fine:
                f_valid = yf_f >= 0
                wf = compute_domain_balanced_weights(
                    np.where(f_valid, yf_f, 0), dm_f, n_fine, "sqrt", isic_w, valid_mask=f_valid
                )
                sc_f, hd_f = fit_linear_head(X_f[f_valid], yf_f[f_valid], wf[f_valid], c_val=c_val, l2_norm=True)
                oof_f[va_idx] = linear_logits(sc_f, hd_f, va_views, l2_norm=True)
            del X_f
            gc.collect()
        return {
            "head_type": "linear",
            "C": c_val,
            "isic_weight": isic_w,
            "class_weight": scheme,
            **selection_score(oof_b, oof_f, y_broad_tr, y_fine_tr, is_derm_tr),
        }

    def cv_mlp(isic_w: float, scheme: str) -> dict:
        oof_b = np.zeros((len(train_items), n_broad), dtype=np.float32)
        oof_f = np.zeros((len(train_items), n_fine), dtype=np.float32)
        for fold_i, (tr_idx, va_idx) in enumerate(tr_folds):
            X_f, yb_f, yf_f, dm_f = assemble_train_matrix(tr_idx, include_aug=True)
            wb = compute_domain_balanced_weights(yb_f, dm_f, n_broad, scheme, isic_w)
            f_valid = yf_f >= 0
            wf = compute_domain_balanced_weights(
                np.where(f_valid, yf_f, 0), dm_f, n_fine, "sqrt", isic_w, valid_mask=f_valid
            )
            # Use validation fold orig view for early stopping
            x_va_fold = np.asarray(x_tr_clean[va_idx, 0], dtype=np.float32)
            w_va_fold = compute_domain_balanced_weights(y_broad_tr[va_idx], is_derm_tr[va_idx], n_broad, scheme, isic_w)
            m_weights = fit_multitask_mlp(
                X_f, yb_f, wb, yf_f, wf,
                x_va_fold, y_broad_tr[va_idx], w_va_fold,
                n_broad=n_broad, n_fine=n_fine,
                seed=SEED + fold_i, l2_norm=True,
            )
            b_log, f_log = mlp_predict_logits([m_weights], np.asarray(x_tr_clean[va_idx], dtype=np.float32))
            oof_b[va_idx] = b_log
            oof_f[va_idx] = f_log
            del X_f, x_va_fold
            gc.collect()
        return {
            "head_type": "multitask_residual_mlp",
            "hidden_dim": 512,
            "isic_weight": isic_w,
            "class_weight": scheme,
            **selection_score(oof_b, oof_f, y_broad_tr, y_fine_tr, is_derm_tr),
        }

    # ------------------------------------------------------------------ head / weighting selection (train CV only)
    cv_tables: dict = {}
    if fixed is not None:
        winner_head = {
            "head_type": fixed.get("head_type", "multitask_residual_mlp"),
            "C": fixed.get("C", 0.005),
            "isic_weight": float(fixed["isic_weight"]),
            "class_weight": fixed["class_weight"],
        }
        log(f"[{bb_name}/{mode}] Using pinned head settings: {winner_head}")
    elif args.head_search == "fast":
        log(f"[{bb_name}/{mode}] Head selection: Multi-Task Residual MLP weighting ({CV_FOLDS}-fold grouped CV on train)")
        grid = ((0.40, "none"), (0.40, "sqrt"), (0.25, "sqrt"))
        # The CV table only depends on the train split, its labels, and the cached features, so a
        # re-run with the same inputs reuses it.
        cv_key = sha256(
            json.dumps(
                {
                    "train": sha256("".join(i.sha for i in train_items).encode("utf-8")).hexdigest(),
                    "labels": sha256(y_broad_tr.tobytes() + y_fine_tr.tobytes()).hexdigest(),
                    "backbone": ex_name,
                    "mode": mode,
                    "aug": [args.aug_views, args.minority_aug_views],
                    "folds": CV_FOLDS,
                    "seed": SEED,
                    "grid": grid,
                    "head": "multitask_residual_mlp/512",
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        cv_path = BUILD_DIR / "cv_cache" / f"{cv_key}.json"
        if cv_path.is_file():
            rows = json.loads(cv_path.read_text(encoding="utf-8"))
            for row in rows:
                log(f"  Head CV (cached): {row}")
        else:
            rows = []
            for isic_w, scheme in grid:
                row = cv_mlp(isic_w, scheme)
                rows.append(row)
                log(f"  Head CV: {row}")
            cv_path.parent.mkdir(parents=True, exist_ok=True)
            cv_path.write_text(json.dumps(rows), encoding="utf-8")
        cv_tables["cv_stage3_head_comparison"] = rows
        winner_head = max(rows, key=lambda r: r["composite"])
    else:
        log(f"[{bb_name}/{mode}] Head selection: exhaustive weighting + linear vs MLP ({CV_FOLDS}-fold grouped CV on train)")
        weighting_results = []
        for isic_w in (0.15, 0.25, 0.40):
            for scheme in ("none", "sqrt", "balanced"):
                row = cv_linear(isic_w, scheme, c_val=0.01, include_aug=False, with_fine=False)
                weighting_results.append(row)
                log(f"  Weighting CV: {row}")
        best_w = max(weighting_results, key=lambda r: 0.55 * r["clinical_broad_macro_f1"] + 0.30 * r["clinical_broad_acc"] + 0.15 * r["dermoscopy_broad_acc"])
        cv_tables["cv_stage2_domain_weighting"] = weighting_results
        rows = []
        for c_val in (0.002, 0.005, 0.015, 0.05):
            row = cv_linear(best_w["isic_weight"], best_w["class_weight"], c_val, include_aug=True, with_fine=True)
            rows.append(row)
            log(f"  Head CV (Linear C={c_val}): {row}")
        mlp_row = cv_mlp(best_w["isic_weight"], best_w["class_weight"])
        rows.append(mlp_row)
        log(f"  Head CV (Multi-Task Residual MLP): {mlp_row}")
        cv_tables["cv_stage3_head_comparison"] = rows
        winner_head = max(rows, key=lambda r: r["composite"])
    log(
        f"[{bb_name}/{mode}] Head: {winner_head['head_type']} "
        f"(isic_weight={winner_head['isic_weight']}, class_weight={winner_head['class_weight']})"
    )

    # ------------------------------------------------------------------ final fit on the full train split
    isic_w, scheme = winner_head["isic_weight"], winner_head["class_weight"]
    all_tr_idx = np.arange(len(train_items))
    X_all, yb_all, yf_all, dm_all = assemble_train_matrix(all_tr_idx, include_aug=True)
    wb_all = compute_domain_balanced_weights(yb_all, dm_all, n_broad, scheme, isic_w)
    f_valid_all = yf_all >= 0
    wf_all = compute_domain_balanced_weights(
        np.where(f_valid_all, yf_all, 0), dm_all, n_fine, "sqrt", isic_w, valid_mask=f_valid_all
    )
    va_views_np = np.asarray(x_va_clean, dtype=np.float32)
    te_views_np = np.asarray(x_te_clean, dtype=np.float32)

    if winner_head["head_type"] == "linear":
        sc_b, hd_b = fit_linear_head(X_all, yb_all, wb_all, c_val=winner_head["C"], l2_norm=True)
        sc_f, hd_f = fit_linear_head(
            X_all[f_valid_all], yf_all[f_valid_all], wf_all[f_valid_all], c_val=winner_head["C"], l2_norm=True
        )
        head_artifacts = {"broad": (sc_b, hd_b), "fine": (sc_f, hd_f)}
        val_b_logits = linear_logits(sc_b, hd_b, va_views_np, l2_norm=True)
        val_f_logits = linear_logits(sc_f, hd_f, va_views_np, l2_norm=True)
        test_b_logits = linear_logits(sc_b, hd_b, te_views_np, l2_norm=True)
        test_f_logits = linear_logits(sc_f, hd_f, te_views_np, l2_norm=True)
        check_b_exp = linear_logits(sc_b, hd_b, te_views_np[:4, :1], l2_norm=True)
        check_f_exp = linear_logits(sc_f, hd_f, te_views_np[:4, :1], l2_norm=True)
        ensemble_seeds = 1
    else:
        x_va_es = np.asarray(x_va_clean[:, 0], dtype=np.float32)
        w_va_es = compute_domain_balanced_weights(data.y_broad_va, data.is_derm_va, n_broad, scheme, isic_w)
        models = []
        for s_i in range(args.ensemble_seeds):
            log(f"  Training Multi-Task Residual MLP seed {s_i + 1}/{args.ensemble_seeds}...")
            models.append(
                fit_multitask_mlp(
                    X_all, yb_all, wb_all, yf_all, wf_all,
                    x_va_es, data.y_broad_va, w_va_es,
                    n_broad=n_broad, n_fine=n_fine,
                    seed=SEED + 100 * s_i, l2_norm=True,
                )
            )
        head_artifacts = {"models": models}
        val_b_logits, val_f_logits = mlp_predict_logits(models, va_views_np)
        test_b_logits, test_f_logits = mlp_predict_logits(models, te_views_np)
        check_b_exp, check_f_exp = mlp_predict_logits(models, te_views_np[:4, :1])
        ensemble_seeds = args.ensemble_seeds
        del x_va_es
    del X_all
    gc.collect()

    temp_broad, temp_fine, threshold = calibrate_on_val(data, val_b_logits, val_f_logits)
    min_margin = 0.03
    final_model = evaluate_model(data, val_b_logits, test_b_logits, test_f_logits, temp_broad, temp_fine, threshold, min_margin)
    val_selection = selection_score(val_b_logits, val_f_logits, data.y_broad_va, data.y_fine_va, data.is_derm_va)
    log(f"[{bb_name}/{mode}] val selection: {val_selection}")
    log(f"[{bb_name}/{mode}] test headline: {headline(final_model)}")

    return {
        "config": {"xml": ex_xml, "backbone": ex_name, "size": ex_size, "mode": mode, "l2_norm": True},
        "head_type": winner_head["head_type"],
        "head_artifacts": head_artifacts,
        "temp_broad": temp_broad,
        "temp_fine": temp_fine,
        "threshold": threshold,
        "min_margin": min_margin,
        "check_logits": (check_b_exp, check_f_exp),
        "cv_tables": cv_tables,
        "val_selection": val_selection,
        "final_model": final_model,
        "selected_configuration": {
            "backbone": ex_name,
            "preprocessing": mode,
            "head_type": winner_head["head_type"],
            "ensemble_seeds": ensemble_seeds,
            "l2_norm": True,
            "isic_weight": isic_w,
            "class_weight": scheme,
            "aug_views_clinical": args.aug_views,
            "aug_views_minority_fine": args.minority_aug_views,
            "temperature_broad": temp_broad,
            "temperature_fine": temp_fine,
            "uncertain_threshold": threshold,
            "min_margin": min_margin,
            "target_accuracy_on_answers": args.target_accuracy,
        },
    }


def run_fusion(data: SplitArrays, names: list[str], mode: str, args, weighting: dict) -> dict:
    """Several backbones behind one head, scored through the fused graph itself.

    The head is trained on each backbone's cached features (L2-normalised per backbone, then
    concatenated). The fused graph resizes the shared input for the smaller backbone, which the
    cached features do not reflect, so val and test are scored end to end through the saved graph.
    """
    n_broad, n_fine = len(BROAD_CLASSES), len(FINE_CLASSES)
    fused_name = "+".join(names)
    y_broad_tr, y_fine_tr, is_derm_tr = data.y_broad_tr, data.y_fine_tr, data.is_derm_tr
    train_items = data.train_items
    isic_w, scheme = float(weighting["isic_weight"]), weighting["class_weight"]
    log(f"[{fused_name}/{mode}] Fusing cached features (weights: isic_weight={isic_w}, class_weight={scheme})")

    clin_tr_indices = np.flatnonzero(~is_derm_tr).tolist()
    fine_counts_tr = Counter(int(v) for v in y_fine_tr if v >= 0)
    minority_fine_ids = {fid for fid, cnt in fine_counts_tr.items() if cnt < 450}
    min_tr_indices = [idx for idx in clin_tr_indices if int(y_fine_tr[idx]) in minority_fine_ids]
    aug_views = [f"aug{k}" for k in range(args.aug_views)]
    extra_views = [f"aug{k}" for k in range(args.aug_views, args.minority_aug_views)]
    use_extra = bool(extra_views and min_tr_indices)

    n_base = 2 * len(train_items)
    n_aug = len(clin_tr_indices) * len(aug_views)
    n_extra = len(min_tr_indices) * len(extra_views) if use_extra else 0
    dims = []
    feat_dir = BUILD_DIR / "features"
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for name in names:
        ex = AsyncExtractor(name)
        prefix = f"{ex.name}__{mode}"
        parts = {
            "train": ex.extract_views_to_array(train_items, mode, ["orig", "flip"], feat_dir / f"{prefix}__train_clean"),
            "val": ex.extract_views_to_array(data.val_items, mode, ["orig", "flip"], feat_dir / f"{prefix}__val_clean"),
        }
        if aug_views:
            parts["aug"] = ex.extract_views_to_array(
                [train_items[i] for i in clin_tr_indices], mode, aug_views,
                feat_dir / f"{prefix}__train_clin_aug{args.aug_views}",
            )
        if use_extra:
            parts["extra"] = ex.extract_views_to_array(
                [train_items[i] for i in min_tr_indices], mode, extra_views,
                feat_dir / f"{prefix}__train_min_aug{args.minority_aug_views}",
            )
        arrays[name] = parts
        dims.append(ex.dim)
        del ex
        gc.collect()

    # One pre-allocated matrix, filled backbone by backbone, keeps peak memory at a single copy.
    total_dim = int(sum(dims))
    wait_until_safe("assembling fused features")
    X_all = np.empty((n_base + n_aug + n_extra, total_dim), dtype=np.float32)
    x_val = np.empty((len(data.val_items), total_dim), dtype=np.float32)
    column = 0
    for name, dim in zip(names, dims):
        parts = arrays[name]
        cols = slice(column, column + dim)
        X_all[:n_base, cols] = l2_normalize(np.asarray(parts["train"], dtype=np.float32)).reshape(n_base, dim)
        if n_aug:
            X_all[n_base : n_base + n_aug, cols] = l2_normalize(np.asarray(parts["aug"], dtype=np.float32)).reshape(n_aug, dim)
        if n_extra:
            X_all[n_base + n_aug :, cols] = l2_normalize(np.asarray(parts["extra"], dtype=np.float32)).reshape(n_extra, dim)
        x_val[:, cols] = l2_normalize(np.asarray(parts["val"][:, 0], dtype=np.float32))
        column += dim
    del arrays
    gc.collect()

    yb_all = np.concatenate([
        np.repeat(y_broad_tr, 2),
        np.repeat(y_broad_tr[clin_tr_indices], len(aug_views)),
        np.repeat(y_broad_tr[min_tr_indices], len(extra_views)) if n_extra else np.empty(0, np.int32),
    ]).astype(np.int32)
    yf_all = np.concatenate([
        np.repeat(y_fine_tr, 2),
        np.repeat(y_fine_tr[clin_tr_indices], len(aug_views)),
        np.repeat(y_fine_tr[min_tr_indices], len(extra_views)) if n_extra else np.empty(0, np.int32),
    ]).astype(np.int32)
    dm_all = np.concatenate([np.repeat(is_derm_tr, 2), np.zeros(n_aug + n_extra, dtype=bool)])
    wb_all = compute_domain_balanced_weights(yb_all, dm_all, n_broad, scheme, isic_w)
    f_valid_all = yf_all >= 0
    wf_all = compute_domain_balanced_weights(
        np.where(f_valid_all, yf_all, 0), dm_all, n_fine, "sqrt", isic_w, valid_mask=f_valid_all
    )
    w_va_es = compute_domain_balanced_weights(data.y_broad_va, data.is_derm_va, n_broad, scheme, isic_w)

    models = []
    for s_i in range(args.ensemble_seeds):
        log(f"  Training fused Multi-Task Residual MLP seed {s_i + 1}/{args.ensemble_seeds}...")
        models.append(
            fit_multitask_mlp(
                X_all, yb_all, wb_all, yf_all, wf_all,
                x_val, data.y_broad_va, w_va_es,
                n_broad=n_broad, n_fine=n_fine,
                seed=SEED + 100 * s_i, l2_norm=False,
            )
        )
    del X_all
    gc.collect()

    # Temperatures come from the cached val features; the Uncertain threshold and every reported
    # number come from the saved graph, which is what the scanner runs.
    native_b, native_f = mlp_predict_logits(models, x_val[:, None, :])
    temp_broad, temp_fine, _ = calibrate_on_val(data, native_b, native_f)
    min_margin = 0.03
    config = {
        "xml": fused_backbone_xml(names),
        "backbone": fused_name,
        "size": max(BACKBONES[name]["size"] for name in names),
        "mode": mode,
        "l2_norm": False,
    }
    head_artifacts = {"models": models}
    graph = build_classifier_graph(config, "multitask_residual_mlp", head_artifacts, temp_broad, temp_fine)
    eval_dir = BUILD_DIR / "fusion_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_xml = eval_dir / "classifier.xml"
    ov.save_model(graph, eval_xml, compress_to_fp16=args.fp16)
    del graph
    scored = as_shipped_evaluation(eval_xml, data, config, temp_broad, temp_fine, 0.36, min_margin)
    raw_val_b, raw_val_f = scored["raw_val_logits"]
    graph_temp_broad, graph_temp_fine, threshold = calibrate_on_val(data, raw_val_b, raw_val_f)
    log(
        f"[{fused_name}/{mode}] temperatures from cached features {temp_broad:.3f}/{temp_fine:.3f}, "
        f"from the fused graph {graph_temp_broad:.3f}/{graph_temp_fine:.3f}"
    )
    scored = as_shipped_evaluation(eval_xml, data, config, temp_broad, temp_fine, threshold, min_margin)
    log(f"[{fused_name}/{mode}] val selection: {scored['val_selection']}")
    log(f"[{fused_name}/{mode}] test headline: {scored['test_headline']}")

    return {
        "config": config,
        "head_type": "multitask_residual_mlp",
        "head_artifacts": head_artifacts,
        "temp_broad": temp_broad,
        "temp_fine": temp_fine,
        "threshold": threshold,
        "min_margin": min_margin,
        "check_logits": (None, None),
        "cv_tables": {},
        "val_selection": scored["val_selection"],
        "final_model": scored["final_model"],
        "selected_configuration": {
            "backbone": fused_name,
            "preprocessing": mode,
            "head_type": "multitask_residual_mlp",
            "ensemble_seeds": args.ensemble_seeds,
            "l2_norm": "per backbone",
            "isic_weight": isic_w,
            "class_weight": scheme,
            "aug_views_clinical": args.aug_views,
            "aug_views_minority_fine": args.minority_aug_views,
            "temperature_broad": temp_broad,
            "temperature_fine": temp_fine,
            "uncertain_threshold": threshold,
            "min_margin": min_margin,
            "target_accuracy_on_answers": args.target_accuracy,
        },
    }


def previous_recipe() -> dict | None:
    """Backbone, preprocessing, and weighting of the currently deployed model (the recipe to beat)."""
    bundle_path = MODEL_DIR / "classifier.zip"
    if not bundle_path.is_file():
        return None
    with ZipFile(bundle_path) as zf:
        if "metadata.json" not in zf.namelist():
            return None
        meta = json.loads(zf.read("metadata.json"))
    # A fused bundle names its backbones joined by "+" (see run_fusion).
    backbones = str(meta.get("backbone") or "").split("+")
    if not all(name in BACKBONES for name in backbones) or meta.get("preprocessing") not in PREPROCESSING_MODES:
        return None
    recipe = {
        "backbone": meta["backbone"],
        "backbones": backbones,
        "mode": meta["preprocessing"],
        "head_type": meta.get("head_type", "multitask_residual_mlp"),
        "isic_weight": 0.40,
        "class_weight": "none",
    }
    evaluation_path = MODEL_DIR / "evaluation.json"
    if evaluation_path.is_file():
        try:
            selected = json.loads(evaluation_path.read_text(encoding="utf-8")).get("selected_configuration", {})
            if selected.get("backbone") == recipe["backbone"]:
                recipe["isic_weight"] = float(selected.get("isic_weight", recipe["isic_weight"]))
                recipe["class_weight"] = selected.get("class_weight", recipe["class_weight"])
        except (OSError, ValueError):
            pass
    if recipe["head_type"] not in ("linear", "multitask_residual_mlp"):
        recipe["head_type"] = "multitask_residual_mlp"
    return recipe


# --------------------------------------------------------------------------- main pipeline
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", type=Path, default=PROJECT_ROOT / "datasets" / "20-skin-diseases.zip")
    parser.add_argument("--backbones", nargs="+", default=["effnetv2b2_260", "dinov2_vits14_224"], choices=sorted(BACKBONES))
    parser.add_argument("--modes", nargs="+", default=["letterbox", "letterbox_sog"], choices=PREPROCESSING_MODES)
    parser.add_argument("--pilot-per-class", type=int, default=420)
    parser.add_argument("--aug-views", type=int, default=1, help="Extra augmented views per clinical train image.")
    parser.add_argument("--minority-aug-views", type=int, default=4, help="Augmented views for minority fine classes.")
    parser.add_argument("--ensemble-seeds", type=int, default=3)
    parser.add_argument("--target-accuracy", type=float, default=0.80)
    parser.add_argument(
        "--head-search", choices=("fast", "full"), default="fast",
        help="fast: CV the MLP head over three weightings; full: exhaustive weighting + linear vs MLP search.",
    )
    parser.add_argument("--skip-baseline", action="store_true", help="Do not re-fit the previously deployed recipe.")
    parser.add_argument("--no-deploy", action="store_true", help="Write the bundle to build/candidate instead of model/.")
    parser.add_argument(
        "--force-deploy", action="store_true",
        help="Replace the deployed model even when it scores higher on the val split than the new one.",
    )
    parser.add_argument(
        "--fusion", action="store_true",
        help="Also evaluate one head over both the selected backbone and the previously deployed backbone.",
    )
    parser.add_argument(
        "--fp16", action="store_true",
        help="Store the exported weights in half precision (halves classifier.zip; scored end to end before it ships).",
    )
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument(
        "--fresh-split", action="store_true",
        help="Draw a new train/val/test split for every photo instead of keeping earlier assignments.",
    )
    parser.add_argument(
        "--require-ac", action="store_true",
        help="On a laptop, pause whenever it runs on battery and resume when the charger is connected.",
    )
    args = parser.parse_args()

    global REQUIRE_AC
    REQUIRE_AC = args.require_ac
    log(f"Using {CPU_THREADS} worker threads" + (" (pauses on battery)" if REQUIRE_AC else ""))
    wait_until_safe("starting")
    np.random.seed(SEED)

    log("Phase 1: Auditing & merging ALL workspace datasets (leakage-safe)")
    items, audit = audit_and_split_datasets(
        args.dataset, force_rebuild=args.rebuild_manifest, fresh_split=args.fresh_split
    )
    data = build_split_arrays(items)
    train_items, test_items = data.train_items, data.test_items
    y_broad_tr, y_fine_tr, is_derm_tr, groups_tr = data.y_broad_tr, data.y_fine_tr, data.is_derm_tr, data.groups_tr

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
            p_views = np.asarray(
                ex.extract_views_to_array(
                    pilot_items, mode, ["orig", "flip"], BUILD_DIR / "pilot_features" / f"{bb_name}__{mode}"
                ),
                dtype=np.float32,
            )
            clin_mask = ~pilot_derm
            f_mask = clin_mask & (pilot_yf >= 0)
            # The broad probe is fitted at two regularisation strengths so backbones with different
            # feature widths are compared at their own best setting.
            best_broad = None
            for c_val in (0.01, 0.05):
                oof_b = np.zeros((len(pilot_items), len(BROAD_CLASSES)), dtype=np.float32)
                for tr_i, va_i in pilot_folds:
                    x_flat = p_views[tr_i].reshape(-1, ex.dim)
                    yb_rep = np.repeat(pilot_yb[tr_i], 2)
                    dm_rep = np.repeat(pilot_derm[tr_i], 2)
                    wb = compute_domain_balanced_weights(yb_rep, dm_rep, len(BROAD_CLASSES), "sqrt", 0.25)
                    sc_b, hd_b = fit_linear_head(x_flat, yb_rep, wb, c_val=c_val, l2_norm=True)
                    oof_b[va_i] = linear_logits(sc_b, hd_b, p_views[va_i], l2_norm=True)
                clin_m = classification_metrics(oof_b[clin_mask].argmax(1), pilot_yb[clin_mask], BROAD_CLASSES)
                derm_m = classification_metrics(oof_b[pilot_derm].argmax(1), pilot_yb[pilot_derm], BROAD_CLASSES)
                broad_part = 0.50 * clin_m["macro_f1"] + 0.25 * clin_m["accuracy"] + 0.10 * derm_m["accuracy"]
                if best_broad is None or broad_part > best_broad[0]:
                    best_broad = (broad_part, c_val, clin_m, derm_m)
            broad_part, best_c, clin_m, derm_m = best_broad

            oof_f = np.zeros((len(pilot_items), len(FINE_CLASSES)), dtype=np.float32)
            for tr_i, va_i in pilot_folds:
                x_flat = p_views[tr_i].reshape(-1, ex.dim)
                dm_rep = np.repeat(pilot_derm[tr_i], 2)
                yf_rep = np.repeat(pilot_yf[tr_i], 2)
                f_valid = yf_rep >= 0
                if len(np.unique(yf_rep[f_valid])) == len(FINE_CLASSES):
                    wf = compute_domain_balanced_weights(
                        np.where(f_valid, yf_rep, 0), dm_rep, len(FINE_CLASSES), "sqrt", 0.25, valid_mask=f_valid
                    )
                    sc_f, hd_f = fit_linear_head(x_flat[f_valid], yf_rep[f_valid], wf[f_valid], c_val=best_c, l2_norm=True)
                    oof_f[va_i] = linear_logits(sc_f, hd_f, p_views[va_i], l2_norm=True)
            fine_m = classification_metrics(oof_f[f_mask].argmax(1), pilot_yf[f_mask], FINE_CLASSES)
            composite = broad_part + 0.15 * fine_m["macro_f1"]
            res = {
                "backbone": bb_name,
                "mode": mode,
                "probe_C": best_c,
                "clinical_broad_acc": round(clin_m["accuracy"], 4),
                "clinical_broad_macro_f1": round(clin_m["macro_f1"], 4),
                "clinical_fine_macro_f1": round(fine_m["macro_f1"], 4),
                "dermoscopy_broad_acc": round(derm_m["accuracy"], 4),
                "composite_score": round(composite, 4),
            }
            stage1_results.append(res)
            log(f"  Stage1 {res}")
            del p_views
        del ex
        gc.collect()

    # Preprocessing modes of the leading backbone within 0.003 of each other are a statistical tie on
    # this pilot; the tie goes to letterbox_sog, whose colour constancy removes lighting colour casts.
    leader = max(stage1_results, key=lambda r: r["composite_score"])
    tied = [
        r for r in stage1_results
        if r["backbone"] == leader["backbone"] and leader["composite_score"] - r["composite_score"] <= 0.003
    ]
    best_s1 = max(tied, key=lambda r: (r["mode"] == "letterbox_sog", r["composite_score"]))
    report["cv_stage1_backbone_preprocessing"] = stage1_results
    log(f"Selected Stage 1 winner: {best_s1['backbone']} / {best_s1['mode']} (composite={best_s1['composite_score']})")

    # ------------------------------------------------------------------ Stages 2-4 for the winner (and the recipe to beat)
    log("Phase 3: Full feature extraction, head selection, val calibration, and one test evaluation")
    candidates = [("selected", run_recipe(data, best_s1["backbone"], best_s1["mode"], args))]

    baseline = None if args.skip_baseline else previous_recipe()
    if baseline is not None and len(baseline["backbones"]) > 1:
        log(
            f"Phase 4: Re-fitting the previously deployed fused recipe ({baseline['backbone']}/{baseline['mode']}) "
            "on the same train split for a like-for-like comparison"
        )
        selected_cfg = candidates[0][1]["selected_configuration"]
        weighting = {"isic_weight": selected_cfg["isic_weight"], "class_weight": selected_cfg["class_weight"]}
        candidates.append(
            ("previous_recipe_refit", run_fusion(data, baseline["backbones"], baseline["mode"], args, weighting))
        )
    elif baseline is not None and (baseline["backbone"], baseline["mode"]) != (best_s1["backbone"], best_s1["mode"]):
        log(
            f"Phase 4: Re-fitting the previously deployed recipe ({baseline['backbone']}/{baseline['mode']}) "
            "on the same train split for a like-for-like comparison"
        )
        candidates.append(("previous_recipe_refit", run_recipe(data, baseline["backbone"], baseline["mode"], args, fixed=baseline)))
        if args.fusion and baseline["backbone"] != best_s1["backbone"] and baseline["mode"] == best_s1["mode"]:
            log("Phase 4b: One head over both backbones, scored through the fused graph")
            selected_cfg = candidates[0][1]["selected_configuration"]
            weighting = {"isic_weight": selected_cfg["isic_weight"], "class_weight": selected_cfg["class_weight"]}
            candidates.append(
                ("fusion", run_fusion(data, [baseline["backbone"], best_s1["backbone"]], best_s1["mode"], args, weighting))
            )

    previous_bundle = evaluate_previous_bundle(MODEL_DIR / "classifier.zip", data, audit.get("split_lineage"))

    # The model that ships is chosen on the val split only; the test split is reported, never used to choose.
    role, chosen = max(candidates, key=lambda entry: entry[1]["val_selection"]["composite"])
    log(f"Chosen for deployment (best val composite): {chosen['config']['backbone']}/{chosen['config']['mode']} [{role}]")

    # A deployed model that provably never saw today's val photos (same split lineage) and scores higher
    # on them stays in place, for example a bundle fine-tuned on a GPU by backend/finetune_gpu.py.
    keep_deployed = (
        not args.no_deploy
        and not args.force_deploy
        and previous_bundle is not None
        and previous_bundle["same_split_lineage"]
        and previous_bundle["val_selection"]["composite"] > chosen["val_selection"]["composite"]
    )
    if keep_deployed:
        log(
            f"The deployed model scores higher on val ({previous_bundle['val_selection']['composite']:.4f} vs "
            f"{chosen['val_selection']['composite']:.4f}), so it stays; the new model goes to build/candidate. "
            "Pass --force-deploy to replace it anyway."
        )

    report.update(chosen["cv_tables"])
    report["selected_configuration"] = chosen["selected_configuration"]
    report["final_model"] = chosen["final_model"]
    report["comparison"] = {
        "protocol": (
            "Every candidate is trained on the same train split, calibrated on the same val split, and "
            "scored once on the same held-out test split. The deployed model is the candidate with the "
            "best val composite. final_model holds the scores of the saved classifier graph itself, run "
            "end to end over the val/test photos."
        ),
        "deployed_role": role,
        "candidates": [
            {
                "role": cand_role,
                "backbone": cand["config"]["backbone"],
                "preprocessing": cand["config"]["mode"],
                "head_type": cand["head_type"],
                "val_selection": cand["val_selection"],
                "test_headline": headline(cand["final_model"]),
            }
            for cand_role, cand in candidates
        ],
        "previous_deployed_bundle": (
            {k: v for k, v in previous_bundle.items() if k != "final_model"} if previous_bundle else None
        ),
    }

    # Fresh quality gate calibration on clinical training photos
    log("Phase 5: Fresh image quality gate calibration on clinical training photos")
    quality_report = calibrate_quality_fresh([i for i in train_items if i.domain == "clinical"])

    log("Phase 6: Exporting dual-output OpenVINO classifier.zip (format 3) & verifying numerical parity")
    archive_cache: dict[str, ZipFile] = {}
    try:
        check_pixels = np.stack(
            [
                prepare(read_item_image(i, archive_cache), chosen["config"]["size"], chosen["config"]["mode"])
                for i in test_items[:4]
            ]
        )
    finally:
        for zf in archive_cache.values():
            zf.close()

    if args.no_deploy or keep_deployed:
        out_dir = BUILD_DIR / "candidate"
    else:
        out_dir = MODEL_DIR
        existing = [MODEL_DIR / name for name in ("classifier.zip", "evaluation.json", "quality.json")]
        if any(p.is_file() for p in existing):
            backup_dir = BUILD_DIR / "model_backups" / datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_dir.mkdir(parents=True, exist_ok=True)
            for p in existing:
                if p.is_file():
                    shutil.copy2(p, backup_dir / p.name)
            log(f"Backed up the previously deployed model files to {backup_dir}")

    export_hierarchical_bundle(
        chosen["config"],
        chosen["head_type"],
        chosen["head_artifacts"],
        chosen["temp_broad"],
        chosen["temp_fine"],
        chosen["threshold"],
        chosen["min_margin"],
        report,
        quality_report,
        check_pixels,
        chosen["check_logits"][0],
        chosen["check_logits"][1],
        out_dir=out_dir,
        compress_fp16=args.fp16,
        verify=lambda xml_path: as_shipped_evaluation(
            xml_path, data, chosen["config"], chosen["temp_broad"], chosen["temp_fine"],
            chosen["threshold"], chosen["min_margin"],
        ),
    )

    final = report["final_model"]
    tc = final["test_clinical"]["broad"]
    tf_m = final["test_clinical"]["fine"]
    td = final["test_dermoscopy"]["broad"]
    to = final["test_overall"]["broad"]
    ts = final["test_clinical"]["broad_selective"]
    rule = final["test_clinical"]["scanner_report_rule"]
    log(
        f"TEST CLINICAL   (7-broad): acc={tc['accuracy']:.3f}  macro-F1={tc['macro_f1']:.3f}  "
        f"top3={tc['top3_accuracy']:.3f}  ECE={tc['ece_15_bins']:.4f} (uncal={tc['uncalibrated_ece_15_bins']:.4f})"
    )
    log(
        f"TEST CLINICAL   (21-fine): acc={tf_m['accuracy']:.3f}  macro-F1={tf_m['macro_f1']:.3f}  "
        f"top3={tf_m['top3_accuracy']:.3f}  hier-acc={tf_m['hierarchical_conditioned_accuracy']:.3f}"
    )
    log(f"TEST DERMOSCOPY (ISIC)   : acc={td['accuracy']:.3f}  macro-F1={td['macro_f1']:.3f}")
    log(f"TEST OVERALL    (7-broad): acc={to['accuracy']:.3f}  macro-F1={to['macro_f1']:.3f}  top3={to['top3_accuracy']:.3f}")
    log(
        f"CLINICAL SELECTIVE (t={ts['threshold']:.2f}): coverage={ts['coverage']:.3f}  "
        f"acc_on_accepted={ts['accuracy_on_accepted']:.3f}"
    )
    log(
        f"SCANNER REPORT RULE (score >= {rule['report_threshold']:.2f}): answers {rule['answered_rate']:.1%} of clinical "
        f"test photos, {rule['accuracy_when_answered']:.1%} of those answers correct"
        if rule["accuracy_when_answered"] is not None
        else "SCANNER REPORT RULE: no clinical test photo reached the report threshold"
    )
    for cand in report["comparison"]["candidates"]:
        log(f"COMPARISON {cand['role']} ({cand['backbone']}/{cand['preprocessing']}): {cand['test_headline']}")
    if previous_bundle:
        note = "never saw these test images" if previous_bundle["same_split_lineage"] else "may have trained on these test images"
        log(f"COMPARISON previous_deployed_bundle ({note}): {previous_bundle['test_headline']}")
    if keep_deployed:
        log("The deployed model in model/ was kept.")
    log(f"Saved classifier.zip, evaluation.json, and quality.json in {out_dir}")


if __name__ == "__main__":
    main()
