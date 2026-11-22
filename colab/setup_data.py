"""Fetch the training photos on a fresh machine (Google Colab) and check them against the manifest.

For each of the five public Kaggle archives this downloads the zip, extracts only the photos that
``build/unified_manifest.json`` lists, into the same relative paths the manifest uses, and compares
every photo's SHA-256 with the manifest. It fails loudly when a photo is missing or differs, because
the train/val/test split is only valid for identical photos.

Usage:
  python colab/setup_data.py                       download, extract, verify
  python colab/setup_data.py --zip-dir datasets    use archives that are already on disk
  python colab/setup_data.py --check-zips          read the archives and verify, but write nothing
  python colab/setup_data.py --check-only          verify the photos already on disk; download nothing

Only the Python standard library is used, so it runs before anything is installed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zipfile import ZipFile, is_zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS = (
    # (Kaggle dataset, folder under datasets/)
    ("haroonalam16/20-skin-diseases-dataset", "20-skin-diseases"),
    ("shubhamgoel27/dermnet", "dermnet"),
    ("ascanipek/skin-diseases", "skin-diseases-6class"),
    ("subirbiswas19/skin-disease-dataset", "skin-disease-dataset"),
    ("pacificrm/skindiseasedataset", "skindiseasedataset"),
)
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/{slug}"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def kaggle_credentials() -> tuple[str, str] | None:
    """Optional Kaggle API token (environment or kaggle.json); public datasets normally need none."""
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if user and key:
        return user, key
    for candidate in (Path.home() / ".kaggle" / "kaggle.json", Path("kaggle.json"), PROJECT_ROOT / "kaggle.json"):
        try:
            token = json.loads(candidate.read_text(encoding="utf-8"))
            return token["username"], token["key"]
        except (OSError, ValueError, KeyError):
            continue
    return None


def download(slug: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    command = ["curl", "-L", "--fail", "--silent", "--show-error", "--retry", "3", "--retry-delay", "5", "-o", str(partial)]
    credentials = kaggle_credentials()
    if credentials:
        command += ["-u", f"{credentials[0]}:{credentials[1]}"]
    command.append(KAGGLE_URL.format(slug=slug))
    started = time.perf_counter()
    result = subprocess.run(command, check=False)
    if result.returncode != 0 or not partial.is_file() or not is_zipfile(partial):
        partial.unlink(missing_ok=True)
        raise SystemExit(
            f"\nCould not download {slug} from Kaggle (curl exit {result.returncode}).\n"
            "If Kaggle asks for a login: on kaggle.com open Settings > API > Create New Token, upload the\n"
            "kaggle.json it gives you next to this notebook (see the optional cell), and run this step again."
        )
    partial.replace(dest)
    log(f"  downloaded {dest.name}: {dest.stat().st_size / 1e9:.2f} GB in {time.perf_counter() - started:.0f}s")


def wanted_files(manifest_path: Path) -> dict[str, dict[str, str]]:
    """Dataset folder -> {path inside that folder: SHA-256} for every photo in the manifest."""
    wanted: dict[str, dict[str, str]] = {folder: {} for _, folder in DATASETS}
    for row in json.loads(manifest_path.read_text(encoding="utf-8"))["items"]:
        if row["storage"] != "file":
            raise SystemExit(f"Unsupported manifest entry (stored inside an archive): {row['rel_path']}")
        parts = row["rel_path"].split("/", 2)
        if len(parts) != 3 or parts[0] != "datasets" or parts[1] not in wanted:
            raise SystemExit(f"Photo outside the known dataset folders: {row['rel_path']}")
        wanted[parts[1]][parts[2]] = row["sha"]
    return wanted


def hash_file(path: Path) -> str | None:
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def check_on_disk(target: Path, wanted: dict[str, str]) -> tuple[list[str], list[str]]:
    """(missing, different) photos of one dataset folder, hashing with a few threads."""
    names = list(wanted)
    with ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 2) * 2)) as pool:
        digests = list(pool.map(lambda name: hash_file(target / name), names))
    missing = [name for name, digest in zip(names, digests) if digest is None]
    different = [name for name, digest in zip(names, digests) if digest is not None and digest != wanted[name]]
    return missing, different


def extract(archive: Path, target: Path, wanted: dict[str, str], write: bool) -> tuple[list[str], list[str]]:
    """Copy the wanted photos out of the archive; returns (missing, different)."""
    found: set[str] = set()
    different: list[str] = []
    with ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            expected = wanted.get(name)
            if expected is None or info.is_dir():
                continue
            data = zf.read(info)
            found.add(name)
            if sha256(data).hexdigest() != expected:
                different.append(name)
                continue
            if write:
                out = target / name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(data)
    return [name for name in wanted if name not in found], different


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip-dir", type=Path, help="Folder that already holds <folder>.zip archives; nothing is downloaded for those.")
    parser.add_argument("--dest-root", type=Path, default=PROJECT_ROOT, help="Where the datasets/ folder is written.")
    parser.add_argument("--keep-zips", action="store_true", help="Keep downloaded archives after extraction.")
    parser.add_argument("--check-zips", action="store_true", help="Verify the archives against the manifest without writing photos.")
    parser.add_argument("--check-only", action="store_true", help="Verify the photos on disk; download and extract nothing.")
    parser.add_argument("--only", nargs="+", help="Limit the run to these dataset folders.")
    args = parser.parse_args()

    wanted = wanted_files(PROJECT_ROOT / "build" / "unified_manifest.json")
    total = sum(len(files) for files in wanted.values())
    log(f"The manifest lists {total} photos in {len(wanted)} datasets")
    problems = 0
    for slug, folder in DATASETS:
        if args.only and folder not in args.only:
            continue
        files = wanted[folder]
        target = args.dest_root / "datasets" / folder
        if not files:
            continue
        if not args.check_zips:
            missing, different = check_on_disk(target, files) if target.is_dir() else (list(files), [])
            if not missing and not different:
                log(f"{folder}: all {len(files)} photos are present and identical")
                continue
            if args.check_only:
                log(f"{folder}: {len(missing)} missing, {len(different)} different, of {len(files)}")
                problems += len(missing) + len(different)
                continue
        local_zip = (args.zip_dir / f"{folder}.zip") if args.zip_dir else None
        downloaded = False
        if local_zip is None or not local_zip.is_file():
            if args.check_zips:
                log(f"{folder}: no archive at {local_zip}; skipped")
                continue
            local_zip = args.dest_root / "datasets" / f"{folder}.zip"
            if not local_zip.is_file():
                log(f"{folder}: downloading {slug}")
                download(slug, local_zip)
                downloaded = True
        started = time.perf_counter()
        missing, different = extract(local_zip, target, files, write=not args.check_zips)
        verb = "verified in the archive" if args.check_zips else "extracted"
        log(f"{folder}: {len(files) - len(missing) - len(different)} of {len(files)} photos {verb} in {time.perf_counter() - started:.0f}s")
        for label, names in (("missing from the archive", missing), ("different from the manifest", different)):
            if names:
                problems += len(names)
                log(f"  {len(names)} {label}, for example: {names[:3]}")
        if downloaded and not args.keep_zips:
            local_zip.unlink(missing_ok=True)

    free_gb = shutil.disk_usage(args.dest_root).free / 1e9
    if problems:
        log(f"PROBLEM: {problems} photos are missing or differ from the manifest. Training on them would not match the laptop's split.")
        return 1
    log(f"All photos match the manifest. Free disk space: {free_gb:.0f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
