"""Build ``build/colab_pack.zip``: the small file to upload to Google Colab before fine-tuning.

It holds the project's Python code, the photo list with the train/val/test split
(``build/unified_manifest.json``), and the installed model's report. The photos themselves are not
included; the notebook downloads them from Kaggle and checks them against the list.

Usage:
  .venv\\Scripts\\python.exe colab\\make_pack.py

Run it again after the datasets change (after ``backend/train.py`` has rebuilt the manifest), so that
Colab trains on the same photos and split as this machine.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "build" / "colab_pack.zip"


def main() -> int:
    manifest_path = PROJECT_ROOT / "build" / "unified_manifest.json"
    if not manifest_path.is_file():
        print("build/unified_manifest.json is missing. Run backend\\train.py once first.")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (manifest.get("audit") or {}).get("split_lineage"):
        print("The manifest has no split lineage id. Run backend\\train.py once so it is rebuilt.")
        return 1

    files = sorted((PROJECT_ROOT / "backend").glob("*.py")) + [
        PROJECT_ROOT / "colab" / "setup_data.py",
        manifest_path,
        PROJECT_ROOT / "model" / "evaluation.json",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUT, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            if path.is_file():
                archive.write(path, path.relative_to(PROJECT_ROOT).as_posix())
    counts = manifest["audit"].get("clean_by_split", {})
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    print(f"Photos in the list: {len(manifest['items'])} (train {counts.get('train')}, val {counts.get('val')}, test {counts.get('test')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
