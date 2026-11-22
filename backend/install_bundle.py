"""Check a classifier bundle that was trained elsewhere (for example on Colab) and install it if it is better.

The bundle is scored on this machine exactly as the scanner runs it (OpenVINO on the CPU, original plus
mirrored photo) on the val and test photos, next to the model that is installed now. It replaces the
installed model only when it scores higher on the val split; the replaced files are backed up in
``build/model_backups/``.

Usage:
  .venv\\Scripts\\python.exe backend\\install_bundle.py colab_result.zip
  .venv\\Scripts\\python.exe backend\\install_bundle.py colab_result.zip --dry-run    compare only
  .venv\\Scripts\\python.exe backend\\install_bundle.py colab_result.zip --force      install even if it is not better

Scoring a new bundle takes about 15 minutes on a laptop CPU. Set SKIN_TRAIN_THREADS (for example 6) first
to cap the CPU threads.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.train import (
    BUILD_DIR,
    MODEL_DIR,
    UnifiedItem,
    build_split_arrays,
    evaluate_previous_bundle,
    log,
    manifest_fingerprint,
)

ROWS = (
    ("val composite (decides)", "val", "composite", "{:.4f}"),
    ("clinical photos: right category", "test", "clinical_broad_acc", "{:.1%}"),
    ("clinical photos: macro-F1", "test", "clinical_broad_macro_f1", "{:.3f}"),
    ("clinical photos: right specific condition", "test", "clinical_fine_acc", "{:.1%}"),
    ("dermoscopy images: right category", "test", "dermoscopy_broad_acc", "{:.1%}"),
    ("photos where a match is shown", "test", "scanner_answered_rate_clinical", "{:.1%}"),
    ("shown matches that are right", "test", "scanner_accuracy_when_answered_clinical", "{:.1%}"),
    ("photos with a correct match on screen", "test", "scanner_correct_answers_shown_rate_clinical", "{:.1%}"),
)


def unpack(source: Path, work: Path) -> tuple[Path, dict]:
    """Bundle path and its training report from either colab_result.zip or a bare classifier.zip."""
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with ZipFile(source) as archive:
        names = set(archive.namelist())
        if "classifier.xml" in names:
            shutil.copy2(source, work / "classifier.zip")
            return work / "classifier.zip", {}
        if "classifier.zip" not in names:
            raise SystemExit(f"{source} holds neither classifier.zip nor classifier.xml")
        (work / "classifier.zip").write_bytes(archive.read("classifier.zip"))
        report = json.loads(archive.read("evaluation.json")) if "evaluation.json" in names else {}
    return work / "classifier.zip", report


def cell(result: dict | None, where: str, key: str, fmt: str) -> str:
    if result is None:
        return "-"
    value = (result["val_selection"] if where == "val" else result["test_headline"]).get(key)
    return "-" if value is None else fmt.format(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=Path, help="colab_result.zip from the notebook, or a classifier.zip")
    parser.add_argument("--dry-run", action="store_true", help="Compare the two models and stop.")
    parser.add_argument("--force", action="store_true", help="Install even when the checks or the comparison say no.")
    args = parser.parse_args()
    if not args.bundle.is_file():
        raise SystemExit(f"File not found: {args.bundle}")

    bundle_path, report = unpack(args.bundle, BUILD_DIR / "incoming")
    with ZipFile(bundle_path) as archive:
        meta = json.loads(archive.read("metadata.json"))
    manifest = json.loads((BUILD_DIR / "unified_manifest.json").read_text(encoding="utf-8"))
    items = [UnifiedItem(**row) for row in manifest["items"]]
    lineage = (manifest.get("audit") or {}).get("split_lineage")
    fingerprint = manifest_fingerprint(items)

    stop = []
    if meta.get("trial_run"):
        stop.append("it is a trial run trained or scored on a subset of the photos")
    if meta.get("manifest_fingerprint") not in (None, fingerprint):
        stop.append(
            "it was trained with a different photo list or split than this machine has now "
            f"({meta.get('manifest_fingerprint')} vs {fingerprint}), so some of today's test photos may have been its training photos"
        )
    for reason in stop:
        log(f"WARNING: this bundle should not be installed: {reason}.")

    data = build_split_arrays(items)
    log(f"New bundle: {meta.get('backbone')} trained {str(meta.get('created_at'))[:16]}")
    new = evaluate_previous_bundle(bundle_path, data, lineage)
    if new is None:
        raise SystemExit("The bundle does not use this scanner's model format or categories.")
    current = evaluate_previous_bundle(MODEL_DIR / "classifier.zip", data, lineage)

    print(f"\n{'':<44}{'installed now':>16}{'new bundle':>16}")
    for label, where, key, fmt in ROWS:
        print(f"{label:<44}{cell(current, where, key, fmt):>16}{cell(new, where, key, fmt):>16}")
    print()

    better = current is None or new["val_selection"]["composite"] > current["val_selection"]["composite"]
    if args.dry_run:
        log(f"Dry run: the new bundle is {'better' if better else 'not better'} on the val split. Nothing was changed.")
        return 0
    if (stop or not better) and not args.force:
        log(
            "Not installed: "
            + ("; ".join(stop) if stop else "the new bundle does not score higher on the val split than the installed model")
            + ". Pass --force to install it anyway."
        )
        return 2

    existing = [MODEL_DIR / name for name in ("classifier.zip", "evaluation.json", "quality.json") if (MODEL_DIR / name).is_file()]
    if existing:
        backup = BUILD_DIR / "model_backups" / datetime.now().strftime("%Y%m%d-%H%M%S")
        backup.mkdir(parents=True, exist_ok=True)
        for path in existing:
            shutil.copy2(path, backup / path.name)
        log(f"Backed up the installed model to {backup}")

    # The published numbers must describe the exact file the scanner loads, scored on this machine.
    report["final_model"] = new["final_model"]
    comparison = report.setdefault("comparison", {})
    comparison["as_shipped"] = {
        "note": "The saved classifier graph scored end to end on val/test (orig + flip), exactly as the scanner runs it.",
        "val_selection": new["val_selection"],
        "test_headline": new["test_headline"],
    }
    comparison["previous_deployed_bundle"] = {k: v for k, v in current.items() if k != "final_model"} if current else None
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundle_path, MODEL_DIR / "classifier.zip")
    (MODEL_DIR / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log("Installed the new bundle in model/. Restart backend/app.py if it is running.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
