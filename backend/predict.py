"""CLI entry point for single-image skin analysis invoked by the PHP frontend."""
import json
from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.analyzer import AnalysisError, analyze_image_bytes


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "Choose an image to upload."}, ensure_ascii=True))
        return 2

    image_path = Path(sys.argv[1])
    if not image_path.is_file():
        print(json.dumps({"ok": False, "error": "Choose an image to upload."}, ensure_ascii=True))
        return 2

    try:
        result = analyze_image_bytes(image_path.read_bytes())
    except AnalysisError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=True))
        return 2
    except Exception:
        traceback.print_exc(file=sys.stderr)
        print(
            json.dumps(
                {"ok": False, "error": "The Python analysis backend could not process this image."},
                ensure_ascii=True,
            )
        )
        return 1

    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
