from __future__ import annotations

import json
import sys
from pathlib import Path


def normalize_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))

    metadata = notebook.get("metadata", {})
    kernelspec = metadata.get(
        "kernelspec",
        {
            "display_name": "Python (robust-portfolio-analytics)",
            "language": "python",
            "name": "robust-portfolio-analytics",
        },
    )
    language_info = metadata.get("language_info", {})
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": kernelspec.get("display_name", "Python (robust-portfolio-analytics)"),
            "language": kernelspec.get("language", "python"),
            "name": kernelspec.get("name", "robust-portfolio-analytics"),
        },
        "language_info": {
            "name": language_info.get("name", "python"),
            "version": language_info.get("version", "3.12"),
        },
    }

    for cell in notebook.get("cells", []):
        cell["metadata"] = {}

    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/normalize_notebook_metadata.py <notebook_path>")
    normalize_notebook(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
