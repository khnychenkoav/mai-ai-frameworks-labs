"""Выполняет лабораторные по порядку и сохраняет результаты в ноутбуках."""

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
NOTEBOOKS = {
    "1": "01_knn.ipynb",
    "2": "02_regression.ipynb",
    "3": "03_decision_tree.ipynb",
    "4": "04_random_forest.ipynb",
    "5": "05_gradient_boosting.ipynb",
    "summary": "06_comparison.ipynb",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labs", choices=[*NOTEBOOKS, "all"], nargs="*", default=None)
    options = parser.parse_args()
    selected = list(NOTEBOOKS) if not options.labs or "all" in options.labs else options.labs
    for lab in selected:
        path = ROOT / NOTEBOOKS[lab]
        print(f"Выполняется {path.name}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        NotebookClient(
            notebook, timeout=1800, kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
            record_timing=False,
        ).execute()
        temporary = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=ROOT, suffix=".ipynb.tmp", delete=False,
            ) as output:
                temporary = Path(output.name)
                nbformat.write(notebook, output)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f"Сохранён {path.name}", flush=True)


if __name__ == "__main__":
    main()
