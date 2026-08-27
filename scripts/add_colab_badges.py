"""Add or refresh Colab badges in every notebook published by Quarto.

The Quarto chapter list is the source of truth for "student-facing" notebooks.
Instructor notebooks and unlisted source notebooks are therefore never modified.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import nbformat
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "_quarto.yml"
BADGE_ASSET = "https://colab.research.google.com/assets/colab-badge.svg"


def published_notebooks(items: list[object]) -> list[str]:
    """Recursively collect notebook paths from the Quarto chapter structure."""
    paths: list[str] = []
    for item in items:
        if isinstance(item, str) and item.endswith(".ipynb"):
            paths.append(item)
        elif isinstance(item, dict):
            paths.extend(published_notebooks(item.get("chapters", [])))
    return paths


def badge_markdown(repo_url: str, branch: str, notebook_path: str) -> str:
    repo_path = repo_url.removeprefix("https://github.com/").rstrip("/")
    encoded_path = quote(notebook_path, safe="/")
    colab_url = (
        f"https://colab.research.google.com/github/{repo_path}/blob/"
        f"{branch}/{encoded_path}#copy=true"
    )
    return f"[![]({BADGE_ASSET})]({colab_url})"


def add_or_update_badge(path: Path, markdown: str) -> None:
    notebook = nbformat.read(path, as_version=4)

    # Remove old badge cells wherever they occur, then add one canonical first cell.
    notebook.cells = [
        cell
        for cell in notebook.cells
        if not (cell.cell_type == "markdown" and BADGE_ASSET in cell.source)
    ]
    badge = nbformat.v4.new_markdown_cell(markdown)
    if notebook.get("nbformat_minor", 0) >= 5:
        badge.id = "open-in-colab"
    else:
        badge.pop("id", None)
    badge.metadata["tags"] = ["colab-button"]
    notebook.cells.insert(0, badge)

    nbformat.validate(notebook)
    nbformat.write(notebook, path)


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    repo_url = config["book"]["repo-url"]
    branch = config["book"].get("repo-branch", "main")
    paths = published_notebooks(config["book"]["chapters"])

    for notebook_path in paths:
        path = ROOT / notebook_path
        if not path.exists():
            raise FileNotFoundError(f"Published notebook does not exist: {notebook_path}")
        if "instructor" in path.stem.lower():
            raise ValueError(f"Instructor notebook found in public chapters: {notebook_path}")

        markdown = badge_markdown(repo_url, branch, notebook_path)
        add_or_update_badge(path, markdown)
        print(f"updated {notebook_path}")

    print(f"Added or refreshed Colab badges in {len(paths)} public notebooks.")


if __name__ == "__main__":
    main()
