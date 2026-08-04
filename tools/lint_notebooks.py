"""Run pylint over the code cells of every notebook.

pylint cannot read .ipynb, so each notebook's code cells are concatenated into
a temporary module, checked, and the findings are mapped back to the cell and
line they came from.

    python tools/lint_notebooks.py

Four checks are switched off because they flag notebook conventions rather
than defects:

* invalid-name, because a cell assigns at module level and pylint then wants
  every name in upper case;
* wrong-import-position, because the first cell extends sys.path before any
  project import can run;
* pointless-statement, because a bare expression on the last line of a cell is
  how a notebook displays a value;
* trailing-newlines, which is an artefact of joining the cells.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"

DISABLED = (
    "invalid-name",
    "wrong-import-position",
    "pointless-statement",
    "trailing-newlines",
)

FINDING_RE = re.compile(r"(.*?):(\d+):(\d+): (\S+): (.*)")


def flatten(notebook: Path, target: Path) -> dict[int, tuple[int, int]]:
    """Write the code cells to `target`; return module line -> (cell, line)."""
    cells = json.loads(notebook.read_text())["cells"]
    lines = [f'"""Code cells of {notebook.name}, flattened for static analysis."""', ""]
    mapping: dict[int, tuple[int, int]] = {}

    cell_number = 0
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        cell_number += 1
        body = "".join(cell["source"]).rstrip("\n")
        if not body.strip():
            continue
        for offset, line in enumerate(body.split("\n"), start=1):
            lines.append(line)
            mapping[len(lines)] = (cell_number, offset)
        lines.append("")

    target.write_text("\n".join(lines) + "\n")
    return mapping


def main() -> int:
    """Check every notebook and print the findings. Returns an exit status."""
    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not notebooks:
        print(f"no notebooks found in {NOTEBOOK_DIR}")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        maps = {}
        for notebook in notebooks:
            target = directory / f"{notebook.stem}.py"
            maps[target.name] = (notebook.name, flatten(notebook, target))

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pylint",
                str(directory),
                f"--rcfile={ROOT / '.pylintrc'}",
                f"--disable={','.join(DISABLED)}",
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=ROOT,
        )

        findings = 0
        for line in completed.stdout.splitlines():
            match = FINDING_RE.match(line)
            if not match:
                continue
            path, number, _, code, message = match.groups()
            name, mapping = maps[Path(path).name]
            cell, offset = mapping.get(int(number), (0, 0))
            print(f"{name}  cell {cell} line {offset}  {code}  {message}")
            findings += 1

    print(f"\n{len(notebooks)} notebooks checked, {findings} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
