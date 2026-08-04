"""Consistency checks on the LaTeX sources.

The report rules require that every float carries a label, that every label is
referred to at least once, that no figure file is included twice under two
different labels, and that the bibliography is actually used. Checking this by
hand before submission has failed before, so it is checked here instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
FIGURE_DIR = ROOT / "results" / "figures"

REPORT_TEX = REPORT_DIR / "report.tex"
SLIDES_TEX = REPORT_DIR / "slides.tex"
REFS_BIB = REPORT_DIR / "refs.bib"

# \resultfig{stem}{caption}{label} often spans several lines, so the groups may
# be separated by whitespace.
RESULTFIG_RE = re.compile(
    r"\\resultfig\s*\{([^}]*)\}\s*\{.*?\}\s*\{(fig:[^}]*)\}", re.DOTALL
)
LABEL_RE = re.compile(r"\\label\{([^}]*)\}")
REF_RE = re.compile(r"\\(?:ref|eqref|autoref)\{([^}]*)\}")
CITE_RE = re.compile(r"\\cite[a-z]*\{([^}]*)\}")
BIB_ENTRY_RE = re.compile(r"^@\w+\{([^,]+),", re.MULTILINE)


def read_report() -> str:
    return REPORT_TEX.read_text(encoding="utf-8")


def declared_labels(text: str) -> list[str]:
    """Labels from \\label and from the third argument of \\resultfig."""
    return LABEL_RE.findall(text) + [label for _, label in RESULTFIG_RE.findall(text)]


def test_labels_are_unique() -> None:
    labels = declared_labels(read_report())
    duplicates = {label for label in labels if labels.count(label) > 1}
    assert not duplicates, f"labels declared more than once: {sorted(duplicates)}"


def test_every_float_is_referenced() -> None:
    """Figures and tables must be mentioned at least once in the body text.

    Section and equation labels are anchors and may stay unreferenced.
    """
    text = read_report()
    referenced = set(REF_RE.findall(text))
    unreferenced = [
        label
        for label in declared_labels(text)
        if label.startswith(("fig:", "tab:")) and label not in referenced
    ]
    assert not unreferenced, f"declared but never referenced: {sorted(unreferenced)}"


def test_every_reference_resolves() -> None:
    text = read_report()
    declared = set(declared_labels(text))
    dangling = sorted(set(REF_RE.findall(text)) - declared)
    assert not dangling, f"referenced but never declared: {dangling}"


def test_no_figure_file_is_included_twice() -> None:
    stems = [stem for stem, _ in RESULTFIG_RE.findall(read_report())]
    duplicates = {stem for stem in stems if stems.count(stem) > 1}
    assert not duplicates, f"figure files included more than once: {sorted(duplicates)}"


@pytest.mark.skipif(not FIGURE_DIR.exists(), reason="figures have not been generated")
def test_included_figures_exist_on_disk() -> None:
    missing = [
        stem
        for stem, _ in RESULTFIG_RE.findall(read_report())
        if not (FIGURE_DIR / f"{stem}.pdf").exists()
    ]
    assert not missing, f"included but missing from results/figures: {sorted(missing)}"


def test_bibliography_is_used() -> None:
    """Every entry in refs.bib is cited, and every citation key exists."""
    cited: set[str] = set()
    for path in (REPORT_TEX, SLIDES_TEX):
        if path.exists():
            for group in CITE_RE.findall(path.read_text(encoding="utf-8")):
                cited.update(key.strip() for key in group.split(","))

    entries = set(BIB_ENTRY_RE.findall(REFS_BIB.read_text(encoding="utf-8")))
    assert not cited - entries, f"cited but absent from refs.bib: {sorted(cited - entries)}"
    assert not entries - cited, f"present in refs.bib but never cited: {sorted(entries - cited)}"
