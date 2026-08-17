"""Prose checks on the LaTeX sources.

These are the style rules that a machine can decide. Keeping them here rather
than in `docs/van-phong-tieng-viet.md` is the point: a rule that lives in a
document has to be remembered and reviewed by hand every time, which is the
approach that already failed once on this project. A rule that lives in a test
costs nothing to enforce.

The hard part is separating prose from markup. A naive scan for a decimal point
reports 39 hits on report.tex, nearly all of them layout parameters such as
`0.5cm` or `rgb{0.58,0,0.82}`. `prose()` below therefore drops the preamble,
the math, and the arguments of every command that does not carry prose.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"

SOURCES = ("report.tex", "slides.tex")

# Commands whose braced argument is text meant for the reader. Everything else
# is treated as markup and dropped together with its arguments.
PROSE_COMMANDS = frozenset(
    {
        "caption", "emph", "textbf", "textit", "text", "footnote", "note",
        "chapter", "section", "subsection", "subsubsection", "paragraph",
        "subparagraph", "item", "frametitle", "title",
    }
)

# Environments whose body is not prose.
NON_PROSE_ENVIRONMENTS = (
    "equation|align|aligned|gather|multline|cases|array|tabular|tabularx"
    "|lstlisting|verbatim|minted|tikzpicture|algorithmic"
)

EM_DASH = "\u2014"

DECIMAL_POINT_RE = re.compile(r"\d\.\d")

# Version numbers keep the dot. The rule is about numbers the report measured,
# not about the names of the tools that produced them.
SOFTWARE_NAMES = (
    "Python", "NumPy", "SciPy", "scikit-learn", "macOS", "TeX Live",
    "pandas", "matplotlib", "Accelerate",
)

# One canonical name per concept. The alternates below are the ones a writer
# reaches for by accident; the report currently uses none of them.
BANNED_SYNONYMS = {
    "số điều kiện": ("chỉ số điều kiện", "hệ số điều kiện"),
    "độ dài bước": ("kích thước bước", "bước nhảy", "độ lớn bước"),
    "hệ số hiệu chỉnh": ("hệ số chính quy", "tham số hiệu chỉnh", "hệ số phạt"),
    "vòng lặp": ("bước lặp",),
    "phân kỳ": ("phân kì",),
    "line search": ("tìm kiếm đường thẳng", "dò tìm bước"),
    "backtracking": ("quay lui",),
}


def _skip_group(text: str, i: int, opener: str, closer: str) -> int:
    """Index just past the group starting at `i`, honouring nesting."""
    depth = 0
    while i < len(text):
        char = text[i]
        if char == "\\":
            i += 2
            continue
        if char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return i


def strip_commands(text: str) -> str:
    """Drop LaTeX commands, keeping the arguments of the prose-bearing ones."""
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] != "\\":
            out.append(text[i])
            i += 1
            continue

        match = re.match(r"\\([A-Za-z@]+)\*?", text[i:])
        if match is None:
            # A control symbol such as \\ or \,. The line-break form takes an
            # optional spacing argument, which is layout rather than prose.
            i += 2
            if i < len(text) and text[i] == "[":
                i = _skip_group(text, i, "[", "]")
            continue

        name = match.group(1)
        i += match.end()
        while i < len(text) and text[i] in " \n\t":
            if text[i] == "\n":
                out.append("\n")
            i += 1

        while i < len(text) and text[i] in "[{":
            opener, closer = ("[", "]") if text[i] == "[" else ("{", "}")
            end = _skip_group(text, i, opener, closer)
            if name in PROSE_COMMANDS and opener == "{":
                out.append(strip_commands(text[i + 1 : end - 1]))
            i = end

    return "".join(out)


def prose(text: str) -> str:
    """The parts of a source file that a reader actually reads."""
    body = text.split(r"\begin{document}", 1)[-1]
    body = re.sub(r"(?<!\\)%.*", " ", body)
    body = re.sub(
        r"\\begin\{(?:%s)\*?\}.*?\\end\{(?:%s)\*?\}"
        % (NON_PROSE_ENVIRONMENTS, NON_PROSE_ENVIRONMENTS),
        " ",
        body,
        flags=re.DOTALL,
    )
    body = re.sub(r"\\\[.*?\\\]", " ", body, flags=re.DOTALL)
    body = re.sub(r"\$\$.*?\$\$", " ", body, flags=re.DOTALL)
    body = re.sub(r"\$[^$]*\$", " ", body, flags=re.DOTALL)
    return strip_commands(body)


def read_source(name: str) -> str:
    """Raw text of one file under report/."""
    path = REPORT_DIR / name
    if not path.exists():
        pytest.skip(f"{name} is not present")
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("name", SOURCES)
def test_no_em_dash(name: str) -> None:
    """Vietnamese punctuation here uses commas, colons or a sentence break."""
    text = read_source(name)
    assert EM_DASH not in text, f"{name} contains an em dash"


@pytest.mark.parametrize("name", SOURCES)
def test_decimal_separator_in_prose_is_a_comma(name: str) -> None:
    """A measured number in a sentence reads 0,773 rather than 0.773.

    Numbers inside math mode keep the dot, and so do software version numbers.
    """
    body = prose(read_source(name))
    offenders = []
    for match in DECIMAL_POINT_RE.finditer(body):
        window = body[max(0, match.start() - 40) : match.start()]
        if any(tool in window for tool in SOFTWARE_NAMES):
            continue
        offenders.append(body[max(0, match.start() - 30) : match.start() + 10].strip())

    assert not offenders, f"{name} uses a decimal point in prose: {offenders}"


@pytest.mark.parametrize("name", SOURCES)
def test_terminology_is_consistent(name: str) -> None:
    """One concept, one name. A second name for it reads as a second concept."""
    body = prose(read_source(name)).lower()
    found = {
        canonical: [word for word in alternates if word in body]
        for canonical, alternates in BANNED_SYNONYMS.items()
    }
    offenders = {key: value for key, value in found.items() if value}
    assert not offenders, f"{name} mixes names for one concept: {offenders}"
