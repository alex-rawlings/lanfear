"""Tidy a Sphinx markdown build for the GitHub wiki.

``sphinx-markdown-builder`` renders NumPy-style "Parameters" fields as
(sometimes deeply nested) bullet lists. GitHub's wiki has no themed styling
to make that scannable, so it reads as a wall of text; a Markdown table reads
far better under GitHub's plain CSS. This rewrites every "Parameters" field
into a table, leaving everything else (headings, Returns, Raises, prose)
untouched.
"""

import re
import sys
from pathlib import Path

_PARAM_HEADER_RE = re.compile(r"^\* \*\*Parameters:\*\*\s*$")
_NEW_ITEM_RE = re.compile(r"^  \* (.*)$")
_NAME_RE = re.compile(r"^\*\*(.+?)\*\*")

# Top-level (### only -- nested members already show a bare name) headings
# carry the fully-qualified `lanfear.module.name`; the module a page covers
# is already obvious from its title, so drop that prefix for scannability.
# The <a id="lanfear.module.name"> anchor just above (markdown_anchor_signatures
# in conf.py) still carries the full name, so cross-page links are unaffected.
_HEADING_PREFIX_RE = re.compile(r"^(### (?:\*\w+\* )*)lanfear\.(?:\w+\.)+")

# A class heading's own constructor arg list is long and, unlike a function's,
# adds little (it's already covered by the class's own Parameters table just
# below) -- drop it. Functions/methods keep their arg list; it's the useful
# part of their heading.
_CLASS_ARGS_RE = re.compile(r"^(### \*class\* \S+)\(.*\)\s*$")


def _match_paren(s: str, open_idx: int) -> int:
    """Index just past the ``)`` matching the ``(`` at ``s[open_idx]``."""
    depth = 0
    for i in range(open_idx, len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(s)


def _split_param(text: str) -> tuple:
    """Split ``**name** (type) -- description`` into its three parts."""
    m = _NAME_RE.match(text)
    if not m:
        return text, "", ""
    name = m.group(0)
    rest = text[m.end() :]
    typ = ""
    if rest.startswith(" ("):
        end = _match_paren(rest, 1)
        typ = rest[2 : end - 1]
        rest = rest[end:]
    desc = rest[3:] if rest.startswith(" \u2013 ") else rest.strip()
    return name, typ, desc.strip()


def _cell(text: str) -> str:
    """Escape a string for use inside a Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _params_to_table(body_lines: list) -> list:
    """Render one field's collected body lines as a Markdown table."""
    entries = []
    current = None
    for line in body_lines:
        m = _NEW_ITEM_RE.match(line)
        if m:
            if current is not None:
                entries.append(current)
            current = m.group(1)
        elif current is not None:
            current += " " + line.strip()
        else:
            current = line[2:] if line.startswith("  ") else line.strip()
    if current is not None:
        entries.append(current)

    rows = [_split_param(e) for e in entries]
    table = ["| Name | Type | Description |", "|---|---|---|"]
    table += [f"| {_cell(n)} | {_cell(t)} | {_cell(d)} |" for n, t, d in rows]
    return table


def tidy(text: str) -> str:
    """Rewrite every "Parameters" field in ``text`` into a Markdown table."""
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        line = _HEADING_PREFIX_RE.sub(r"\1", line)
        line = _CLASS_ARGS_RE.sub(r"\1", line)
        if _PARAM_HEADER_RE.match(line):
            j = i + 1
            body = []
            while j < len(lines) and lines[j].startswith("  "):
                body.append(lines[j])
                j += 1
            out += ["**Parameters:**", ""]
            out += _params_to_table(body)
            out.append("")
            i = j
            continue
        out.append(line)
        i += 1
    return "\n".join(out) + "\n"


def main() -> None:
    build_dir = Path(sys.argv[1])
    for md in build_dir.glob("*.md"):
        md.write_text(tidy(md.read_text()))


if __name__ == "__main__":
    main()
