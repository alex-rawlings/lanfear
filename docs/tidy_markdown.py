"""Tidy a Sphinx markdown build for the GitHub wiki.

Two independent problems with the raw ``sphinx-markdown-builder`` output on
GitHub's wiki:

* NumPy-style "Parameters" fields render as (sometimes deeply nested) bullet
  lists. GitHub's wiki has no themed styling to make that scannable, so it
  reads as a wall of text; a Markdown table reads far better under GitHub's
  plain CSS.

* Cross-references point at ``#lanfear.module.Class.method``-style fragments
  (matching the ``<a id="...">`` anchors ``markdown_anchor_signatures`` in
  conf.py emits ahead of each heading) -- but GitHub's wiki sanitizer strips
  those custom anchors and generates its own from the heading *text* instead
  (e.g. ``user-content-class-particlesystem``), so every internal link is
  silently dead. This recomputes GitHub's actual anchor slugs (reverse
  engineered against real rendered pages, including its per-page duplicate
  suffixing) and rewrites every link to match, then drops the now-useless
  ``<a id="...">`` lines.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

_PARAM_HEADER_RE = re.compile(r"^\* \*\*Parameters:\*\*\s*$")
_NEW_ITEM_RE = re.compile(r"^  \* (.*)$")
_NAME_RE = re.compile(r"^\*\*(.+?)\*\*")

# Top-level (### only -- nested members already show a bare name) headings
# carry the fully-qualified `lanfear.module.name`; the module a page covers
# is already obvious from its title, so drop that prefix for scannability.
# The <a id="lanfear.module.name"> anchor just above (markdown_anchor_signatures
# in conf.py) still carries the full name, used below to fix up cross-links,
# before being stripped from the final output.
_HEADING_PREFIX_RE = re.compile(r"^(### (?:\*\w+\* )*)lanfear\.(?:\w+\.)+")

# A class heading's own constructor arg list is long and, unlike a function's,
# adds little (it's already covered by the class's own Parameters table just
# below) -- drop it. Functions/methods keep their arg list; it's the useful
# part of their heading.
_CLASS_ARGS_RE = re.compile(r"^(### \*class\* \S+)\(.*\)\s*$")

_ANCHOR_RE = re.compile(r'^<a id="([^"]+)"></a>\s*$')
_HEADING_RE = re.compile(r"^(#{1,6}) (.*)$")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_CODE_RE = re.compile(r"`([^`]*)`")
_NON_SLUG_RE = re.compile(r"[^\w\s-]")
_DOTTED_LINK_RE = re.compile(r"(\]\((?:[\w./]*)?#)(lanfear\.[\w.]+)(\))")


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


def _heading_text(raw: str) -> str:
    """Reduce a heading line's Markdown to the plain text GitHub slugs."""
    text = _MD_LINK_RE.sub(r"\1", raw)
    text = _MD_CODE_RE.sub(r"\1", text)
    text = text.replace("*", "").replace("\\", "")
    return text


def _github_slug(heading_text: str, seen: dict) -> str:
    """GitHub's wiki heading-anchor slug, including its de-dup suffixing.

    Reverse engineered against real rendered wiki pages: lowercase, drop
    everything but word characters/spaces/hyphens (deleting, not replacing
    with a space -- ``"a (b)"`` -> ``"a-b"`` but ``"a: b"`` -> ``"a--b"``,
    since only the ``:`` is deleted and *both* surrounding spaces survive to
    become hyphens), then turn each remaining space into a hyphen. A slug
    already used earlier on the same page gets ``-1``, ``-2``, ... appended.
    """
    slug = _NON_SLUG_RE.sub("", heading_text.lower()).replace(" ", "-")
    n = seen[slug]
    seen[slug] += 1
    return slug if n == 0 else f"{slug}-{n}"


def _collect_anchors(md_files: list) -> dict:
    """Map each symbol's dotted name to its (filename, real GitHub slug)."""
    targets = {}
    for path in md_files:
        seen = defaultdict(int)
        pending_id = None
        for line in path.read_text().splitlines():
            m = _HEADING_RE.match(line)
            if m:
                slug = _github_slug(_heading_text(m.group(2)), seen)
                if pending_id:
                    targets[pending_id] = (path.name, slug)
                pending_id = None
                continue
            m = _ANCHOR_RE.match(line)
            pending_id = m.group(1) if m else pending_id
    return targets


def _fix_links(text: str, this_file: str, targets: dict) -> str:
    """Point cross-reference links at GitHub's real anchor slugs."""

    def repl(m):
        dotted = m.group(2)
        if dotted not in targets:
            return m.group(0)
        target_file, slug = targets[dotted]
        prefix = "](#" if target_file == this_file else f"]({target_file}#"
        return f"{prefix}{slug}{m.group(3)}"

    return _DOTTED_LINK_RE.sub(repl, text)


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
    md_files = sorted(build_dir.glob("*.md"))

    # Pass 1: Parameters tables + heading simplification. Anchor lines are
    # kept for now -- pass 2 needs them to know each heading's dotted name.
    for path in md_files:
        path.write_text(tidy(path.read_text()))

    # Pass 2: now that headings are in their final (shortened) form, compute
    # GitHub's real anchor slugs for them, repoint every cross-reference link
    # at the right one, and drop the anchor lines (GitHub strips them anyway).
    targets = _collect_anchors(md_files)
    for path in md_files:
        text = _fix_links(path.read_text(), path.name, targets)
        text = "\n".join(
            line for line in text.splitlines() if not _ANCHOR_RE.match(line)
        )
        path.write_text(text + "\n")


if __name__ == "__main__":
    main()
