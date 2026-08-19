"""Sphinx configuration for the lanfear API documentation."""

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(".."))

# Read __version__ from source rather than `import lanfear`: the CI runner
# builds these docs without compiling the C++ extension, and `import lanfear`
# pulls in `lanfear._core` (see autodoc_mock_imports below for how autodoc
# itself copes with that).
_init_py = (
    Path(__file__).resolve().parent.parent / "lanfear" / "__init__.py"
).read_text()
release = re.search(r'__version__ = "(.*?)"', _init_py).group(1)

project = "lanfear"
copyright = "2026, Alex Rawlings"
author = "Alex Rawlings"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_markdown_builder",
]

# The compiled extension isn't built in the docs CI job; mock it so the pure
# Python modules that do `from . import _core` can still be imported and
# documented.
autodoc_mock_imports = ["lanfear._core"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
# Some classes document their constructor params on the class docstring
# (dataclasses), others on __init__'s (e.g. Potential) -- merge both so
# neither convention ends up with a blank, type-only Parameters table.
autoclass_content = "both"

# Markdown build (rendered by the repo wiki): emit an <a id="..."> anchor
# ahead of each heading using the symbol's dotted name, matching what
# cross-reference links point at -- GitHub slugifies heading *text* for its
# auto-anchors, which won't match once docs/tidy_markdown.py shortens the
# visible heading text below.
markdown_anchor_signatures = True

napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

exclude_patterns = ["_build"]

html_theme = "furo"
