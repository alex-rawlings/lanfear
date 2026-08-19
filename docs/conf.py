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
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# The compiled extension isn't built in the docs CI job; mock it so the pure
# Python modules that do `from . import _core` can still be imported and
# documented.
autodoc_mock_imports = ["lanfear._core"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"

napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

exclude_patterns = ["_build"]

html_theme = "furo"
