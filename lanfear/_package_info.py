"""Package version/build info, for diagnostics and bug reports."""

from __future__ import annotations

import os
import subprocess


def _git_hash() -> str:
    """Return the current git commit hash, or ``"unknown"`` if unavailable.

    Returns
    -------
    githash : str
        The full HEAD commit hash of the repository containing this file, or
        ``"unknown"`` if git is unavailable or this is not a git checkout
        (e.g. a package installed from a wheel without the ``.git`` folder).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def print_package_info() -> None:
    """Print the lanfear version and current git commit hash."""
    from . import __version__  # deferred: avoids a circular import at load time

    githash = _git_hash()
    print("LANFEAR")
    print(f"> version {__version__}")
    print(f"> githash {githash[:8]}")
