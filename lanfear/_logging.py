"""Package-wide logging for lanfear.

A single logger named ``"lanfear"`` is configured when the package is imported;
every module logs through a child of it (``logging.getLogger(__name__)``), so all
records flow to one handler. Control the verbosity from a calling script with
:func:`set_verbosity`::

    import lanfear as lf

    lf.set_verbosity("INFO")

The default level is ``WARNING``. Messages are written to stderr and, because the
package logger does not propagate, do not interfere with the application's root
logging configuration.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

_LOGGER_NAME = "lanfear"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return the package logger or a named child of it.

    Parameters
    ----------
    name : str, optional
        A dotted module name (typically ``__name__``). If None or equal to the
        package name, the root package logger is returned.

    Returns
    -------
    logger : logging.Logger
        The requested logger; a child of the ``"lanfear"`` logger.
    """
    if not name or name == _LOGGER_NAME:
        return logging.getLogger(_LOGGER_NAME)
    return logging.getLogger(name)


def configure(level: Union[int, str] = logging.WARNING) -> logging.Logger:
    """Attach the default stderr handler to the package logger.

    Called once when the package is imported. It is idempotent: calling it again
    will not add a duplicate handler.

    Parameters
    ----------
    level : int or str, optional
        Initial verbosity level (default ``logging.WARNING``).

    Returns
    -------
    logger : logging.Logger
        The configured package logger.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if not any(getattr(h, "_lanfear_handler", False) for h in logger.handlers):
        handler = logging.StreamHandler()  # -> stderr
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        handler._lanfear_handler = True  # tag so we do not add it twice
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False  # our handler owns lanfear output
    return logger


def set_verbosity(level: Union[int, str]) -> int:
    """Set the verbosity of the lanfear package logger.

    Parameters
    ----------
    level : int or str
        A logging level: an integer (e.g. ``logging.INFO``) or a level name such
        as ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"`` or ``"CRITICAL"``.

    Returns
    -------
    level : int
        The numeric level that was set.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if isinstance(level, str):
        level = level.upper()
    logger.setLevel(level)
    return logger.level
