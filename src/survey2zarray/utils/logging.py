"""Configure logging for the package."""

import logging

# Full module paths disambiguate repeated names such as build.py and edit.py.
LOG_FORMAT: str = "[%(levelname)s][%(name)s] %(message)s"
ROOT_LOGGER_NAME: str = "survey2zarray"


class ProgressAwareHandler(logging.StreamHandler):
    """Preserve active progress bars while emitting records.

    Use ``tqdm.write`` to clear and redraw active bars around each record.
    Fall back to standard stream handling when tqdm is unavailable.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit one log record around any active progress bar.

        Parameters
        ----------
        record
            Log record to emit.
        """
        try:
            from tqdm import tqdm
        except ImportError:
            super().emit(record)
            return
        try:
            tqdm.write(self.format(record), file=self.stream, end=self.terminator)
            self.flush()
        except Exception:  # noqa: BLE001 (Logging handlers must not raise.)
            self.handleError(record)


def configure(level: int = logging.INFO, stream=None) -> logging.Logger:
    """Attach one package-level handler.

    Replace any handler installed by an earlier call to prevent duplicate
    records.

    Parameters
    ----------
    level
        Level for the package logger.
    stream
        Where records go. Defaults to standard error.

    Returns
    -------
    result
        The package's root logger.
    """
    root = logging.getLogger(ROOT_LOGGER_NAME)
    for existing in list(root.handlers):
        if getattr(existing, "_survey2zarray", False):
            root.removeHandler(existing)
    handler = ProgressAwareHandler(stream)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    handler._survey2zarray = True
    root.addHandler(handler)
    root.setLevel(level)
    return root


def warn_many(
    logger: logging.Logger,
    items,
    summary: str,
    limit: int = 3,
) -> None:
    """Report a count and representative identifiers in one record.

    Parameters
    ----------
    logger
        Where to report.
    items
        One entry per thing to report. Nothing is logged when it is empty.
    summary
        What happened, taking ``{n}``, the number of things.
    limit
        How many to name before counting the rest.
    """
    items = list(items)
    if not items:
        return
    named = ", ".join(str(item) for item in items[:limit])
    if len(items) > limit:
        named += f", and {len(items) - limit} more"
    logger.warning(f"{summary.format(n=len(items))}. Examples are {named}.")


def get_logger(name: str) -> logging.Logger:
    """Return a module logger that propagates to the package handler.

    Applications can redirect or silence the full package with

    >>> logging.getLogger("survey2zarray").setLevel(logging.WARNING)

    which a module attaching its own handler would defeat.

    Parameters
    ----------
    name
        The module's ``__name__``.

    Returns
    -------
    result
        Module logger.
    """
    return logging.getLogger(name)


# Configure package output for command-line and notebook use by default.
configure()
