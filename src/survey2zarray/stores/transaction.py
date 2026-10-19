"""Protect store edits against interruption.

Both store types edit in place by resizing arrays and rewriting attributes, so
an interruption partway through would leave a store whose arrays disagree with
one another. An edit records what it is about to touch, and a store that opens
holding an unfinished one is put back the way it was."""

import fcntl
import os
import shutil
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

import zarr

from survey2zarray.stores.store import consolidate, open_store
from survey2zarray.utils.logging import get_logger

logger = get_logger(__name__)

BACKUP_SUFFIX = ".orig"
EDIT_FLAG: str = "edit_in_progress"
# Durable states distinguish rollback from published-commit finalisation.
WRITING: str = "writing"
COMMITTED: str = "committed"
# Place the lock beside the store to exclude it from Zarr metadata.
LOCK_SUFFIX: str = ".edit.lock"


@contextmanager
def store_edit_lock(path: str | Path):
    """Hold the exclusive process lock for one local store.

    Parameters
    ----------
    path
        Store path.

    Yields
    ------
    item
        Control while the process lock is held.
    """
    store_path = Path(path)
    lock_path = store_path.with_name(store_path.name + LOCK_SUFFIX)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def serialised_store_edit(function: Callable) -> Callable:
    """Run a store-editing entry point under its store's process lock.

    Acquire the lock before recovery so concurrent writers wait for live edits.

    Parameters
    ----------
    function
        Store-editing callable.

    Returns
    -------
    result
        Callable serialised by store path.
    """

    @wraps(function)
    def wrapped(store_path, *args, **kwargs):
        """Execute one store edit under its path lock.

        Parameters
        ----------
        store_path
            Store path.
        *args
            Positional arguments for the wrapped edit.
        **kwargs
            Keyword arguments for the wrapped edit.

        Returns
        -------
        result
            Return value from the wrapped edit.
        """
        path = Path(store_path)
        with store_edit_lock(path):
            recover_pending_edit(path, _already_locked=True)
            return function(path, *args, **kwargs)

    return wrapped


def store_directory(root: zarr.Group) -> Path:
    """Return the directory a store is written to.

    Parameters
    ----------
    root
        Store root group.

    Returns
    -------
    result
        Directory backing the store.

    Raises
    ------
    TypeError
        Where the group is not backed by a directory, which the file-level parts
        of an edit need.
    """
    location = getattr(root.store, "root", None)
    if location is None:
        raise TypeError(
            f"{type(root.store).__name__} is not backed by a directory, so files "
            f"beside the arrays cannot take part in an edit."
        )
    return Path(location)


def begin_edit(
    root: zarr.Group,
    array_paths: list[str],
    attrs: dict[str, Any] | None = None,
    groups: list[str] | None = None,
    files: list[str] | None = None,
) -> dict[str, int]:
    """Open an edit, recording enough state to undo it.

    Copies are taken before the flag names them. A crash in between leaves an
    unreferenced copy, which is harmless, whereas naming a copy that does not
    exist yet would make revert restore nothing.

    Attributes are written through immediately, so the record is durable before
    any array is touched.

    Parameters
    ----------
    root
        Root group being edited.
    array_paths
        Paths of every array the edit will extend. Their current lengths
        become the marks to revert to.
    attrs
        Prior values of any attributes the edit will change, restored on
        revert. A key of the form ``'group/path:name'`` targets a subgroup.
    groups
        Paths the edit creates whole, which revert deletes outright since they
        have no prior length to truncate to.
    files
        Plain files beside the arrays that the edit will rewrite, as paths
        relative to the store. Each is copied aside first and restored on
        revert, so a byte map cannot be left describing images the metadata was
        rolled back past.

    Returns
    -------
    result
        The recorded marks, keyed by array path.
    """
    if root.attrs.get(EDIT_FLAG):
        raise RuntimeError("Cannot begin an edit while another edit is recorded.")
    marks = {path: int(root[path].shape[0]) for path in array_paths}
    saved = []
    if files:
        directory = store_directory(root)
        for relative in files:
            source = directory / relative
            existed = source.exists()
            if existed:
                backup = source.with_name(source.name + BACKUP_SUFFIX)
                temporary = backup.with_name(backup.name + ".tmp")
                shutil.copyfile(source, temporary)
                os.replace(temporary, backup)
            saved.append([relative, existed])
    root.attrs[EDIT_FLAG] = {
        "phase": WRITING,
        "marks": marks,
        "attrs": attrs or {},
        "groups": list(groups or []),
        "files": saved,
    }
    return marks


def commit_edit(root: zarr.Group) -> None:
    """Close an edit that completed successfully.

    The completed arrays are consolidated before the transaction becomes
    committed. The committed state is then consolidated too, so recovery can
    distinguish a write which must be reverted from a published commit which
    only needs finalising. Backups remain until that distinction is durable.

    Parameters
    ----------
    root
        Root group being edited.
    """
    state = dict(root.attrs.get(EDIT_FLAG) or {})
    if not state:
        raise RuntimeError("Cannot commit because no edit is in progress.")

    # Retain rollback state until resized and created members are published.
    consolidate(root)
    state["phase"] = COMMITTED
    root.attrs[EDIT_FLAG] = state
    # Mark successful publication so recovery completes commit cleanup.
    consolidate(root)

    _remove_backups(root, state)
    del root.attrs[EDIT_FLAG]
    # Refresh consolidated metadata to clear the reader-visible edit flag.
    consolidate(root)


def _remove_backups(root: zarr.Group, state: dict[str, Any]) -> None:
    """Delete file backups belonging to an edit which will not roll back."""
    directory = None
    for relative, _ in state.get("files", []):
        if directory is None:
            directory = store_directory(root)
        backup = directory / (relative + BACKUP_SUFFIX)
        if backup.exists():
            backup.unlink()


def _finish_committed_edit(root: zarr.Group, state: dict[str, Any]) -> None:
    """Complete cleanup after a commit interrupted during finalisation."""
    _remove_backups(root, state)
    if root.attrs.get(EDIT_FLAG):
        del root.attrs[EDIT_FLAG]
    consolidate(root)


def recover_edit(root: zarr.Group) -> bool:
    """Undo an unfinished edit, if one is recorded.

    Every recorded array is truncated together, since reverting only some would
    leave maps pointing past their data, or data nothing references. The revert is
    idempotent and the flag is cleared only once it finishes, so an interrupted
    revert is safely re-run.

    Parameters
    ----------
    root
        Root group to inspect.

    Returns
    -------
    result
        True if an edit was reverted.
    """
    state = root.attrs.get(EDIT_FLAG)
    if not state:
        return False
    state = dict(state)
    if state.get("phase") == COMMITTED:
        logger.warning("Committed edit needs finalising. Completing its cleanup.")
        _finish_committed_edit(root, state)
        return True
    logger.warning(
        "Unfinished edit detected, so arrays are reverted to their high-water marks."
    )
    for path in state.get("groups", []):
        if path in root:
            del root[path]
    for path, mark in state["marks"].items():
        array = root[path]
        if array.shape[0] > mark:
            array.resize((mark, *array.shape[1:]))
    directory = None
    for relative, existed in state.get("files", []):
        if directory is None:
            directory = store_directory(root)
        target = directory / relative
        backup = directory / (relative + BACKUP_SUFFIX)
        if backup.exists():
            # Atomic rename completes each file restoration once.
            os.replace(backup, target)
        elif not existed and target.exists():
            target.unlink()

    for key, value in state.get("attrs", {}).items():
        # Parse group/path:attr keys while bare keys address root attributes.
        if ":" in key:
            group_path, attr_name = key.split(":", 1)
            root[group_path].attrs[attr_name] = value
        else:
            root.attrs[key] = value
    # Keep rollback state active until the restored layout is published.
    consolidate(root)
    del root.attrs[EDIT_FLAG]
    consolidate(root)
    return True


def recover_pending_edit(path: str | Path, _already_locked: bool = False) -> bool:
    """Undo an unfinished edit on a store at a given path.

    The flag is checked read-only first, so opening a healthy store for reading
    never requires write access.

    Parameters
    ----------
    path
        Location of the store.

    Returns
    -------
    result
        True if an edit was reverted.
    """
    root = open_store(path, mode="r")
    observed = root.attrs.get(EDIT_FLAG)
    if not observed:
        return False
    if not _already_locked:
        with store_edit_lock(path):
            return recover_pending_edit(path, _already_locked=True)

    root = open_store(path, mode="r+")
    if root.attrs.get(EDIT_FLAG):
        return recover_edit(root)

        # A committed record authorises metadata refresh and backup removal.
    observed = dict(observed)
    if observed.get("phase") == COMMITTED:
        _remove_backups(root, observed)
        consolidate(root)
        return True
    return False
