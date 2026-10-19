"""Provide shared Zarr store creation, opening, and consolidation."""

import warnings
from pathlib import Path

import zarr
from zarr.errors import ZarrUserWarning


def open_store(path: str | Path, mode: str = "r") -> zarr.Group:
    """Open an existing store's root group.

    Readers use consolidated metadata to load array definitions in one request.
    Writers use live metadata so edits expose newly added arrays.

    Parameters
    ----------
    path
        Directory the store was written to.
    mode
        Passed to ``zarr.open_group``. ``"r"`` for reading, ``"r+"`` to edit
        a store that already exists.

    Returns
    -------
    result
        Root group of the store.
    """
    return zarr.open_group(
        str(path), mode=mode, use_consolidated=None if mode == "r" else False
    )


def create_store(path: str | Path) -> zarr.Group:
    """Create a store's root group, discarding anything already at the path.

    Parameters
    ----------
    path
        Directory to write the store to.

    Returns
    -------
    result
        Empty root group.
    """
    return zarr.group(store=str(path), overwrite=True)


def consolidate(root: zarr.Group) -> zarr.Group:
    """Gather a store's metadata into its root, so opening it is one read.

    Combine small-array metadata into one request. Rebuild the consolidated
    copy after adding, removing, or resizing an array.

    zarr warns twice on every call, once that consolidated metadata is an
    extension to the format, and once for each CSV record beside the arrays.
    Suppress both layout warnings during store construction.

    Parameters
    ----------
    root
        Root group of a finished store or edit.

    Returns
    -------
    result
        The root group, reopened from the consolidated metadata.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ZarrUserWarning)
        return zarr.consolidate_metadata(root.store)


def apply_coalesce_gap(gap_bytes: int | None) -> None:
    """Set how far zarr reads through a shard between the chunks it wants.

    zarr's own default is 1 MiB, which is larger than a whole shard at the
    geometry these stores use, so leaving it merges every wanted chunk with
    everything between them and serves the bytes sharding exists to skip.

    The process-wide Zarr setting applies to every open array and passes to
    forked DataLoader workers.

    Parameters
    ----------
    gap_bytes
        Maximum gap to include in one read, in bytes. None retains the Zarr
        default.
    """
    if gap_bytes is None:
        return
    key = "array.sharding_coalesce_max_gap_bytes"
    if zarr.config.get(key) != int(gap_bytes):
        zarr.config.set({key: int(gap_bytes)})
