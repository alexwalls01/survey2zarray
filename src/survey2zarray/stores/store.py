"""Provide shared Zarr store creation, opening, consolidation, and versioning."""

import warnings
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from pathlib import Path

import zarr
from zarr.errors import ZarrUserWarning

# Bump only when the on-disk layout changes, and register a migration with it.
SURVEY_SCHEMA_VERSION: int = 2
DATASET_SCHEMA_VERSION: int = 2
SURVEY_UPGRADE_COMMAND: str = "s2z survey upgrade"
DATASET_UPGRADE_COMMAND: str = "s2z dataset upgrade"
# Stores written before versioning hold no tag.
UNTAGGED_SCHEMA_VERSION: int = 1


class StoreVersionError(Exception):
    """Represent a store whose schema version this code cannot use."""


def package_version() -> str:
    """Return the installed survey2zarray version.

    Returns
    -------
    result
        Version string, or ``'unknown'`` where the package is not installed.
    """
    try:
        return installed_version("survey2zarray")
    except PackageNotFoundError:
        return "unknown"


def schema_version(root: zarr.Group) -> int:
    """Return the schema version a store was written in.

    Parameters
    ----------
    root
        Root group of the store.

    Returns
    -------
    result
        Stored ``schema_version``, or `UNTAGGED_SCHEMA_VERSION` where absent.
    """
    value = root.attrs.get("schema_version")
    if value is None:
        return UNTAGGED_SCHEMA_VERSION
    return int(value)


def stamp_version(root: zarr.Group, current: int) -> None:
    """Record the schema version and package version a store is written with.

    Parameters
    ----------
    root
        Root group of a store being created.
    current
        Schema version of the store type.
    """
    root.attrs["schema_version"] = int(current)
    root.attrs["created_with"] = package_version()


def check_readable(root: zarr.Group, current: int, path: str | Path) -> int:
    """Refuse a store written in a newer schema than this code reads.

    Parameters
    ----------
    root
        Root group of the store.
    current
        Newest schema version this code reads for the store type.
    path
        Location of the store, used in error messages.

    Returns
    -------
    result
        The store's schema version.

    Raises
    ------
    StoreVersionError
        Where the store's schema version is newer than ``current``.
    """
    stored = schema_version(root)
    if stored > current:
        created_with = root.attrs.get("created_with", "an unknown version")
        raise StoreVersionError(
            f"{path} uses schema version {stored}, and this installation "
            f"(survey2zarray {package_version()}) reads up to version {current}. "
            f"The store was built with survey2zarray {created_with}, so install "
            f"that version or later."
        )
    return stored


def check_editable(
    root: zarr.Group, current: int, path: str | Path, upgrade_command: str
) -> None:
    """Refuse to edit a store written in any schema other than the current one.

    Parameters
    ----------
    root
        Root group of the store.
    current
        Schema version this code writes for the store type.
    path
        Location of the store, used in error messages.
    upgrade_command
        CLI command that upgrades the store type, such as ``'s2z survey upgrade'``.

    Raises
    ------
    StoreVersionError
        Where the store is newer than this code, or older and not yet upgraded.
    """
    stored = check_readable(root, current, path)
    if stored < current:
        raise StoreVersionError(
            f"{path} uses schema version {stored}, and edits need version "
            f"{current}. Run '{upgrade_command} {path}' first."
        )


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
