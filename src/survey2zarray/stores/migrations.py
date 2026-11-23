"""Upgrade stores between schema versions, and adapt older stores for reading.

A store records the schema version it was written in. Readers adapt any older
version through the read adapters, edits require the current version, and the
upgrade functions here rewrite a store to the current version one step at a
time, each step a recoverable edit."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from survey2zarray.stores.store import (
    DATASET_SCHEMA_VERSION,
    SURVEY_SCHEMA_VERSION,
    StoreVersionError,
    open_store,
    package_version,
    schema_version,
)
from survey2zarray.stores.transaction import (
    attr_target,
    begin_edit,
    commit_edit,
    recover_edit,
    serialised_store_edit,
)
from survey2zarray.utils.logging import get_logger

logger = get_logger(__name__)

# Schema version to the release that introduced it and what changed.
SCHEMA_HISTORY: dict[str, dict[int, tuple[str, str]]] = {
    "survey": {
        1: ("0.1.0.dev0", "Initial layout."),
        2: (
            "0.1.0.dev2",
            "Resolution arrays renamed from bmaj, bmin, and bpa. Band and "
            "zeropoint arrays, and band, blank_value, resolution, and "
            "band_pattern attributes, added.",
        ),
    },
    "dataset": {
        1: ("0.1.0.dev0", "Initial layout."),
        2: (
            "0.1.0.dev2",
            "Resolution arrays renamed from bmaj, bmin, and bpa. Band and "
            "zeropoint arrays, and a band attribute per sub-store, added.",
        ),
    },
}


@dataclass(frozen=True)
class Migration:
    """Describe one upgrade step, from a schema version to the next.

    Parameters
    ----------
    apply
        Rewrite the opened root group in place.
    attrs
        Return the attributes the step changes or adds, as names for the root
        or ``'group/path:name'`` for a subgroup.
    groups
        Return the paths of arrays or groups the step creates whole.
    members
        Return the paths of existing arrays or groups the step rewrites or
        deletes.
    files
        Return the paths of plain files beside the arrays the step rewrites.
    """

    apply: Callable[[zarr.Group], None]
    attrs: Callable[[zarr.Group], list[str]] = field(default=lambda root: [])
    groups: Callable[[zarr.Group], list[str]] = field(default=lambda root: [])
    members: Callable[[zarr.Group], list[str]] = field(default=lambda root: [])
    files: Callable[[zarr.Group], list[str]] = field(default=lambda root: [])


# Version 1 resolution arrays and their version 2 names:
V1_RESOLUTION_NAMES: dict[str, str] = {
    "bmaj": "resolution_major",
    "bmin": "resolution_minor",
    "bpa": "resolution_pa",
}
V2_METADATA_ARRAYS: tuple[str, ...] = (
    *V1_RESOLUTION_NAMES.values(),
    "band",
    "zeropoint",
)


def v1_metadata(metadata: Mapping[str, Any], rows_from: str) -> dict[str, Any]:
    """Present version 1 metadata under version 2 names.

    Parameters
    ----------
    metadata
        Version 1 metadata group.
    rows_from
        Array whose length gives the row count.

    Returns
    -------
    result
        Arrays keyed by version 2 name, with no band and no zeropoint.
    """
    view = {name: metadata[name] for name in metadata.array_keys()}
    for old, new in V1_RESOLUTION_NAMES.items():
        view[new] = view.pop(old)
    n_rows = view[rows_from].shape[0]
    view["band"] = np.full(n_rows, "", dtype=np.dtypes.StringDType())
    view["zeropoint"] = np.full(n_rows, np.nan)
    return view


def rename_resolution_arrays(metadata: zarr.Group) -> None:
    """Rename version 1 resolution arrays to their version 2 names.

    Parameters
    ----------
    metadata
        Metadata group, rewritten in place.
    """
    for old, new in V1_RESOLUTION_NAMES.items():
        array = metadata[old]
        metadata.create_array(new, data=array[:], chunks=array.chunks)
        del metadata[old]


def single_band(bands: list[str], where: str) -> str:
    """Return the band shared by every image, or an empty string.

    Parameters
    ----------
    bands
        Band per image.
    where
        Store or sub-store described in the warning for mixed bands.

    Returns
    -------
    result
        The shared band, ``''`` where there is none or more than one.
    """
    distinct = sorted(set(bands))
    if len(distinct) > 1:
        logger.warning(
            f"{where} holds bands {distinct}, so it is given no single band. Rebuild "
            f"it to split it by band."
        )
        return ""
    return distinct[0] if distinct else ""


def image_bands(survey_root: zarr.Group) -> tuple[list[str], list[float]]:
    """Return each image's band and zeropoint in a survey store of any version.

    Parameters
    ----------
    survey_root
        Root group of the survey store.

    Returns
    -------
    bands
        Band per image.
    zeropoints
        Zeropoint per image.
    """
    from astropy.io import fits

    from survey2zarray.stores.survey_store.fits import resolve_band, resolve_zeropoint

    metadata = survey_root["metadata"]
    if "band" in metadata:
        return (
            [str(b) for b in metadata["band"][:]],
            [float(z) for z in metadata["zeropoint"][:]],
        )
    headers = [fits.Header.fromstring(str(text)) for text in metadata["header"][:]]
    return (
        [resolve_band(header) for header in headers],
        [resolve_zeropoint(header) for header in headers],
    )


def survey_v1_to_v2(root: zarr.Group) -> None:
    """Upgrade a survey store from version 1 to version 2.

    Bands and zeropoints come from the stored image headers. A band recorded only
    in a primary header was not stored, so those images get no band.

    Parameters
    ----------
    root
        Root group of the store.
    """
    metadata = root["metadata"]
    rename_resolution_arrays(metadata)
    bands, zeropoints = image_bands(root)
    chunks = metadata["crval"].chunks[:1]
    metadata.create_array(
        "band", data=np.array(bands, dtype=np.dtypes.StringDType()), chunks=chunks
    )
    metadata.create_array(
        "zeropoint", data=np.array(zeropoints, dtype=np.float64), chunks=chunks
    )
    root.attrs["band"] = single_band(bands, "The survey store")


def dataset_v1_to_v2(root: zarr.Group) -> None:
    """Upgrade a dataset store from version 1 to version 2.

    Each sub-store takes its bands and zeropoints from the survey store it was
    built from. Where that store cannot be opened, they are left empty.

    Parameters
    ----------
    root
        Root group of the store.
    """
    from survey2zarray.stores.dataset_store.dataset_store import PADDING

    paths = dict(
        zip(root.attrs["surveys"], root.attrs["survey_store_paths"], strict=True)
    )
    for survey in root["surveys"].group_keys():
        sub = root["surveys"][survey]
        metadata = sub["metadata"]
        rename_resolution_arrays(metadata)
        n_rows = metadata["crpix"].shape[0]
        bands = [""] * n_rows
        zeropoints = [np.nan] * n_rows
        covered = np.flatnonzero(sub["index_map"][:] != PADDING)
        try:
            source_bands, source_zeropoints = image_bands(
                open_store(paths[survey], mode="r")
            )
        except Exception as error:  # noqa: BLE001 (reported, then left empty)
            logger.warning(
                f"Cannot read survey store {paths.get(survey)} ({error}), so "
                f"{survey} is given no bands or zeropoints."
            )
        else:
            image_index = metadata["image_index"][:]
            for row in covered:
                bands[row] = source_bands[int(image_index[row])]
                zeropoints[row] = source_zeropoints[int(image_index[row])]
        chunks = metadata["crpix"].chunks[:1]
        metadata.create_array(
            "band", data=np.array(bands, dtype=np.dtypes.StringDType()), chunks=chunks
        )
        metadata.create_array(
            "zeropoint", data=np.array(zeropoints, dtype=np.float64), chunks=chunks
        )
        sub.attrs["band"] = single_band(
            [bands[row] for row in covered], f"Sub-store {survey}"
        )


def survey_names(root: zarr.Group) -> list[str]:
    """Return the sub-store names of a dataset store.

    Parameters
    ----------
    root
        Root group of the store.

    Returns
    -------
    result
        Sub-store names.
    """
    return list(root["surveys"].group_keys())


# Upgrade steps, keyed by the version each one upgrades from:
SURVEY_MIGRATIONS: dict[int, Migration] = {
    1: Migration(
        apply=survey_v1_to_v2,
        attrs=lambda root: ["band"],
        groups=lambda root: [f"metadata/{name}" for name in V2_METADATA_ARRAYS],
        members=lambda root: [f"metadata/{name}" for name in V1_RESOLUTION_NAMES],
    ),
}
DATASET_MIGRATIONS: dict[int, Migration] = {
    1: Migration(
        apply=dataset_v1_to_v2,
        attrs=lambda root: [f"surveys/{s}:band" for s in survey_names(root)],
        groups=lambda root: [
            f"surveys/{s}/metadata/{name}"
            for s in survey_names(root)
            for name in V2_METADATA_ARRAYS
        ],
        members=lambda root: [
            f"surveys/{s}/metadata/{name}"
            for s in survey_names(root)
            for name in V1_RESOLUTION_NAMES
        ],
    ),
}

MetadataAdapter = Callable[[Mapping[str, Any]], Mapping[str, Any]]
# Read adapters, keyed by the version each one adapts from:
SURVEY_READ_ADAPTERS: dict[int, MetadataAdapter] = {
    1: lambda metadata: v1_metadata(metadata, rows_from="crval"),
}
DATASET_READ_ADAPTERS: dict[int, MetadataAdapter] = {
    1: lambda metadata: v1_metadata(metadata, rows_from="crpix"),
}


def adapt_metadata(
    metadata: Mapping[str, Any],
    stored: int,
    current: int,
    adapters: Mapping[int, MetadataAdapter],
) -> Mapping[str, Any]:
    """Present an older store's metadata in the current layout.

    Parameters
    ----------
    metadata
        Metadata group of the store, or of one dataset sub-store.
    stored
        Schema version the store was written in.
    current
        Schema version this code reads.
    adapters
        Read adapters for the store type.

    Returns
    -------
    result
        Mapping from current array names to arrays. The group itself where no
        adapter applies.
    """
    for version in range(stored, current):
        if version in adapters:
            metadata = adapters[version](metadata)
    return metadata


def upgrade(
    path: Path,
    current: int,
    migrations: Mapping[int, Migration],
) -> int:
    """Upgrade a store to the current schema version, one step per edit.

    A step that fails is reverted before the error propagates, leaving the store
    at the last version it completed.

    Parameters
    ----------
    path
        Location of the store, already locked by the caller.
    current
        Schema version this code writes for the store type.
    migrations
        Upgrade steps for the store type.

    Returns
    -------
    result
        Schema version of the store afterwards.

    Raises
    ------
    RuntimeError
        Where the store was not completely written, or a step is missing.
    StoreVersionError
        Where the store is newer than this code.
    """
    root = open_store(path, mode="r+")
    if not root.attrs.get("complete", False):
        raise RuntimeError(f"{path} is incomplete. Rebuild it before upgrading.")
    stored = schema_version(root)
    if stored > current:
        raise StoreVersionError(
            f"{path} uses schema version {stored}, which is newer than this "
            f"installation (survey2zarray {package_version()}) writes."
        )
    if stored == current:
        logger.info(f"{path} is already at schema version {current}.")
        return stored

    missing = [
        version for version in range(stored, current) if version not in migrations
    ]
    if missing:
        raise RuntimeError(
            f"No upgrade step is registered from schema version {missing[0]}."
        )
    for version in range(stored, current):
        step = migrations[version]
        prior, added = {}, []
        for key in [*step.attrs(root), "schema_version", "upgraded_with"]:
            group, name = attr_target(root, key)
            if name in group.attrs:
                prior[key] = group.attrs[name]
            else:
                added.append(key)
        begin_edit(
            root,
            [],
            attrs=prior,
            added_attrs=added,
            groups=step.groups(root),
            members=step.members(root),
            files=step.files(root),
        )
        try:
            step.apply(root)
            root.attrs["schema_version"] = version + 1
            root.attrs["upgraded_with"] = package_version()
            commit_edit(root)
        except BaseException:
            recover_edit(root)
            raise
        logger.info(f"Upgraded {path} to schema version {version + 1}.")
    return current


@serialised_store_edit
def upgrade_survey_store(store_path: str | Path) -> int:
    """Upgrade a survey store to the current schema version.

    Parameters
    ----------
    store_path
        Location of the store.

    Returns
    -------
    result
        Schema version of the store afterwards.
    """
    return upgrade(Path(store_path), SURVEY_SCHEMA_VERSION, SURVEY_MIGRATIONS)


@serialised_store_edit
def upgrade_dataset_store(store_path: str | Path) -> int:
    """Upgrade a dataset store to the current schema version.

    Parameters
    ----------
    store_path
        Location of the store.

    Returns
    -------
    result
        Schema version of the store afterwards.
    """
    return upgrade(Path(store_path), DATASET_SCHEMA_VERSION, DATASET_MIGRATIONS)
