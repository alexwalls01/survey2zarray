"""Append tiles to a survey store and retry unreadable inputs."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr
from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord

from survey2zarray.stores.store import open_store
from survey2zarray.stores.survey_store.build import (
    STRING_METADATA_KEYS,
    build_header_index,
    extend_reference_map,
    resolve_fits_list,
    write_image_data_to_zarr_array,
)
from survey2zarray.stores.survey_store.fits import (
    IncompatibleTilesError,
    check_compatibility,
)
from survey2zarray.stores.survey_store.references import REFERENCE_MAP_PATH
from survey2zarray.stores.transaction import (
    begin_edit,
    commit_edit,
    serialised_store_edit,
)
from survey2zarray.utils.logging import get_logger

logger = get_logger(__name__)


def find_duplicate_tiles(
    image_catalogue: list[dict[str, Any]],
    root: zarr.Group,
    tolerance_px: float = 0.1,
) -> list[bool]:
    """Flag candidate images whose footprint is already in the store.

    A duplicate is a practically identical footprint, meaning the same centre
    within the tolerance and the same shape. Overlap alone is not enough, since survey
    images are meant to overlap at their edges. Locations are deliberately not
    compared, being fragile across re-downloads, moved archives, and changing
    URLs.

    Parameters
    ----------
    image_catalogue
        Metadata dicts for the candidate images.
    root
        Root group of the store to compare against.
    tolerance_px
        Centre agreement required, as a fraction of a pixel.

    Returns
    -------
    result
        True where the candidate duplicates a stored image.
    """
    metadata = root["metadata"]
    centres = metadata["pixel_centre"][:]
    shapes = metadata["pixel_shape"][:]
    scales = metadata["pixel_scale"][:]
    if len(centres) == 0:
        return [False] * len(image_catalogue)
    existing = SkyCoord(ra=centres[:, 0] * u.deg, dec=centres[:, 1] * u.deg, frame=ICRS)
    tolerance = tolerance_px * np.max(scales, axis=1)
    duplicates = []
    for m in image_catalogue:
        centre = SkyCoord(
            ra=m["pixel_centre"][0] * u.deg,
            dec=m["pixel_centre"][1] * u.deg,
            frame=ICRS,
        )
        separation = centre.separation(existing).arcsec
        same_centre = separation < tolerance
        same_shape = np.all(shapes == np.asarray(m["pixel_shape"]), axis=1)
        duplicates.append(bool(np.any(same_centre & same_shape)))
    return duplicates


def append_metadata_arrays(
    root: zarr.Group,
    image_catalogue: list[dict[str, Any]],
    start: int,
) -> None:
    """Extend every metadata array with rows for new images.

    Iterate stored array names so missing extractor fields raise immediately.

    Parameters
    ----------
    root
        Root group of the store.
    image_catalogue
        Metadata dicts for the new images.
    start
        Row index at which the new images begin.
    """
    meta = root["metadata"]
    n_new = len(image_catalogue)
    for key in meta.array_keys():
        values = [m[key] for m in image_catalogue]
        array = meta[key]
        array.resize((start + n_new, *array.shape[1:]))
        if key in STRING_METADATA_KEYS:
            array[start:] = np.array(values, dtype=np.dtypes.StringDType())
        else:
            array[start:] = np.array(values, dtype=array.dtype)


@serialised_store_edit
def append_tiles(
    store_path: str | Path,
    source: str | Path | list[str],
    tolerance_px: float = 0.1,
    max_workers: int | None = None,
    storage_options: dict[str, dict] | None = None,
) -> zarr.Group:
    """Add images to an existing survey store.

    Warn and skip duplicate images. Grow the image axis and spatial dimensions
    as required, leaving added spatial regions at the fill value.

    Parameters
    ----------
    store_path
        Location of the store.
    source
        New images, specified as for the store's own access mode.
    tolerance_px
        Centre agreement for duplicate rejection, as a fraction of a pixel.
    max_workers
        Headers to read at once, as in `build_survey_store`.
    storage_options
        Filesystem options keyed by protocol, as in `build_survey_store`.
        Defaults to what the store recorded when it was built, so an append
        reuses the archive configuration.

    Returns
    -------
    result
        Root group of the store.

    Raises
    ------
    RuntimeError
        Where the store was not completely written.
    IncompatibleTilesError
        Where the new images disagree with each other or with the store.
    """
    store_path = Path(store_path)
    root = open_store(store_path, mode="r+")
    if not root.attrs.get("complete", False):
        raise RuntimeError(
            f"Conversion of {store_path} is incomplete. Reconvert it before use."
        )
    observation_type = root.attrs["observation_type"]
    access_mode = root.attrs["access_mode"]
    n_chan = root.attrs.get("n_chan")

    # Caller-supplied storage options override persisted archive options.
    if storage_options is None:
        storage_options = dict(root.attrs.get("storage_options", {})) or None
    fits_paths = resolve_fits_list(source, storage_options)
    image_catalogue, image_valid_df, metadata_valid_df = build_header_index(
        fits_paths,
        require_referenceable=(access_mode == "reference"),
        max_workers=max_workers,
        storage_options=storage_options,
    )
    new_type, new_n_chan = check_compatibility(image_catalogue)
    if new_type != observation_type:
        raise IncompatibleTilesError(
            f"Appended images are {new_type}. Store {store_path} is {observation_type}."
        )
    if observation_type == "spectral" and new_n_chan != n_chan:
        raise IncompatibleTilesError(
            f"Appended images have n_chan={new_n_chan}. Store {store_path} has "
            f"n_chan={n_chan}."
        )

    duplicates = find_duplicate_tiles(image_catalogue, root, tolerance_px=tolerance_px)
    skipped_as_duplicate = set()
    for m, dup in zip(image_catalogue, duplicates, strict=True):
        if dup:
            logger.warning(
                f"Skipping duplicate image {m['image_location']}. Its footprint "
                f"is already present."
            )
            skipped_as_duplicate.add(str(m["image_location"]))
    image_catalogue = [
        m for m, dup in zip(image_catalogue, duplicates, strict=True) if not dup
    ]
    if len(image_catalogue) == 0:
        logger.info("No new images to append.")
        return root

    n_old = int(root.attrs["n_images"])
    n_new = len(image_catalogue)

    array_paths = [f"metadata/{key}" for key in root["metadata"].array_keys()]
    files = []
    if access_mode == "copy":
        array_paths = ["data"] + array_paths
    else:
        # Include the byte map in the transaction to keep metadata addresses valid.
        files = [REFERENCE_MAP_PATH]
    begin_edit(root, array_paths, attrs={"n_images": n_old}, files=files)

    # Writing data first makes an interrupted image safe to append again.
    if access_mode == "copy":
        data = root["data"]
        new_ny = max(int(m["pixel_shape"][0]) for m in image_catalogue)
        new_nx = max(int(m["pixel_shape"][1]) for m in image_catalogue)
        ny = max(data.shape[-2], new_ny)
        nx = max(data.shape[-1], new_nx)
        data.resize((n_old + n_new, *data.shape[1:-2], ny, nx))
        write_image_data_to_zarr_array(
            _OffsetArray(data, n_old), image_catalogue, storage_options
        )
    else:
        # Extend the byte map with image rows to preserve valid pixel addresses.
        extend_reference_map(store_path, image_catalogue)
    append_metadata_arrays(root, image_catalogue, n_old)

    root.attrs["n_images"] = n_old + n_new
    commit_edit(root)

    # Renumber array_index from the rows assigned during this append.
    image_valid_df = image_valid_df.copy()
    written = {
        str(m["image_location"]): n_old + i for i, m in enumerate(image_catalogue)
    }
    image_valid_df["array_index"] = [
        written.get(str(location)) for location in image_valid_df["image_location"]
    ]
    is_duplicate = (
        image_valid_df["image_location"].astype(str).isin(skipped_as_duplicate)
    )
    image_valid_df.loc[is_duplicate, "converted"] = False
    image_valid_df.loc[is_duplicate, "error"] = (
        "Duplicate footprint, already in the store"
    )
    for name, df in [
        ("image_valid.csv", image_valid_df),
        ("metadata_valid.csv", metadata_valid_df),
    ]:
        csv_path = store_path / name
        if csv_path.exists():
            df = pd.concat([pd.read_csv(csv_path), df], ignore_index=True)
        df.to_csv(csv_path, index=False)

    logger.info(f"Appended {n_new} images to {store_path}.")
    return root


class _OffsetArray:
    """Shift first-axis writes by a fixed offset.

    Present appended rows as a zero-based destination to the shared writer.

    Parameters
    ----------
    array
        Destination array.
    offset
        Added to the first index of every write.
    """

    def __init__(self, array: zarr.Array, offset: int) -> None:
        self.array = array
        self.offset = offset

    def __setitem__(self, key: tuple, value: np.ndarray) -> None:
        index, *rest = key
        self.array[(index + self.offset, *rest)] = value


def retry_failed_tiles(
    store_path: str | Path,
    max_workers: int | None = None,
    storage_options: dict[str, dict] | None = None,
) -> zarr.Group:
    """Read again the files that failed when a survey store was built.

    Re-read locations recorded as failures in `image_valid.csv` and append
    newly successful images.

    Files that fail again keep their place in the record, so this can be run
    more than once and only ever narrows the gap.

    Parameters
    ----------
    store_path
        Location of the store.
    max_workers
        Headers to read at once, as in `build_survey_store`.
    storage_options
        Filesystem options keyed by protocol, as in `build_survey_store`.
        Defaults to what the store recorded; a credential was not recorded, so
        a private archive needs it given here.

    Returns
    -------
    result
        Root group of the store.

    Raises
    ------
    FileNotFoundError
        Where the store holds no record of which files were used.
    """
    store_path = Path(store_path)
    record_path = store_path / "image_valid.csv"
    if not record_path.exists():
        raise FileNotFoundError(
            f"{record_path} does not exist, so there is no record of what failed."
        )

    record = pd.read_csv(record_path)
    unread = record.loc[~record["converted"].astype(bool), "image_location"].tolist()
    if not unread:
        logger.info(f"Nothing to retry. Every file of {store_path.name} was read.")
        return open_store(store_path, mode="r+")

    logger.info(
        f"Retrying {len(unread)} file(s) that {store_path.name} could not read."
    )
    try:
        root = append_tiles(
            store_path, unread, max_workers=max_workers, storage_options=storage_options
        )
    except RuntimeError:
        logger.warning(f"None of the {len(unread)} file(s) could be read this time.")
        return open_store(store_path, mode="r+")

    # Retain failure records only for files that remain unreadable.
    record = pd.read_csv(record_path)
    read_now = set(record.loc[record["converted"].astype(bool), "image_location"])
    stale = record["image_location"].isin(read_now & set(unread)) & ~record[
        "converted"
    ].astype(bool)
    record[~stale].to_csv(record_path, index=False)

    recovered = len(read_now & set(unread))
    logger.info(
        f"Recovered {recovered} of {len(unread)} file(s), leaving "
        f"{len(unread) - recovered} unread."
    )
    return root
