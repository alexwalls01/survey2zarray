"""Append tiles to a survey store and retry unreadable inputs."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr
from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord

from survey2zarray.stores.store import (
    SURVEY_SCHEMA_VERSION,
    SURVEY_UPGRADE_COMMAND,
    check_editable,
    open_store,
)
from survey2zarray.stores.survey_store.build import (
    STRING_METADATA_KEYS,
    apply_band_pattern,
    apply_resolution,
    build_header_index,
    compile_band_pattern,
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
    skip_other_bands: bool = False,
) -> zarr.Group:
    """Add images to an existing survey store.

    Warn and skip duplicate images. Grow the image axis and spatial dimensions
    as required, leaving added spatial regions at the fill value. The store's
    ``band_pattern`` and ``resolution`` apply to the new images as they did at
    build.

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
    skip_other_bands
        Whether to record images of another band as unwritten and append the
        rest. Otherwise any such image refuses the whole append.

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
    check_editable(root, SURVEY_SCHEMA_VERSION, store_path, SURVEY_UPGRADE_COMMAND)
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
    band_pattern = compile_band_pattern(root.attrs.get("band_pattern"))
    if band_pattern is not None:
        apply_band_pattern(image_catalogue, band_pattern)
    if root.attrs.get("resolution") is not None:
        apply_resolution(image_catalogue, float(root.attrs["resolution"]))

    store_band = root.attrs.get("band", "")
    other_bands = {
        str(m["image_location"]): m["band"]
        for m in image_catalogue
        if m["band"] != store_band
    }
    if other_bands and not skip_other_bands:
        raise IncompatibleTilesError(
            other_bands_message(store_path, store_band, other_bands)
        )
    image_catalogue = [
        m for m in image_catalogue if str(m["image_location"]) not in other_bands
    ]
    if not image_catalogue:
        logger.info("No images of this store's band to append.")
        write_append_records(
            store_path, image_valid_df, metadata_valid_df, {}, {}, other_bands
        )
        return root

    new_type, new_n_chan, _ = check_compatibility(image_catalogue)
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

    written = {
        str(m["image_location"]): n_old + i for i, m in enumerate(image_catalogue)
    }
    write_append_records(
        store_path,
        image_valid_df,
        metadata_valid_df,
        written,
        skipped_as_duplicate,
        other_bands,
    )
    logger.info(f"Appended {n_new} images to {store_path}.")
    return root


def other_bands_message(
    store_path: Path, store_band: str, other_bands: dict[str, str]
) -> str:
    """Describe images whose band differs from a store's.

    Parameters
    ----------
    store_path
        Location of the store.
    store_band
        Band of the store, ``''`` where it has none.
    other_bands
        Band of each refused image, keyed by location.

    Returns
    -------
    result
        Message naming the images and, where the store holds one band, the
        sibling store each band belongs in.
    """
    by_band: dict[str, list[str]] = {}
    for location, band in other_bands.items():
        by_band.setdefault(band, []).append(location)
    lines = [f"Store {store_path} holds band {store_band or 'none'}."]
    for band, locations in sorted(by_band.items()):
        if store_band and band:
            sibling = sibling_band_path(store_path, store_band, band)
            lines.append(
                f"Band {band} applies to {locations} and belongs in {sibling}."
            )
        else:
            lines.append(f"Band {band or 'none'} applies to {locations}.")
    return " ".join(lines)


def sibling_band_path(store_path: Path, store_band: str, band: str) -> Path:
    """Return the store a split build would have written another band to.

    Parameters
    ----------
    store_path
        Location of a store holding one band.
    store_band
        Band of that store.
    band
        Another band.

    Returns
    -------
    result
        ``store_path`` with its ``_<store_band>`` name ending replaced, or with
        ``_<band>`` added where its name has no such ending.
    """
    suffix = ".zarr" if store_path.suffix == ".zarr" else ""
    stem = store_path.name.removesuffix(suffix)
    ending = f"_{store_band}"
    if stem.endswith(ending):
        stem = stem.removesuffix(ending)
    return store_path.with_name(f"{stem}_{band}{suffix}")


def write_append_records(
    store_path: Path,
    image_valid_df: pd.DataFrame,
    metadata_valid_df: pd.DataFrame,
    written: dict[str, int],
    duplicates: set[str],
    other_bands: dict[str, str],
) -> None:
    """Add an append's provenance records to the store's CSV files.

    Parameters
    ----------
    store_path
        Location of the store.
    image_valid_df, metadata_valid_df
        Records from the append's header pass.
    written
        Row each appended image was written to, keyed by location.
    duplicates
        Locations skipped as duplicates of stored footprints.
    other_bands
        Band of each image skipped for belonging to another store, keyed by
        location.
    """
    image_valid_df = image_valid_df.copy()
    locations = image_valid_df["image_location"].astype(str)
    image_valid_df["array_index"] = [written.get(location) for location in locations]
    is_duplicate = locations.isin(duplicates)
    image_valid_df.loc[is_duplicate, "converted"] = False
    image_valid_df.loc[is_duplicate, "error"] = (
        "Duplicate footprint, already in the store"
    )
    for location, band in other_bands.items():
        refused = locations == location
        image_valid_df.loc[refused, "converted"] = False
        image_valid_df.loc[refused, "error"] = (
            f"Band {band or 'none'}, not this store's"
        )
    metadata_valid_df = metadata_valid_df[
        ~metadata_valid_df["image_location"].astype(str).isin(other_bands)
    ]
    for name, df in [
        ("image_valid.csv", image_valid_df),
        ("metadata_valid.csv", metadata_valid_df),
    ]:
        csv_path = store_path / name
        if csv_path.exists():
            df = pd.concat([pd.read_csv(csv_path), df], ignore_index=True)
        df.to_csv(csv_path, index=False)


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
    check_editable(
        open_store(store_path, mode="r"),
        SURVEY_SCHEMA_VERSION,
        store_path,
        SURVEY_UPGRADE_COMMAND,
    )
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
    n_before = len(record)
    try:
        root = append_tiles(
            store_path,
            unread,
            max_workers=max_workers,
            storage_options=storage_options,
            skip_other_bands=True,
        )
    except RuntimeError:
        logger.warning(f"None of the {len(unread)} file(s) could be read this time.")
        return open_store(store_path, mode="r+")

    # Keep only the latest record of each file this retry recorded:
    record = pd.read_csv(record_path)
    latest = record.iloc[n_before:]
    read_now = set(latest.loc[latest["converted"].astype(bool), "image_location"])
    superseded = (record.index < n_before) & record["image_location"].isin(
        set(latest["image_location"]) & set(unread)
    )
    record[~superseded].to_csv(record_path, index=False)

    recovered = len(read_now & set(unread))
    logger.info(
        f"Recovered {recovered} of {len(unread)} file(s), leaving "
        f"{len(unread) - recovered} unread."
    )
    return root
