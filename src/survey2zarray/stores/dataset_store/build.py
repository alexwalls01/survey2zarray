"""Build a dataset store, matching a catalogue to images and extracting cutouts."""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
import zarr
from astropy.coordinates import ICRS, SkyCoord
from astropy.table import Table
from tqdm import tqdm
from zarr.codecs import BloscCodec

from survey2zarray.stores.dataset_store.buckets import (
    BucketingRule,
    bucket_order,
    build_rule,
    distinct_strata,
    write_rule,
)
from survey2zarray.stores.dataset_store.coverage import (
    match_coords_to_images,
)
from survey2zarray.stores.dataset_store.dataset_store import PADDING
from survey2zarray.stores.dataset_store.duplicates import (
    find_catalogue_duplicates,
    matched_beams,
)
from survey2zarray.stores.dataset_store.geometry import (
    blank_outside_disc,
    disc_diameter,
    edge_padding_for,
    square_size,
    valid_fraction,
)
from survey2zarray.stores.dataset_store.layout import (
    is_power_of_two,
    shard_rows,
)
from survey2zarray.stores.store import consolidate, create_store, open_store
from survey2zarray.stores.survey_store.build import REMOTE_MINIMUM_WORKERS
from survey2zarray.stores.survey_store.fits import quiet_fits_warnings
from survey2zarray.stores.survey_store.references import is_remote
from survey2zarray.stores.survey_store.survey_store import SurveyStore
from survey2zarray.utils.logging import get_logger, warn_many
from survey2zarray.utils.parallel import polite_workers

logger = get_logger(__name__)

# Preserve source-image metadata independently from cutout geometry.
CARRIED_METADATA_KEYS: list[str] = [
    "crval",
    "cd",
    "ctype",
    "pixel_scale",
    "telescope",
    "frequency",
    "bandwidth",
    "date",
    "bmaj",
    "bmin",
    "bpa",
    "bunit",
    "equinox",
    "radesys",
]
STRING_METADATA_KEYS: list[str] = ["ctype", "telescope", "bunit", "radesys"]

# Match HEADER_WORKERS to cap archive connections across both build passes.
EXTRACTION_WORKERS: int = 8
EXTRACTION_BUDGET: int = 4

# Bound unwritten extraction work while amortising array resizing.
CUTOUT_BATCH: int = 64

# Bridging one inner chunk trades 25% more bytes for 40% fewer reads at the
# representative 128 by 128 cutout and 32-row shard size.
COALESCE_CHUNKS: int = 1


def report_bucketing(rule: BucketingRule, df: pd.DataFrame) -> None:
    """Report the bucketing rule and split granularity.

    Parameters
    ----------
    rule
        Rule the sources were dealt by.
    df
        Normalised catalogue, in the order the store holds it.
    """
    strata = distinct_strata(df, rule.discrete)
    logger.info(
        f"{len(df)} source(s) across {strata} stratum/strata on "
        f"{list(rule.columns) or 'no column'}, {rule.n_buckets} bucket(s), "
        f"split fractions accurate to within {100 / rule.n_buckets:.1f}%."
    )


def coalesce_gap_bytes(surveys_group: zarr.Group, surveys: Sequence[str]) -> int:
    """Derive how far a read may run through a shard between the cutouts it wants.

    Taken over the smallest inner chunk of any survey, since the setting is
    zarr's and one number covers every array a run reads. A gap derived from
    the largest would be several chunks of the survey with the smallest, which
    is the merging this exists to bound.

    Parameters
    ----------
    surveys_group
        Group holding the per-survey sub-stores.
    surveys
        Sub-stores to derive over.

    Returns
    -------
    result
        Bytes, recorded on the store and applied by `apply_coalesce_gap` when
        it is opened.
    """
    inner = [
        int(np.prod(surveys_group[survey]["data"].chunks))
        * np.dtype(surveys_group[survey]["data"].dtype).itemsize
        for survey in surveys
    ]
    return COALESCE_CHUNKS * min(inner, default=0)


def resumable_build(out_path: str | Path, recipe: dict[str, Any]) -> bool:
    """Report whether an unfinished build matches a requested recipe.

    Require an incomplete, readable store with identical build attributes.

    Parameters
    ----------
    out_path
        Where the store would be written.
    recipe
        The attributes this build would set that a resumed one must match.

    Returns
    -------
    result
        Whether the build can resume.
    """
    out_path = Path(out_path)
    if not (out_path / "zarr.json").exists():
        return False
    try:
        root = open_store(out_path, mode="r")
        if root.attrs.get("complete", False):
            logger.info(f"{out_path} is already complete, so it is built again.")
            return False
        for key, wanted in recipe.items():
            if root.attrs.get(key) != wanted:
                logger.warning(
                    f"Cannot resume {out_path}. Stored {key}="
                    f"{root.attrs.get(key)!r}, requested {wanted!r}. Starting again."
                )
                return False
        if "sources" not in root:
            return False
    except Exception as error:  # noqa: BLE001 (reported, then rebuilt)
        logger.warning(
            f"Cannot read {out_path} to resume it ({error}), so it starts again."
        )
        return False
    return True


def open_catalogue_file(catalogue_path: str | Path) -> pd.DataFrame:
    """Read a source catalogue into a data frame.

    Parameters
    ----------
    catalogue_path
        Catalogue file, in FITS, whitespace-delimited text, or CSV format,
        recognised by suffix.

    Returns
    -------
    result
        Catalogue rows.

    Raises
    ------
    ValueError
        Where the path is not a file, or its format is unrecognised.
    """
    path = Path(catalogue_path)
    if path.is_file():
        if path.suffix == ".fits":
            table = Table.read(path, memmap=True, format="fits")
        elif path.suffix == ".txt":
            table = Table.read(path, format="ascii.commented_header")
        elif path.suffix == ".csv":
            table = Table.read(path, format="ascii.csv")
        else:
            raise ValueError("Unsupported catalogue format.")
    else:
        raise ValueError(f"Expected a catalogue file, got {catalogue_path}.")
    return table.to_pandas()


def normalise_catalogue(
    df: pd.DataFrame,
    catalogue_path: str | Path,
) -> pd.DataFrame:
    """Standardise catalogue columns and drop repeated identifiers.

    Column names are matched case-insensitively.

    Parameters
    ----------
    df
        Catalogue as read from file.
    catalogue_path
        Source of the catalogue, used in messages.

    Returns
    -------
    result
        Reindexed, with ``id``, ``ra``, and ``dec`` columns present under those
        names. Other columns are left alone.

    Raises
    ------
    ValueError
        Where the catalogue is empty, or lacks an identifier or coordinates.
    """
    if df.shape[0] == 0:
        raise ValueError(f"No rows found in catalogue {catalogue_path}.")
    cols = {c.lower(): c for c in df.columns}
    if "id" not in cols:
        raise ValueError(f"Missing ID column in catalogue {catalogue_path}.")
    if "ra" not in cols or "dec" not in cols:
        raise ValueError(f"Missing coordinates in catalogue {catalogue_path}.")
    rename = {cols["id"]: "id", cols["ra"]: "ra", cols["dec"]: "dec"}
    df = df.rename(columns=rename)
    n_duplicates = df["id"].duplicated().sum()
    if n_duplicates > 0:
        logger.warning(
            f"{n_duplicates} duplicate source ID(s) found in the catalogue, so "
            f"only the first instance of each source is kept."
        )
        df = df.drop_duplicates(subset="id", keep="first")
    return df.reset_index(drop=True)


def get_cutout(
    store: SurveyStore,
    image_index: int,
    coords: tuple[float, float],
    diameter: float,
) -> tuple[np.ndarray, int, int, float]:
    """Extract the disc of sky of a given diameter around a source.

    Read pixels through the store's active access mode. Enforce the coverage
    boundary invariant at runtime. Centre the disc on the source's true pixel
    position so each survey blanks the same sky region.

    Parameters
    ----------
    store
        Survey holding the image.
    image_index
        Index of the image.
    coords
        Source position as ``(ra, dec)`` in degrees.
    diameter
        Diameter of the disc, in arcsec.

    Returns
    -------
    result
        Cutout, integer origin, and valid-pixel fraction. Continuum cutouts have
        shape ``(H, W)`` and spectral cutouts have shape ``(n_chan, H, W)``.
    """
    image_index = int(image_index)
    sky = SkyCoord(ra=coords[0] * u.deg, dec=coords[1] * u.deg, frame=ICRS)

    px, py = store.pixel_scale[image_index]
    nx = max(1, int(np.ceil(diameter / px)))
    ny = max(1, int(np.ceil(diameter / py)))

    wcs = store.get_wcs(image_index).celestial
    cx, cy = wcs.world_to_pixel(sky)
    x0 = int(round(cx - nx / 2))
    y0 = int(round(cy - ny / 2))
    x1, y1 = x0 + nx, y0 + ny

    h, w = store.pixel_shape[image_index]
    if not (0 <= x0 and x1 <= w and 0 <= y0 and y1 <= h):
        raise RuntimeError(
            f"Cutout exceeds image {image_index} for source "
            f"({coords[0]}, {coords[1]}). Requested x=[{x0}, {x1}) and "
            f"y=[{y0}, {y1}) within shape ({h}, {w}). Coverage matching "
            f"must exclude this source."
        )
    cutout = store.get_cutout_region(image_index, x0, x1, y0, y1)
    centre = (float(cx) - x0, float(cy) - y0)
    cutout = blank_outside_disc(cutout, (px, py), diameter, centre=centre)
    valid = valid_fraction(cutout, (px, py), diameter, centre=centre)

    return cutout, x0, y0, valid


def shift_crpix(crpix: Sequence[float], x0: int, y0: int) -> tuple[float, ...]:
    """Move a reference pixel from image coordinates into cutout coordinates.

    Only the two celestial axes shift, so any further axis such as frequency
    keeps its reference pixel and the sky mapping is otherwise unchanged.

    Parameters
    ----------
    crpix
        Image reference pixel.
    x0, y0
        Origin of the cutout within the image, as (column, row).

    Returns
    -------
    result
        Shifted reference pixel.
    """
    shifted = list(map(float, crpix))
    shifted[0] -= x0
    shifted[1] -= y0
    return tuple(shifted)


def write_sources_group(root: zarr.Group, df: pd.DataFrame) -> zarr.Group:
    """Write the source table that indexes the whole dataset.

    Row order defines the source index used everywhere else, and is
    independent of survey coverage. Catalogue columns beyond the standard three
    pass through unchanged, integers as integers, other numbers as floats, and
    the rest as strings.

    Parameters
    ----------
    root
        Root group of the store.
    df
        Normalised catalogue.

    Returns
    -------
    result
        The sources group.
    """
    compressors = [BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")]
    sources = root.require_group("sources")
    n = len(df)
    sources.create_array(
        "source_id",
        data=np.array([str(v) for v in df["id"]], dtype=np.dtypes.StringDType()),
        chunks=(n,),
    )
    sources.create_array("ra", data=df["ra"].to_numpy(dtype=np.float64), chunks=(n,))
    sources.create_array("dec", data=df["dec"].to_numpy(dtype=np.float64), chunks=(n,))
    for column in df.columns:
        if column in ("id", "ra", "dec"):
            continue
        values = df[column]
        if pd.api.types.is_numeric_dtype(values):
            # Preserve integer columns for direct use as indices.
            dtype = np.int64 if pd.api.types.is_integer_dtype(values) else np.float64
            sources.create_array(
                column,
                data=values.to_numpy(dtype=dtype),
                chunks=(n,),
                compressors=compressors,
            )
        else:
            sources.create_array(
                column,
                # An empty string distinguishes absent values from a "nan" category.
                data=np.array(
                    ["" if pd.isna(v) else str(v) for v in values],
                    dtype=np.dtypes.StringDType(),
                ),
                chunks=(n,),
            )
    return sources


def extraction_plan(
    store: SurveyStore,
    image_index: int,
    n_requests: int,
) -> tuple[int, int]:
    """Allocate the extraction connection budget across workers and ranges.

    Use concurrent workers for remote reference reads. Divide
    `EXTRACTION_BUDGET` between worker-level and range-level concurrency.

    Parameters
    ----------
    store
        Survey to extract from.
    image_index
        Any image the extraction will read, for resolving pixel storage.
    n_requests
        Cutouts to extract.

    Returns
    -------
    workers
        Cutouts to extract at once.
    concurrency
        Ranges in flight within each of them.
    """
    remote = store.access_mode == "reference" and is_remote(
        str(store.image_record(image_index)["location"])
    )
    if not remote:
        return 1, store.concurrency

    workers = polite_workers(
        min(EXTRACTION_WORKERS, EXTRACTION_BUDGET),
        ceiling=n_requests,
        minimum=REMOTE_MINIMUM_WORKERS,
    )
    return workers, max(1, EXTRACTION_BUDGET // workers)


def extract_cutouts(
    store: SurveyStore,
    requests: Sequence[tuple[int, tuple[float, float]]],
    diameter: float,
    description: str,
    progress: tqdm | None = None,
) -> list[tuple[np.ndarray, int, int, float]]:
    """Extract source discs under a shared concurrency budget.

    Overlap remote reference reads and keep copied local reads serial. Divide
    connections across simultaneous cutouts and byte ranges.

    Parameters
    ----------
    store
        Survey to extract from.
    requests
        Per cutout, the image index and the source's ``(ra, dec)`` in degrees.
    diameter
        Diameter of the disc, in arcsec.
    description
        Progress bar label, used only where no bar is supplied.
    progress
        Progress bar to advance once per cutout.

    Returns
    -------
    result
        One ``(cutout, x0, y0, valid)`` per request, in the order given.
    """
    if not requests:
        return []
    workers, concurrency = extraction_plan(store, requests[0][0], len(requests))

    previous = store.concurrency
    store.concurrency = concurrency
    owned = progress is None
    if owned:
        logger.info(
            f"Extracting with {workers} worker(s), with {concurrency} range(s) "
            f"in flight each."
        )
        progress = tqdm(total=len(requests), desc=description)
    try:
        if workers == 1:
            results = []
            for index, coords in requests:
                results.append(get_cutout(store, index, coords, diameter))
                progress.update(1)
            return results
        # Position-indexed results preserve input order while progress follows
        # completion.
        results = [None] * len(requests)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(get_cutout, store, index, coords, diameter): position
                for position, (index, coords) in enumerate(requests)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                progress.update(1)
        return results
    finally:
        if owned:
            progress.close()
        store.concurrency = previous


def create_empty_substore(
    surveys_group: zarr.Group,
    survey: str,
    store: SurveyStore,
    diameter: float,
    n_sources: int,
    n_covered: int,
    shard_size: int | None = None,
) -> zarr.Group:
    """Create an aligned sub-store with one row per source.

    Keep source index ``i`` at row ``i`` across surveys and leave uncovered
    rows at the fill value. Size spatial dimensions from the survey's finest
    pixel scale because Zarr fixes chunk shape at array creation. Store one
    cutout per inner chunk and multiple rows per shard.

    Parameters
    ----------
    surveys_group
        Group holding the per-survey sub-stores.
    survey
        Name for this sub-store.
    store
        Survey the cutouts will come from, which fixes the metadata shapes.
    diameter
        Diameter of the disc each cutout holds, in arcsec.
    n_sources
        Rows every array is given, so that row ``i`` holds source ``i``.
    n_covered
        Sources this survey covers, which is what its shard is sized by.
    shard_size
        Rows held in one shard of the pixel array. Defaults to what
        `shard_rows` derives from this survey's cutout size and coverage.

    Returns
    -------
    result
        Created survey sub-store.

    Raises
    ------
    ValueError
        Where ``shard_size`` is not a power of two.
    """
    if shard_size is not None and not is_power_of_two(shard_size):
        raise ValueError(
            f"shard_size must be a power-of-two row count, got {shard_size}. "
            f"Other values break survey alignment."
        )
    n_sources = int(n_sources)
    compressors = [BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")]
    sub = surveys_group.require_group(survey)
    sub.attrs["observation_type"] = store.observation_type
    if store.n_chan is not None:
        sub.attrs["n_chan"] = int(store.n_chan)
    sub.attrs["n_cutouts"] = 0

    finest = np.min(store.pixel_scale, axis=0)
    max_w = max(1, int(np.ceil(float(diameter) / float(finest[0]))))
    max_h = max(1, int(np.ceil(float(diameter) / float(finest[1]))))
    if store.observation_type == "spectral":
        cutout = (store.n_chan, max_h, max_w)
    else:
        cutout = (max_h, max_w)
    cutout_bytes = int(np.prod(cutout)) * np.dtype(np.float32).itemsize
    coverage = int(n_covered) / max(1, n_sources)
    if shard_size is None:
        shard_size = shard_rows(cutout_bytes, coverage)
    shard_size = int(shard_size)
    logger.info(
        f"{survey} uses shards of {shard_size} row(s) at "
        f"{100 * coverage:.0f}% coverage, "
        f"{shard_size * coverage * cutout_bytes / 1e6:.1f} MB served, "
        f"one cutout to a chunk."
    )
    sub.create_array(
        "data",
        shape=(n_sources, *cutout),
        chunks=(1, *cutout),
        shards=(shard_size, *cutout),
        dtype=np.float32,
        fill_value=np.nan,
        compressors=compressors,
    )
    # Fill values mark unwritten rows as uncovered sources.
    sub.create_array(
        "index_map",
        shape=(n_sources,),
        chunks=(CUTOUT_BATCH,),
        dtype=np.int64,
        fill_value=PADDING,
    )

    meta = sub.require_group("metadata")
    naxes = store.crval.shape[1]
    meta.create_array(
        "crpix",
        shape=(n_sources, naxes),
        chunks=(CUTOUT_BATCH, naxes),
        dtype=np.float64,
    )
    meta.create_array(
        "pixel_shape", shape=(n_sources, 2), chunks=(CUTOUT_BATCH, 2), dtype=np.int32
    )
    meta.create_array(
        "image_index", shape=(n_sources,), chunks=(CUTOUT_BATCH,), dtype=np.int64
    )
    for key in CARRIED_METADATA_KEYS:
        if key in STRING_METADATA_KEYS:
            meta.create_array(
                key,
                shape=(n_sources,),
                chunks=(CUTOUT_BATCH,),
                dtype=str,
            )
        else:
            trailing = tuple(getattr(store, key).shape[1:])
            meta.create_array(
                key,
                shape=(n_sources, *trailing),
                chunks=(CUTOUT_BATCH, *trailing),
                dtype=np.float64,
            )
    return sub


def write_survey_substore(
    surveys_group: zarr.Group,
    survey: str,
    store: SurveyStore,
    df: pd.DataFrame,
    source_image_map: dict[Any, Any],
    diameter: float,
    resume: bool = False,
    shard_size: int | None = None,
) -> zarr.Group:
    """Extract and store every cutout one survey can supply.

    Keep row ``i`` aligned with source ``i`` and leave uncovered rows at their
    fill values. Pad varying cutout shapes and store each true shape. Write the
    index map after each pixel batch so it marks only complete rows.

    Parameters
    ----------
    surveys_group
        Group holding the per-survey sub-stores.
    survey
        Name for this sub-store.
    store
        Survey to extract from.
    df
        Normalised catalogue, in source index order.
    source_image_map
        Source identifier to matched image index, NaN where absent.
    diameter
        Diameter of the disc to extract, in arcsec.
    resume
        Whether to keep what an earlier run already stored and extract only
        the rest.
    shard_size
        Rows held in one shard of the pixel array. Defaults to what
        `shard_rows` derives from this survey's cutout size and coverage, and
        is ignored when resuming, since a shard shape is fixed when the array
        is made.

    Returns
    -------
    result
        The sub-store group.
    """
    present = []
    for source_index, source_id in enumerate(df["id"]):
        image_index = source_image_map[source_id]
        if not pd.isna(image_index):
            present.append((source_index, int(image_index)))

    if resume and survey in surveys_group and "index_map" in surveys_group[survey]:
        sub = surveys_group[survey]
        done = set(np.flatnonzero(sub["index_map"][:] != PADDING).tolist())
        sub.attrs["n_cutouts"] = len(done)
        remaining = [pair for pair in present if pair[0] not in done]
        logger.info(
            f"{survey} has {len(done)} stored cutout(s), with "
            f"{len(remaining)} remaining."
        )
        present = remaining
    else:
        sub = create_empty_substore(
            surveys_group,
            survey,
            store,
            diameter,
            len(df),
            len(present),
            shard_size,
        )

    n_blank = 0
    # Batching bounds unwritten cutout memory while retaining one survey-wide plan.
    if present:
        workers, concurrency = extraction_plan(store, present[0][1], len(present))
        logger.info(
            f"Extracting {survey} with {workers} worker(s), {concurrency} range(s) "
            f"in flight each."
        )
    with (
        quiet_fits_warnings(),
        tqdm(total=len(present), desc=f"Extracting {survey} cutouts") as progress,
    ):
        for start in range(0, len(present), CUTOUT_BATCH):
            batch = present[start : start + CUTOUT_BATCH]
            extracted = extract_cutouts(
                store,
                [
                    (
                        image_index,
                        (
                            float(df["ra"].iloc[source_index]),
                            float(df["dec"].iloc[source_index]),
                        ),
                    )
                    for source_index, image_index in batch
                ],
                diameter,
                f"Extracting {survey} cutouts",
                progress=progress,
            )
            kept, cutouts, origins = [], [], []
            for (source_index, image_index), (cutout, x0, y0, valid) in zip(
                batch, extracted, strict=True
            ):
                # Pixel validity enforces a fully populated sky disc across surveys.
                if valid < 1.0:
                    n_blank += 1
                    continue
                kept.append((source_index, image_index))
                cutouts.append(cutout)
                origins.append((x0, y0))
            if kept:
                write_substore_rows(sub, store, kept, cutouts, origins)

    if n_blank:
        # The index map encodes dropped sources through their absence.
        logger.warning(
            f"{n_blank} source(s) dropped from {survey} because their discs "
            f"contain blank pixels."
        )
    return sub


def write_cutout_rows(data: zarr.Array, rows: Sequence[int], cutouts: list) -> None:
    """Write a batch of cutouts into the pixel array at their own rows.

    Cutouts of differing size are padded by the array's fill value, and the
    array widens if one is larger than any stored so far. The row count is the
    source count and never grows here; only `edit.extend_substore` changes it.

    The batch is assembled in memory and written in one call. With several rows
    to a shard, a row at a time would read, decompress, and rewrite the whole
    shard for each of them.

    Parameters
    ----------
    data
        The sub-store's pixel array.
    rows
        Row each cutout belongs at, which is its source index. Ascending.
    cutouts
        Cutouts to store, each ``(H, W)`` or ``(n_chan, H, W)``.
    """
    max_h = max(max(c.shape[-2] for c in cutouts), data.shape[-2])
    max_w = max(max(c.shape[-1] for c in cutouts), data.shape[-1])
    if (max_h, max_w) != tuple(data.shape[-2:]):
        data.resize((data.shape[0], *data.shape[1:-2], max_h, max_w))
    block = np.full((len(cutouts), *data.shape[1:]), data.fill_value, dtype=data.dtype)
    for offset, cutout in enumerate(cutouts):
        h, w = cutout.shape[-2:]
        if cutout.ndim == 2:
            block[offset, :h, :w] = cutout
        else:
            block[offset, :, :h, :w] = cutout
    data.oindex[list(rows)] = block


def write_substore_rows(
    sub: zarr.Group,
    store: SurveyStore,
    present: list,
    cutouts: list,
    origins: list,
) -> None:
    """Write a batch of cutouts, their metadata, and their map entries.

    Pixels, then metadata, then the index map, so what the map names is
    complete and an interruption can only leave rows nothing refers to.

    Parameters
    ----------
    sub
        The sub-store to write into.
    store
        Survey the cutouts came from.
    present
        ``(row, image_index)`` per cutout, the row being the source index.
        Ascending, and every row unwritten so far.
    cutouts
        The extracted cutouts, in the same order.
    origins
        Each cutout's ``(x0, y0)`` origin within its image.
    """
    rows = [row for row, _ in present]
    n_new = len(rows)
    write_cutout_rows(sub["data"], rows, cutouts)

    meta = sub["metadata"]
    naxes = meta["crval"].shape[1]

    def write(array, values):
        """Write values at the current source rows.

        Parameters
        ----------
        array
            Destination metadata array.
        values
            Values for the current source rows.
        """
        array.oindex[rows] = values

    write(
        meta["crpix"],
        np.array(
            [
                shift_crpix(store.crpix[image_index], x0, y0)
                for (_, image_index), (x0, y0) in zip(present, origins, strict=True)
            ],
            dtype=np.float64,
        ).reshape(n_new, naxes),
    )
    write(
        meta["pixel_shape"],
        np.array([c.shape[-2:] for c in cutouts], dtype=np.int32).reshape(n_new, 2),
    )
    write(
        meta["image_index"],
        np.array([image_index for _, image_index in present], dtype=np.int64),
    )
    for key in CARRIED_METADATA_KEYS:
        values = [getattr(store, key)[image_index] for _, image_index in present]
        if key in STRING_METADATA_KEYS:
            write(
                meta[key],
                np.array([str(v) for v in values], dtype=np.dtypes.StringDType()),
            )
        else:
            write(
                meta[key],
                np.array(values, dtype=np.float64).reshape(n_new, *meta[key].shape[1:]),
            )

    # Covered rows use the source index to preserve cross-survey alignment.
    sub["index_map"].oindex[rows] = np.array(rows, dtype=np.int64)
    sub.attrs["n_cutouts"] = int(sub.attrs["n_cutouts"]) + n_new


def build_dataset_store(
    survey_store_paths: list[str | Path],
    catalogue: str | Path,
    max_cutout_size_arcsec: float,
    out_path: str | Path,
    survey_names: list[str] | None = None,
    margin_px: int = 2,
    resume: bool = False,
    coarsest_pixel_scale_arcsec: float | None = None,
    shard_size: int | None = None,
    bucket_by: Sequence[str] | None = None,
    n_buckets: int | None = None,
    bucket_seed: int = 0,
) -> zarr.Group:
    """Build a dataset store from survey stores and a source catalogue.

    Extract each cutout and resolve duplicate sources before writing the source
    table. Later reads use the dataset store directly.

    A store knows nothing about splits. It is written in bucket order, each
    bucket a miniature of the catalogue, and `S2ZDataset` cuts the splits it
    was configured for across that sequence.

    Store discs to preserve one sky region across rotated survey grids.

    Parameters
    ----------
    survey_store_paths
        Survey stores to extract cutouts from.
    catalogue
        Source catalogue file.
    max_cutout_size_arcsec
        Largest square, in arcsec across, the store must be able to serve at
        any rotation. Every later read asks for this or less. The disc actually
        stored is its diagonal across, so the arrays written are ``sqrt(2)``
        times this on a side.
    out_path
        Destination, overwritten if it exists.
    survey_names
        Names for the sub-stores. Defaults to the survey store directory names,
        which must then be distinct.
    margin_px
        Pixels the disc must stay clear of each image edge by.
    resume
        Whether to continue a matching unfinished build at ``out_path``.
    coarsest_pixel_scale_arcsec
        Coarsest pixel scale, in arcsec per pixel, the store should be able to
        serve ``max_cutout_size_arcsec`` in full for. Defaults to the coarsest of
        the surveys being built. Set it higher to pre-pay for a coarser survey
        added later by ``append_survey``, which cannot enlarge the disc.
    shard_size
        Rows held in one shard of every survey's pixel array, which is the file
        a read opens. Must be a power of two, or the surveys' shards stop
        dividing one another. Defaults to `shard_rows` of the survey's own
        cutout size and coverage, so a shard serves roughly
        `TARGET_SHARD_BYTES` no matter what a survey's cutouts measure and
        however little of the catalogue it covers. Fixed at build time, since reshaping
        means rebuilding.
    bucket_by
        Catalogue columns the sources are sorted and dealt by, most important
        first. Defaults to every column the catalogue holds beyond the
        standard three, discrete ones ahead of continuous.
    n_buckets
        Buckets to deal the sorted catalogue into, which is the granularity a
        later split fraction is accurate to. Defaults to what
        `buckets.choose_bucket_count` derives from the catalogue.
    bucket_seed
        Seed for the tie-break within the sort and the shuffle within each
        bucket.

    Returns
    -------
    result
        Root group of the finished store.

    Raises
    ------
    TypeError
        Where ``max_cutout_size_arcsec`` is not a scalar.
    ValueError
        Where ``max_cutout_size_arcsec`` is not positive, ``shard_size`` is not
        a power of two, survey names repeat, or the catalogue is unusable.
    """
    if not np.isscalar(max_cutout_size_arcsec):
        raise TypeError(
            f"max_cutout_size_arcsec must be one disc diameter, got "
            f"{max_cutout_size_arcsec!r}."
        )
    cutout_size = float(max_cutout_size_arcsec)
    if cutout_size <= 0:
        raise ValueError(f"max_cutout_size_arcsec must be positive, got {cutout_size}.")
    if shard_size is not None:
        shard_size = int(shard_size)
        if not is_power_of_two(shard_size):
            raise ValueError(
                f"shard_size must be a power-of-two row count, got {shard_size}. "
                f"Other values break survey alignment."
            )
    survey_store_paths = [Path(p) for p in survey_store_paths]
    if survey_names is None:
        survey_names = [p.name for p in survey_store_paths]
    if len(survey_names) != len(set(survey_names)):
        raise ValueError(f"survey_names must be unique, got {survey_names}.")

    df = normalise_catalogue(open_catalogue_file(catalogue), catalogue)
    stores = {
        name: SurveyStore.open(path)
        for name, path in zip(survey_names, survey_store_paths, strict=True)
    }

    # margin_px extends the disc beyond square corners for bilinear interpolation.
    survey_coarsest = max(float(np.max(store.pixel_scale)) for store in stores.values())
    pad_scale = (
        survey_coarsest
        if coarsest_pixel_scale_arcsec is None
        else float(coarsest_pixel_scale_arcsec)
    )
    if pad_scale < survey_coarsest:
        logger.warning(
            (
                "coarsest_pixel_scale_arcsec {} is finer than the coarsest "
                "survey ({} arcsec/pixel). That survey can serve {:.1f} arcsec "
                "from the requested {} arcsec."
            ).format(
                pad_scale,
                survey_coarsest,
                square_size(
                    disc_diameter(
                        cutout_size,
                        edge_padding_for(pad_scale, margin_px),
                    ),
                    edge_padding_for(survey_coarsest, margin_px),
                ),
                cutout_size,
            )
        )
    padding = edge_padding_for(pad_scale, margin_px)
    diameter = disc_diameter(cutout_size, padding)
    logger.info(
        f"Storing {cutout_size:.1f} arcsec squares servable at any rotation as "
        f"{diameter:.1f} arcsec discs padded for {pad_scale:.2f} arcsec/pixel."
    )

    # Duplicate checks require matched-image beams from every survey.
    source_image_maps = {}
    beams = {}
    for survey in survey_names:
        store = stores[survey]
        logger.info(f"Matching catalogue source coordinates to {survey} images ...")
        source_image_map = match_coords_to_images(
            df, store, survey, diameter, margin_px
        )
        matches = [source_image_map[source_id] for source_id in df["id"]]
        source_image_maps[survey] = matches
        beams[survey] = matched_beams(store, matches)

    duplicates = find_catalogue_duplicates(df, source_image_maps, beams)
    if duplicates.any():
        warn_many(
            logger,
            df["id"][duplicates],
            "{n} duplicate source(s) dropped, unresolvable from another "
            "within the beam of the finest common survey",
        )
        keep = ~duplicates
        df = df[keep].reset_index(drop=True)
        source_image_maps = {
            survey: [m for m, k in zip(matches, keep, strict=True) if k]
            for survey, matches in source_image_maps.items()
        }

    rule = build_rule(df, bucket_by, seed=bucket_seed, n_buckets=n_buckets)
    order, source_buckets = bucket_order(df, rule)
    df = df.iloc[order].reset_index(drop=True)
    source_image_maps = {
        survey: [matches[position] for position in order]
        for survey, matches in source_image_maps.items()
    }
    report_bucketing(rule, df)

    out_path = Path(out_path)
    logger.info(f"Creating dataset store at {out_path}.")
    recipe = {
        "max_cutout_size_arcsec": cutout_size,
        "disc_diameter_arcsec": diameter,
        "surveys": list(survey_names),
        "n_sources": len(df),
        "margin_px": margin_px,
        # Existing shard shapes require a fresh build when the request changes.
        "shard_size": shard_size,
        # A changed bucketing rule requires reordered source rows.
        "bucketing": rule.as_dict(),
    }
    resuming = resume and resumable_build(out_path, recipe)
    if resuming:
        root = open_store(out_path, mode="r+")
        logger.info(f"Resuming the unfinished build at {out_path}.")
    else:
        root = create_store(out_path)
    root.attrs["complete"] = False
    root.attrs["max_cutout_size_arcsec"] = cutout_size
    root.attrs["disc_diameter_arcsec"] = diameter
    # Record padding scale so appends can detect a reduced servable square.
    root.attrs["disc_padding_arcsec"] = padding
    root.attrs["padded_for_pixel_scale_arcsec"] = pad_scale
    root.attrs["surveys"] = list(survey_names)
    root.attrs["survey_store_paths"] = [str(p) for p in survey_store_paths]
    root.attrs["n_sources"] = len(df)
    root.attrs["margin_px"] = margin_px
    root.attrs["shard_size"] = shard_size

    if not resuming:
        # Source rows define the shared index required by every sub-store.
        write_sources_group(root, df)
        write_rule(root, rule, source_buckets)

    surveys_group = root.require_group("surveys")
    for survey in survey_names:
        source_image_map = dict(zip(df["id"], source_image_maps[survey], strict=True))
        logger.info(f"Writing {survey} cutout substore ...")
        write_survey_substore(
            surveys_group,
            survey,
            stores[survey],
            df,
            source_image_map,
            diameter,
            resume=resuming,
            shard_size=shard_size,
        )

    # The smallest completed survey chunk defines cross-survey shard alignment.
    root.attrs["coalesce_gap_bytes"] = coalesce_gap_bytes(surveys_group, survey_names)
    root.attrs["complete"] = True
    consolidate(root)
    logger.info(f"Finished creating dataset store at {out_path}.")
    return root
