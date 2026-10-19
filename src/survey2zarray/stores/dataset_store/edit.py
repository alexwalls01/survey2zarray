"""Edit a dataset store, appending sources or attaching another survey."""

from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import pandas as pd
import zarr
from astropy.coordinates import ICRS, SkyCoord
from tqdm import tqdm

from survey2zarray.stores.dataset_store.buckets import (
    BUCKET_ARRAY,
    bucket_order,
    read_rule,
    validate_columns,
)
from survey2zarray.stores.dataset_store.build import (
    CUTOUT_BATCH,
    coalesce_gap_bytes,
    extract_cutouts,
    extraction_plan,
    normalise_catalogue,
    open_catalogue_file,
    write_substore_rows,
    write_survey_substore,
)
from survey2zarray.stores.dataset_store.coverage import (
    match_coords_to_images,
)
from survey2zarray.stores.dataset_store.dataset_store import (
    PADDING,
    check_index_consistency,
)
from survey2zarray.stores.dataset_store.duplicates import (
    find_catalogue_duplicates,
    find_stored_duplicates,
    matched_beams,
)
from survey2zarray.stores.dataset_store.geometry import edge_padding_for, square_size
from survey2zarray.stores.store import open_store
from survey2zarray.stores.survey_store.fits import quiet_fits_warnings
from survey2zarray.stores.survey_store.survey_store import SurveyStore
from survey2zarray.stores.transaction import (
    begin_edit,
    commit_edit,
    serialised_store_edit,
)
from survey2zarray.utils.logging import get_logger, warn_many

logger = get_logger(__name__)


def _is_string_array(array: zarr.Array) -> bool:
    return array.dtype.kind == "T"


def extend_substore(sub: zarr.Group, n_sources: int) -> None:
    """Give a sub-store a row for every source, up to a new source count.

    Extend every survey to preserve source-row alignment. Fill-only rows require
    no stored chunks.

    Parameters
    ----------
    sub
        The sub-store to extend.
    n_sources
        Rows every array should now have.
    """
    for _, array in list(sub.arrays()) + list(sub["metadata"].arrays()):
        array.resize((int(n_sources), *array.shape[1:]))


def append_cutout_rows(
    sub: zarr.Group,
    store: SurveyStore,
    df_new: pd.DataFrame,
    source_image_map: dict[Any, Any],
    diameter: float,
    source_index_offset: int,
) -> None:
    """Extract cutouts for new sources and write them at their own rows.

    Write the index map after cutout data so it marks only complete rows. Grow
    stored spatial dimensions when required.

    A shard may hold both the region the store was built with and one appended
    later, which adds no overhead. Reads are orthogonal, so one fetches only the
    rows it asked for no matter what else the shard holds, and blocks are cut from a
    single split's served sources so none spans a split boundary.

    Parameters
    ----------
    sub
        The sub-store to write into, already extended by `extend_substore`.
    store
        Survey to extract from.
    df_new
        New sources, in the order they were appended to the source table.
    source_image_map
        Source identifier to matched image index, NaN where absent.
    diameter
        Diameter of the disc to extract, in arcsec.
    source_index_offset
        Source index the new rows start at.
    """
    present = []
    for position, source_id in enumerate(df_new["id"]):
        image_index = source_image_map[source_id]
        if not pd.isna(image_index):
            present.append((source_index_offset + position, int(image_index)))
    if len(present) == 0:
        return

    workers, concurrency = extraction_plan(store, present[0][1], len(present))
    logger.info(
        f"Extracting appended cutouts with {workers} worker(s), {concurrency} "
        "range(s) in flight each."
    )
    n_blank = 0
    with (
        quiet_fits_warnings(),
        tqdm(total=len(present), desc="Extracting appended cutouts") as progress,
    ):
        for start in range(0, len(present), CUTOUT_BATCH):
            batch = present[start : start + CUTOUT_BATCH]
            extracted = extract_cutouts(
                store,
                [
                    (
                        image_index,
                        (
                            float(df_new["ra"].iloc[row - source_index_offset]),
                            float(df_new["dec"].iloc[row - source_index_offset]),
                        ),
                    )
                    for row, image_index in batch
                ],
                diameter,
                "Extracting appended cutouts",
                progress=progress,
            )
            kept, cutouts, origins = [], [], []
            for pair, (cutout, x0, y0, valid) in zip(batch, extracted, strict=True):
                if valid < 1.0:
                    n_blank += 1
                    continue
                kept.append(pair)
                cutouts.append(cutout)
                origins.append((x0, y0))
            if kept:
                write_substore_rows(sub, store, kept, cutouts, origins)

    if n_blank:
        logger.warning(
            f"{n_blank} appended source(s) dropped because their discs contain "
            "blank pixels."
        )


def filter_sources(
    df: pd.DataFrame,
    maps: dict[str, list[Any]],
    beams: dict[str, np.ndarray],
    keep: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, list[Any]], dict[str, np.ndarray]]:
    """Drop rows from a catalogue and its per-survey arrays together.

    Filtering all three at once is what keeps them index-aligned.

    Parameters
    ----------
    df
        Catalogue.
    maps
        Per survey, the matched image index per row.
    beams
        Per survey, the matched image's beam major axis per row.
    keep
        Rows to retain.

    Returns
    -------
    result
        The filtered catalogue, maps, and beams.
    """
    return (
        df[keep].reset_index(drop=True),
        {
            survey: [m for m, k in zip(matches, keep, strict=True) if k]
            for survey, matches in maps.items()
        },
        {survey: values[keep] for survey, values in beams.items()},
    )


def read_source_table(root: zarr.Group) -> pd.DataFrame:
    """Read the source table back into a data frame.

    Parameters
    ----------
    root
        Root group of the store.

    Returns
    -------
    result
        Rows in source index order, with the ``id``, ``ra``, and ``dec`` columns
        the source matcher expects.
    """
    sources = root["sources"]
    return pd.DataFrame(
        {
            "id": [str(v) for v in sources["source_id"][:]],
            "ra": sources["ra"][:],
            "dec": sources["dec"][:],
        }
    )


@serialised_store_edit
def append_survey(
    store_path: str | Path,
    survey_store_path: str | Path,
    survey_name: str | None = None,
) -> zarr.Group:
    """Attach another survey to an existing dataset store.

    Cutouts are extracted for the sources already stored; the source table is
    unchanged. This widens a dataset across surveys, where `append_sources`
    lengthens it across sources.

    A survey covering none of the sources still attaches, with an empty
    sub-store and a warning.

    Parameters
    ----------
    store_path
        Location of the dataset store.
    survey_store_path
        Survey store to extract from.
    survey_name
        Name for the new sub-store. Defaults to the survey store's directory
        name.

    Returns
    -------
    result
        Root group of the store.

    Raises
    ------
    RuntimeError
        Where the store was not completely written.
    ValueError
        Where a survey of that name is already attached.
    """
    store_path = Path(store_path)
    root = open_store(store_path, mode="r+")
    if not root.attrs.get("complete", False):
        raise RuntimeError(
            f"Dataset store {store_path} is incomplete. Rebuild it before use."
        )

    survey_store_path = Path(survey_store_path)
    survey_name = survey_name or survey_store_path.name
    surveys = list(root.attrs["surveys"])
    if survey_name in surveys:
        raise ValueError(
            f"Dataset store {store_path} already has survey {survey_name!r}. "
            f"Pass a different name or rebuild to replace it."
        )
    # Reuse store geometry to align every sub-store.
    diameter = float(root.attrs["disc_diameter_arcsec"])
    margin_px = root.attrs["margin_px"]
    survey_store_paths = list(root.attrs["survey_store_paths"])
    # Preserve per-survey shard sizing when the store has no shared size.
    shard_size = root.attrs.get("shard_size")

    store = SurveyStore.open(survey_store_path)
    df = read_source_table(root)

    # Coarser appended surveys reduce the servable square within fixed discs.
    padded_for = float(root.attrs["padded_for_pixel_scale_arcsec"])
    coarsest = float(np.max(store.pixel_scale)) if len(store.pixel_scale) else 0.0
    if coarsest > padded_for > 0.0:
        stored_size = float(root.attrs["max_cutout_size_arcsec"])
        reachable = square_size(diameter, edge_padding_for(coarsest, margin_px))
        logger.warning(
            f"{survey_name} is {coarsest:.2f} arcsec/pixel. This store was padded "
            f"for "
            f"{padded_for:.2f}. With it selected, cutouts serve {reachable:.1f} arcsec "
            f"from the requested {stored_size:.1f}. Rebuild with "
            f"coarsest_pixel_scale_arcsec={coarsest:.2f} to keep the full size."
        )

    logger.info(f"Matching catalogue source coordinates to {survey_name} images ...")
    source_image_map = match_coords_to_images(
        df, store, survey_name, diameter, margin_px
    )

    # Readers only enumerate sub-stores published through the surveys attribute.
    begin_edit(
        root,
        [],
        attrs={
            "surveys": surveys,
            "survey_store_paths": survey_store_paths,
            "coalesce_gap_bytes": root.attrs.get("coalesce_gap_bytes"),
        },
        groups=[f"surveys/{survey_name}"],
    )

    write_survey_substore(
        root.require_group("surveys"),
        survey_name,
        store,
        df,
        source_image_map,
        diameter,
        shard_size=shard_size,
    )

    root.attrs["surveys"] = surveys + [survey_name]
    root.attrs["survey_store_paths"] = survey_store_paths + [str(survey_store_path)]
    # The smallest survey chunk limits the shared read gap.
    root.attrs["coalesce_gap_bytes"] = coalesce_gap_bytes(
        root["surveys"], root.attrs["surveys"]
    )
    check_index_consistency(root)
    commit_edit(root)

    n_cutouts = int(root["surveys"][survey_name].attrs["n_cutouts"])
    if n_cutouts == 0:
        logger.warning(f"No sources fall inside {survey_name}. The sub-store is empty.")
    logger.info(f"Appended survey {survey_name} ({n_cutouts} cutouts) to {store_path}.")
    return root


@serialised_store_edit
def append_sources(
    store_path: str | Path,
    catalogue: str | Path | pd.DataFrame,
    survey_store_paths: list[str | Path] | None = None,
) -> zarr.Group:
    """Add sources to an existing dataset store.

    Sources already present, whether by identifier or by position, are skipped
    with a warning; the incoming catalogue is also de-duplicated against
    itself. Existing source indices are immutable and never reused, and removing
    sources is not supported.

    New sources form a region at the tail, sorted by bucket. Every survey's
    arrays lengthen to the new source count, so a survey that covers none of
    them still grows in shape; only the rows it does cover are written.

    Parameters
    ----------
    store_path
        Location of the dataset store.
    catalogue
        New sources.
    survey_store_paths
        Survey stores to extract from. Defaults to the paths recorded when the
        dataset store was built.

    Returns
    -------
    result
        Root group of the store.

    Raises
    ------
    RuntimeError
        Where the store was not completely written.
    ValueError
        Where the catalogue lacks a column the store holds or buckets on, or
        a bucketing column arrives with a different dtype or designation.
    """
    store_path = Path(store_path)
    root = open_store(store_path, mode="r+")
    if not root.attrs.get("complete", False):
        raise RuntimeError(
            f"Dataset store {store_path} is incomplete. Rebuild it before use."
        )

    surveys = list(root.attrs["surveys"])
    diameter = float(root.attrs["disc_diameter_arcsec"])
    margin_px = root.attrs["margin_px"]
    n_sources = int(root.attrs["n_sources"])
    if survey_store_paths is None:
        survey_store_paths = root.attrs["survey_store_paths"]
    stores = {
        name: SurveyStore.open(path)
        for name, path in zip(surveys, survey_store_paths, strict=True)
    }

    if isinstance(catalogue, pd.DataFrame):
        df_new = normalise_catalogue(catalogue.copy(), "<dataframe>")
    else:
        df_new = normalise_catalogue(open_catalogue_file(catalogue), catalogue)

    # Requiring every stored column avoids inventing missing integer values.
    stored_columns = [
        name
        for name in sorted(root["sources"].array_keys())
        if name not in ("source_id", "ra", "dec", BUCKET_ARRAY)
    ]
    absent = [name for name in stored_columns if name not in df_new.columns]
    if absent:
        raise ValueError(
            f"Appended catalogue is missing column(s) {absent} that dataset store "
            f"{store_path} holds for its existing sources. Add them to the "
            f"catalogue or rebuild with a reduced schema."
        )
    unknown = [
        name
        for name in df_new.columns
        if name not in ("id", "ra", "dec") and name not in stored_columns
    ]
    if unknown:
        logger.warning(
            f"Appended catalogue has column(s) {unknown} the store does not hold, "
            f"so they are dropped, since its arrays were fixed when it was built."
        )
    # The persisted bucketing rule places appends against the original distribution.
    rule = read_rule(root)
    validate_columns(df_new, rule)

    existing_ids = set(str(v) for v in root["sources"]["source_id"][:])
    already = df_new["id"].astype(str).isin(existing_ids)
    if already.any():
        warn_many(
            logger, df_new["id"][already], "{n} source(s) already in the store, by id"
        )
        df_new = df_new[~already].reset_index(drop=True)
    if len(df_new) == 0:
        logger.info("No new sources to append.")
        return root

    new_maps = {}
    new_beams = {}
    for survey in surveys:
        store = stores[survey]
        source_image_map = match_coords_to_images(
            df_new, store, survey, diameter, margin_px
        )
        matches = [source_image_map[source_id] for source_id in df_new["id"]]
        new_maps[survey] = matches
        new_beams[survey] = matched_beams(store, matches)

    # Compare with stored sources first to keep incremental and one-shot results equal.
    existing_coords = SkyCoord(
        ra=root["sources"]["ra"][:] * u.deg,
        dec=root["sources"]["dec"][:] * u.deg,
        frame=ICRS,
    )
    existing_presence = {}
    existing_beams = {}
    for survey in surveys:
        sub = root["surveys"][survey]
        entries = sub["index_map"][:]
        held = entries != PADDING
        index_map = entries[held]
        presence = np.zeros(n_sources, dtype=bool)
        presence[index_map] = True
        beams = np.full(n_sources, np.nan)
        beams[index_map] = sub["metadata"]["bmaj"][:][held]
        existing_presence[survey] = presence
        existing_beams[survey] = beams
    duplicates = find_stored_duplicates(
        df_new, new_maps, new_beams, existing_coords, existing_presence, existing_beams
    )
    warn_many(
        logger,
        df_new["id"][duplicates],
        "{n} duplicate source(s) skipped, unresolvable from one already stored "
        "within the beam of the finest common survey",
    )
    df_new, new_maps, new_beams = filter_sources(
        df_new, new_maps, new_beams, ~duplicates
    )
    if len(df_new) == 0:
        logger.info("No new sources to append.")
        return root

    # Only candidates retained from the stored comparison may suppress later rows.
    internal = find_catalogue_duplicates(df_new, new_maps, new_beams)
    if internal.any():
        warn_many(
            logger,
            df_new["id"][internal],
            "{n} source(s) skipped, duplicated within the appended catalogue",
        )
        df_new, new_maps, new_beams = filter_sources(
            df_new, new_maps, new_beams, ~internal
        )
        if len(df_new) == 0:
            logger.info("No new sources to append.")
            return root

    # Continue the persisted deal so appended rows form a bucket-sorted tail.
    order, new_buckets = bucket_order(df_new, rule, start_rank=n_sources)
    df_new = df_new.iloc[order].reset_index(drop=True)
    new_maps = {
        survey: [matches[i] for i in order] for survey, matches in new_maps.items()
    }
    new_beams = {survey: values[order] for survey, values in new_beams.items()}

    # One transaction state covers source rows, cutouts, and index maps.
    array_paths = [f"sources/{key}" for key in root["sources"].array_keys()]
    attrs = {"n_sources": n_sources}
    for survey in surveys:
        sub = root["surveys"][survey]
        array_paths.append(f"surveys/{survey}/data")
        array_paths.append(f"surveys/{survey}/index_map")
        array_paths.extend(
            f"surveys/{survey}/metadata/{key}" for key in sub["metadata"].array_keys()
        )
        attrs[f"surveys/{survey}:n_cutouts"] = int(sub.attrs["n_cutouts"])
    begin_edit(root, array_paths, attrs=attrs)

    # Source-table length defines every new shared index.
    sources = root["sources"]
    n_new = len(df_new)
    for key in sources.array_keys():
        array = sources[key]
        array.resize((n_sources + n_new,))
        if key == "source_id":
            array[n_sources:] = np.array(
                [str(v) for v in df_new["id"]], dtype=np.dtypes.StringDType()
            )
        elif key in ("ra", "dec"):
            array[n_sources:] = df_new[key].to_numpy(dtype=np.float64)
        elif key == BUCKET_ARRAY:
            array[n_sources:] = new_buckets
        else:
            if _is_string_array(array):
                array[n_sources:] = np.array(
                    [str(v) for v in df_new[key]], dtype=np.dtypes.StringDType()
                )
            else:
                array[n_sources:] = df_new[key].to_numpy(dtype=array.dtype)

    # Extend every survey to preserve one aligned row per source.
    for survey in surveys:
        extend_substore(root["surveys"][survey], n_sources + n_new)
        source_image_map = {
            source_id: m
            for source_id, m in zip(df_new["id"], new_maps[survey], strict=True)
        }
        append_cutout_rows(
            root["surveys"][survey],
            stores[survey],
            df_new,
            source_image_map,
            diameter,
            source_index_offset=n_sources,
        )

    root.attrs["n_sources"] = n_sources + n_new
    check_index_consistency(root)
    commit_edit(root)
    logger.info(f"Appended {n_new} sources to {store_path}.")
    return root
