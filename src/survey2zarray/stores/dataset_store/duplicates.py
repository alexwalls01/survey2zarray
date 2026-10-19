"""Decide when two catalogue entries are one source.

Two sources are duplicates if they fall within the beam of the
finest-resolution survey observing both. Where the best-resolving instrument
covering both cannot separate them, they are one source. The radius is
therefore survey-dependent, never fixed and never the coarsest beam, which
would merge sources a finer survey resolves."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import ICRS, SkyCoord

if TYPE_CHECKING:
    from survey2zarray.stores.survey_store.survey_store import SurveyStore

from survey2zarray.utils.logging import get_logger

logger = get_logger(__name__)


def matched_beams(store: "SurveyStore", matches: list[Any]) -> np.ndarray:
    """Return the beam major axis of the image each source matched.

    Parameters
    ----------
    store
        Survey the sources were matched against.
    matches
        Matched image index per source, NaN where the source is absent.

    Returns
    -------
    result
        Beam major axis in degrees, NaN where the source is absent.
    """
    return np.array([store.bmaj[int(m)] if not pd.isna(m) else np.nan for m in matches])


def presence_mask(matches: list[Any]) -> np.ndarray:
    """Return which sources matched an image.

    Parameters
    ----------
    matches
        Matched image index per source, NaN where the source is absent.

    Returns
    -------
    result
        Boolean coverage mask.
    """
    return np.array([not pd.isna(m) for m in matches], dtype=bool)


def search_limit(*beam_sets: dict[str, np.ndarray]) -> u.Quantity | None:
    """Return a separation bound covering every possible duplicate pair.

    The largest beam bounds every pair threshold and preserves all duplicate
    candidates in the neighbour search.

    Parameters
    ----------
    *beam_sets
        Mappings of survey name to per-source beam major axis in degrees.

    Returns
    -------
    result
        The largest beam, or None where no beam is recorded anywhere, in which
        case nothing can be declared a duplicate.
    """
    beams = [values for beams in beam_sets for values in beams.values()]
    if not beams:
        return None
    all_beams = np.concatenate(beams)
    if all_beams.size == 0 or np.all(np.isnan(all_beams)):
        return None
    return np.nanmax(all_beams) * u.deg


def pair_thresholds(
    surveys: Iterable[str],
    presence_a: dict[str, np.ndarray],
    beams_a: dict[str, np.ndarray],
    idx_a: np.ndarray,
    presence_b: dict[str, np.ndarray],
    beams_b: dict[str, np.ndarray],
    idx_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the separation below which each candidate pair is one source.

    A pair with no common survey is deliberately never a duplicate, since
    nothing can compare them, and neither is a pair whose common survey records
    no beam, since the resulting NaN threshold compares false.

    Parameters
    ----------
    surveys
        Survey names to consider.
    presence_a, presence_b
        Per survey, which sources on each side matched an image.
    beams_a, beams_b
        Per survey, beam major axis in degrees per source on each side.
    idx_a, idx_b
        Indices of the two members of each candidate pair.

    Returns
    -------
    has_common
        Whether any survey observes both members.
    threshold
        Finest common beam in degrees, ``inf`` where no survey observes both.
    """
    has_common = np.zeros(len(idx_a), dtype=bool)
    threshold = np.full(len(idx_a), np.inf)
    for survey in surveys:
        both = presence_a[survey][idx_a] & presence_b[survey][idx_b]
        # Use the larger beam to retain every possible duplicate pair.
        pair_beam = np.maximum(beams_a[survey][idx_a], beams_b[survey][idx_b])
        threshold[both] = np.minimum(threshold[both], pair_beam[both])
        has_common |= both
    return has_common, threshold


def find_catalogue_duplicates(
    df: pd.DataFrame,
    maps: dict[str, list[Any]],
    beams: dict[str, np.ndarray],
) -> np.ndarray:
    """Flag sources that duplicate an earlier source in the same catalogue.

    The first member of each group survives, in catalogue order, matching how
    repeated identifiers are resolved. Groups chain transitively, since a row
    already flagged cannot absorb a later one.

    Parameters
    ----------
    df
        Catalogue with ``ra`` and ``dec`` columns in degrees.
    maps
        Per survey, the matched image index per row, NaN where absent.
    beams
        Per survey, the matched image's beam major axis in degrees per row.

    Returns
    -------
    result
        True for rows to drop.
    """
    duplicates = np.zeros(len(df), dtype=bool)
    seplimit = search_limit(beams)
    if len(df) < 2 or not maps or seplimit is None:
        return duplicates
    coords = SkyCoord(
        ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg, frame=ICRS
    )
    idx1, idx2, sep2d, _ = coords.search_around_sky(coords, seplimit=seplimit)
    # Evaluate each unordered pair once with the earlier row first.
    ordered = idx1 < idx2
    idx1, idx2, separation = idx1[ordered], idx2[ordered], sep2d[ordered].deg
    if len(idx1) == 0:
        return duplicates
    presence = {survey: presence_mask(matches) for survey, matches in maps.items()}
    has_common, threshold = pair_thresholds(
        maps, presence, beams, idx1, presence, beams, idx2
    )
    is_duplicate = has_common & (separation < threshold)
    for position in np.argsort(idx1, kind="stable"):
        if is_duplicate[position] and not duplicates[idx1[position]]:
            duplicates[idx2[position]] = True
    return duplicates


def find_stored_duplicates(
    df_new: pd.DataFrame,
    new_maps: dict[str, list[Any]],
    new_beams: dict[str, np.ndarray],
    existing_coords: SkyCoord,
    existing_presence: dict[str, np.ndarray],
    existing_beams: dict[str, np.ndarray],
) -> np.ndarray:
    """Flag new sources that duplicate one already in the store.

    Applies the same rule as `find_catalogue_duplicates`, so building a store
    in one pass and building it incrementally give the same result.

    Parameters
    ----------
    df_new
        New sources, with ``ra`` and ``dec`` columns in degrees.
    new_maps
        Per survey, the matched image index per new source, NaN where absent.
    new_beams
        Per survey, the matched image's beam major axis in degrees per new
        source.
    existing_coords
        Stored source positions, in canonical index order.
    existing_presence
        Per survey, which stored sources that survey holds a cutout for.
    existing_beams
        Per survey, beam major axis in degrees per stored source.

    Returns
    -------
    result
        True for new sources to drop.
    """
    duplicates = np.zeros(len(df_new), dtype=bool)
    seplimit = search_limit(new_beams, existing_beams)
    if len(df_new) == 0 or len(existing_coords) == 0 or seplimit is None:
        return duplicates
    new_coords = SkyCoord(
        ra=df_new["ra"].values * u.deg, dec=df_new["dec"].values * u.deg, frame=ICRS
    )
    # Neighbour search avoids a full cross-product against large stores.
    idx_new, idx_existing, sep2d, _ = existing_coords.search_around_sky(
        new_coords, seplimit=seplimit
    )
    if len(idx_new) == 0:
        return duplicates
    new_presence = {
        survey: presence_mask(matches) for survey, matches in new_maps.items()
    }
    has_common, threshold = pair_thresholds(
        new_maps,
        new_presence,
        new_beams,
        idx_new,
        existing_presence,
        existing_beams,
        idx_existing,
    )
    duplicates[idx_new[has_common & (sep2d.deg < threshold)]] = True
    return duplicates
