"""Decide which image of a survey can supply each source's cutout."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord

if TYPE_CHECKING:
    from survey2zarray.stores.survey_store.survey_store import SurveyStore

from survey2zarray.utils.logging import get_logger, warn_many

logger = get_logger(__name__)


def cutouts_inside(
    x: np.ndarray,
    y: np.ndarray,
    h: int,
    w: int,
    pixel_scale: Sequence[float],
    diameter: float,
    margin_px: int = 2,
) -> np.ndarray:
    """Test a vector of source positions against one image boundary.

    Accept projected vectors to amortise WCS conversion across candidates.

    Parameters
    ----------
    x, y
        Source positions in image pixels, as (column, row).
    h, w
        Image shape as (row, column).
    pixel_scale
        Image pixel scale in arcsec per pixel, as (x, y).
    diameter
        Diameter of the disc, in arcsec.
    margin_px
        Pixels to inset from every edge.

    Returns
    -------
    result
        Boolean, True where the whole disc plus margin lies inside the image.
    """
    half_x = (diameter / 2) / pixel_scale[0]
    half_y = (diameter / 2) / pixel_scale[1]
    x = np.asarray(x)
    y = np.asarray(y)
    return (
        (x - half_x >= margin_px)
        & (x + half_x <= w - 1 - margin_px)
        & (y - half_y >= margin_px)
        & (y + half_y <= h - 1 - margin_px)
    )


def match_coords_to_images(
    df: pd.DataFrame,
    store: "SurveyStore",
    survey: str,
    diameter: float,
    margin_px: int = 2,
) -> dict:
    """Assign each catalogue source the image its cutout can be taken from.

    Candidates come from a neighbour search against image centres, bounded by
    the largest image half-diagonal, then the closest candidate whose bounds
    hold the whole cutout wins. Where the closest image cannot supply one, the
    remaining candidates are tried in order of distance, which recovers sources
    lying near the edge of one image but well inside an overlapping neighbour.

    Log unmatched sources and return NaN entries to represent partial coverage.

    Parameters
    ----------
    df
        Catalogue with ``id``, ``ra``, and ``dec`` columns, coordinates in
        degrees.
    store
        Survey to match against.
    survey
        Survey name, used in messages.
    diameter
        Diameter in arcsec of the disc each source needs.
    margin_px
        Pixels a disc must stay clear of each image edge by.

    Returns
    -------
    result
        Source identifier to matched image index, NaN for sources no image can
        supply a full disc for.
    """
    coords = SkyCoord(
        ra=df["ra"].values * u.deg,
        dec=df["dec"].values * u.deg,
        frame=ICRS,
    )
    # Precomputed image centres avoid WCS projection during candidate search.
    centres = store.pixel_centre
    shapes = store.pixel_shape
    scales = store.pixel_scale
    half_h = shapes[:, 0] / 2 * scales[:, 1]
    half_w = shapes[:, 1] / 2 * scales[:, 0]
    half_radii = np.hypot(half_h, half_w)

    logger.info(f"Locating sources in {survey} images ...")
    # Size buckets prevent oversized images from inflating every search radius.
    scale_bands = np.floor(np.log2(half_radii / max(half_radii.min(), 1e-9))).astype(
        int
    )
    found_sources, found_images, found_distances = [], [], []
    for band in np.unique(scale_bands):
        rows = np.flatnonzero(scale_bands == band)
        band_centres = SkyCoord(
            ra=centres[rows, 0] * u.deg, dec=centres[rows, 1] * u.deg, frame=ICRS
        )
        matched_sources, matched_images, separations, _ = (
            band_centres.search_around_sky(
                coords, seplimit=half_radii[rows].max() * u.arcsec
            )
        )
        found_sources.append(matched_sources)
        found_images.append(rows[matched_images])
        found_distances.append(separations.arcsec)
    source_indices = (
        np.concatenate(found_sources) if found_sources else np.empty(0, dtype=int)
    )
    image_indices = (
        np.concatenate(found_images) if found_images else np.empty(0, dtype=int)
    )
    dist_2d = np.concatenate(found_distances) if found_distances else np.empty(0)

    # Grouping candidates by image amortises each WCS projection over many sources.
    logger.info(f"Testing which {survey} images can supply each source ...")
    fits_inside = np.zeros(len(source_indices), dtype=bool)
    by_image = np.argsort(image_indices, kind="stable")
    if by_image.size:
        splits = np.flatnonzero(np.diff(image_indices[by_image])) + 1
        for rows in np.split(by_image, splits):
            image_index = int(image_indices[rows[0]])
            wcs = store.get_wcs(image_index).celestial
            x, y = wcs.world_to_pixel(coords[source_indices[rows]])
            height, width = store.pixel_shape[image_index]
            fits_inside[rows] = cutouts_inside(
                x, y, height, width, store.pixel_scale[image_index], diameter, margin_px
            )

    # Distance order selects the nearest image that supplies the complete disc.
    source_image_map = {source_id: np.nan for source_id in df["id"]}
    usable = np.flatnonzero(fits_inside)
    if usable.size:
        ordered = usable[np.lexsort((dist_2d[usable], source_indices[usable]))]
        nearest = ordered[np.unique(source_indices[ordered], return_index=True)[1]]
        identifiers = df["id"].to_numpy()
        for row in nearest:
            source_image_map[identifiers[source_indices[row]]] = int(image_indices[row])

    missing_sources = [
        source for source, index in source_image_map.items() if pd.isna(index)
    ]
    warn_many(
        logger, missing_sources, f"{{n}} source(s) not covered by any {survey} image"
    )
    return source_image_map
