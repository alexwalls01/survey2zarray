"""Provide rotation-invariant disc geometry for dataset cutouts.

Store a sky disc around each source and define requested size as the side of an
inscribed square. The square remains inside the shared disc at every rotation.
"""

from collections.abc import Sequence

import numpy as np

SQRT2 = float(np.sqrt(2.0))


def disc_diameter(square_size: float, edge_padding: float = 0.0) -> float:
    """Return the diameter of the disc a square fits inside at any rotation.

    Add the reader's interpolation margin beyond the square corners.

    Parameters
    ----------
    square_size
        Side of the square, in arcsec.
    edge_padding
        Extra radius in arcsec, beyond where the square's corners fall.

    Returns
    -------
    result
        Diameter in arcsec, being the square's diagonal plus twice the padding.
    """
    return float(square_size) * SQRT2 + 2.0 * float(edge_padding)


def square_size(diameter: float, edge_padding: float = 0.0) -> float:
    """Return the side of the largest square a disc can serve.

    Parameters
    ----------
    diameter
        Diameter of the disc, in arcsec.
    edge_padding
        Radius in arcsec to keep clear at the edge.

    Returns
    -------
    result
        Side in arcsec. Inverse of :func:`disc_diameter`.
    """
    return (float(diameter) - 2.0 * float(edge_padding)) / SQRT2


def edge_padding_for(pixel_scale: float, margin_px: int) -> float:
    """Return the padding a survey of a given pixel scale needs at the disc edge.

    Tie the stored disc radius to the coarsest supported survey grid.

    Parameters
    ----------
    pixel_scale
        Coarsest pixel scale to support, in arcsec per pixel.
    margin_px
        Pixels a reader insets by before interpolating.

    Returns
    -------
    result
        Padding in arcsec.
    """
    return float(margin_px) * float(pixel_scale)


def disc_mask(
    shape: Sequence[int],
    pixel_scale: Sequence[float],
    diameter: float,
    centre: Sequence[float] | None = None,
) -> np.ndarray:
    """Return a mask of the pixels lying within a disc.

    The radius is angular, so surveys of differing pixel scale mask the same
    patch of sky to within the coarser survey's pixel. Testing pixel centres
    keeps each survey's region inside the true disc.

    Parameters
    ----------
    shape
        Array shape as ``(H, W)``.
    pixel_scale
        Pixel scale in arcsec per pixel, as (x, y).
    diameter
        Diameter of the disc in arcsec.
    centre
        Centre in pixel coordinates as (x, y). Defaults to the array centre.

    Returns
    -------
    result
        Boolean array of ``shape``, True inside the disc.
    """
    height, width = int(shape[0]), int(shape[1])
    scale_x, scale_y = float(pixel_scale[0]), float(pixel_scale[1])
    if centre is None:
        centre_x, centre_y = (width - 1) / 2.0, (height - 1) / 2.0
    else:
        centre_x, centre_y = float(centre[0]), float(centre[1])
    radius = float(diameter) / 2.0

    y, x = np.ogrid[:height, :width]
    dx = (x - centre_x) * scale_x
    dy = (y - centre_y) * scale_y
    return (dx * dx + dy * dy) <= radius * radius


def valid_fraction(
    array: np.ndarray,
    pixel_scale: Sequence[float],
    diameter: float,
    centre: Sequence[float] | None = None,
) -> float:
    """Return the fraction of a disc that holds real data.

    An image is a rectangular array, but its data need not fill that rectangle.
    A mosaic edge, a primary-beam cutoff, or a blanked region all leave holes
    the image's bounds say nothing about. Measuring what came back is the only way
    to tell, and the pixels have been read by this point anyway.

    Parameters
    ----------
    array
        ``(H, W)`` or ``(C, H, W)``, already blanked outside the disc.
    pixel_scale
        Pixel scale in arcsec per pixel, as (x, y).
    diameter
        Diameter of the disc in arcsec.
    centre
        Centre in pixel coordinates as (x, y). Defaults to the array centre.

    Returns
    -------
    result
        Between 0 and 1, measured over the disc alone. Zero for an empty array.
    """
    array = np.asarray(array)
    mask = disc_mask(array.shape[-2:], pixel_scale, diameter, centre)
    inside = array[..., mask]
    if inside.size == 0:
        return 0.0
    return float(np.isfinite(inside).mean())


def blank_outside_disc(
    array: np.ndarray,
    pixel_scale: Sequence[float],
    diameter: float,
    centre: Sequence[float] | None = None,
) -> np.ndarray:
    """Blank array corners outside a disc.

    Encode the disc with non-finite pixels so statistics and transforms use the
    shared sky region.

    Parameters
    ----------
    array
        ``(H, W)`` or ``(C, H, W)``. Integer input is promoted to float, since
        the blanked pixels are NaN.
    pixel_scale
        Pixel scale in arcsec per pixel, as (x, y).
    diameter
        Diameter of the disc in arcsec.
    centre
        Centre in pixel coordinates as (x, y). Defaults to the array centre.

    Returns
    -------
    result
        Copy with everything outside the disc set to NaN.
    """
    array = np.asarray(array)
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
    else:
        array = array.copy()
    mask = disc_mask(array.shape[-2:], pixel_scale, diameter, centre)
    array[..., ~mask] = np.nan
    return array
