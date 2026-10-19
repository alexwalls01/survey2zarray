"""Regression tests for the project's dfreproject fork."""

import warnings

import numpy as np
from astropy.wcs import WCS
from dfreproject import calculate_reprojection


def _sin_wcs(size: int, scale_arcsec: float, rotation_deg: float) -> WCS:
    """Construct a rotated two-dimensional SIN WCS with a CD matrix."""
    scale = scale_arcsec / 3600.0
    angle = np.deg2rad(rotation_deg)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [(size + 1) / 2, (size + 1) / 2]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.cd = np.array([[-scale, 0.0], [0.0, scale]]) @ rotation
    wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    return wcs


def test_rotated_cd_sin_reprojection_matches_astropy_coordinates():
    """Match an affine image evaluated at Astropy-derived source pixels."""
    input_size = 41
    output_size = 23
    source_wcs = _sin_wcs(input_size, scale_arcsec=1.0, rotation_deg=31.0)
    target_wcs = _sin_wcs(output_size, scale_arcsec=0.8, rotation_deg=-13.0)

    source_y, source_x = np.mgrid[:input_size, :input_size]
    source = (2.5 * source_x - 1.75 * source_y + 11.0).astype(np.float64)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="cdelt will be ignored",
            category=RuntimeWarning,
        )
        actual = calculate_reprojection(
            (source, source_wcs),
            target_wcs,
            shape_out=(output_size, output_size),
            order="bilinear",
            device="cpu",
            conserve_flux=False,
            compute_jacobian=False,
            show_log=False,
        )

    actual = np.asarray(actual).reshape(output_size, output_size)

    target_y, target_x = np.mgrid[:output_size, :output_size]
    ra, dec = target_wcs.pixel_to_world_values(target_x, target_y)
    input_x, input_y = source_wcs.world_to_pixel_values(ra, dec)
    expected = 2.5 * input_x - 1.75 * input_y + 11.0

    interior = (
        (input_x >= 1)
        & (input_x <= input_size - 2)
        & (input_y >= 1)
        & (input_y <= input_size - 2)
    )
    assert np.any(interior)
    assert np.all(np.isfinite(actual[interior]))
    np.testing.assert_allclose(
        actual[interior],
        expected[interior],
        rtol=0.0,
        atol=2e-4,
    )
