from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

# ---------------------------------------------------------------------------
# Synthetic FITS fixtures. Images are tiny (tens of pixels), with only the
# dimensions needed to test behaviour: 2D continuum images, 4D
# (stokes, freq, y, x) continuum images with degenerate axes, and small
# spectral cubes. No external astronomy datasets are required.
# ---------------------------------------------------------------------------

IMAGE_RA = 150.0  # deg, image grid reference
IMAGE_DEC = 2.0
PIXEL_SCALE_ARCSEC = 1.0  # arcsec / pixel


def make_header(
    shape_yx=(64, 64),
    ra=IMAGE_RA,
    dec=IMAGE_DEC,
    pixel_scale=PIXEL_SCALE_ARCSEC,
    naxis_extra=(),  # e.g. [("FREQ", 1), ("STOKES", 1)] appended as FITS axes 3, 4
    stokes_crval=1.0,
    stokes_crpix=1.0,
    stokes_cdelt=1.0,
    projection="TAN",
    freq_hz=1.4e9,
    with_beam=True,
    rotation=0.0,  # deg, the grid's position angle on the sky
):
    ny, nx = shape_yx
    scale_deg = pixel_scale / 3600.0
    header = fits.Header()
    header["NAXIS"] = 2 + len(naxis_extra)
    header["NAXIS1"] = nx
    header["NAXIS2"] = ny
    header["CTYPE1"] = f"RA---{projection}"
    header["CTYPE2"] = f"DEC--{projection}"
    header["CRVAL1"] = ra
    header["CRVAL2"] = dec
    header["CRPIX1"] = nx / 2 + 0.5
    header["CRPIX2"] = ny / 2 + 0.5
    angle = np.radians(rotation)
    cd = np.array([[-scale_deg, 0.0], [0.0, scale_deg]]) @ np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    header["CD1_1"] = cd[0, 0]
    header["CD1_2"] = cd[0, 1]
    header["CD2_1"] = cd[1, 0]
    header["CD2_2"] = cd[1, 1]
    for i, (ctype, n) in enumerate(naxis_extra, start=3):
        header[f"NAXIS{i}"] = n
        header[f"CTYPE{i}"] = ctype
        if ctype == "FREQ":
            header[f"CRVAL{i}"] = freq_hz
            header[f"CRPIX{i}"] = 1.0
            header[f"CDELT{i}"] = 1e6
        elif ctype == "STOKES":
            header[f"CRVAL{i}"] = stokes_crval
            header[f"CRPIX{i}"] = stokes_crpix
            header[f"CDELT{i}"] = stokes_cdelt
        else:
            header[f"CRVAL{i}"] = 1.0
            header[f"CRPIX{i}"] = 1.0
            header[f"CDELT{i}"] = 1.0
    header["RADESYS"] = "ICRS"
    header["RESTFRQ"] = freq_hz  # Keyword fallback when there is no FREQ axis
    header["BUNIT"] = "JY/BEAM"
    header["TELESCOP"] = "TESTSCOPE"
    header["DATE-OBS"] = "2020-01-01T00:00:00"
    if with_beam:
        header["BMAJ"] = 6.0 / 3600.0  # deg
        header["BMIN"] = 5.0 / 3600.0
        header["BPA"] = 10.0
    return header


def image_pixels(shape, seed=0):
    # Deterministic, position-dependent values so cutouts can be compared
    # against direct image slices exactly
    rng = np.random.default_rng(seed)
    return (
        rng.normal(0.0, 1.0, size=shape)
        + np.arange(np.prod(shape)).reshape(shape) * 1e-3
    ).astype(np.float32)


def write_fits(path, data, header):
    fits.PrimaryHDU(data=data, header=header).writeto(path, overwrite=True)
    return Path(path)


def make_continuum_fits(
    path, shape_yx=(64, 64), ra=IMAGE_RA, dec=IMAGE_DEC, seed=0, **kwargs
):
    # Plain 2D continuum image
    header = make_header(shape_yx=shape_yx, ra=ra, dec=dec, **kwargs)
    data = image_pixels(shape_yx, seed=seed)
    return write_fits(path, data, header), data


def make_continuum_4d_fits(
    path,
    shape_yx=(64, 64),
    ra=IMAGE_RA,
    dec=IMAGE_DEC,
    seed=0,
    stokes_first=True,
    n_stokes=1,
    **kwargs,
):
    # 4D (stokes, freq, y, x) image with degenerate freq axis - still continuum
    ny, nx = shape_yx
    if stokes_first:
        # FITS axes: RA, DEC, FREQ, STOKES -> numpy (stokes, freq, y, x)
        naxis_extra = [("FREQ", 1), ("STOKES", n_stokes)]
        data = image_pixels((n_stokes, 1, ny, nx), seed=seed)
    else:
        # FITS axes: RA, DEC, STOKES, FREQ -> numpy (freq, stokes, y, x)
        naxis_extra = [("STOKES", n_stokes), ("FREQ", 1)]
        data = image_pixels((1, n_stokes, ny, nx), seed=seed)
    header = make_header(
        shape_yx=shape_yx, ra=ra, dec=dec, naxis_extra=naxis_extra, **kwargs
    )
    return write_fits(path, data, header), data


def make_spectral_fits(
    path, shape_yx=(48, 48), n_chan=3, ra=IMAGE_RA, dec=IMAGE_DEC, seed=0, **kwargs
):
    # (stokes, freq, y, x) cube with a non-degenerate frequency axis
    ny, nx = shape_yx
    naxis_extra = [("FREQ", n_chan), ("STOKES", 1)]
    data = image_pixels((1, n_chan, ny, nx), seed=seed)
    header = make_header(
        shape_yx=shape_yx, ra=ra, dec=dec, naxis_extra=naxis_extra, **kwargs
    )
    return write_fits(path, data, header), data


# Test catalogues hold a handful of sources, far below what the derived bucket
# count needs, so a fixture names the bucketing instead of taking the default.
# Sorting on the identifier is exact rather than binned, so the store order is
# the catalogue's wherever the identifiers are in order, and a bucket count
# above the source count leaves one source to a bucket, which is what lets a
# fixture say which split a source falls in.
TEST_BUCKET_BY = ["id"]
TEST_N_BUCKETS = 64


def build_test_dataset_store(survey_stores, catalogue, cutout_size, out, **kwargs):
    """Build a dataset store bucketed one source to a bucket, in identifier order.

    Parameters
    ----------
    survey_stores : list of pathlib.Path
        Survey stores to extract from.
    catalogue : pathlib.Path or pandas.DataFrame
        Source catalogue.
    cutout_size : float
        Largest square to serve, in arcsec.
    out : pathlib.Path
        Destination.
    **kwargs
        Passed on, and any of the bucketing arguments override the defaults.

    Returns
    -------
    zarr.Group
    """
    from survey2zarray.stores.dataset_store.build import build_dataset_store

    kwargs.setdefault("bucket_by", TEST_BUCKET_BY)
    kwargs.setdefault("n_buckets", TEST_N_BUCKETS)
    return build_dataset_store(survey_stores, catalogue, cutout_size, out, **kwargs)


def split_config(train=1.0, val=0.0, test=0.0, seed=0):
    """Return a splits config section.

    Parameters
    ----------
    train, val, test : float
        Share of the served sources each split takes.
    seed : int, default 0
        Seed for the rotation the cut points start from.

    Returns
    -------
    dict
    """
    return {"train": train, "val": val, "test": test, "seed": seed}


@pytest.fixture
def continuum_fits_dir(tmp_path):
    # Two overlapping-footprint continuum images of different sizes plus a 4D
    # degenerate-axis variant; centres offset so each covers distinct sources
    fits_dir = tmp_path / "fits"
    fits_dir.mkdir()
    _, data_a = make_continuum_fits(
        fits_dir / "image_a.fits", shape_yx=(64, 64), ra=IMAGE_RA, dec=IMAGE_DEC, seed=1
    )
    _, data_b = make_continuum_fits(
        fits_dir / "image_b.fits",
        shape_yx=(48, 56),
        ra=IMAGE_RA + 0.02,
        dec=IMAGE_DEC,
        seed=2,
    )
    _, data_c = make_continuum_4d_fits(
        fits_dir / "image_c.fits",
        shape_yx=(64, 64),
        ra=IMAGE_RA,
        dec=IMAGE_DEC + 0.02,
        seed=3,
    )
    return fits_dir, {"image_a": data_a, "image_b": data_b, "image_c": data_c}


@pytest.fixture
def spectral_fits_dir(tmp_path):
    fits_dir = tmp_path / "fits_spectral"
    fits_dir.mkdir()
    _, data_a = make_spectral_fits(
        fits_dir / "cube_a.fits",
        shape_yx=(48, 48),
        n_chan=3,
        ra=IMAGE_RA,
        dec=IMAGE_DEC,
        seed=4,
    )
    _, data_b = make_spectral_fits(
        fits_dir / "cube_b.fits",
        shape_yx=(40, 40),
        n_chan=3,
        ra=IMAGE_RA + 0.015,
        dec=IMAGE_DEC,
        seed=5,
    )
    return fits_dir, {"cube_a": data_a, "cube_b": data_b}
