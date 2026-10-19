"""Read FITS images, extracting header metadata and selecting Stokes I."""

import re
import sys
import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from astropy.time import Time
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales

from survey2zarray.stores.survey_store.references import (
    UnreferenceableFITSError,
    check_referenceable,
    data_offset,
    is_remote,
    open_remote,
    raw_array_spec,
    remote_header_buffer,
)
from survey2zarray.utils.logging import get_logger

logger = get_logger(__name__)


class InvalidFITSError(Exception):
    """Represent a FITS conversion failure."""


class FITSOpenError(InvalidFITSError):
    """Represent a FITS open failure.

    Archive load can cause transient open failures.
    """


class IncompatibleTilesError(Exception):
    """Represent images incompatible with one survey store."""


def resolve_observation_type(header: fits.Header) -> str | None:
    """Classify an image as continuum or spectral from its header.

    A one-channel cube and a continuum image are shape-identical, so only the
    header distinguishes them.

    Parameters
    ----------
    header
        Header of the image HDU.

    Returns
    -------
    result
        ``'spectral'`` if a non-degenerate frequency axis is present,
        ``'continuum'`` otherwise, or None where the type is undeterminable.
    """
    naxis = header.get("NAXIS", 0)
    if naxis < 2:
        return None
    for i in range(1, naxis + 1):
        if "FREQ" in str(header.get(f"CTYPE{i}", "")).upper():
            if header.get(f"NAXIS{i}", 1) > 1:
                return "spectral"
    return "continuum"


def resolve_telescope(header: fits.Header) -> str:
    """Return the telescope or observatory name, or an empty string.

    Parameters
    ----------
    header
        FITS header.

    Returns
    -------
    result
        Telescope name, or None when absent.
    """
    val = header.get("TELESCOP") or header.get("OBSERVAT")
    if val:
        return str(val)
    return ""


def resolve_date(header: fits.Header) -> float:
    """Return the observation date as a Modified Julian Date.

    Parameters
    ----------
    header
        FITS header.

    Returns
    -------
    result
        MJD, or NaN where no date could be read.
    """
    val = header.get("DATE-OBS") or header.get("DATE_OBS")
    if val:
        for fmt in (None, "fits"):
            try:
                return float(Time(val, format=fmt).mjd)
            except Exception:
                continue
    mjd = header.get("MJD-OBS")
    try:
        return float(mjd) if mjd is not None else np.nan
    except Exception:
        return np.nan


def resolve_frequency(header: fits.Header) -> float:
    """Return the observing frequency in Hz.

    Prefers the reference value of the frequency axis, falling back to common
    frequency keywords.

    Parameters
    ----------
    header
        FITS header.

    Returns
    -------
    result
        Frequency in Hz, or NaN where unavailable.
    """
    naxis = header.get("NAXIS", 0)
    for i in range(1, naxis + 1):
        ctype = str(header.get(f"CTYPE{i}", "")).upper()
        if "FREQ" in ctype:
            freq = header.get(f"CRVAL{i}")
            if freq is not None:
                return float(freq)
    for key in ["RESTFREQ", "RESTFRQ", "CENTFREQ", "FREQ"]:
        freq = header.get(key)
        if freq is not None:
            return float(freq)
    return np.nan


def resolve_bandwidth(header: fits.Header) -> float:
    """Return the channel bandwidth in Hz.

    Prefers the increment of the frequency axis, falling back to common
    bandwidth keywords.

    Parameters
    ----------
    header
        FITS header.

    Returns
    -------
    result
        Bandwidth in Hz, or NaN where unavailable.
    """
    naxis = header.get("NAXIS", 0)
    for i in range(1, naxis + 1):
        ctype = str(header.get(f"CTYPE{i}", "")).upper()
        if "FREQ" in ctype:
            bandwidth = header.get(f"CDELT{i}")
            if bandwidth is not None:
                return float(bandwidth)
    for key in ["BANDWID", "BANDWIDT", "CHAN_BW", "FREQRES"]:
        val = header.get(key)
        if val is not None:
            return float(val)
    return np.nan


def resolve_beam(header: fits.Header) -> dict[str, float]:
    """Return the restoring beam, parsing HISTORY when keywords are absent.

    Parameters
    ----------
    header
        FITS header.

    Returns
    -------
    result
        Keys ``bmaj``, ``bmin``, and ``bpa``, all in degrees, each NaN where
        unavailable.
    """
    if all(k in header for k in ["BMAJ", "BMIN"]):
        return {
            "bmaj": float(header["BMAJ"]),
            "bmin": float(header["BMIN"]),
            "bpa": float(header.get("BPA", np.nan)),
        }
    history = header.get("HISTORY", [])
    # FITS readers may expose HISTORY as one string or a sequence.
    if isinstance(history, str):
        history = [history]
    for line in history:
        if "BMAJ" in line and "BMIN" in line:
            bmaj = re.search(r"BMAJ=\s*([-\d.Ee+]+)", line)
            bmin = re.search(r"BMIN=\s*([-\d.Ee+]+)", line)
            bpa = re.search(r"BPA=\s*([-\d.Ee+]+)", line)
            if bmaj and bmin:
                return {
                    "bmaj": float(bmaj.group(1)),
                    "bmin": float(bmin.group(1)),
                    "bpa": float(bpa.group(1)) if bpa else np.nan,
                }
    return {
        "bmaj": np.nan,
        "bmin": np.nan,
        "bpa": np.nan,
    }


def resolve_bunit(header: fits.Header) -> str:
    """Return the pixel brightness unit, or an empty string.

    Parameters
    ----------
    header
        FITS header.

    Returns
    -------
    result
        Brightness unit, or None when absent.
    """
    val = header.get("BUNIT")
    if val:
        return str(val).strip()
    return ""


def resolve_equinox(header: fits.Header) -> float:
    """Return the coordinate equinox as a year.

    Parameters
    ----------
    header
        FITS header.

    Returns
    -------
    result
        Equinox year, or NaN where unavailable.
    """
    val = header.get("EQUINOX")
    if val is None:
        # Accept EPOCH when EQUINOX is absent.
        val = header.get("EPOCH")
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            return np.nan
    return np.nan


def resolve_radesys(header: fits.Header) -> str:
    """Return the celestial reference frame, inferring it when unstated.

    Parameters
    ----------
    header
        FITS header.

    Returns
    -------
    result
        Frame name, e.g. ``'ICRS'``, ``'FK5'`` or ``'FK4'``.
    """
    # Accept RADECSYS as an alternate frame key.
    val = header.get("RADESYS") or header.get("RADECSYS")
    if val:
        return str(val).strip().upper()
    equinox = resolve_equinox(header)
    if not np.isnan(equinox):
        return "FK5" if equinox >= 1984.0 else "FK4"
    # FITS defaults a missing EQUINOX to ICRS.
    return "ICRS"


def resolve_wcs_params(
    wcs: WCS,
    path: str,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[tuple[float, ...], ...], str]:
    """Extract the numeric parameters needed to rebuild a WCS.

    Parameters
    ----------
    wcs
        World coordinate system.
    path
        Source location, used in error messages.

    Returns
    -------
    crval
        World coordinates at the reference pixel.
    crpix
        Reference pixel, one-based as per the FITS convention.
    cd
        Linear transformation matrix for all stored axes.
    ctype
        Axis types joined by ``'|'``.

    Raises
    ------
    InvalidFITSError
        Where the parameters cannot be read, or the WCS contains a distortion
        or projection parameter the numeric store schema cannot reconstruct.
    """
    try:
        crval = tuple(map(float, wcs.wcs.crval))
        crpix = tuple(map(float, wcs.wcs.crpix))
        cd = tuple(tuple(map(float, row)) for row in wcs.pixel_scale_matrix)
        ctype = "|".join(str(c) for c in wcs.wcs.ctype)
    except Exception as error:
        raise InvalidFITSError(
            f"Could not extract WCS parameters for {path}."
        ) from error

    unsupported = []
    if wcs.has_distortion:
        unsupported.append("SIP or lookup-table distortion")
    if wcs.wcs.get_pv():
        unsupported.append("PV projection parameters")
    if wcs.wcs.get_ps():
        unsupported.append("PS projection parameters")

    # Reject pole overrides outside the compact WCS schema.
    candidate = WCS(naxis=len(crval))
    candidate.wcs.crval = crval
    candidate.wcs.crpix = crpix
    candidate.wcs.cd = cd
    candidate.wcs.ctype = list(ctype.split("|"))
    candidate.wcs.set()
    wcs.wcs.set()
    for name in ("lonpole", "latpole"):
        actual = float(getattr(wcs.wcs, name))
        rebuilt = float(getattr(candidate.wcs, name))
        if np.isfinite(actual) and not np.isclose(actual, rebuilt, atol=1e-12):
            unsupported.append(name.upper())
    original_units = [str(unit) for unit in wcs.wcs.cunit]
    rebuilt_units = [str(unit) for unit in candidate.wcs.cunit]
    if original_units != rebuilt_units:
        unsupported.append("non-standard axis units")

    if unsupported:
        raise InvalidFITSError(
            f"WCS for {path} uses {', '.join(unsupported)}, which the store's "
            "CRVAL/CRPIX/CD/CTYPE schema cannot reconstruct exactly."
        )
    return crval, crpix, cd, ctype


def resolve_pixel_scale(wcs: WCS, path: str) -> tuple[float, float]:
    """Return the celestial pixel scale in arcsec per pixel.

    Parameters
    ----------
    wcs
        Celestial WCS.
    path
        Source location, used in error messages.

    Returns
    -------
    result
        Scale as ``(x, y)``, i.e. (column, row).

    Raises
    ------
    InvalidFITSError
        Where the scale cannot be read, or the WCS is not two-dimensional.
    """
    try:
        scales = proj_plane_pixel_scales(wcs)
    except Exception as error:
        raise InvalidFITSError(f"Could not extract pixel scale for {path}.") from error
    if len(scales) != 2:
        raise InvalidFITSError(f"Could not extract pixel scale for {path}.")
    return float(scales[0]) * 3600.0, float(scales[1]) * 3600.0


def resolve_pixel_centre(
    wcs: WCS, path: str, image_shape: tuple[int, int]
) -> tuple[float, float]:
    """Return the sky coordinates of the image centre.

    Parameters
    ----------
    wcs
        Celestial WCS.
    path
        Source location, used in error messages.
    image_shape
        Image shape as ``(H, W)``, i.e. (row, column).

    Returns
    -------
    result
        Centre as ``(ra, dec)`` in degrees.

    Raises
    ------
    InvalidFITSError
        Where the centre cannot be computed.
    """
    try:
        ny, nx = image_shape
        ra, dec = wcs.all_pix2world((nx - 1) / 2.0, (ny - 1) / 2.0, 0)
        return float(ra), float(dec)
    except Exception as error:
        raise InvalidFITSError(f"Could not extract image centre for {path}.") from error


def get_image_hdu(hdul: fits.HDUList) -> fits.PrimaryHDU | fits.ImageHDU:
    """Return the first HDU holding a two-dimensional or larger image.

    Selection reads headers only, so remote opens never fetch pixel bytes.

    Parameters
    ----------
    hdul
        Open FITS HDU list.

    Returns
    -------
    result
        First image HDU with at least two dimensions.

    Raises
    ------
    ValueError
        Where no image HDU is present.
    """
    for hdu in hdul:
        if hdu.is_image and hdu.header.get("NAXIS", 0) >= 2:
            return hdu
    raise ValueError("No image HDU found.")


def locate_stokes_axis(header: fits.Header) -> int | None:
    """Return the FITS axis number of the STOKES axis, or None if absent.

    Parameters
    ----------
    header
        FITS header.

    Returns
    -------
    result
        One-based FITS axis number.
    """
    naxis = header.get("NAXIS", 0)
    for i in range(1, naxis + 1):
        if str(header.get(f"CTYPE{i}", "")).upper().startswith("STOKES"):
            return i
    return None


def stokes_i_index(header: fits.Header, stokes_axis: int, n: int) -> int:
    """Return the index along the STOKES axis holding Stokes I.

    Stokes values follow the FITS convention I=1, Q=2, U=3, V=4, evaluated as
    ``CRVAL + (k + 1 - CRPIX) * CDELT``. I is not necessarily index 0.

    Parameters
    ----------
    header
        FITS header.
    stokes_axis
        One-based FITS axis number of the STOKES axis.
    n
        Length of that axis.

    Returns
    -------
    result
        Zero-based index of the Stokes I plane.

    Raises
    ------
    InvalidFITSError
        Where Stokes I is not among the stored planes.
    """
    crval = header.get(f"CRVAL{stokes_axis}", 1.0)
    crpix = header.get(f"CRPIX{stokes_axis}", 1.0)
    cdelt = header.get(f"CDELT{stokes_axis}", 1.0)
    stokes_values = [crval + (k + 1 - crpix) * cdelt for k in range(n)]
    i_indices = [k for k, v in enumerate(stokes_values) if round(v) == 1]
    if not i_indices:
        raise InvalidFITSError(
            f"Stokes I is absent. STOKES axis values are {stokes_values}."
        )
    return i_indices[0]


def stokes_i_planes(
    shape: Sequence[int], header: fits.Header
) -> tuple[list[int], tuple[int, ...]]:
    """Return which planes of an image hold Stokes I, and the shape they make.

    Treat axes before the final celestial pair as a C-order image stack. Select
    Stokes I planes during read planning to skip other polarisations.

    Parameters
    ----------
    shape
        On-disk array shape in numpy axis order.
    header
        FITS header.

    Returns
    -------
    planes
        Flat plane numbers in C order, ascending, holding Stokes I.
    shape
        Shape those planes form once stacked, before the two celestial axes and
        before degenerate axes are squeezed out.

    Raises
    ------
    InvalidFITSError
        Where a STOKES axis exists but holds no Stokes I plane.
    """
    shape = tuple(int(n) for n in shape)
    leading = shape[:-2]
    if not leading:
        return [0], ()

    stokes_axis = locate_stokes_axis(header)
    if stokes_axis is None:
        return list(range(int(np.prod(leading)))), leading

    # FITS axis i maps to leading NumPy axis ndim - i.
    numpy_axis = len(shape) - stokes_axis
    if numpy_axis >= len(leading):
        return list(range(int(np.prod(leading)))), leading

    index = stokes_i_index(header, stokes_axis, shape[numpy_axis])
    planes = [
        int(np.ravel_multi_index(combination, leading))
        for combination in np.ndindex(*leading)
        if combination[numpy_axis] == index
    ]
    kept = leading[:numpy_axis] + leading[numpy_axis + 1 :]
    return planes, kept


def extract_stokes_i(data: np.ndarray, header: fits.Header) -> np.ndarray:
    """Reduce a data array to its Stokes I plane or planes.

    Removes only the Stokes axis, keeping any frequency axis, then squeezes
    remaining degenerate axes. Data with no STOKES axis is assumed to be
    Stokes I already. That assumption is worth knowing about, but a survey
    whose headers all omit the axis would report it once per file, so it is
    counted during the metadata pass and reported once for the survey.

    Parameters
    ----------
    data
        Pixel data in numpy axis order.
    header
        FITS header.

    Returns
    -------
    result
        ``(H, W)`` for continuum, ``(n_chan, H, W)`` for spectral.

    Raises
    ------
    InvalidFITSError
        Where a STOKES axis exists but holds no Stokes I plane.
    """
    stokes_axis = locate_stokes_axis(header)

    if stokes_axis is None:
        return np.squeeze(data)

    # FITS axis i maps to NumPy axis ndim - i.
    np_axis = data.ndim - stokes_axis
    i_index = stokes_i_index(header, stokes_axis, data.shape[np_axis])

    sliced = np.take(data, i_index, axis=np_axis)
    return np.squeeze(sliced)


def stokes_i_shape(header: fits.Header) -> tuple[int, ...]:
    """Return the post-extraction data shape, from the header alone.

    Reading no pixel data keeps the metadata pass cheap for remote files.

    Parameters
    ----------
    header
        FITS header.

    Returns
    -------
    result
        Shape in numpy axis order, matching what `extract_stokes_i` produces.

    Raises
    ------
    InvalidFITSError
        Where a STOKES axis exists but holds no Stokes I plane.
    """
    naxis = header.get("NAXIS", 0)
    shape = [header.get(f"NAXIS{i}", 1) for i in range(naxis, 0, -1)]
    stokes_axis = locate_stokes_axis(header)
    if stokes_axis is None:
        pass  # Report implicit Stokes I through the caller's survey-level count.
    else:
        # FITS axis i maps to NumPy axis ndim - i.
        np_axis = naxis - stokes_axis
        stokes_i_index(header, stokes_axis, shape[np_axis])
        del shape[np_axis]
    return tuple(s for s in shape if s > 1)


def header_sources(path: str, storage_options: dict[str, dict] | None = None) -> list:
    """Return ways of opening a file for its header, cheapest first.

    A served file is first tried as a buffer of its opening bytes, which spends
    one request. That holds every header of almost every file, but not of one
    whose headers run past it or sit behind another HDU's data, so a seekable
    open follows it.

    Parameters
    ----------
    path
        Local path or URL of the FITS file.
    storage_options
        Filesystem options keyed by protocol, as `options_for` takes them.

    Returns
    -------
    result
        Each returns something `astropy.io.fits.open` accepts.
    """
    if not is_remote(path):
        return [lambda: path]
    return [
        lambda: remote_header_buffer(path, storage_options=storage_options),
        lambda: open_remote(path, storage_options=storage_options),
    ]


def warning_writer(previous):
    """Wrap a ``showwarning`` so that it keeps clear of any progress bar.

    Use ``tqdm.write`` around active bars. Delegate to ``previous`` for other
    warnings so caller-installed routing remains active.

    Parameters
    ----------
    previous
        Existing ``warnings.showwarning`` for warnings outside progress bars.

    Returns
    -------
    result
        A replacement ``showwarning``.
    """

    def show(message, category, filename, lineno, file=None, line=None):
        """Write one warning while preserving active progress bars.

        Parameters
        ----------
        message
            Warning message.
        category
            Warning category.
        filename
            Source filename.
        lineno
            Source line number.
        file
            Destination stream.
        line
            Source line text.
        """
        try:
            from tqdm import tqdm
        except ImportError:
            bars = ()
        else:
            bars = getattr(tqdm, "_instances", ())
        if not bars:
            previous(message, category, filename, lineno, file, line)
            return
        text = warnings.formatwarning(message, category, filename, lineno, line)
        tqdm.write(text.rstrip("\n"), file=sys.stderr)

    return show


@contextmanager
def quiet_fits_warnings(expect_truncated: bool = False):
    """Silence astropy's complaints about headers for the duration of a block.

    Apply one process-wide warning context around each header pass to avoid
    filter-state races between reader threads. Route unmatched warnings through
    `tqdm` to preserve the progress display.

    Parameters
    ----------
    expect_truncated
        Whether the pass intentionally reads partial headers.

    Yields
    ------
    item
        Control while the warning filters are active.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=VerifyWarning)
        warnings.simplefilter("ignore", category=FITSFixedWarning)
        if expect_truncated:
            warnings.filterwarnings(
                "ignore",
                message="File may have been truncated",
                category=AstropyUserWarning,
            )
        warnings.showwarning = warning_writer(warnings.showwarning)
        yield


def extract_metadata(
    path: str,
    require_referenceable: bool = False,
    storage_options: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Extract all stored metadata for one FITS image.

    Use one extraction path for local and remote inputs. Store the WCS after
    Stokes selection, including celestial axes and the spectral frequency axis.
    Callers wrap concurrent passes in `quiet_fits_warnings` to avoid global
    warning-filter races.

    Parameters
    ----------
    path
        Local path or URL of the FITS file.
    require_referenceable
        Whether this file must be addressable as byte ranges, as a reference
        store needs. Compressed files are refused when it is.
    storage_options
        Filesystem options keyed by protocol, as `options_for` takes them.
        Ignored for a file on disk, which is read directly.

    Returns
    -------
    result
        Metadata fields, keyed by the names used for the stored arrays.

    Raises
    ------
    InvalidFITSError
        Where the file cannot be opened or read, holds no image HDU, has no
        valid WCS, or does not have the dimensionality implied by its type.
    """
    try:
        if require_referenceable:
            # Byte maps require directly addressable, uncompressed pixel ranges.
            try:
                check_referenceable(path)
            except UnreferenceableFITSError as error:
                raise InvalidFITSError(str(error)) from error
        # Fetch opening byte ranges for remote headers before opening full files.
        sources = header_sources(path, storage_options=storage_options)
        header, pixel_offset = None, None
        for index, source in enumerate(sources):
            last = index == len(sources) - 1
            try:
                with fits.open(source(), memmap=False) as hdul:
                    hdu = get_image_hdu(hdul)
                    header = hdu.header.copy()
                    try:
                        pixel_offset = data_offset(hdu)
                    except UnreferenceableFITSError:
                        pixel_offset = None
                break
            # Try each header source because short buffers and missing files
            # share OSError.
            except (TimeoutError, OSError):
                if last:
                    raise
            except Exception as error:
                if last:
                    raise InvalidFITSError(
                        f"Could not read a header from {path}. {error}."
                    ) from error
    except (TimeoutError, OSError) as error:
        # Treat fsspec FSTimeoutError as a transient read failure.
        raise FITSOpenError(f"Failed to open {path}. {error!r}.") from error

    image_shape = stokes_i_shape(header)
    observation_type = resolve_observation_type(header)
    if observation_type == "continuum" and len(image_shape) != 2:
        raise InvalidFITSError(
            f"{path} is not 2D after Stokes extraction (shape {image_shape})."
        )
    if observation_type == "spectral" and len(image_shape) != 3:
        raise InvalidFITSError(
            f"{path} is not 3D after Stokes extraction (shape {image_shape})."
        )
    n_chan = int(image_shape[0]) if observation_type == "spectral" else None
    celestial_shape = image_shape[-2:]
    try:
        wcs_full = WCS(header)
        # Remove the selected Stokes axis from the stored WCS.
        if observation_type == "spectral":
            wcs = wcs_full.sub(["longitude", "latitude", "spectral"])
        else:
            wcs = wcs_full.celestial
    except Exception as error:
        raise InvalidFITSError(f"Could not construct valid WCS for {path}.") from error
    try:
        raw_shape, raw_dtype = raw_array_spec(header)
    except UnreferenceableFITSError as e:
        logger.debug(f"{path} cannot be referenced. {e}.")
        raw_shape, raw_dtype, pixel_offset = None, None, None
    crval, crpix, cd, ctype = resolve_wcs_params(wcs, path)
    pixel_scale = resolve_pixel_scale(wcs.celestial, path)
    pixel_centre = resolve_pixel_centre(wcs.celestial, path, celestial_shape)
    telescope = resolve_telescope(header)
    date = resolve_date(header)
    frequency = resolve_frequency(header)
    bandwidth = resolve_bandwidth(header)
    beam = resolve_beam(header)
    bmaj = beam["bmaj"]
    bmin = beam["bmin"]
    bpa = beam["bpa"]
    metadata = {
        "crval": crval,
        "crpix": crpix,
        "cd": cd,
        "ctype": ctype,
        "image_location": str(path),
        "pixel_shape": celestial_shape,
        "n_chan": n_chan,
        "pixel_centre": pixel_centre,
        "pixel_scale": pixel_scale,
        "telescope": telescope,
        "date": date,
        "frequency": frequency,
        "bandwidth": bandwidth,
        "bmaj": bmaj,
        "bmin": bmin,
        "bpa": bpa,
        "bunit": resolve_bunit(header),
        "equinox": resolve_equinox(header),
        "radesys": resolve_radesys(header),
        "header": header.tostring(),
        "observation_type": observation_type,
        # Reference mode records the on-disk array before Stokes selection.
        "data_offset": pixel_offset,
        "raw_shape": raw_shape,
        "raw_dtype": raw_dtype,
        "stokes_assumed": locate_stokes_axis(header) is None,
    }
    return metadata


def check_compatibility(
    image_catalogue: list[dict[str, Any]],
) -> tuple[str | None, int | None]:
    """Check that a set of images can share one store, or raise.

    Height and width may differ between images; padding handles that. Every
    problem is reported at once, before anything is written, so a failed run
    leaves no partial store behind.

    Parameters
    ----------
    image_catalogue
        Metadata dicts as returned by `extract_metadata`.

    Returns
    -------
    observation_type
        The type shared by every image.
    n_chan
        Channel count shared by every image, None for continuum.

    Raises
    ------
    IncompatibleTilesError
        Where types are mixed or undeterminable, or spectral channel counts
        differ. The message names the offending images.
    """
    problems = []
    types = {}
    for m in image_catalogue:
        types.setdefault(m["observation_type"], []).append(m["image_location"])
    if None in types:
        problems.append(
            f"Header metadata cannot determine observation_type for {types[None]}."
        )
    resolved_types = sorted(t for t in types if t is not None)
    if len(resolved_types) > 1:
        listing = ". ".join(f"{t} applies to {types[t]}" for t in resolved_types)
        problems.append(f"All images must share one observation_type. Found {listing}.")
    observation_type = resolved_types[0] if len(resolved_types) == 1 else None
    n_chan = None
    if observation_type == "spectral":
        n_chans = {}
        for m in image_catalogue:
            n_chans.setdefault(m["n_chan"], []).append(m["image_location"])
        if len(n_chans) > 1:
            listing = ". ".join(
                f"n_chan={k} applies to {v}" for k, v in sorted(n_chans.items())
            )
            problems.append(f"All spectral images must share n_chan. Found {listing}.")
        else:
            n_chan = next(iter(n_chans))
    if problems:
        raise IncompatibleTilesError("\n".join(problems))
    return observation_type, n_chan
