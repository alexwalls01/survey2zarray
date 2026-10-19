"""Provide fsspec byte maps for FITS pixel data.

Record each image's pixel offset and layout, then read FITS byte ranges through
fsspec. Plan row-major regions arithmetically and fetch whole-row spans to limit
request counts.
"""

import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlparse

from astropy.io import fits

from survey2zarray.utils.logging import get_logger

logger = get_logger(__name__)

BITPIX_DTYPES: dict[int, str] = {
    8: "|u1",
    16: ">i2",
    32: ">i4",
    64: ">i8",
    -32: ">f4",
    -64: ">f8",
}
# Bound request memory while keeping ordinary cutouts within one request.
MAX_REQUEST_BYTES: int = 32_000_000

# Cap concurrent coroutines to protect shared archives.
REMOTE_CONCURRENCY: int = 16


COMPRESSED_SUFFIXES: tuple[str, ...] = (".gz", ".bz2", ".xz", ".zst", ".z")

SIZE_PROBE_TIMEOUT: float = 60.0

# Fetch enough leading bytes to cover typical FITS headers in one request.
HEADER_BYTES: int = 1 << 16

REFERENCE_MAP_PATH: str = "data/references.json"


def write_json_atomically(path, payload) -> None:
    """Replace a JSON file in one step.

    Write and flush a neighbouring temporary file before an atomic rename. The
    target always contains one complete version.

    Parameters
    ----------
    path
        Destination.
    payload
        JSON-serialisable content.
    """
    import os
    from pathlib import Path as _Path

    path = _Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class UnreferenceableFITSError(Exception):
    """Represent pixel data unavailable through direct byte ranges."""


def raw_array_spec(header: fits.Header) -> tuple[tuple[int, ...], str]:
    """Return the on-disk array shape and dtype of an image HDU.

    Reject ``BSCALE`` and ``BZERO`` because byte maps expose raw stored values.

    Parameters
    ----------
    header
        FITS header.

    Returns
    -------
    shape
        Shape in numpy axis order, degenerate axes included. This is the array
        as it is laid out in the file, before any Stokes selection.
    dtype
        Big-endian numpy dtype string.

    Raises
    ------
    UnreferenceableFITSError
        Where the pixels are scaled or the bit depth is unrecognised.
    """
    bitpix = int(header.get("BITPIX", 0))
    if bitpix not in BITPIX_DTYPES:
        raise UnreferenceableFITSError(f"Unsupported BITPIX {bitpix}.")
    scale = header.get("BSCALE", 1)
    zero = header.get("BZERO", 0)
    if scale != 1 or zero != 0:
        raise UnreferenceableFITSError(
            f"Pixels use BSCALE={scale} and BZERO={zero}. A byte map cannot apply "
            f"this scaling. Build this survey in copy mode."
        )
    naxis = int(header.get("NAXIS", 0))
    if naxis < 2:
        raise UnreferenceableFITSError(f"Expected at least 2 axes, found {naxis}.")
    shape = tuple(int(header.get(f"NAXIS{i}", 1)) for i in range(naxis, 0, -1))
    return shape, BITPIX_DTYPES[bitpix]


def data_offset(hdu: fits.PrimaryHDU | fits.ImageHDU) -> int:
    """Return the byte offset of an HDU's pixel data within its file.

    Parameters
    ----------
    hdu
        Image HDU.

    Returns
    -------
    result
        Byte offset of the pixel array.

    Raises
    ------
    UnreferenceableFITSError
        Where the offset cannot be determined, which happens when the HDU did
        not come from a file.
    """
    try:
        location = hdu.fileinfo()["datLoc"]
    except (AttributeError, TypeError, KeyError):
        location = None
    if location is None:
        raise UnreferenceableFITSError(
            "Could not locate the pixel data within the file."
        )
    return int(location)


def compression_suffix(location: str) -> str | None:
    """Return the compression suffix of a location, if it has one.

    Parameters
    ----------
    location
        Path or URL of a file.

    Returns
    -------
    result
        The suffix, lower-cased, or None where the file is not compressed.
    """
    name = urlparse(str(location)).path.rsplit("/", 1)[-1].lower()
    for suffix in COMPRESSED_SUFFIXES:
        if name.endswith(suffix):
            return suffix
    return None


def check_referenceable(location: str) -> None:
    """Refuse a file whose pixels cannot be addressed where they lie.

    Compressed streams provide no fixed byte offset for each pixel row. Require
    directly addressable stored ranges.

    Parameters
    ----------
    location
        Path or URL of the FITS file.

    Raises
    ------
    UnreferenceableFITSError
        Where the file is compressed.
    """
    suffix = compression_suffix(location)
    if suffix is not None:
        raise UnreferenceableFITSError(
            f"{location} uses {suffix} compression. A byte map requires directly "
            f"addressable stored ranges. Build this survey in copy mode."
        )


@dataclass(frozen=True)
class ReadPlan:
    """Store byte ranges in reassembly order.

    Attributes
    ----------
    fetch
        Half-open byte ranges to ask the server for, in assembly order.
    fetched_bytes
        Bytes the requests will move. A range is whole rows of one wanted
        plane, so this is also what the caller asked for and nothing fetched
        is discarded.
    """

    fetch: list[tuple[int, int]]
    fetched_bytes: int


def region_ranges(
    offset: int,
    shape: Sequence[int],
    itemsize: int,
    planes: Sequence[int],
    y0: int,
    y1: int,
    max_bytes: int = MAX_REQUEST_BYTES,
) -> ReadPlan:
    """Plan the read of a rectangular region of the given planes.

    FITS stores each run of whole rows contiguously. Plan one or more bounded
    ranges per selected plane and slice columns after transfer. Whole-row reads
    limit archive request counts.

    Each plane is planned separately, so the planes between two that are wanted
    are never read through. Those are the polarisations a Stokes I read exists
    to skip, and reading through them would transfer four times the bytes.

    Compute range boundaries directly from shape, offset, and item size.

    Parameters
    ----------
    offset
        Byte offset of the pixel data within the file.
    shape
        On-disk array shape in numpy axis order.
    itemsize
        Bytes per pixel.
    planes
        Flat plane numbers, as returned by `stokes_i_planes`.
    y0, y1
        Row bounds, half-open.
    max_bytes
        Most any one request may span.

    Returns
    -------
    result
        Byte-range read plan.
    """
    ny, nx = int(shape[-2]), int(shape[-1])
    row_bytes = nx * int(itemsize)
    plane_bytes = ny * row_bytes
    y0, y1 = int(y0), int(y1)
    rows = max(0, y1 - y0)

    # Whole-row pieces preserve pixel boundaries above the request bound.
    per_piece = max(1, int(max_bytes) // row_bytes) if row_bytes else 1

    # Separate plane ranges skip unselected polarisations.
    fetch: list[tuple[int, int]] = []
    for plane in planes:
        base = int(offset) + int(plane) * plane_bytes + y0 * row_bytes
        for first in range(0, rows, per_piece):
            length = min(per_piece, rows - first) * row_bytes
            start = base + first * row_bytes
            fetch.append((start, start + length))
    return ReadPlan(fetch, sum(end - start for start, end in fetch))


def is_remote(location: str) -> bool:
    """Return whether a location uses remote storage.

    Determine storage location independently of survey access mode.

    Parameters
    ----------
    location
        Path or URL of a file.

    Returns
    -------
    result
        Whether the location uses remote storage.
    """
    return storage_protocol(location) != "file"


def options_for(protocol: str, storage_options: dict[str, dict] | None) -> dict:
    """Return the filesystem options that apply to one protocol.

    Select protocol-specific options because one survey may span filesystems.

    Parameters
    ----------
    protocol
        Storage protocol, as `storage_protocol` reports it.
    storage_options
        Options keyed by protocol, as ``{'s3': {'anon': True}}``.

    Returns
    -------
    result
        A copy, so what uses the options cannot mutate the caller's mapping.
    """
    if not storage_options:
        return {}
    return dict(storage_options.get(str(protocol), {}))


def probe_size(location: str) -> int | None:
    """Ask a server how large a file is, with a one-byte range request.

    ``Content-Range: bytes 0-0/<total>`` states the total unambiguously, which
    ``Content-Length`` beside a ``Content-Encoding`` header does not. fsspec
    discards a length that might describe an encoded body, and some archives
    label their responses with codings that are not codings at all.

    Parameters
    ----------
    location
        HTTP or HTTPS URL.

    Returns
    -------
    result
        Size in bytes, or None where the server did not say.
    """
    from urllib.request import Request, urlopen

    request = Request(
        str(location),
        method="GET",
        headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
    )
    try:
        with urlopen(request, timeout=SIZE_PROBE_TIMEOUT) as response:
            content_range = response.headers.get("Content-Range")
    except Exception:
        return None
    if not content_range or "/" not in content_range:
        return None
    total = content_range.rsplit("/", 1)[-1].strip()
    return int(total) if total.isdigit() else None


def remote_header_buffer(
    location: str,
    nbytes: int | None = None,
    storage_options: dict[str, dict] | None = None,
):
    """Fetch the front of a served file, where its headers are.

    Fetch the leading bytes in one range request to minimise archive latency.

    Parameters
    ----------
    location
        Path or URL of the file.
    nbytes
        Bytes to fetch. None resolves to the current `HEADER_BYTES` value.
    storage_options
        Filesystem options keyed by protocol, as `options_for` takes them.

    Returns
    -------
    result
        The bytes fetched, ready to be parsed.
    """
    import fsspec

    nbytes = HEADER_BYTES if nbytes is None else nbytes
    protocol = storage_protocol(location)
    filesystem = fsspec.filesystem(protocol, **options_for(protocol, storage_options))
    return io.BytesIO(filesystem.cat_file(location, start=0, end=int(nbytes)))


def open_remote(location: str, storage_options: dict[str, dict] | None = None):
    """Open a file through fsspec, seekable in every case the size can be established.

    fsspec falls back to a streaming, unseekable file when it cannot determine
    a size, and a header cannot be read from one of those. Where the filesystem
    cannot say, the server is asked directly and the size passed in.

    Parameters
    ----------
    location
        Path or URL of the file.
    storage_options
        Filesystem options keyed by protocol, as `options_for` takes them.

    Returns
    -------
    result
        Open for reading in binary.
    """
    import fsspec

    protocol = storage_protocol(location)
    filesystem = fsspec.filesystem(protocol, **options_for(protocol, storage_options))
    try:
        size = filesystem.info(location).get("size")
    except Exception:
        size = None
    if size is None and protocol in ("http", "https"):
        size = probe_size(location)
        if size is not None:
            logger.debug(f"A range request reported {size} bytes for {location}.")
    if size is None:
        return filesystem.open(location)
    return filesystem.open(location, size=int(size))


def storage_protocol(url: str) -> str:
    """Return the fsspec protocol a URL should be opened with.

    Parameters
    ----------
    url
        Path or URL.

    Returns
    -------
    result
        ``'file'`` for a bare path, otherwise the URL's scheme.
    """
    scheme = urlparse(str(url)).scheme
    # Treat parsed one-character schemes as Windows drive letters.
    return scheme if len(scheme) > 1 else "file"
