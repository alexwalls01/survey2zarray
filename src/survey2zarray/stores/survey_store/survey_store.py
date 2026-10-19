"""Provide survey store reads for both access modes."""

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from astropy.io import fits
from astropy.wcs import WCS

from survey2zarray.stores.store import open_store
from survey2zarray.stores.survey_store.fits import stokes_i_planes
from survey2zarray.stores.survey_store.references import (
    MAX_REQUEST_BYTES,
    REFERENCE_MAP_PATH,
    REMOTE_CONCURRENCY,
    options_for,
    region_ranges,
    storage_protocol,
)
from survey2zarray.stores.wcs import reconstruct_wcs
from survey2zarray.utils.logging import get_logger

logger = get_logger(__name__)


# Bound archive requests so stalled connections produce visible failures.
REQUEST_TIMEOUT: float = 60.0
CONNECT_TIMEOUT: float = 15.0


@dataclass(frozen=True)
class ImageRead:
    """Store immutable inputs for one referenced image read.

    Cache byte-map and header-derived values once per image.

    Attributes
    ----------
    location
        Path or URL of the FITS file.
    offset
        Byte offset of the pixel data within it.
    shape
        On-disk array shape in numpy axis order.
    dtype
        On-disk element type.
    planes
        Flat plane numbers holding Stokes I.
    kept
        Shape those planes form once stacked, before the celestial axes.
    """

    location: str
    offset: int
    shape: tuple[int, ...]
    dtype: np.dtype
    planes: tuple[int, ...]
    kept: tuple[int, ...]


class SurveyStore:
    """Provide read access to survey images and metadata.

    Metadata arrays are loaded once on construction; pixels are read per
    request through `get_cutout_region`.

    The two access modes differ only in how pixel bytes are read. Structure,
    attributes, and metadata are identical.

    Parameters
    ----------
    root
        Opened root group of the store.
    path
        Location of the store.

    Attributes
    ----------
    observation_type
        Read from the stored attribute, never inferred from array shape.
    access_mode
        Whether pixels live in the store or in the source files.
    n_images
        Number of images.
    n_chan
        Channel count for spectral stores.

    Raises
    ------
    RuntimeError
        Where the store was not completely written.
    """

    def __init__(
        self,
        root: zarr.Group,
        path: str | Path,
        max_request_bytes: int = MAX_REQUEST_BYTES,
        concurrency: int = REMOTE_CONCURRENCY,
        request_timeout: float | None = REQUEST_TIMEOUT,
        connect_timeout: float = CONNECT_TIMEOUT,
        storage_options: dict[str, dict] | None = None,
    ) -> None:
        self.root = root
        self.path = Path(path)
        if not root.attrs.get("complete", False):
            raise RuntimeError(
                f"Conversion of {path} is incomplete. Reconvert it before use."
            )
        # Build metadata fixes the observation type independently of array shape.
        self.observation_type = root.attrs["observation_type"]
        self.access_mode = root.attrs["access_mode"]
        # Cache byte maps and filesystems once under the shared reader lock.
        self._lock = threading.Lock()
        self._reference_map = None
        self._filesystems = {}
        # Cache immutable referenced-read specifications per image.
        self._image_reads: dict[int, ImageRead] = {}
        self.max_request_bytes = int(max_request_bytes)
        self.concurrency = max(1, int(concurrency))
        self.request_timeout = request_timeout
        self.connect_timeout = float(connect_timeout)
        # Caller options override stored archive options and supply credentials.
        self.stored_storage_options = dict(root.attrs.get("storage_options", {}))
        self.storage_options = dict(storage_options or {})
        self.n_images = root.attrs["n_images"]
        self.n_chan = root.attrs.get("n_chan")

        # Single metadata chunks require one read per array.
        metadata = root["metadata"]
        self.crval = metadata["crval"][:]
        self.crpix = metadata["crpix"][:]
        self.cd = metadata["cd"][:]
        self.ctype = metadata["ctype"][:]
        self.image_location = metadata["image_location"][:]
        self.pixel_shape = metadata["pixel_shape"][:]
        self.pixel_centre = metadata["pixel_centre"][:]
        self.pixel_scale = metadata["pixel_scale"][:]
        self.telescope = metadata["telescope"][:]
        self.frequency = metadata["frequency"][:]
        self.bandwidth = metadata["bandwidth"][:]
        self.date = metadata["date"][:]
        self.bmaj = metadata["bmaj"][:]
        self.bmin = metadata["bmin"][:]
        self.bpa = metadata["bpa"][:]
        self.bunit = metadata["bunit"][:]
        self.equinox = metadata["equinox"][:]
        self.radesys = metadata["radesys"][:]
        # Load Stokes metadata lazily for reference-mode reductions.
        self._headers = metadata["header"]

    @classmethod
    def open(
        cls,
        path: str | Path,
        max_request_bytes: int = MAX_REQUEST_BYTES,
        concurrency: int = REMOTE_CONCURRENCY,
        request_timeout: float | None = REQUEST_TIMEOUT,
        connect_timeout: float = CONNECT_TIMEOUT,
        storage_options: dict[str, dict] | None = None,
    ) -> "SurveyStore":
        """Open a survey store for reading.

        Any edit left unfinished by an interrupted run is reverted first.

        Parameters
        ----------
        path
            Location of the store.
        max_request_bytes
            Most any one request may span.
        concurrency
            Requests in flight at once against one archive.
        request_timeout
            Seconds a single request may take before it fails. None leaves the
            HTTP client's own default, which is five minutes and reads as a
            hang.
        connect_timeout
            Seconds to wait for a connection.
        storage_options
            Filesystem options keyed by protocol, as ``{'s3': {'anon': True}}``.
            Merged over what the build recorded, so this is where a
            credential goes, since a store never holds one.

        Returns
        -------
        result
            Opened survey store.
        """
        from survey2zarray.stores.transaction import recover_pending_edit

        recover_pending_edit(path)
        root = open_store(path, mode="r")
        return cls(
            root,
            path,
            max_request_bytes=max_request_bytes,
            concurrency=concurrency,
            request_timeout=request_timeout,
            connect_timeout=connect_timeout,
            storage_options=storage_options,
        )

    def get_wcs(self, image_index: int) -> WCS:
        """Return a newly reconstructed WCS for one image.

        Reconstruction avoids sharing mutable WCS objects across readers and
        takes negligible time relative to pixel access.

        Parameters
        ----------
        image_index
            Index of the image.

        Returns
        -------
        result
            A new object each call.
        """
        key = int(image_index)
        return reconstruct_wcs(
            self.crval[key],
            self.crpix[key],
            self.cd[key],
            self.ctype[key],
            self.radesys[key],
            self.equinox[key],
        )

    def get_cutout_region(
        self, image_index: int, x0: int, x1: int, y0: int, y1: int
    ) -> np.ndarray:
        """Read one pixel region of one image.

        This is the single point where the access mode matters. A copy store
        slices its own array, and a reference store fetches these bytes from
        the FITS file no matter where it lives. Callers need not know which.

        Parameters
        ----------
        image_index
            Index of the image.
        x0, x1
            Column bounds, half-open.
        y0, y1
            Row bounds, half-open.

        Returns
        -------
        result
            ``(y1 - y0, x1 - x0)`` for continuum, with a leading channel axis
            for spectral data.
        """
        image_index = int(image_index)
        if self.access_mode == "copy":
            if self.observation_type == "spectral":
                cutout = self.root["data"][image_index, :, y0:y1, x0:x1]
            else:
                cutout = self.root["data"][image_index, y0:y1, x0:x1]
            return np.asarray(cutout)
        return self.read_referenced_region(image_index, x0, x1, y0, y1)

    def read_referenced_region(
        self, image_index: int, x0: int, x1: int, y0: int, y1: int
    ) -> np.ndarray:
        """Read a region through the byte map.

        Fetch contiguous full-width row ranges and slice requested columns
        locally. Select Stokes I planes during range planning.

        Parameters
        ----------
        image_index
            Index of the image.
        x0, x1
            Column bounds, half-open.
        y0, y1
            Row bounds, half-open.

        Returns
        -------
        result
            Stokes I, with a leading channel axis for spectral data.
        """
        image_index = int(image_index)
        spec = self.image_read_spec(image_index)
        location = spec.location
        filesystem = self.filesystem(location)
        plan = region_ranges(
            spec.offset,
            spec.shape,
            spec.dtype.itemsize,
            spec.planes,
            y0,
            y1,
            max_bytes=self.max_request_bytes,
        )

        blocks = filesystem.cat_ranges(
            [location] * len(plan.fetch),
            [start for start, _ in plan.fetch],
            [end for _, end in plan.fetch],
            batch_size=self.concurrency,
        )
        for block, (start, end) in zip(blocks, plan.fetch, strict=True):
            # Promote cat_ranges exception values into contextual read failures.
            if isinstance(block, BaseException):
                raise RuntimeError(
                    f"Could not read bytes {start} to {end} of {location}. "
                    f"{type(block).__name__}. {str(block) or 'No detail given'}."
                ) from block
            if len(block) != end - start:
                raise RuntimeError(
                    f"Read {len(block)} bytes for range {start} to {end} of "
                    f"{location}, expected {end - start}. The file may have changed."
                )
        rows, columns = int(y1) - int(y0), spec.shape[-1]
        region = np.frombuffer(b"".join(blocks), dtype=spec.dtype).reshape(
            spec.kept + (rows, columns)
        )
        region = region[..., :, int(x0) : int(x1)]
        # Preserve singleton celestial dimensions while removing leading axes.
        kept = tuple(n for n in spec.kept if n > 1)
        # Copy the buffer because callers mutate returned cutouts.
        return region.reshape(kept + region.shape[-2:]).copy()

    def image_read_spec(self, image_index: int) -> ImageRead:
        """Return cached byte-range inputs for one referenced image.

        Derive immutable values outside the lock to preserve concurrent first
        reads. Claim one result under the lock with ``setdefault``.

        Parameters
        ----------
        image_index
            Index of the image.

        Returns
        -------
        result
            Cached immutable read specification.
        """
        key = int(image_index)
        with self._lock:
            spec = self._image_reads.get(key)
        if spec is not None:
            return spec

        record = self.image_record(key)
        header = fits.Header.fromstring(str(self._headers[key]))
        shape = tuple(int(n) for n in record["shape"])
        planes, kept = stokes_i_planes(shape, header)
        spec = ImageRead(
            location=str(record["location"]),
            offset=int(record["offset"]),
            shape=shape,
            dtype=np.dtype(record["dtype"]),
            planes=tuple(planes),
            kept=tuple(kept),
        )
        with self._lock:
            return self._image_reads.setdefault(key, spec)

    def image_record(self, image_index: int) -> dict:
        """Return where one image's pixels live and how they are laid out.

        Parameters
        ----------
        image_index
            Index of the image.

        Returns
        -------
        result
            ``location``, ``offset``, ``shape``, and ``dtype``.

        Raises
        ------
        RuntimeError
            Where the store holds no byte map, which means it was built to copy.
        """
        with self._lock:
            if self._reference_map is None:
                path = Path(self.path) / REFERENCE_MAP_PATH
                if not path.exists():
                    raise RuntimeError(
                        f"Survey store {self.path} has no byte map at {path}. Its "
                        f"access mode is {self.access_mode!r}."
                    )
                with open(path) as handle:
                    self._reference_map = json.load(handle)
        return self._reference_map["images"][int(image_index)]

    def filesystem(self, location: str):
        """Return the fsspec filesystem serving a location.

        Cache one filesystem and connection pool per protocol.

        Parameters
        ----------
        location
            Path or URL of a FITS file.

        Returns
        -------
        result
            Filesystem for the location protocol.
        """
        import fsspec

        protocol = storage_protocol(location)
        with self._lock:
            if protocol not in self._filesystems:
                self._filesystems[protocol] = fsspec.filesystem(
                    protocol, **self.filesystem_options(protocol)
                )
            return self._filesystems[protocol]

    def filesystem_options(self, protocol: str) -> dict[str, Any]:
        """Return the options a filesystem of one protocol is created with.

        Merge stored archive options, reader overrides, and default HTTP
        timeouts. Preserve caller-supplied timeouts.

        Parameters
        ----------
        protocol
            Storage protocol, as `storage_protocol` reports it.

        Returns
        -------
        result
            Merged filesystem options.
        """
        options = options_for(protocol, self.stored_storage_options)
        options.update(options_for(protocol, self.storage_options))
        if protocol not in ("http", "https") or self.request_timeout is None:
            return options
        try:
            import aiohttp
        except ImportError:
            return options
        client_kwargs = dict(options.get("client_kwargs", {}))
        client_kwargs.setdefault(
            "timeout",
            aiohttp.ClientTimeout(
                total=float(self.request_timeout),
                connect=float(self.connect_timeout),
            ),
        )
        options["client_kwargs"] = client_kwargs
        return options
