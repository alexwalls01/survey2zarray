"""Read access to a dataset store of pre-extracted cutouts."""

from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import astropy.units as u
import numpy as np
import zarr
from astropy.coordinates import ICRS, SkyCoord
from astropy.wcs import WCS

from survey2zarray.stores.dataset_store.buckets import (
    BUCKET_ARRAY,
    BucketingRule,
    read_rule,
)
from survey2zarray.stores.dataset_store.geometry import (
    blank_outside_disc,
    disc_diameter,
    edge_padding_for,
    square_size,
)
from survey2zarray.stores.store import apply_coalesce_gap, open_store
from survey2zarray.stores.wcs import reconstruct_wcs
from survey2zarray.utils.logging import get_logger

logger = get_logger(__name__)

# PADDING marks aligned source rows lacking survey coverage.
PADDING: int = -1


def check_index_consistency(root: zarr.Group) -> None:
    """Verify every index map against the cutouts and the source table.

    Require one row per source and an identity index map over covered rows. The
    identity constraint preserves cross-survey alignment and shard-aware
    sampling.

    Parameters
    ----------
    root
        Root group of the store.

    Raises
    ------
    RuntimeError
        Where any map entry has no row, any row has no entry, entries repeat, an
        entry falls outside the source table, a survey has other than one row
        per source, or a map entry names a source other than its own row.
    """
    n_sources = int(root.attrs["n_sources"])
    for survey in root.attrs["surveys"]:
        sub = root["surveys"][survey]
        index_map = sub["index_map"][:]
        n_rows = sub["data"].shape[0]
        if len(index_map) != n_rows:
            raise RuntimeError(
                f"Inconsistent {survey} sub-store. Found {len(index_map)} map "
                f"entries for {n_rows} rows."
            )
        if n_rows != n_sources:
            raise RuntimeError(
                f"Unaligned {survey} sub-store. Found {n_rows} rows for "
                f"{n_sources} source(s). Every survey requires one row per source."
            )
        held = index_map[index_map != PADDING]
        if len(held) != len(np.unique(held)):
            raise RuntimeError(
                f"Inconsistent {survey} sub-store. The index map has duplicate entries."
            )
        if len(held) > 0 and (held.min() < 0 or held.max() >= n_sources):
            raise RuntimeError(
                f"Inconsistent {survey} sub-store. Index map entries fall outside "
                f"the source table."
            )
        covered = np.flatnonzero(index_map != PADDING)
        if not np.array_equal(held, covered):
            raise RuntimeError(
                f"Unaligned {survey} sub-store. The index map must be the identity "
                f"over covered sources. Row "
                f"{int(covered[held != covered][0])} holds source "
                f"{int(held[held != covered][0])}."
            )


class SurveySubStore:
    """Provide read access to one survey within a dataset store.

    Parameters
    ----------
    group
        The sub-store group.
    wcs_cache_size
        Number of reconstructed WCS objects to keep.
    coalesce_gap
        Bytes a read may run through a shard between the cutouts it wants, as
        recorded by the store. Applied before every batched read, including in
        spawned workers.

    Attributes
    ----------
    observation_type
        Read from the stored attribute, never inferred from array shape.
    n_chan
        Channel count for spectral surveys.
    n_cutouts
        Cutouts this survey holds, which is the count of sources it covers.
        Never ``data.shape[0]``, since every survey has a row per source and
        the uncovered ones are fill.
    shard_size
        Rows in one shard of the pixel array, a power of two, read off the array
        itself. A shard is one file, so this is what a batch aligns to; the rows
        it holds are chunked one apiece, so only the ones asked for are
        decompressed.
    index_map
        Source index to position in this survey's arrays, which is the identity
        on the sources it covers. A source this survey does not cover is simply
        absent, which is how coverage is recorded.
    """

    def __init__(
        self,
        group: zarr.Group,
        wcs_cache_size: int = 512,
        coalesce_gap: int | None = None,
    ) -> None:
        self.group = group
        self.coalesce_gap = coalesce_gap
        # Build metadata fixes the observation type independently of array shape.
        self.observation_type = group.attrs["observation_type"]
        self.n_chan = group.attrs.get("n_chan")
        self.n_cutouts = group.attrs["n_cutouts"]
        self.data = group["data"]
        self.shard_size = int((self.data.shards or self.data.chunks)[0])

        # Covered source indices map identically onto aligned cutout rows.
        map_values = group["index_map"][:]
        self.index_map = {
            int(source_index): local
            for local, source_index in enumerate(map_values)
            if int(source_index) != PADDING
        }

        # Single metadata chunks require one read per array.
        metadata = group["metadata"]
        self.crval = metadata["crval"][:]
        self.crpix = metadata["crpix"][:]
        self.cd = metadata["cd"][:]
        self.ctype = metadata["ctype"][:]
        self.pixel_shape = metadata["pixel_shape"][:]
        self.pixel_scale = metadata["pixel_scale"][:]
        self.image_index = metadata["image_index"][:]
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

        # Lazy WCS reconstruction avoids one cached object per cutout.
        self._wcs_cache_size = int(wcs_cache_size)
        self._wcs_cache = OrderedDict()

        self._rows: np.ndarray | None = None
        self._row_of: dict[int, int] = {}

    @property
    def row_bytes(self) -> int:
        """Return the uncompressed byte size of one stored row.

        Returns
        -------
        result
            Uncompressed row size in bytes.
        """
        return int(np.prod(self.data.shape[1:])) * self.data.dtype.itemsize

    def load_rows(self, local_indices: Sequence[int]) -> int:
        """Hold the given stored rows in memory, so later reads never open a file.

        Hold complete stored rows while preserving per-read cropping, disc
        blanking, and preprocessing. Read one shard at a time and select only
        served rows.

        Parameters
        ----------
        local_indices
            Positions in this survey's arrays to hold. Repeats are held once.

        Returns
        -------
        result
            Bytes now held.
        """
        wanted = sorted(dict.fromkeys(int(index) for index in local_indices))
        rows = np.empty((len(wanted), *self.data.shape[1:]), dtype=self.data.dtype)
        row_of = {local_index: row for row, local_index in enumerate(wanted)}

        apply_coalesce_gap(self.coalesce_gap)
        shards: dict[int, list[int]] = {}
        for local_index in wanted:
            shards.setdefault(local_index // self.shard_size, []).append(local_index)
        for group in shards.values():
            rows[[row_of[local_index] for local_index in group]] = self.data.oindex[
                group
            ]

        self._rows = rows
        self._row_of = row_of
        return rows.nbytes

    def drop_rows(self) -> None:
        """Release the rows `load_rows` held, so reads go back to the store."""
        self._rows = None
        self._row_of = {}

    def cached_row(self, local_index: int) -> np.ndarray | None:
        """Return a held row when available.

        Parameters
        ----------
        local_index
            Position in the survey arrays.

        Returns
        -------
        result
            Held row, or None when unavailable.
        """
        if self._rows is None:
            return None
        row = self._row_of.get(int(local_index))
        return None if row is None else self._rows[row]

    def crop_bounds(
        self,
        local_index: int,
        coords: tuple[float, float],
        cutout_size: float,
        edge_padding: float = 0.0,
    ) -> tuple[int, int, int, int]:
        """Return the pixel bounds of a smaller disc centred on a source.

        ``world_to_pixel`` returns (x, y) = (column, row) while ``pixel_shape``
        is (H, W) = (row, column). The box is clamped inside the stored cutout,
        so the result is always the full requested size.

        Parameters
        ----------
        local_index
            Position of the cutout in this survey's arrays.
        coords
            Source position as ``(ra, dec)`` in degrees.
        cutout_size
            Requested square side in arcsec. The disc pulled out is its
            diagonal across, plus twice the padding.
        edge_padding
            Radius in arcsec to keep beyond the square's corners, matching what
            the store was built with.

        Returns
        -------
        x0, y0
            Origin of the crop within the stored cutout, as (column, row).
        nx, ny
            Crop size in pixels, as (column, row).

        Raises
        ------
        ValueError
            Where no coordinates were given.
        """
        if coords is None:
            raise ValueError("coords are required to crop a cutout to a smaller size.")
        local_index = int(local_index)
        h, w = self.pixel_shape[local_index]
        px, py = self.pixel_scale[local_index]
        diameter = disc_diameter(cutout_size, edge_padding)
        nx = min(int(w), max(1, int(np.ceil(diameter / px))))
        ny = min(int(h), max(1, int(np.ceil(diameter / py))))
        cx, cy = self.source_pixel(local_index, coords)
        x0 = int(np.clip(round(cx - nx / 2), 0, int(w) - nx))
        y0 = int(np.clip(round(cy - ny / 2), 0, int(h) - ny))
        return x0, y0, nx, ny

    def source_pixel(
        self, local_index: int, coords: tuple[float, float]
    ) -> tuple[float, float]:
        """Locate a source within its stored cutout.

        Use the source's true WCS position to retain sub-pixel centring.

        Parameters
        ----------
        local_index
            Position of the cutout in this survey's arrays.
        coords
            Source position as ``(ra, dec)`` in degrees.

        Returns
        -------
        result
            Position as ``(x, y)`` in cutout pixels, to sub-pixel precision.
        """
        sky = SkyCoord(ra=coords[0] * u.deg, dec=coords[1] * u.deg, frame=ICRS)
        cx, cy = self.get_wcs(int(local_index)).celestial.world_to_pixel(sky)
        return float(cx), float(cy)

    def get_cutout(
        self,
        local_index: int,
        coords: tuple[float, float] | None = None,
        cutout_size: float | None = None,
        edge_padding: float = 0.0,
    ) -> tuple[np.ndarray, WCS]:
        """Return one cutout and the WCS describing it.

        Re-crop smaller requests around the source and blank the resulting disc.
        Return the array and its translated WCS together.

        Parameters
        ----------
        local_index
            Position of the cutout in this survey's arrays.
        coords
            Source position as ``(ra, dec)`` in degrees. Required only when
            re-cropping.
        cutout_size
            Square side in arcsec. None returns the whole stored disc.
        edge_padding
            Radius in arcsec to keep beyond the square's corners.

        Returns
        -------
        cutout
            ``(H, W)`` for continuum, ``(n_chan, H, W)`` for spectral, trimmed
            of any padding.
        wcs
            WCS for exactly the pixels returned.
        """
        local_index = int(local_index)
        x0, y0, nx, ny = self.cutout_bounds(
            local_index, coords, cutout_size, edge_padding
        )
        row = self.cached_row(local_index)
        if row is not None:
            # Copy cached rows because preprocessing mutates returned arrays.
            cutout = row[..., y0 : y0 + ny, x0 : x0 + nx].copy()
        elif self.observation_type == "spectral":
            cutout = self.data[local_index, :, y0 : y0 + ny, x0 : x0 + nx]
        else:
            cutout = self.data[local_index, y0 : y0 + ny, x0 : x0 + nx]
        return self.finish_cutout(
            local_index, np.asarray(cutout), coords, cutout_size, edge_padding, (x0, y0)
        )

    def get_cutouts(
        self,
        local_indices: Sequence[int],
        coords: Sequence[tuple[float, float] | None],
        cutout_size: float | None = None,
        edge_padding: float = 0.0,
    ) -> list[tuple[np.ndarray, WCS]]:
        """Return several cutouts, decompressing each chunk they share only once.

        Group reads by shard and request exact cutout indices. Open each touched
        shard once and decompress selected chunks. Deduplicate repeated indices
        and serve preloaded rows from memory.

        Parameters
        ----------
        local_indices
            Positions of the cutouts in this survey's arrays.
        coords
            Source position as ``(ra, dec)`` in degrees, one per index.
            Required only when re-cropping.
        cutout_size
            Square side in arcsec. None returns the whole stored disc.
        edge_padding
            Radius in arcsec to keep beyond the square's corners.

        Returns
        -------
        result
            ``(cutout, wcs)`` per index, in the order asked for, each as
            `get_cutout` would have returned it.
        """
        local_indices = [int(index) for index in local_indices]
        if len(coords) != len(local_indices):
            raise ValueError(
                f"Got {len(local_indices)} cutout index/indices but {len(coords)} "
                f"coordinate(s). Each cutout index requires one coordinate."
            )
        apply_coalesce_gap(self.coalesce_gap)
        shards: dict[int, list[int]] = {}
        held: list[int] = []
        for position, local_index in enumerate(local_indices):
            if self.cached_row(local_index) is not None:
                held.append(position)
            else:
                shards.setdefault(local_index // self.shard_size, []).append(position)

        results: list[tuple[np.ndarray, WCS] | None] = [None] * len(local_indices)

        def crop(position: int, row: np.ndarray) -> tuple[np.ndarray, WCS]:
            """Finish one row selected from the current batch.

            Parameters
            ----------
            position
                Position within the current batch.
            row
                Stored cutout row.

            Returns
            -------
            result
                Finished cutout and WCS.
            """
            local_index = local_indices[position]
            x0, y0, nx, ny = self.cutout_bounds(
                local_index, coords[position], cutout_size, edge_padding
            )
            return self.finish_cutout(
                local_index,
                np.array(row[..., y0 : y0 + ny, x0 : x0 + nx]),
                coords[position],
                cutout_size,
                edge_padding,
                (x0, y0),
            )

        for position in held:
            results[position] = crop(position, self.cached_row(local_indices[position]))
        for positions in shards.values():
            wanted = sorted({local_indices[position] for position in positions})
            row_of = {local_index: row for row, local_index in enumerate(wanted)}
            rows = np.asarray(self.data.oindex[wanted])
            for position in positions:
                results[position] = crop(
                    position, rows[row_of[local_indices[position]]]
                )
        return results

    def cutout_bounds(
        self,
        local_index: int,
        coords: tuple[float, float] | None,
        cutout_size: float | None,
        edge_padding: float = 0.0,
    ) -> tuple[int, int, int, int]:
        """Return the pixel bounds of what one read should return.

        Use stored true shape for full-cutout requests because the array has
        survey-wide padding.

        Parameters
        ----------
        local_index
            Position of the cutout in this survey's arrays.
        coords
            Source position as ``(ra, dec)`` in degrees. Required only when
            re-cropping.
        cutout_size
            Square side in arcsec. None takes the whole stored cutout.
        edge_padding
            Radius in arcsec to keep beyond the square's corners.

        Returns
        -------
        x0, y0, nx, ny
            Origin and size of the crop within the stored row, as (column, row)
            then (columns, rows).
        """
        if cutout_size is None:
            h, w = self.pixel_shape[int(local_index)]
            return 0, 0, int(w), int(h)
        return self.crop_bounds(local_index, coords, cutout_size, edge_padding)

    def finish_cutout(
        self,
        local_index: int,
        cutout: np.ndarray,
        coords: tuple[float, float] | None,
        cutout_size: float | None,
        edge_padding: float,
        origin: tuple[int, int],
    ) -> tuple[np.ndarray, WCS]:
        """Blank a cropped cutout back to a disc and pair it with its WCS.

        Cropping preserves the larger disc boundary inside the new square.
        Reapply the smaller disc mask and return its WCS with the array.

        Parameters
        ----------
        local_index
            Position of the cutout in this survey's arrays.
        cutout
            Pixels already cropped to ``origin`` and the requested size.
        coords
            Source position as ``(ra, dec)`` in degrees.
        cutout_size
            Square side in arcsec. None leaves the pixels as they are.
        edge_padding
            Radius in arcsec kept beyond the square's corners.
        origin
            Origin ``(x0, y0)`` the cutout was cropped at.

        Returns
        -------
        cutout
            Blanked cutout at its true shape.
        wcs
            WCS translated to the cutout origin.
        """
        x0, y0 = origin
        if cutout_size is not None:
            px, py = self.pixel_scale[int(local_index)]
            cx, cy = self.source_pixel(local_index, coords)
            cutout = blank_outside_disc(
                cutout,
                (px, py),
                disc_diameter(cutout_size, edge_padding),
                centre=(cx - x0, cy - y0),
            )
        return cutout, self.get_wcs(local_index, origin=(x0, y0))

    def get_wcs(self, local_index: int, origin: tuple[int, int] = (0, 0)) -> WCS:
        """Return the WCS of one cutout, optionally shifted to a sub-crop.

        Shifting keeps the same sky mapping and moves only the reference pixel,
        so any further axis such as frequency is untouched.

        Parameters
        ----------
        local_index
            Position of the cutout in this survey's arrays.
        origin
            Origin ``(x0, y0)`` of a sub-crop within the cutout.

        Returns
        -------
        result
            At the default origin this is the cached object, which should be
            treated as read-only; otherwise it is a fresh copy.
        """
        key = int(local_index)
        if key in self._wcs_cache:
            self._wcs_cache.move_to_end(key)
            wcs = self._wcs_cache[key]
        else:
            wcs = reconstruct_wcs(
                self.crval[key],
                self.crpix[key],
                self.cd[key],
                self.ctype[key],
                self.radesys[key],
                self.equinox[key],
            )
            self._wcs_cache[key] = wcs
            if len(self._wcs_cache) > self._wcs_cache_size:
                self._wcs_cache.popitem(last=False)
        x0, y0 = origin
        if x0 == 0 and y0 == 0:
            return wcs
        wcs = wcs.deepcopy()  # Preserve the cached WCS.
        crpix = list(wcs.wcs.crpix)  # Preserve reference pixels on non-celestial axes.
        crpix[0] -= x0
        crpix[1] -= y0
        wcs.wcs.crpix = crpix
        return wcs


class DatasetStore:
    """Provide read access to sources and per-survey cutouts.

    The source table is the whole dataset's index and is independent of survey
    coverage, so a source appears here even when no survey holds a cutout of it.

    Parameters
    ----------
    root
        Opened root group of the store.
    path
        Location of the store.

    Attributes
    ----------
    surveys
        Survey names, in the order their sub-stores were added.
    max_cutout_size
        Side in arcsec of the square this store can serve at any rotation.
    disc_diameter
        Diameter in arcsec of the disc actually stored, which is the side of
        the arrays on disk.
    disc_padding
        Radius in arcsec the disc extends beyond the square's corners, so that
        a reader can inset before interpolating and still reach the full
        ``cutout_size``.
    padded_for_pixel_scale
        Coarsest pixel scale, in arcsec per pixel, the padding covers.
    n_sources
        Number of rows in the source table.
    margin_px
        Pixels a disc was kept clear of each image edge by when matching.
    survey_store_paths
        Where each sub-store's survey came from, reopened by later appends.
    coalesce_gap
        Bytes a read may run through a shard between the chunks it wants, as
        the store recorded it. Applied to zarr's own configuration when the
        store is opened.
    shard_size
        Rows to a shard, over every sub-store. The smallest of them, and every
        one a power of two, so this many rows fall inside one shard of every
        survey.
    source_id, ra, dec
        Source table columns, in source index order. How the sources are
        divided is not among them. A store holds no splits, and `S2ZDataset`
        cuts them across the buckets.
    extra_columns
        What the catalogue held beyond ``id``, ``ra``, and ``dec``, in
        source index order. Empty when it held none.
    bucketing
        How a source's catalogue values decided which bucket it went in.
    buckets
        Bucket per source, in source index order and ascending except across
        an appended region.

    Raises
    ------
    RuntimeError
        Where the store was not completely written, or an index map is
        inconsistent.
    """

    def __init__(self, root: zarr.Group, path: str | Path) -> None:
        self.root = root
        self.path = Path(path)
        if not root.attrs.get("complete", False):
            raise RuntimeError(
                f"Dataset store {path} is incomplete. Rebuild it before use."
            )
        check_index_consistency(root)
        self.surveys = list(root.attrs["surveys"])
        self.max_cutout_size = float(root.attrs["max_cutout_size_arcsec"])
        self.disc_diameter = float(root.attrs["disc_diameter_arcsec"])
        self.margin_px = int(root.attrs["margin_px"])
        self.disc_padding = float(root.attrs["disc_padding_arcsec"])
        self.padded_for_pixel_scale = float(root.attrs["padded_for_pixel_scale_arcsec"])

        self.n_sources = int(root.attrs["n_sources"])
        self.survey_store_paths = list(root.attrs["survey_store_paths"])

        sources = root["sources"]
        self.source_id = sources["source_id"][:]
        self.ra = sources["ra"][:]
        self.dec = sources["dec"][:]
        self.bucketing: BucketingRule = read_rule(root)
        self.buckets = np.asarray(sources[BUCKET_ARRAY][:], dtype=np.int64)
        self.extra_columns = {
            name: sources[name][:]
            for name in sorted(sources.array_keys())
            if name not in ("source_id", "ra", "dec", BUCKET_ARRAY)
        }

        # Apply build-time geometry through the store interface.
        self.coalesce_gap = root.attrs.get("coalesce_gap_bytes")
        apply_coalesce_gap(self.coalesce_gap)

        self._substores = {
            survey: SurveySubStore(
                root["surveys"][survey], coalesce_gap=self.coalesce_gap
            )
            for survey in self.surveys
        }
        # Array shards remain authoritative after sub-store appends.
        self.shard_size = min(
            (sub.shard_size for sub in self._substores.values()), default=1
        )

    def servable_square(self, pixel_scale: float) -> float:
        """Return the largest square servable to a survey of a given pixel scale.

        A survey coarser than the store was padded for insets further before
        interpolating, so it reaches less of the disc.

        Parameters
        ----------
        pixel_scale
            Coarsest pixel scale in play, in arcsec per pixel.

        Returns
        -------
        result
            Side in arcsec, never more than ``cutout_size``.
        """
        reachable = square_size(
            self.disc_diameter, edge_padding_for(pixel_scale, self.margin_px)
        )
        return min(self.max_cutout_size, reachable)

    @classmethod
    def open(cls, path: str | Path) -> "DatasetStore":
        """Open a dataset store for reading.

        Any edit left unfinished by an interrupted run is reverted first.

        Parameters
        ----------
        path
            Location of the store.

        Returns
        -------
        result
            Opened dataset store.
        """
        from survey2zarray.stores.transaction import recover_pending_edit

        recover_pending_edit(path)
        root = open_store(path, mode="r")
        return cls(root, path)

    def survey(self, name: str) -> SurveySubStore:
        """Return one survey's sub-store.

        Parameters
        ----------
        name
            Survey name.

        Returns
        -------
        result
            Requested survey sub-store.
        """
        return self._substores[name]
