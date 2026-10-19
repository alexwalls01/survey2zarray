"""Serve augmented samples from a dataset store."""

import multiprocessing
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.coordinates import ICRS, SkyCoord
from tqdm import tqdm

from survey2zarray.dataset.augmentation import S2ZAugmentation
from survey2zarray.dataset.postprocessing import S2ZPostprocessing
from survey2zarray.dataset.preprocessing import S2ZPreprocessing
from survey2zarray.stores.dataset_store.buckets import (
    SPLIT_NAMES,
    balance_warnings,
    plan_splits,
    read_split_config,
)
from survey2zarray.stores.dataset_store.dataset_store import DatasetStore
from survey2zarray.stores.dataset_store.filters import (
    count_views,
    resolve_subset,
    select_sources,
)
from survey2zarray.stores.dataset_store.geometry import disc_diameter
from survey2zarray.utils.configs import load_config
from survey2zarray.utils.logging import get_logger
from survey2zarray.utils.parallel import MEMORY_FRACTION, available_memory

logger = get_logger(__name__)


class S2ZDataset(torch.utils.data.Dataset):
    """Serve augmented multi-survey views of each source.

    Reads cutouts and their metadata from a dataset store, then preprocesses,
    augments and postprocesses them.

    Parameters
    ----------
    config
        Configuration, as a mapping or a path to a YAML file. Recognised keys
        are ``dataset_store``, a ``splits`` section, ``include`` and
        ``exclude`` subset sections, ``cutout_size``, ``min_views``,
        ``label_by``, ``label_order``, ``final_pixel_size``, ``preprocessing``,
        ``augmentation``, and ``postprocessing``.
    split
        Which split to serve, one of ``train``, ``val`` and ``test``.
    statistics
        Precomputed per-survey ``(mean, standard deviation)`` for `normalise`.
    in_memory
        Cutout storage mode. True forces memory and False streams batches.
        ``"auto"`` remains unresolved until a loader is constructed or the
        `in_memory` property is first read. The loader makes a worker- and
        local-rank-aware decision; direct use assumes one process and no
        workers.
    distributed_statistics
        Whether automatic normalisation-statistics measurement is shared over
        the default ``torch.distributed`` process group. This is explicit: an
        initialised process group alone never enables a collective. When
        automatic measurement is needed, every rank in the group must
        construct the dataset and enter it.

    Attributes
    ----------
    surveys
        Surveys served, after any subsetting.
    cutout_size
        Square side served, in arcsec, valid at any rotation.
    servable_cutout_size
        Side in arcsec of the largest square this dataset can serve at any
        rotation. Equal to ``cutout_size`` unless a selected survey is coarser
        than the store's disc was padded for.
    shard_size
        Minimum rows per shard across selected surveys. All shard counts are
        powers of two.
    fractions
        Share of the served sources each split takes, as the config asked for
        them.
    split_plan
        Where the cut points fell across the store's buckets, given this
        configuration's filtering.
    min_views
        Views a source must have among the selected surveys to be served. One
        by default, which drops sources no selected survey covers, since every
        view of those would be None. Zero serves them anyway.
    final_pixel_size
        Side in pixels every view is resampled to, after preprocessing and
        augmentation. None leaves each survey at its own pixel count.
    label_classes
        Categories of the label column, where position is the integer label.
        None unless ``label_by`` is set.
    in_memory
        Whether the split's cutouts are held in memory. Reading this property
        resolves a pending ``"auto"`` request for direct, loader-free use.
    memory_bytes
        Bytes the split's stored cutouts occupy uncompressed, held or not.
    has_freq_axis
        Whether each survey's channel axis represents frequency.

    Raises
    ------
    TypeError
        If the arguments are of the wrong type.
    ValueError
        If the split is not one of `SPLIT_NAMES`, the ``splits`` section is
        unusable, the subsetting selects nothing, or no source in the split has
        ``min_views`` views.
    RuntimeError
        If distributed statistics are requested without an available,
        initialised process group.
    """

    @staticmethod
    def resolve_cutout_size(
        requested: float | None,
        stored: float,
        tolerance: float = 1e-9,
    ) -> float:
        """Resolve a requested cutout size against what the store holds.

        A size smaller than the store's is served by cropping to a smaller disc.
        A larger one is refused.

        Parameters
        ----------
        requested
            Square side in arcsec. None takes the stored size.
        stored
            Side the store can serve, in arcsec.
        tolerance
            Slack allowed when comparing against the stored size.

        Returns
        -------
        result
            Square side in arcsec.

        Raises
        ------
        TypeError
            If the request is not a scalar.
        ValueError
            If the request is not positive, or exceeds the stored size.
        """
        if requested is None:
            return stored
        if not np.isscalar(requested):
            raise TypeError(
                f"cutout_size must be one disc diameter, got {requested!r}."
            )
        requested = float(requested)
        if requested <= 0:
            raise ValueError(f"cutout_size must be positive, got {requested} arcsec.")
        if requested > stored + tolerance:
            raise ValueError(
                f"cutout_size {requested} arcsec exceeds the stored {stored} arcsec. "
                f"The extra pixels are unavailable. "
                f"Rebuild the store with a larger --max-cutout-size to read them."
            )
        return requested

    @staticmethod
    def is_missing(value: Any) -> bool:
        """Report whether a catalogue value is absent.

        Parameters
        ----------
        value
            Catalogue value to test.

        Returns
        -------
        result
            True for None, a NaN, or a string that is empty or all
            whitespace.
        """
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        if isinstance(value, (float, np.floating)):
            return bool(np.isnan(value))
        return False

    @staticmethod
    def resolve_label_classes(
        values: Sequence[Any],
        order: Sequence[Any] | None = None,
    ) -> list:
        """Fix which category each integer label stands for.

        Preserve positions named in ``order`` across dataset subsets.

        Parameters
        ----------
        values
            Label value for every served source, in source index order.
        order
            Categories in integer order. Additional categories follow in order
            of first appearance.

        Returns
        -------
        result
            Categories, where position is the integer label.

        Raises
        ------
        ValueError
            If a served source has no category.
        """
        values = list(values)
        missing = sum(1 for value in values if S2ZDataset.is_missing(value))
        if missing:
            raise ValueError(
                f"{missing} served source(s) have no value in the label column. "
                f"Every source needs a category."
            )
        classes = list(order) if order is not None else []
        for value in values:
            if value not in classes:
                classes.append(value)
        return classes

    @staticmethod
    def resize_views(groups_per_survey: list, valid: Sequence[bool], size: int) -> list:
        """Resample every view onto a common pixel grid.

        Every view comes back at the same pixel count, regardless of each
        survey's pixel scale. The surveys already cover the same sky by this
        point, so only how finely that sky is sampled changes.

        A view holding NaN picks up more of it afterwards, so fill with the
        ``nan_fill`` signal transform first if that matters.

        Parameters
        ----------
        groups_per_survey
            Per survey, a list of ``(tensor, has_freq_axis)`` groups, or None
            where the survey does not cover the source.
        valid
            Per survey, whether it covers the source.
        size
            Side in pixels every view is resampled to.

        Returns
        -------
        result
            Same structure, each tensor ``(C, size, size)``.
        """
        output = []
        for groups, covered in zip(groups_per_survey, valid, strict=True):
            if not covered or groups is None:
                output.append(groups)
                continue
            resized = []
            for tensor, has_freq_axis in groups:
                if tuple(tensor.shape[-2:]) == (size, size):
                    resized.append((tensor, has_freq_axis))
                    continue
                # interpolate requires a batch axis. Antialiasing applies during
                # downsampling.
                scaled = torch.nn.functional.interpolate(
                    tensor.unsqueeze(0),
                    size=(size, size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                ).squeeze(0)
                resized.append((scaled.to(tensor.dtype), has_freq_axis))
            output.append(resized)
        return output

    def __init__(
        self,
        config: str | Path | dict,
        split: str,
        statistics: dict[str, tuple[float, float]] | None = None,
        in_memory: bool | str = "auto",
        distributed_statistics: bool = False,
    ) -> None:
        if not isinstance(config, (str, Path, dict)):
            raise TypeError(
                f"Expected str, Path, or dict for config, but got "
                f"{type(config).__name__}."
            )
        if not isinstance(split, str):
            raise TypeError(f"Expected str for split, but got {type(split).__name__}.")
        if split not in SPLIT_NAMES:
            raise ValueError(
                f"split must be one of {list(SPLIT_NAMES)}, got {split!r}."
            )
        if not isinstance(distributed_statistics, bool):
            raise TypeError(
                "distributed_statistics must be a boolean, got "
                f"{type(distributed_statistics).__name__}."
            )

        self.split = split
        self.config = load_config(config)
        if "splits" not in self.config:
            raise ValueError(
                "The config needs a splits section giving the train, val, and "
                "test fractions and a seed. A store holds no splits of its own."
            )
        self.fractions, self.split_seed = read_split_config(self.config["splits"])

        self.store = DatasetStore.open(self.config["dataset_store"])

        include = self.config.get("include") or {}
        exclude = self.config.get("exclude") or {}
        for name, section in (("include", include), ("exclude", exclude)):
            if not isinstance(section, dict):
                raise TypeError(
                    f"'{name}' must be a mapping keyed by 'surveys', 'sources' or a "
                    f"catalogue column, got {type(section).__name__}."
                )

        self.surveys = resolve_subset(
            self.store.surveys,
            include.get("surveys"),
            exclude.get("surveys"),
            "surveys",
        )
        if not self.surveys:
            raise ValueError("No surveys selected.")
        source_ids = [str(v) for v in self.store.source_id]

        logger.info(f"Building dataset from {self.store.path} ...")
        logger.info(f"Using surveys {self.surveys}.")

        self.substores = {survey: self.store.survey(survey) for survey in self.surveys}

        self.has_freq_axis = {
            survey: self.substores[survey].observation_type == "spectral"
            for survey in self.surveys
        }

        # Survey selection can reduce a source below the required view count.
        self.min_views = int(self.config.get("min_views", 1))
        views = count_views(
            {survey: list(self.substores[survey].index_map) for survey in self.surveys},
            self.store.n_sources,
        )

        selection = select_sources(
            source_ids,
            self.store.extra_columns,
            views,
            include,
            exclude,
            self.min_views,
            len(self.surveys),
        )
        for reason, count in selection.reasons.items():
            logger.info(f"Excluding {count} source(s) because {reason}.")
        selected_ids = selection.selected_ids
        column_keep = selection.column_keep
        self._eligible = selection.eligible

        self.label_by = self.config.get("label_by")
        self.label_classes = None
        self._labels = None
        if self.label_by is not None:
            if self.label_by not in self.store.extra_columns:
                raise ValueError(
                    f"Cannot label by column {self.label_by!r}. The dataset store "
                    f"holds {sorted(self.store.extra_columns) or 'no extra columns'}."
                )
            column = [
                v.decode() if isinstance(v, bytes) else v
                for v in self.store.extra_columns[self.label_by]
            ]
            order = self.config.get("label_order")
            self.label_classes = self.resolve_label_classes(
                [
                    value
                    for value, keep in zip(column, self._eligible, strict=True)
                    if keep
                ],
                order,
            )
            lookup = {value: index for index, value in enumerate(self.label_classes)}
            # The -1 sentinel exposes accidental reads of excluded sources to losses.
            self._labels = np.array(
                [lookup.get(value, -1) for value in column], dtype=np.int64
            )
            mapping = ", ".join(
                f"{value} -> {index}" for index, value in enumerate(self.label_classes)
            )
            logger.info(f"Labels from '{self.label_by}' are {mapping}.")
            if order is None:
                logger.warning(
                    "Catalogue order defines label numbering because label_order "
                    "is absent. Set "
                    "label_order to fix it, or a differently ordered catalogue "
                    "will number them differently."
                )

        # Splitting after filtering preserves the configured fractions.
        self.split_plan = plan_splits(
            self.store.buckets,
            self._eligible,
            self.fractions,
            self.split_seed,
            self.store.bucketing.n_buckets,
        )
        self.split_values = self.split_plan.of_source(self.store.buckets)
        self._report_splits()

        # DataLoader workers require these serialisable constructor inputs.
        self._source_ids = source_ids
        self._selected_ids = selected_ids
        self._column_keep = column_keep
        self._views = views
        self.subset_indices = self._indices_for(self.split)
        self._sample_index_by_source_id: dict[str, int] | None = None
        if len(self.subset_indices) == 0:
            raise ValueError(
                f"No sources found in catalogue for split '{split}' with at least "
                f"{self.min_views} view(s) among {self.surveys}."
            )
        self.pixel_scale = {
            survey: self.substores[survey].pixel_scale for survey in self.surveys
        }
        # Excluded surveys cannot constrain the active shard size.
        self.shard_size = min(
            (self.substores[survey].shard_size for survey in self.surveys), default=1
        )
        self.stored_cutout_size = float(self.store.max_cutout_size)
        self.cutout_size = self.resolve_cutout_size(
            self.config.get("cutout_size"), self.stored_cutout_size
        )
        self._crop_size = (
            None
            if np.isclose(self.cutout_size, self.stored_cutout_size)
            else self.cutout_size
        )
        self.disc_diameter = disc_diameter(self.cutout_size, self.store.disc_padding)

        self.preprocess = S2ZPreprocessing(
            self.config.get("preprocessing"), self.pixel_scale
        )

        # Two coarse pixels provide the bilinear interpolation margin.
        coarsest = max(
            np.max(self.pixel_scale[s])
            for s in self.surveys
            if len(self.pixel_scale[s]) > 0
        )
        margin = self.store.margin_px * coarsest
        # Rotated square corners reach disc_diameter / 2 - margin.
        # The configured size caps the servable square.
        self.servable_cutout_size = min(
            (self.disc_diameter / 2 - margin) * np.sqrt(2),
            self.cutout_size,
        )
        if self.servable_cutout_size < self.cutout_size - 1e-6:
            logger.warning(
                f"Selected surveys reach {self.servable_cutout_size:.1f} arcsec of "
                f"the {self.cutout_size:.1f} arcsec configured, since the coarsest "
                f"is {coarsest:.2f} arcsec/pixel but the store was padded for "
                f"{self.store.padded_for_pixel_scale:.2f}. Rebuild with "
                f"coarsest_pixel_scale_arcsec={coarsest:.2f} to serve the full size."
            )

        augmentation_config = self.config.get("augmentation", {}).get(split, {})
        if not augmentation_config:
            logger.warning(f"No augmentations found for split '{split}'.")
        self.augment = S2ZAugmentation(augmentation_config, self.servable_cutout_size)

        self.final_pixel_size = self.config.get("final_pixel_size")
        if self.final_pixel_size is not None:
            self.final_pixel_size = int(self.final_pixel_size)
            if self.final_pixel_size <= 0:
                raise ValueError(
                    f"final_pixel_size must be positive, got {self.final_pixel_size}."
                )
            self._warn_if_downsampling(augmentation_config)

        self._rng = None
        self._augmentation_seed: int | None = None

        if not isinstance(in_memory, bool) and in_memory != "auto":
            raise ValueError(
                f"in_memory must be True, False or 'auto', got {in_memory!r}."
            )
        self._in_memory_request = in_memory
        self._in_memory: bool | None = None
        self._memory_rows = {
            survey: [
                self.substores[survey].index_map[source_index]
                for source_index in self.subset_indices
                if source_index in self.substores[survey].index_map
            ]
            for survey in self.surveys
        }
        self.memory_bytes = sum(
            len(indices) * self.substores[survey].row_bytes
            for survey, indices in self._memory_rows.items()
        )
        # Explicit modes remain eager. Only "auto" moves to the loader (or to
        # the lazy property below for direct dataset use).
        if isinstance(in_memory, bool):
            self.resolve_in_memory()

        self.postprocess = S2ZPostprocessing(
            self.config.get("postprocessing"),
            surveys=self.surveys,
            channels={s: self.substores[s].n_chan for s in self.surveys},
        )
        if self.postprocess.statistics_split is not None:
            self.postprocess.set_statistics(
                statistics
                if statistics is not None
                else self.measure_statistics(
                    self.postprocess.statistics_split,
                    distributed=distributed_statistics,
                )
            )

        logger.info(f"Finished building dataset using surveys {self.surveys}.")

    @property
    def in_memory(self) -> bool:
        """Whether cutouts are held, resolving ``"auto"`` for direct use."""
        return self.resolve_in_memory(num_workers=0, local_world_size=1)

    def resolve_in_memory(
        self,
        num_workers: int = 0,
        local_world_size: int | None = 1,
        multiprocessing_start_method: str | None = None,
    ) -> bool:
        """Resolve the storage mode and preload selected cutouts when selected.

        Compare the estimated node-wide copies of the uncompressed selected
        rows with available process memory. Preloaded rows permit unrestricted
        sample ordering. Streamed rows use shard-aligned batches to limit file
        opens.

        Parameters
        ----------
        num_workers
            Reader processes this rank's loader will create.
        local_world_size
            Training ranks sharing the node and memory budget. None means the
            topology is unknown, in which case ``"auto"`` streams.
        multiprocessing_start_method
            How DataLoader workers are created. None uses the platform's
            active/default method. Non-fork methods copy held rows to workers.

        Returns
        -------
        result
            Whether the cutouts are now held.

        Raises
        ------
        ValueError
            If a worker count or known local world size is invalid.
        """
        num_workers = int(num_workers)
        if num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {num_workers}.")
        if local_world_size is not None:
            local_world_size = int(local_world_size)
            if local_world_size < 1:
                raise ValueError(
                    f"local_world_size must be positive, got {local_world_size}."
                )

        requested = self._in_memory_request
        if requested is False:
            self._in_memory = False
            return False

        if requested == "auto" and local_world_size is None:
            logger.info(
                f"Reading '{self.split}' from the store because local rank "
                f"topology is unknown. Pass local_world_size to S2ZDataLoader "
                f"to enable a node-aware in_memory='auto' decision."
            )
            self._drop_memory_rows()
            self._in_memory = False
            return False

        # Reuse one measurement for the decision and its message.
        spare = available_memory()
        if multiprocessing_start_method is None:
            multiprocessing_start_method = multiprocessing.get_start_method(
                allow_none=True
            )
            if multiprocessing_start_method is None:
                multiprocessing_start_method = (
                    multiprocessing.get_context().get_start_method()
                )
        copies_per_rank = (
            num_workers + 1
            if num_workers > 0 and multiprocessing_start_method != "fork"
            else 1
        )
        ranks = local_world_size or 1
        copies = ranks * copies_per_rank
        required = self.memory_bytes * copies
        fits = spare is not None and required <= spare * MEMORY_FRACTION

        if self._in_memory is not None:
            if self._in_memory and requested == "auto" and not fits:
                logger.info(
                    f"Returning '{self.split}' to streamed reads because its "
                    f"estimated {required / 1e6:,.0f} MB node-wide held "
                    f"footprint no longer fits the loader's resource budget."
                )
                self._drop_memory_rows()
                self._in_memory = False
            elif self._in_memory and not fits:
                self._warn_memory_pressure(required, spare, copies)
            return self._in_memory

        if requested == "auto" and not fits:
            against = (
                "memory this platform does not report"
                if spare is None
                else f"{spare / 1e6:,.0f} MB this process can still take"
            )
            logger.info(
                f"Reading '{self.split}' from the store. Its cutouts require an "
                f"estimated {required / 1e6:,.0f} MB across {copies} local "
                f"process copy/copies against {against}. Batches want "
                f"BlockAlignedBatchSampler, which keeps them shard aligned."
            )
            self._in_memory = False
            return False
        if not fits:
            self._warn_memory_pressure(required, spare, copies)

        for survey, indices in self._memory_rows.items():
            self.substores[survey].load_rows(indices)
        self._in_memory = True
        logger.info(
            f"Holding {self.memory_bytes / 1e6:,.0f} MB of '{self.split}' cutouts "
            f"in memory, so no batch opens a file. Sources may be drawn in any "
            f"order, which a plain shuffle does and BlockAlignedBatchSampler "
            f"does not need to."
        )
        return True

    def _drop_memory_rows(self) -> None:
        """Release any rows this dataset previously asked its substores to hold."""
        for substore in self.substores.values():
            substore.drop_rows()

    def _warn_memory_pressure(
        self, required: int, spare: int | None, copies: int
    ) -> None:
        """Warn that an explicit or already-resolved held mode exceeds budget."""
        against = (
            "memory this platform does not report"
            if spare is None
            else f"{spare * MEMORY_FRACTION / 1e6:,.0f} MB reserved-memory budget"
        )
        logger.warning(
            f"Holding '{self.split}' requires an estimated "
            f"{required / 1e6:,.0f} MB across {copies} local process copy/copies, "
            f"against {against}. Pass in_memory='auto' to let the loader stream "
            f"when the node-wide estimate does not fit."
        )

    def _report_splits(self) -> None:
        """Report split sizes and distribution imbalance.

        Evaluate distributions over each filtered split because filters can
        skew the served population after bucket construction.
        """
        plan = self.split_plan
        served = int(self._eligible.sum())
        sizes = ", ".join(
            f"{name} {plan.counts[name]} ({plan.counts[name] / max(served, 1):.1%})"
            for name in SPLIT_NAMES
        )
        logger.info(
            f"Splitting {served} served source(s) over "
            f"{self.store.bucketing.n_buckets} bucket(s) with sizes {sizes}."
        )
        if plan.counts[self.split] == 0:
            logger.warning(
                f"No bucket falls in '{self.split}' at a fraction of "
                f"{self.fractions[self.split]:.3g} over "
                f"{self.store.bucketing.n_buckets} bucket(s). Rebuild the store "
                f"with more buckets to express a share this small."
            )
        reports = balance_warnings(
            self.store.extra_columns,
            self.store.bucketing,
            self._eligible,
            self.split_values,
        )
        if reports:
            logger.warning(
                f"Filtering leaves the splits unbalanced. {'. '.join(reports)}."
            )

    def _warn_if_downsampling(self, augmentation_config: dict) -> None:
        """Warn where ``final_pixel_size`` resamples a survey below its own grid.

        Parameters
        ----------
        augmentation_config
            This split's augmentation section, read for its crop fraction.
        """
        crop = augmentation_config.get("sky_transforms", {}).get("random_crop")
        # Crop parameters are fractions of the largest servable square.
        largest_crop = self.servable_cutout_size * (
            float(crop["max_crop"])
            if crop and crop.get("max_crop") is not None
            else 1.0
        )
        losing = {}
        for survey in self.surveys:
            # Zero scales mark shard padding and cannot define the finest grid.
            scales = np.asarray(self.pixel_scale[survey])
            scales = scales[scales > 0]
            if scales.size == 0:
                continue
            native = largest_crop / float(np.min(scales))
            if self.final_pixel_size < native - 1e-9:
                losing[survey] = native
        if losing:
            detail = ", ".join(
                f"{name} up to {native:.0f} px"
                for name, native in sorted(losing.items())
            )
            logger.warning(
                f"final_pixel_size {self.final_pixel_size} downsamples a "
                f"{largest_crop:.0f} arcsec crop of {detail}. Raise it to "
                f"{max(losing.values()):.0f} to keep every survey's resolution."
            )

    @property
    def rng(self) -> np.random.Generator:
        """Return the random source for augmentation.

        DataLoader worker seeds produce independent parallel streams. A loader
        supplies a separate, rank-aware seed for main-process loading; direct
        dataset use falls back to Torch's process seed. A fixed seed, rank and
        loader topology reproduce these streams. Worker-count changes can
        reassign streams to samples.

        Returns
        -------
        result
            Generator for augmentation parameters.
        """
        if self._rng is None:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                self._rng = np.random.default_rng(worker_info.seed % 2**32)
            else:
                seed = (
                    torch.initial_seed()
                    if self._augmentation_seed is None
                    else self._augmentation_seed
                )
                self._rng = np.random.default_rng(seed % 2**32)
        return self._rng

    def set_augmentation_seed(self, seed: int) -> None:
        """Select and reset the main-process augmentation stream.

        `S2ZDataLoader` calls this with a seed derived separately from sampling
        and incorporating the training rank. Worker processes instead receive
        their rank-aware seeds from PyTorch's DataLoader generator.

        Parameters
        ----------
        seed
            Non-negative seed for this loader and rank.

        Raises
        ------
        ValueError
            If the seed is negative.
        """
        seed = int(seed)
        if seed < 0:
            raise ValueError(f"augmentation seed must be non-negative, got {seed}.")
        self._augmentation_seed = seed
        # A dataset may have been sampled directly before loader construction.
        # Do not carry that process-global stream into the loader contract.
        self._rng = None

    def _read_cutouts(
        self, source_indices: Sequence[int]
    ) -> dict[str, dict[int, tuple]]:
        """Read batched survey cutouts shard by shard.

        Issue one batched call per survey to reuse shard reads across cutouts.

        Parameters
        ----------
        source_indices
            Store source indices to read. Repeats are read once.

        Returns
        -------
        result
            Per survey, ``source index -> (cutout, wcs)``, holding only the
            sources that survey covers.
        """
        reads: dict[str, dict[int, tuple]] = {}
        for survey in self.surveys:
            substore = self.substores[survey]
            covered = [
                (source_index, substore.index_map[source_index])
                for source_index in dict.fromkeys(source_indices)
                if source_index in substore.index_map
            ]
            if not covered:
                continue
            cutouts = substore.get_cutouts(
                [local_index for _, local_index in covered],
                [
                    (
                        float(self.store.ra[source_index]),
                        float(self.store.dec[source_index]),
                    )
                    for source_index, _ in covered
                ],
                self._crop_size,
                self.store.disc_padding,
            )
            reads[survey] = {
                source_index: cutout
                for (source_index, _), cutout in zip(covered, cutouts, strict=True)
            }
        return reads

    def _assemble(self, source_index: int, reads: dict | None = None):
        """Return cutouts, WCS, pixel scales, and coverage for one source."""
        coords = (
            float(self.store.ra[source_index]),
            float(self.store.dec[source_index]),
        )
        centre = SkyCoord(ra=coords[0] * u.deg, dec=coords[1] * u.deg, frame=ICRS)

        surveys, mask = [], []
        groups_per_survey, wcs_per_survey, pixel_scales = [], [], []
        for survey in self.surveys:
            substore = self.substores[survey]
            local_index = substore.index_map.get(source_index)
            surveys.append(survey)
            if local_index is None:
                groups_per_survey.append(None)
                wcs_per_survey.append(None)
                pixel_scales.append(None)
                mask.append(False)
            else:
                read = (reads or {}).get(survey, {}).get(source_index)
                cutout, wcs = (
                    read
                    if read is not None
                    else substore.get_cutout(
                        local_index, coords, self._crop_size, self.store.disc_padding
                    )
                )
                groups_per_survey.append(
                    self.preprocess(cutout, self.has_freq_axis[survey])
                )
                wcs_per_survey.append(wcs.celestial)
                pixel_scales.append(tuple(self.pixel_scale[survey][local_index]))
                mask.append(True)
        return (
            coords,
            centre,
            surveys,
            mask,
            groups_per_survey,
            wcs_per_survey,
            pixel_scales,
        )

    def _views_for(self, source_index, augment, rng, reads: dict | None = None):
        """Return one source's views under a given augmentation."""
        _, centre, _, mask, groups, wcs, scales = self._assemble(source_index, reads)
        return augment(groups, wcs, scales, mask, centre, rng), mask

    def _shard_groups(self, source_indices: Sequence[int]) -> list[list[int]]:
        """Group source indices within the smallest selected survey shard.

        Dataset-store rows and every survey's power-of-two shards share their
        origin. Grouping by the smallest selected shard therefore keeps each
        group inside one shard of every selected survey while bounding the
        cutouts held during a sequential pass.

        Parameters
        ----------
        source_indices
            Store source indices to group. Their order is preserved within
            each group, and repeated indices are preserved.

        Returns
        -------
        result
            Groups in order of first appearance of their shard.
        """
        groups: dict[int, list[int]] = {}
        for source_index in source_indices:
            source_index = int(source_index)
            groups.setdefault(source_index // self.shard_size, []).append(source_index)
        return list(groups.values())

    @staticmethod
    def _accumulate_statistics(
        totals: dict[str, list[float]],
        survey: str,
        tensor: torch.Tensor,
    ) -> None:
        """Add finite pixels to a survey's running totals.

        Accumulate three scalar values per survey to bound memory use.

        Parameters
        ----------
        totals
            Per survey, ``[sum, sum of squares, count]``, extended in place.
        survey
            Survey the view belongs to.
        tensor
            One view, as ``(C, H, W)``.
        """
        values = tensor.detach().to(torch.float64)
        finite = torch.isfinite(values)
        if not bool(finite.any()):
            return
        kept = values[finite]
        running = totals.setdefault(survey, [0.0, 0.0, 0.0])
        running[0] += float(kept.sum())
        running[1] += float((kept * kept).sum())
        running[2] += float(kept.numel())

    @staticmethod
    def _finish_statistics(
        totals: dict[str, list[float]],
        epsilon: float = 1e-6,
    ) -> dict[str, tuple[float, float]]:
        """Convert running totals to per-survey means and standard deviations.

        Derive variance from double-precision sums in one pass.

        Parameters
        ----------
        totals
            Per survey, ``[sum, sum of squares, count]``.
        epsilon
            Floor on the standard deviation, guarding a constant survey.

        Returns
        -------
        result
            Per-survey mean and standard deviation.
        """
        statistics = {}
        for survey, (total, total_squares, count) in totals.items():
            if count <= 0:
                statistics[survey] = (0.0, 1.0)
                continue
            mean = total / count
            variance = max(total_squares / count - mean * mean, 0.0)
            statistics[survey] = (mean, max(float(np.sqrt(variance)), epsilon))
        return statistics

    def _all_reduce_statistics(
        self, totals: dict[str, list[float]]
    ) -> dict[str, list[float]]:
        """Sum statistics totals over the default distributed process group.

        Survey order is fixed by ``self.surveys`` so every rank presents the
        collective with an identically shaped tensor. NCCL can only reduce CUDA
        tensors; CPU-capable backends keep this small tensor on the CPU.
        """
        distributed = torch.distributed
        backend = str(distributed.get_backend()).lower()
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if backend == "nccl"
            else torch.device("cpu")
        )
        packed = torch.tensor(
            [totals[survey] for survey in self.surveys],
            dtype=torch.float64,
            device=device,
        )
        distributed.all_reduce(packed, op=distributed.ReduceOp.SUM)
        return {
            survey: row
            for survey, row in zip(self.surveys, packed.cpu().tolist(), strict=True)
        }

    def measure_statistics(
        self, split: str, *, distributed: bool = False
    ) -> dict[str, tuple[float, float]]:
        """Measure each survey's mean and standard deviation over one split.

        Measured on views that have been preprocessed and brought onto the
        canonical grid, with no randomisation and with the final resize where
        one is configured, so the numbers describe what a model is handed. In
        distributed mode, whole shard-aligned groups are divided between ranks
        and their totals are summed over the default process group.

        Parameters
        ----------
        split
            Split to measure over, ordinarily the one a model is fit on.
        distributed
            Opt into distributed partitioning and an all-reduce. Every rank in
            the default process group must call this method in the same survey
            order and for the same split, including ranks assigned no sources.
            Merely initialising a process group does not enable this mode.

        Returns
        -------
        result
            Per survey, its ``(mean, standard deviation)``.

        Raises
        ------
        TypeError
            If ``distributed`` is not boolean.
        ValueError
            If the split holds no source this dataset would serve.
        RuntimeError
            If distributed measurement is requested without an available,
            initialised ``torch.distributed`` process group.
        """
        if not isinstance(distributed, bool):
            raise TypeError(
                f"distributed must be a boolean, got {type(distributed).__name__}."
            )
        indices = self._indices_for(split)
        if len(indices) == 0:
            raise ValueError(
                f"Cannot measure normalisation statistics. No source this dataset "
                f"serves is in split '{split}'. Name another with "
                f"postprocessing.normalise.split, or give it a share in the "
                f"config's splits section."
            )
        rank, world_size = 0, 1
        if distributed:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Distributed statistics require torch.distributed support."
                )
            if not torch.distributed.is_initialized():
                raise RuntimeError(
                    "Distributed statistics require an initialised "
                    "torch.distributed process group."
                )
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        augment = S2ZAugmentation({}, self.servable_cutout_size)
        rng = np.random.default_rng(0)

        totals: dict[str, list[float]] = {
            survey: [0.0, 0.0, 0.0] for survey in self.surveys
        }
        source_groups = self._shard_groups(indices)
        if distributed:
            source_groups = source_groups[rank::world_size]
        local_sources = sum(len(group) for group in source_groups)
        distribution = (
            f" on rank {rank}/{world_size} ({local_sources} local)"
            if distributed
            else ""
        )
        logger.info(
            f"Measuring normalisation statistics over {len(indices)} source(s) "
            f"in '{split}'{distribution} ..."
        )
        with tqdm(
            total=local_sources,
            desc="Measuring statistics",
            disable=distributed and rank != 0,
        ) as progress:
            for source_group in source_groups:
                reads = self._read_cutouts(source_group)
                for source_index in source_group:
                    views, mask = self._views_for(
                        source_index, augment, rng, reads=reads
                    )
                    if self.final_pixel_size is not None:
                        views = self.resize_views(views, mask, self.final_pixel_size)
                    for survey, groups, covered in zip(
                        self.surveys, views, mask, strict=True
                    ):
                        if not covered or groups is None:
                            continue
                        for tensor, _ in groups:
                            self._accumulate_statistics(totals, survey, tensor)
                    progress.update()

        if distributed:
            totals = self._all_reduce_statistics(totals)
        statistics = self._finish_statistics(totals)
        unseen = [survey for survey in self.surveys if totals[survey][2] == 0]
        if unseen:
            logger.warning(
                f"No source in '{split}' is covered by survey(s) {unseen}, so their "
                f"statistics cannot be measured."
            )
        for survey in self.surveys:
            mean, deviation = statistics[survey]
            logger.info(
                f"{survey} has mean {mean:.6g} and standard deviation {deviation:.6g}."
            )
        return statistics

    def _indices_for(self, split_name: str) -> np.ndarray:
        """Return the source indices one split serves.

        Parameters
        ----------
        split_name
            Split to select.

        Returns
        -------
        result
            Source indices, in store order. Position within it is the sample
            index; the value is the source index, which each survey's map then
            turns into a position of its own.
        """
        return np.flatnonzero(
            self._eligible & (self.split_values == split_name)
        ).astype(np.int64)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return every view of one source.

        The views are the surveys, so a source covered by four of them yields four
        views, each through a different beam and pixel grid, aligned with each other
        on the sky.

        One geometric state is drawn per source and applied to every view, which is
        what leaves them aligned.

        Parameters
        ----------
        index
            Sample index, within this split and subset.

        Returns
        -------
        result
            ``data``, per survey a ``(C, H, W)`` tensor, or None where the
            survey does not cover the source; ``mask``, per survey whether it
            covers the source; ``surveys``, the survey names in served order;
            ``catalogue``, this source's value for each catalogue column
            beyond the standard four; ``source_coords`` as ``(ra, dec)`` in
            degrees; and ``source_index``, the store's own index for it. With
            ``label_by`` set, also ``label``, the integer standing for this
            source's category in ``label_classes``.

            Whether a survey's channel axis is frequency is fixed per survey
            and not per sample, and is read from `has_freq_axis`.
        """
        # A loader resolves this before iteration. Direct dataset use reaches
        # the same lazy single-process policy on its first requested sample.
        if self._in_memory is None:
            self.resolve_in_memory(num_workers=0, local_world_size=1)
        return self._sample(int(self.subset_indices[index]))

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        """Return every view of several sources, reading them together.

        Torch DataLoader invokes this method for batched fetching.

        Parameters
        ----------
        indices
            Sample indices, within this split and subset.

        Returns
        -------
        result
            One sample per index, in the order asked for, each as
            ``__getitem__`` would have returned it.
        """
        if self._in_memory is None:
            self.resolve_in_memory(num_workers=0, local_world_size=1)
        source_indices = [int(self.subset_indices[index]) for index in indices]
        reads = self._read_cutouts(source_indices)
        return [self._sample(source_index, reads) for source_index in source_indices]

    def _sample(self, source_index: int, reads: dict | None = None) -> dict[str, Any]:
        """Return one source's sample, from cutouts already read where given."""
        (
            coords,
            centre,
            surveys,
            mask,
            groups_per_survey,
            wcs_per_survey,
            pixel_scales,
        ) = self._assemble(source_index, reads)

        # One geometric state aligns every survey view spatially.
        views = self.augment(
            groups_per_survey,
            wcs_per_survey,
            pixel_scales,
            mask,
            centre,
            self.rng,
        )

        if self.final_pixel_size is not None:
            views = self.resize_views(views, mask, self.final_pixel_size)
        views = self.postprocess(views, mask)

        sample = {
            "data": dict(zip(surveys, self._unwrap(surveys, views, mask), strict=True)),
            "mask": dict(zip(surveys, mask, strict=True)),
            "surveys": surveys,
            "catalogue": {
                name: values[source_index].item()
                if hasattr(values[source_index], "item")
                else values[source_index]
                for name, values in self.store.extra_columns.items()
            },
            "source_coords": coords,
            "source_index": source_index,
        }
        if self._labels is not None:
            sample["label"] = int(self._labels[source_index])
        return sample

    @staticmethod
    def _unwrap(surveys: Sequence[str], views: list, mask: Sequence[bool]) -> list:
        """Take the image tensor out of each survey's groups.

        Parameters
        ----------
        surveys
            Survey names, used in messages.
        views
            Per survey, its ``(tensor, has_freq_axis)`` groups, or None.
        mask
            Whether each survey covers the source.

        Returns
        -------
        result
            Per survey, its tensor, or None where it does not cover the source.

        Raises
        ------
        NotImplementedError
            If a survey yields more than one group, which a served sample has
            no room for.
        """
        output = []
        for survey, groups, covered in zip(surveys, views, mask, strict=True):
            if not covered or groups is None:
                output.append(None)
                continue
            if len(groups) != 1:
                raise NotImplementedError(
                    f"Survey {survey!r} produced {len(groups)} groups. A sample "
                    f"holds the image alone."
                )
            output.append(groups[0][0])
        return output

    def __len__(self) -> int:
        """Return the number of sources served.

        Returns
        -------
        result
            Number of served sources.
        """
        return len(self.subset_indices)

    def _display_norm(self, image: np.ndarray, scale: str | None):
        """Build the intensity normalisation for one displayed image.

        Parameters
        ----------
        image
            Image the range is measured over.
        scale
            Stretch to apply.

        Returns
        -------
        result
            None for linear scaling, for an unrecognised scale, or where the
            image holds no finite pixel to measure a range from.
        """
        if scale is None or scale == "linear":
            return None
        finite = image[np.isfinite(image)]
        if finite.size == 0:
            return None
        vmin, vmax = finite.min(), finite.max()
        if scale == "log":
            from matplotlib.colors import LogNorm

            # LogNorm requires a strictly positive vmin.
            positive = finite[finite > 0]
            if positive.size == 0:
                return None
            return LogNorm(vmin=positive.min(), vmax=vmax)
        if scale == "arcsinh":
            from matplotlib.colors import AsinhNorm

            return AsinhNorm(vmin=vmin, vmax=vmax)
        logger.warning(
            f"Unknown scale '{scale}', falling back to linear.", stacklevel=2
        )
        return None

    def _sample_index_for(self, source: int | str) -> int:
        """Return the sample index one source is served at.

        An integer is already a sample index, so it is taken as it stands;
        anything else is a source ID, looked up among the sources this split
        and subset serves.

        Parameters
        ----------
        source
            Sample index, within this split and subset, or a source ID.

        Returns
        -------
        result
            Sample index, within this split and subset.

        Raises
        ------
        KeyError
            Where the ID is not one this split and subset serves, saying
            whether the store holds it at all.
        """
        if isinstance(source, (int, np.integer)) and not isinstance(source, bool):
            return int(source)
        source_id = str(source)
        if self._sample_index_by_source_id is None:
            self._sample_index_by_source_id = {
                self._source_ids[source_index]: sample_index
                for sample_index, source_index in enumerate(self.subset_indices)
            }
        sample_index = self._sample_index_by_source_id.get(source_id)
        if sample_index is None:
            where = (
                f"is not in split '{self.split}', or was "
                f"removed by this configuration's filtering"
                if source_id in set(self._source_ids)
                else "is not in the dataset store"
            )
            raise KeyError(f"Source ID {source_id!r} {where}.")
        return sample_index

    def plot(
        self,
        source: int | str,
        show: bool = True,
        scale: str | None = None,
        cmap: str = "gray",
    ) -> "plt.Figure":
        """Show one view of a source, one panel per survey.

        Parameters
        ----------
        source
            Sample index, within this split and subset, as ``__getitem__``
            takes; or the source's ID, as a string, which is looked up among
            the sources this split and subset serves. A numeric ID is still a
            string, since an integer is read as an index.
        show
            Whether to display the figure. Pass False to embed it somewhere
            else, such as an ``ipywidgets.Output``, where letting the backend
            display it as well would draw it twice.
        scale
            Intensity scaling for the displayed images. None or 'linear' shows
            the data as-is. 'log' applies a logarithmic stretch (LogNorm),
            'arcsinh' an inverse-hyperbolic-sine stretch.
        cmap
            Colormap for the displayed images. Any matplotlib colormap name is
            accepted, and the three listed are the intended options.

        Returns
        -------
        result
            The figure, displayed or not. Close it when done with it if this
            was called with ``show=False``.
        """
        out = self.__getitem__(self._sample_index_for(source))
        coords = out["source_coords"]

        panels = []
        for survey in out["surveys"]:
            if not out["mask"][survey]:
                panels.append((f"{survey}\n(missing)", None))
                continue
            tensor = out["data"][survey]
            arr = (
                tensor.cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
            )
            if arr.shape[0] == 1 or not self.has_freq_axis[survey]:
                panels.append((survey, arr[0]))
            else:
                c = arr.shape[0] // 2
                logger.warning(
                    f"Cutout from {survey} is a spectral cube with "
                    f"{arr.shape[0]} channels, so only the middle channel "
                    f"({c}) is shown.",
                    stacklevel=2,
                )
                panels.append((f"{survey}\nchannel {c}/{arr.shape[0]}", arr[c]))

        n = len(panels)
        fig, axes = plt.subplots(1, n, figsize=(n * 3, 3), squeeze=False)
        title = f"RA = {coords[0]:.4f}, Dec = {coords[1]:.4f}"
        if self.label_classes is not None:
            title += f"    {self.label_by} = {self.label_classes[out['label']]}"
        fig.suptitle(title)
        axes = axes[0]
        for ax, (title, image) in zip(axes, panels, strict=True):
            if image is None:
                ax.text(
                    0.5,
                    0.5,
                    "Source not in survey",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                )
                ax.set_box_aspect(1)
            else:
                image = np.squeeze(image)
                norm = self._display_norm(image, scale)
                ax.imshow(image, origin="lower", cmap=cmap, norm=norm)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        if show:
            plt.show()
        return fig
