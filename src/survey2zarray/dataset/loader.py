"""Load batches according to dataset storage layout."""

import multiprocessing

import torch

from survey2zarray.dataset.collate import collate_views
from survey2zarray.dataset.sampler import (
    BlockAlignedBatchSampler,
    RankAwareBatchSampler,
)
from survey2zarray.utils.logging import get_logger
from survey2zarray.utils.parallel import LOADER_CAP, loader_workers

logger = get_logger(__name__)

# Keep augmentation draws independent of the sampling generator even for rank
# zero. The odd rank stride is invertible modulo 2**63, so supported rank values
# cannot alias one another for a fixed base seed.
_SEED_MODULUS = 2**63
_AUGMENTATION_STREAM_OFFSET = 0x2D358DCCAA6C78A5
_RANK_SEED_STRIDE = 0x1E3779B97F4A7C15


class S2ZDataLoader(torch.utils.data.DataLoader):
    """Load batches using the dataset's resolved storage mode.

    Use shard-aligned blocks for streamed data to limit file opens. Use plain
    shuffling for in-memory data to draw each batch from the full split.

    Parameters
    ----------
    dataset
        Dataset to load from.
    batch_size
        Samples a batch holds.
    shuffle
        Whether to draw a fresh order when `set_epoch` selects a new epoch on
        a rank-aware sampler.
    drop_last
        Whether to omit the final partial batch.
    block_aligned
        Whether to use shard-aligned blocks. ``"auto"`` follows the resolved
        storage mode.
    block_size
        Sources per shard-aligned block.
    group_size
        Store rows per shard-aligned sampling group.
    generator
        Base random source. Its initial seed defines the rank-independent
        sampling plan and separate rank-aware augmentation/worker streams.
    rank, world_size
        Explicit training-process topology. Values are never inferred from an
        initialised process group. More than one rank partitions a single
        deterministic epoch plan and requires `set_epoch` each epoch.
    local_world_size
        Training ranks sharing this node. Required to let distributed
        ``in_memory="auto"`` hold data. If omitted for a multi-rank job, the
        loader streams and conservatively shares readers across `world_size`.
    num_workers
        Reader processes per rank. When omitted, `loader_workers` chooses a
        polite node-aware default. An explicit value is honoured.
    collate_fn
        Batch collation function. Defaults to `collate_views`.
    **kwargs
        Passed to `torch.utils.data.DataLoader`, such as ``pin_memory`` and
        ``persistent_workers``.

    Attributes
    ----------
    block_aligned
        Resolved shard-aligned batching mode.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        block_aligned: bool | str = "auto",
        block_size: int | None = None,
        group_size: int | None = None,
        generator: torch.Generator | None = None,
        collate_fn=None,
        rank: int = 0,
        world_size: int = 1,
        local_world_size: int | None = None,
        num_workers: int | None = None,
        **kwargs,
    ) -> None:
        if not isinstance(block_aligned, bool) and block_aligned != "auto":
            raise ValueError(
                f"block_aligned must be True, False or 'auto', got {block_aligned!r}."
            )
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.world_size < 1:
            raise ValueError(f"world_size must be positive, got {self.world_size}.")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"rank must be between 0 and {self.world_size - 1}, got {self.rank}."
            )
        if local_world_size is not None:
            local_world_size = int(local_world_size)
            if not 1 <= local_world_size <= self.world_size:
                raise ValueError(
                    f"local_world_size must be between 1 and world_size "
                    f"({self.world_size}), got {local_world_size}."
                )
        elif self.world_size == 1:
            local_world_size = 1
        self.local_world_size = local_world_size

        # Without node topology, assume all ranks may contend for CPUs and I/O.
        # Memory auto-resolution is stricter and streams rather than making that
        # potentially very large shared allocation.
        resource_ranks = local_world_size or self.world_size
        if local_world_size is None:
            logger.warning(
                f"local_world_size was omitted for world_size={self.world_size}. "
                f"in_memory='auto' will stream, and automatic reader capacity "
                f"will conservatively assume all ranks share this node."
            )
        if num_workers is None:
            num_workers = loader_workers(
                ceiling=len(dataset), local_world_size=resource_ranks
            )
        else:
            num_workers = int(num_workers)
            if num_workers < 0:
                raise ValueError(
                    f"num_workers must be non-negative, got {num_workers}."
                )
            self._warn_if_workers_exceed_cap(num_workers, resource_ranks)

        resolver = getattr(dataset, "resolve_in_memory", None)
        start_method = self._multiprocessing_start_method(
            kwargs.get("multiprocessing_context")
        )
        if callable(resolver):
            in_memory = bool(
                resolver(
                    num_workers=num_workers,
                    local_world_size=self.local_world_size,
                    multiprocessing_start_method=start_method,
                )
            )
        else:
            in_memory = bool(getattr(dataset, "in_memory", False))
        if block_aligned == "auto":
            block_aligned = not in_memory
        self.block_aligned = bool(block_aligned)

        if collate_fn is None:
            collate_fn = collate_views
        self._warn_if_workers_copy(dataset, in_memory, num_workers, start_method)
        kwargs["num_workers"] = num_workers

        base_seed = int(
            generator.initial_seed() if generator is not None else torch.initial_seed()
        )
        augmentation_seed = self._augmentation_seed(base_seed, self.rank)
        augmentation_generator = self._generator(augmentation_seed)
        set_augmentation_seed = getattr(dataset, "set_augmentation_seed", None)
        if callable(set_augmentation_seed):
            set_augmentation_seed(augmentation_seed)

        if self.block_aligned:
            sampler = BlockAlignedBatchSampler(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                generator=generator,
                block_size=block_size,
                group_size=group_size,
                drop_last=drop_last,
                rank=self.rank,
                world_size=self.world_size,
            )
            # PyTorch rejects loader batching options alongside a batch sampler.
            super().__init__(
                dataset,
                batch_sampler=sampler,
                collate_fn=collate_fn,
                generator=augmentation_generator,
                **kwargs,
            )
        elif self.world_size != 1 or self.rank != 0:
            sampler = RankAwareBatchSampler(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                generator=generator,
                drop_last=drop_last,
                rank=self.rank,
                world_size=self.world_size,
            )
            super().__init__(
                dataset,
                batch_sampler=sampler,
                collate_fn=collate_fn,
                generator=augmentation_generator,
                **kwargs,
            )
        else:
            sampler = None
            if shuffle:
                # DataLoader otherwise gives one generator to both RandomSampler
                # and worker base-seed creation. Keep those streams independent.
                sampler = torch.utils.data.RandomSampler(
                    dataset, generator=self._generator(base_seed)
                )
            super().__init__(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler,
                drop_last=drop_last,
                generator=augmentation_generator,
                collate_fn=collate_fn,
                **kwargs,
            )

    @staticmethod
    def _generator(seed: int) -> torch.Generator:
        """Return an independent CPU generator at one seed."""
        return torch.Generator().manual_seed(int(seed) % _SEED_MODULUS)

    @staticmethod
    def _augmentation_seed(base_seed: int, rank: int) -> int:
        """Domain-separate augmentation and incorporate the training rank."""
        return (
            int(base_seed) + _AUGMENTATION_STREAM_OFFSET + int(rank) * _RANK_SEED_STRIDE
        ) % _SEED_MODULUS

    @staticmethod
    def _warn_if_workers_exceed_cap(num_workers: int, local_world_size: int) -> None:
        """Warn when an explicit per-rank override exceeds the node I/O cap."""
        total = int(num_workers) * int(local_world_size)
        if total <= LOADER_CAP:
            return
        logger.warning(
            f"Explicit num_workers={num_workers} across {local_world_size} local "
            f"rank(s) creates {total} concurrent readers, above the polite "
            f"node-wide cap of {LOADER_CAP}. The explicit override is honoured."
        )

    @staticmethod
    def _multiprocessing_start_method(context) -> str:
        """Resolve a DataLoader context or the platform's active default."""
        if isinstance(context, str):
            return context
        if context is not None:
            return context.get_start_method()
        method = multiprocessing.get_start_method(allow_none=True)
        if method is not None:
            return method
        return multiprocessing.get_context().get_start_method()

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic sample plan for a training epoch.

        Call this on every rank before iterating a distributed training epoch.
        It is a no-op for the ordinary single-rank plain sampler.
        """
        set_epoch = getattr(self.batch_sampler, "set_epoch", None)
        if set_epoch is not None:
            set_epoch(epoch)

    @staticmethod
    def _warn_if_workers_copy(
        dataset,
        in_memory: bool,
        num_workers: int,
        start_method: str,
    ) -> None:
        """Warn when spawned readers copy in-memory cutouts.

        Spawned workers each receive a dataset copy. Forked workers share
        read-only rows through copy-on-write memory.

        Parameters
        ----------
        dataset
            Dataset the loader was given.
        in_memory
            Whether the dataset holds its cutouts in memory.
        num_workers
            Reader processes the loader was asked for.
        start_method
            Multiprocessing method the loader will use.
        """
        if not in_memory or int(num_workers) < 1:
            return
        if start_method == "fork":
            return
        held = getattr(dataset, "memory_bytes", 0)
        logger.warning(
            f"The platform spawns {num_workers} worker process(es). "
            f"Each is sent its own copy of the {held / 1e6:,.0f} MB held in memory, "
            f"so the loader wants {held * (int(num_workers) + 1) / 1e9:,.1f} GB. "
            f"Use in_memory=False for store reads, or load in the main "
            f"process with num_workers=0."
        )
