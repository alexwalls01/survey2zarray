"""Batch sources so that each batch is a whole number of aligned row groups."""

from collections.abc import Iterator, Sequence

import numpy as np
import torch

from survey2zarray.stores.dataset_store.layout import is_power_of_two
from survey2zarray.utils.logging import get_logger

logger = get_logger(__name__)

# Four blocks draw each batch from four groups with one quarter of the file
# opens required by per-sample shuffling.
BLOCKS_PER_BATCH: int = 4

# Two blocks per group preserve a choice when drawing within each group.
MIN_BLOCKS_PER_GROUP: int = 2

# Eight group rows per batch sample bound excess batchmate repetition at 0.78%
# under block_size**2 / (batch_size * group_size).
GROUP_ROWS_PER_SAMPLE: int = 8


def _validate_topology(rank: int, world_size: int) -> tuple[int, int]:
    """Return a valid explicit rank topology."""
    rank, world_size = int(rank), int(world_size)
    if world_size < 1:
        raise ValueError(f"world_size must be at least 1, got {world_size}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"rank must be between 0 and world_size - 1, got rank={rank} "
            f"and world_size={world_size}."
        )
    return rank, world_size


def _repeat_to_length(values: list[int], length: int) -> list[int]:
    """Repeat the beginning of a non-empty plan until it reaches a length."""
    if not values or length <= len(values):
        return values[:length]
    missing = length - len(values)
    repeats = (missing + len(values) - 1) // len(values)
    return values + (values * repeats)[:missing]


class RankAwareBatchSampler(torch.utils.data.Sampler):
    """Partition one deterministic plain-shuffle epoch plan across ranks.

    Rank and world size are explicit so an initialised process group never
    silently changes data loading. For more than one rank, ``drop_last=True``
    drops the global tail that cannot form one full local batch per rank;
    ``drop_last=False`` repeats the minimum prefix needed to form it.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1 sample, got {batch_size}.")
        self.batch_size = batch_size
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.rank, self.world_size = _validate_topology(rank, world_size)
        self.seed = int(
            generator.initial_seed() if generator is not None else torch.initial_seed()
        )
        self.epoch = 0
        self._size = len(dataset)
        self.n_batches = self._count_batches()
        self._report_tail()

    def _count_batches(self) -> int:
        geometry = self.batch_size * self.world_size
        if self.drop_last:
            return self._size // geometry
        return (self._size + geometry - 1) // geometry

    def _report_tail(self) -> None:
        if self.world_size == 1 or not self._size:
            return
        total = self.n_batches * self.batch_size * self.world_size
        if self.drop_last and total < self._size:
            logger.info(
                f"Distributed batching drops {self._size - total} sample(s) from "
                f"the deterministic global epoch tail across {self.world_size} ranks."
            )
        elif total > self._size:
            logger.info(
                f"Distributed batching repeats {total - self._size} sample(s) from "
                f"the deterministic global epoch prefix across {self.world_size} ranks."
            )

    def _generator(self) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed((self.seed + self.epoch) % 2**63)
        return generator

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic shuffle used for a training epoch."""
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle:
            plan = torch.randperm(self._size, generator=self._generator()).tolist()
        else:
            plan = list(range(self._size))
        if self.world_size == 1 and not self.drop_last:
            return iter(
                [
                    plan[start : start + self.batch_size]
                    for start in range(0, len(plan), self.batch_size)
                ]
            )
        total = self.n_batches * self.batch_size * self.world_size
        plan = _repeat_to_length(plan, total)
        geometry = self.batch_size * self.world_size
        start = self.rank * self.batch_size
        return iter(
            [
                plan[offset + start : offset + start + self.batch_size]
                for offset in range(0, total, geometry)
            ]
        )

    def __len__(self) -> int:
        return self.n_batches


class BlockAlignedBatchSampler(torch.utils.data.Sampler):
    """Batch samples in blocks aligned to store shards.

    Group rows by ``group_size``, shuffle within each group, and redraw blocks
    each epoch. Pack whole blocks into each batch and combine leftover samples
    from adjacent groups. This layout limits shard opens while varying
    blockmates between epochs. Store ordering must contain no model input
    signal because groups constrain which sources can share a batch.

    A batch opens ``(batch_size / block_size) * max(1, group_size /
    shard_rows)`` files per fully covered survey. Empty shards require no read.

    Parameters
    ----------
    dataset
        Dataset with ``subset_indices`` and per-survey sub-stores.
    batch_size
        Largest number of samples a batch may hold.
    shuffle
        Whether to draw fresh blocks within each group when `set_epoch`
        selects a new epoch.
    generator
        Random source. Defaults to the global torch generator.
    block_size
        Sources per block. Must divide ``batch_size``. Larger values reduce
        shard opens and the number of groups represented per batch.
    group_size
        Store rows per sampling group. Must be a power of two. Larger values
        widen the blockmate pool and may increase shard opens.
    drop_last
        Whether to omit the final partial batch.
    rank, world_size
        Explicit distributed topology. One global block-aligned epoch plan is
        partitioned across ranks. These values are never inferred from
        ``torch.distributed``.

    Attributes
    ----------
    blocks
        Sample indices per block for the current epoch.
    block_size
        Sources to a block, as resolved.
    blocks_per_batch
        ``batch_size // block_size``.
    group_size
        Rows per sampling group.
    n_batches
        Batches yielded per epoch.
    epoch
        Epoch selected by `set_epoch`, initially zero.

    Raises
    ------
    TypeError
        If the dataset lacks ``subset_indices``.
    ValueError
        If sizes violate the batching constraints.
    """

    @staticmethod
    def default_group_size(batch_size: int) -> int:
        """Return the rows a group should span, for a batch of a given size.

        Scale groups with batch size to bound excess batchmate repetition at
        0.78% under ``block_size**2 / (batch_size * group_size)``.

        Parameters
        ----------
        batch_size
            Largest number of samples a batch may hold.

        Returns
        -------
        result
            ``GROUP_ROWS_PER_SAMPLE`` rows a sample, rounded up to a power of
            two.
        """
        rows = GROUP_ROWS_PER_SAMPLE * int(batch_size)
        return 1 << (rows - 1).bit_length()

    @staticmethod
    def survey_shard_rows(dataset) -> dict[str, int]:
        """Return shard rows for each survey served by a dataset.

        Parameters
        ----------
        dataset
            Dataset being batched.

        Returns
        -------
        result
            Rows per shard and survey. Empty when no sub-stores are available.
        """
        substores = getattr(dataset, "substores", None) or {}
        return {
            survey: int(substores[survey].shard_size)
            for survey in getattr(dataset, "surveys", None) or []
            if survey in substores
        }

    @staticmethod
    def choose_block_size(sources_per_group: int, batch_size: int) -> int:
        """Return the block a batch should be built from.

        Choose the largest divisor of ``batch_size`` within the group and batch
        diversity limits. Larger blocks reduce shard opens.

        Parameters
        ----------
        sources_per_group
            Median served sources per group.
        batch_size
            Largest number of samples a batch may hold.

        Returns
        -------
        result
            Block size, dividing ``batch_size`` exactly and no larger than
            either a `MIN_BLOCKS_PER_GROUP` share of the group or a
            `BLOCKS_PER_BATCH` share of the batch.
        """
        limit = min(
            max(1, int(sources_per_group) // MIN_BLOCKS_PER_GROUP),
            max(1, int(batch_size) // BLOCKS_PER_BATCH),
        )
        for size in range(limit, 0, -1):
            if int(batch_size) % size == 0:
                return size
        return 1

    @staticmethod
    def served_per_group(source_indices: Sequence[int], group_size: int) -> int:
        """Return median served sources across populated groups.

        The median limits the influence of partial groups at split boundaries.

        Parameters
        ----------
        source_indices
            Store source index per sample.
        group_size
            Rows a group spans.

        Returns
        -------
        result
            Median served sources per group, at least one.
        """
        keys = np.asarray(source_indices, dtype=np.int64) // max(1, int(group_size))
        if keys.size == 0:
            return 1
        _, counts = np.unique(keys, return_counts=True)
        return max(1, int(np.median(counts)))

    @staticmethod
    def build_blocks(source_indices: Sequence[int], block_size: int) -> list[list[int]]:
        """Cut the served samples into consecutive blocks of one size.

        Cut by split position to fill blocks after source filtering. Preserve
        store order for sequential reads.

        Parameters
        ----------
        source_indices
            Store source index per sample, in sample index order.
        block_size
            Sources to a block.

        Returns
        -------
        result
            Sample indices per block in store order. The final block may be
            partial.
        """
        n = len(source_indices)
        return [
            list(range(start, min(start + block_size, n)))
            for start in range(0, n, block_size)
        ]

    @staticmethod
    def resample_blocks(
        source_indices: Sequence[int],
        block_size: int,
        group_size: int,
        generator: torch.Generator | None = None,
    ) -> list[list[int]]:
        """Draw blocks from within each group, redrawing what shares one.

        Draw full blocks within each group. Combine leftovers in group order so
        partial blocks span adjacent groups. This produces
        ``ceil(served / block_size)`` blocks.

        Parameters
        ----------
        source_indices
            Store source index per sample, in sample index order.
        block_size
            Sources to a block.
        group_size
            Rows a group spans, which decides which sources may share a block.
        generator
            Source for the shuffle within each group.

        Returns
        -------
        result
            Sample indices per block. The final block may be partial.
        """
        keys = np.asarray(source_indices, dtype=np.int64) // max(1, int(group_size))
        # Ascending group positions preserve store order among leftover samples.
        order = np.argsort(keys, kind="stable")
        groups = np.split(order, np.flatnonzero(np.diff(keys[order])) + 1)

        blocks: list[list[int]] = []
        leftovers: list[np.ndarray] = []
        for group in groups:
            drawn = group[torch.randperm(len(group), generator=generator).numpy()]
            whole = len(drawn) - len(drawn) % block_size
            blocks.extend(
                drawn[start : start + block_size].tolist()
                for start in range(0, whole, block_size)
            )
            leftovers.append(drawn[whole:])

        spare = np.concatenate(leftovers) if leftovers else np.empty(0, dtype=np.int64)
        blocks.extend(
            spare[start : start + block_size].tolist()
            for start in range(0, len(spare), block_size)
        )
        return blocks

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
        block_size: int | None = None,
        group_size: int | None = None,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        subset_indices = getattr(dataset, "subset_indices", None)
        if subset_indices is None:
            raise TypeError(
                f"Expected a dataset with subset_indices, but got "
                f"{type(dataset).__name__}. This sampler requires S2ZDataset."
            )
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1 sample, got {batch_size}.")
        source_indices = np.asarray(subset_indices, dtype=np.int64)
        if group_size is None:
            group_size = self.default_group_size(batch_size)
        group_size = int(group_size)
        if not is_power_of_two(group_size):
            raise ValueError(
                f"group_size must be a power-of-two row count, got {group_size}. "
                f"Other values can make blocks straddle store shards."
            )
        if block_size is None:
            served = self.served_per_group(source_indices, group_size)
            block_size = self.choose_block_size(served, batch_size)
            logger.info(
                f"Blocks are drawn at {block_size} source(s) from groups of "
                f"{group_size} row(s) serving {served} of them, against a batch of "
                f"{batch_size}, so that {batch_size // block_size} of them build a "
                f"batch. A larger block_size opens fewer files and draws each batch "
                f"from fewer groups."
            )
        block_size = int(block_size)
        if block_size < 1:
            raise ValueError(f"block_size must be at least 1 source, got {block_size}.")
        if batch_size % block_size:
            divisors = [n for n in range(1, batch_size + 1) if batch_size % n == 0]
            raise ValueError(
                f"block_size must divide batch_size, got {block_size} and "
                f"{batch_size}. "
                f"Use one of {divisors}."
            )

        self.batch_size = batch_size
        self.block_size = block_size
        self.shuffle = bool(shuffle)
        self.seed = int(
            generator.initial_seed() if generator is not None else torch.initial_seed()
        )
        self.epoch = 0
        self.drop_last = bool(drop_last)
        self.rank, self.world_size = _validate_topology(rank, world_size)
        self.group_size = group_size
        self._source_indices = source_indices
        self.blocks = self.build_blocks(source_indices, block_size)

        # Only the split remainder can produce a partial block.
        self.blocks_per_batch = batch_size // block_size
        self.n_batches = self._count_batches()
        self._report_opens(dataset)
        self._report_tail()

    def _generator(self) -> torch.Generator:
        """Return the sampling stream for the selected epoch."""
        generator = torch.Generator()
        generator.manual_seed((self.seed + self.epoch) % 2**63)
        return generator

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic block draw used for a training epoch."""
        self.epoch = int(epoch)

    def _report_tail(self) -> None:
        """Report intentional distributed tail dropping or repetition."""
        if self.world_size == 1 or not len(self._source_indices):
            return
        served = len(self._source_indices)
        total = self.n_batches * self.batch_size * self.world_size
        if self.drop_last and total < served:
            logger.info(
                f"Distributed block batching drops {served - total} sample(s) from "
                f"the deterministic global epoch tail across {self.world_size} ranks."
            )
        elif total > served:
            logger.info(
                f"Distributed block batching repeats {total - served} sample(s) from "
                f"the deterministic global epoch prefix across {self.world_size} ranks."
            )

    def _report_opens(self, dataset) -> None:
        """Report expected shard opens for each survey.

        Expose the I/O effect of ``group_size`` while retaining its sampling
        effect.

        Parameters
        ----------
        dataset
            Dataset being batched.
        """
        rows = self.survey_shard_rows(dataset)
        if not rows:
            return
        opens = ", ".join(
            f"{self.blocks_per_batch * max(1, self.group_size // shard)} of {survey} "
            f"(shards of {shard} row(s))"
            for survey, shard in rows.items()
        )
        logger.info(
            f"A batch of {self.batch_size} takes {self.blocks_per_batch} block(s) of "
            f"{self.block_size} source(s) from groups of {self.group_size} row(s) and "
            f"opens {opens}. A smaller group_size opens fewer files and draws "
            f"blockmates from a narrower pool."
        )

    @staticmethod
    def pack(
        blocks: Sequence[Sequence[int]], order: Sequence[int], batch_size: int
    ) -> list[list[int]]:
        """Group whole blocks into batches of no more than a given size.

        Preserve block order and place an overflowing block in the next batch.
        A block larger than ``batch_size`` forms one batch.

        Parameters
        ----------
        blocks
            Sample indices per block.
        order
            Block keys, in the order to pack them.
        batch_size
            Largest number of samples a batch may hold.

        Returns
        -------
        result
            Block keys per batch, in the order given.
        """
        batches: list[list[int]] = []
        current: list[int] = []
        held = 0
        for key in order:
            size = len(blocks[key])
            if current and held + size > batch_size:
                batches.append(current)
                current, held = [], 0
            current.append(key)
            held += size
        if current:
            batches.append(current)
        return batches

    def _block_order(
        self,
        blocks: Sequence[Sequence[int]],
        generator: torch.Generator | None = None,
    ) -> list[int]:
        """Return the block keys in the order to pack this epoch.

        Shuffle blocks, then sort by size while preserving shuffled order among
        equal sizes. This confines the partial block to a batch end.
        """
        if not self.shuffle:
            return list(range(len(blocks)))
        order = torch.randperm(len(blocks), generator=generator).tolist()
        return sorted(order, key=lambda key: -len(blocks[key]))

    def _count_batches(self) -> int:
        """Return how many batches an epoch yields, from the block count alone.

        Both layouts produce ``ceil(served / block_size)`` blocks.
        """
        served = sum(len(block) for block in self.blocks)
        geometry = self.batch_size * self.world_size
        if self.drop_last:
            return served // geometry
        return (served + geometry - 1) // geometry

    def draw_blocks(self, generator: torch.Generator | None = None) -> list[list[int]]:
        """Return the blocks for one epoch, drawn fresh where shuffling.

        Returns
        -------
        result
            Sample indices per block.
        """
        if not self.shuffle:
            return self.build_blocks(self._source_indices, self.block_size)
        return self.resample_blocks(
            self._source_indices,
            self.block_size,
            self.group_size,
            generator,
        )

    def __iter__(self) -> Iterator[list[int]]:
        """Return an iterator over one epoch's batches of sample indices.

        Returns
        -------
        result
            Sample indices per batch.
        """
        generator = self._generator()
        self.blocks = self.draw_blocks(generator)

        # Sorting the sole partial block last confines a short batch to the end.
        keys = self._block_order(self.blocks, generator)
        if self.world_size == 1:
            batches = self.pack(self.blocks, keys, self.batch_size)
            if self.drop_last and batches:
                held = sum(len(self.blocks[key]) for key in batches[-1])
                if held < self.batch_size:
                    batches = batches[:-1]
            return iter(
                [
                    [sample for key in batch for sample in self.blocks[key]]
                    for batch in batches
                ]
            )

        plan = [sample for key in keys for sample in self.blocks[key]]
        total = self.n_batches * self.batch_size * self.world_size
        plan = _repeat_to_length(plan, total)
        # Expose the actual distributed plan as complete logical blocks. The
        # only synthetic block is the padded tail when the split is uneven.
        self.blocks = [
            plan[start : start + self.block_size]
            for start in range(0, len(plan), self.block_size)
        ]
        geometry = self.batch_size * self.world_size
        start = self.rank * self.batch_size
        return iter(
            [
                plan[offset + start : offset + start + self.batch_size]
                for offset in range(0, total, geometry)
            ]
        )

    def __len__(self) -> int:
        """Return the number of batches an epoch yields.

        Exact, and the same every epoch. Every draw gives the same number of
        blocks all but one of which are full, and `_block_order` puts the short
        one last, so the packing runs the same sequence of sizes however the
        blocks fell.

        Returns
        -------
        result
            Batches per epoch.
        """
        return self.n_batches
