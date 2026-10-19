"""Rank-aware sampling without requiring a live process group."""

from collections import Counter

import numpy as np
import pytest
import torch

from survey2zarray.dataset.loader import S2ZDataLoader
from survey2zarray.dataset.sampler import (
    BlockAlignedBatchSampler,
    RankAwareBatchSampler,
)


class SampleIndices:
    """Small dataset exposing the row mapping required by the block sampler."""

    def __init__(self, size: int):
        self.subset_indices = np.arange(size)
        self.substores = {}
        self.surveys = []
        self.augmentation_seed = None

    def __len__(self) -> int:
        return len(self.subset_indices)

    def __getitem__(self, index: int) -> int:
        return index

    def set_augmentation_seed(self, seed: int) -> None:
        self.augmentation_seed = int(seed)


def samplers(size, *, drop_last=False, block_aligned=True):
    """Return equivalent samplers for three explicit ranks."""
    sampler_type = BlockAlignedBatchSampler if block_aligned else RankAwareBatchSampler
    extra = {"block_size": 2} if block_aligned else {}
    return [
        sampler_type(
            SampleIndices(size),
            batch_size=4,
            generator=torch.Generator().manual_seed(17),
            drop_last=drop_last,
            rank=rank,
            world_size=3,
            **extra,
        )
        for rank in range(3)
    ]


def global_plan(rank_samplers):
    """Interleave rank-local batches back into global-step order."""
    epochs = [list(sampler) for sampler in rank_samplers]
    return [
        sample
        for step in zip(*epochs, strict=True)
        for batch in step
        for sample in batch
    ]


@pytest.mark.parametrize("block_aligned", [False, True])
def test_padding_repeats_only_the_minimum_global_tail(block_aligned):
    rank_samplers = samplers(23, block_aligned=block_aligned)
    plan = global_plan(rank_samplers)
    counts = Counter(plan)

    assert {len(sampler) for sampler in rank_samplers} == {2}
    assert len(plan) == 24
    assert set(plan) == set(range(23))
    assert sum(counts.values()) - len(counts) == 1


@pytest.mark.parametrize("block_aligned", [False, True])
def test_drop_last_has_equal_full_batches_and_no_duplicates(block_aligned):
    rank_samplers = samplers(29, drop_last=True, block_aligned=block_aligned)
    epochs = [list(sampler) for sampler in rank_samplers]
    plan = [sample for rank in epochs for batch in rank for sample in batch]

    assert {len(sampler) for sampler in rank_samplers} == {2}
    assert all(len(batch) == 4 for rank in epochs for batch in rank)
    assert len(plan) == len(set(plan)) == 24


def test_rank_local_streamed_batches_are_made_of_whole_logical_blocks():
    rank_samplers = samplers(23, block_aligned=True)
    epochs = [list(sampler) for sampler in rank_samplers]

    for sampler, batches in zip(rank_samplers, epochs, strict=True):
        whole_batches = {
            tuple(
                sample
                for block in sampler.blocks[start : start + sampler.blocks_per_batch]
                for sample in block
            )
            for start in range(0, len(sampler.blocks), sampler.blocks_per_batch)
        }
        assert all(tuple(batch) in whole_batches for batch in batches)


@pytest.mark.parametrize("block_aligned", [False, True])
def test_set_epoch_is_reproducible_and_changes_the_global_plan(block_aligned):
    rank_samplers = samplers(23, block_aligned=block_aligned)
    first = global_plan(rank_samplers)
    assert global_plan(rank_samplers) == first

    for sampler in rank_samplers:
        sampler.set_epoch(1)
    second = global_plan(rank_samplers)

    assert second != first
    assert len(second) == len(first)


def test_invalid_rank_topology_is_rejected():
    dataset = SampleIndices(8)
    with pytest.raises(ValueError, match="world_size"):
        RankAwareBatchSampler(dataset, batch_size=2, world_size=0)
    with pytest.raises(ValueError, match="rank must be between"):
        BlockAlignedBatchSampler(dataset, batch_size=2, rank=2, world_size=2)


@pytest.mark.parametrize("in_memory", [False, True])
def test_loader_wires_rank_sampling_into_both_storage_modes(in_memory):
    dataset = SampleIndices(23)
    dataset.in_memory = in_memory
    loaders = [
        S2ZDataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(17),
            rank=rank,
            world_size=3,
            collate_fn=lambda samples: samples,
        )
        for rank in range(3)
    ]
    sampler_type = RankAwareBatchSampler if in_memory else BlockAlignedBatchSampler
    assert all(isinstance(loader.batch_sampler, sampler_type) for loader in loaders)

    first = global_plan([loader.batch_sampler for loader in loaders])
    assert global_plan([loader.batch_sampler for loader in loaders]) == first
    for loader in loaders:
        loader.set_epoch(1)
    assert global_plan([loader.batch_sampler for loader in loaders]) != first


@pytest.mark.parametrize("in_memory", [False, True])
def test_loader_separates_global_sampling_from_rank_augmentation(in_memory):
    datasets = [SampleIndices(23) for _ in range(3)]
    datasets[0].in_memory = in_memory
    datasets[1].in_memory = in_memory
    datasets[2].in_memory = in_memory
    loaders = [
        S2ZDataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(17),
            rank=rank,
            world_size=3,
            num_workers=0,
            collate_fn=lambda samples: samples,
        )
        for rank, dataset in enumerate(datasets)
    ]

    # Sampling starts from the same global plan on every rank. Augmentation and
    # worker seeds are a distinct stream and differ across ranks.
    assert {loader.batch_sampler.seed for loader in loaders} == {17}
    augmentation_seeds = [dataset.augmentation_seed for dataset in datasets]
    assert len(set(augmentation_seeds)) == 3
    assert augmentation_seeds == [loader.generator.initial_seed() for loader in loaders]
    assert 17 not in augmentation_seeds


def test_rank_augmentation_seed_is_reproducible():
    def seeds():
        datasets = [SampleIndices(8) for _ in range(2)]
        for rank, dataset in enumerate(datasets):
            dataset.in_memory = False
            S2ZDataLoader(
                dataset,
                batch_size=2,
                generator=torch.Generator().manual_seed(99),
                rank=rank,
                world_size=2,
                num_workers=0,
                collate_fn=lambda samples: samples,
            )
        return [dataset.augmentation_seed for dataset in datasets]

    assert seeds() == seeds()


def test_single_rank_plain_shuffle_has_separate_generators():
    dataset = SampleIndices(8)
    dataset.in_memory = True
    loader = S2ZDataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        generator=torch.Generator().manual_seed(17),
        num_workers=0,
        collate_fn=lambda samples: samples,
    )

    assert isinstance(loader.sampler, torch.utils.data.RandomSampler)
    assert loader.sampler.generator.initial_seed() == 17
    assert loader.generator.initial_seed() == dataset.augmentation_seed
    assert loader.generator.initial_seed() != loader.sampler.generator.initial_seed()
