"""Provide the dataset class and its transforms."""

from survey2zarray.dataset.augmentation import S2ZAugmentation
from survey2zarray.dataset.collate import (
    collate_catalogue,
    collate_views,
    select_covered,
)
from survey2zarray.dataset.dataset import S2ZDataset
from survey2zarray.dataset.loader import S2ZDataLoader
from survey2zarray.dataset.preprocessing import S2ZPreprocessing
from survey2zarray.dataset.sampler import (
    BlockAlignedBatchSampler,
    RankAwareBatchSampler,
)
from survey2zarray.stores.dataset_store.filters import resolve_subset

__all__ = [
    "S2ZDataset",
    "resolve_subset",
    "S2ZPreprocessing",
    "S2ZAugmentation",
    "collate_views",
    "collate_catalogue",
    "select_covered",
    "BlockAlignedBatchSampler",
    "RankAwareBatchSampler",
    "S2ZDataLoader",
]
