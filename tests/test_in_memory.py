"""Holding a split in memory, and the loader that follows from it."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from survey2zarray.dataset import dataset as dataset_module
from survey2zarray.dataset import loader as loader_module
from survey2zarray.dataset.dataset import S2ZDataset
from survey2zarray.dataset.loader import S2ZDataLoader
from survey2zarray.dataset.sampler import BlockAlignedBatchSampler
from survey2zarray.stores.dataset_store.dataset_store import DatasetStore
from survey2zarray.stores.survey_store.build import build_survey_store
from tests.conftest import (
    IMAGE_DEC,
    IMAGE_RA,
    build_test_dataset_store,
    split_config,
)

CUTOUT_ARCSEC = 16.0


@pytest.fixture
def store_path(continuum_fits_dir, tmp_path):
    continuum_dir, _ = continuum_fits_dir
    survey = tmp_path / "cont"
    build_survey_store(continuum_dir, survey, mode="copy")
    catalogue = tmp_path / "catalogue.csv"
    pd.DataFrame(
        [
            {
                "ID": f"source_{n}",
                "RA": IMAGE_RA,
                "DEC": IMAGE_DEC + (14.0 * n) / 3600.0,
            }
            for n in range(6)
        ]
    ).to_csv(catalogue, index=False)
    out = tmp_path / "dataset"
    build_test_dataset_store([survey], catalogue, CUTOUT_ARCSEC, out)
    return out


def make_config(store_path, **overrides):
    config = {
        "dataset_store": str(store_path),
        "splits": split_config(train=1.0),
        "preprocessing": {"linear_rescale": {}},
        "augmentation": {},
    }
    config.update(deepcopy(overrides))
    return config


def flatten(obj, out=None):
    """Every array in a sample, however deeply the sample nests it."""
    out = [] if out is None else out
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        out.append(np.asarray(obj))
    elif isinstance(obj, dict):
        for value in obj.values():
            flatten(value, out)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            flatten(value, out)
    return out


class TestHeldRows:
    """A held row is the stored row, and reads of it are reads of the store."""

    def test_a_held_read_matches_the_stored_read(self, store_path):
        store = DatasetStore.open(store_path)
        sub = store.survey("cont")
        indices = list(sub.index_map.values())
        coords = [
            (float(store.ra[source]), float(store.dec[source]))
            for source in sub.index_map
        ]
        from_store = sub.get_cutouts(indices, coords, CUTOUT_ARCSEC / 2, 0.0)

        sub.load_rows(indices)
        held = sub.get_cutouts(indices, coords, CUTOUT_ARCSEC / 2, 0.0)
        singly = [
            sub.get_cutout(index, coord, CUTOUT_ARCSEC / 2, 0.0)
            for index, coord in zip(indices, coords, strict=True)
        ]
        for (want, want_wcs), (batched, wcs), (one, one_wcs) in zip(
            from_store, held, singly, strict=True
        ):
            assert np.array_equal(batched, want, equal_nan=True)
            assert np.array_equal(one, want, equal_nan=True)
            assert wcs.to_header() == want_wcs.to_header()
            assert one_wcs.to_header() == want_wcs.to_header()

    def test_rows_not_held_are_still_read(self, store_path):
        store = DatasetStore.open(store_path)
        sub = store.survey("cont")
        indices = list(sub.index_map.values())
        coords = [
            (float(store.ra[source]), float(store.dec[source]))
            for source in sub.index_map
        ]
        want = sub.get_cutouts(indices, coords, CUTOUT_ARCSEC / 2, 0.0)

        # Half held, half not: a batch mixing the two comes back in its order
        sub.load_rows(indices[::2])
        mixed = sub.get_cutouts(indices, coords, CUTOUT_ARCSEC / 2, 0.0)
        for (cutout, _), (expected, _) in zip(mixed, want, strict=True):
            assert np.array_equal(cutout, expected, equal_nan=True)

    def test_the_held_row_survives_a_caller_writing_to_what_it_returned(
        self, store_path
    ):
        sub = DatasetStore.open(store_path).survey("cont")
        index = next(iter(sub.index_map.values()))
        sub.load_rows([index])
        first, _ = sub.get_cutout(index)
        first[:] = 0.0
        again, _ = sub.get_cutout(index)
        assert not np.array_equal(first, again, equal_nan=True)

    def test_dropping_them_goes_back_to_the_store(self, store_path):
        sub = DatasetStore.open(store_path).survey("cont")
        index = next(iter(sub.index_map.values()))
        sub.load_rows([index])
        assert sub.cached_row(index) is not None
        sub.drop_rows()
        assert sub.cached_row(index) is None


class TestInMemory:
    """The dataset holds its split, or reads it, and says which."""

    def test_a_held_dataset_serves_what_the_store_would_have(self, store_path):
        config = make_config(store_path)
        from_store = S2ZDataset(config, split="train", in_memory=False)
        held = S2ZDataset(config, split="train", in_memory=True)
        assert from_store.in_memory is False
        assert held.in_memory is True

        torch.manual_seed(0)
        want = flatten(from_store[1])
        torch.manual_seed(0)
        got = flatten(held[1])
        assert len(got) == len(want)
        for array, expected in zip(got, want, strict=True):
            assert np.allclose(array, expected, equal_nan=True)

        # And through the batched path, which reads shard by shard
        torch.manual_seed(0)
        want = flatten(from_store.__getitems__([0, 1, 2]))
        torch.manual_seed(0)
        got = flatten(held.__getitems__([0, 1, 2]))
        for array, expected in zip(got, want, strict=True):
            assert np.allclose(array, expected, equal_nan=True)

    def test_it_measures_the_split_uncompressed(self, store_path):
        dataset = S2ZDataset(make_config(store_path), split="train", in_memory=False)
        substore = dataset.substores["cont"]
        assert dataset.memory_bytes == len(dataset.subset_indices) * substore.row_bytes

    def test_auto_holds_what_fits(self, store_path, monkeypatch):
        monkeypatch.setattr(dataset_module, "available_memory", lambda: 1 << 40)
        dataset = S2ZDataset(make_config(store_path), split="train")
        assert dataset._in_memory is None
        assert dataset.in_memory is True

    def test_auto_reads_what_does_not_fit(self, store_path, monkeypatch):
        monkeypatch.setattr(dataset_module, "available_memory", lambda: 1024)
        assert S2ZDataset(make_config(store_path), split="train").in_memory is False

    def test_auto_reads_where_memory_cannot_be_measured(self, store_path, monkeypatch):
        # An unmeasured machine is not one to fill
        monkeypatch.setattr(dataset_module, "available_memory", lambda: None)
        assert S2ZDataset(make_config(store_path), split="train").in_memory is False

    def test_direct_sample_access_lazily_resolves_auto(self, store_path, monkeypatch):
        monkeypatch.setattr(dataset_module, "available_memory", lambda: 1 << 40)
        dataset = S2ZDataset(make_config(store_path), split="train")
        assert dataset._in_memory is None

        dataset[0]

        assert dataset._in_memory is True

    def test_true_holds_them_whatever_the_machine_says(self, store_path, monkeypatch):
        monkeypatch.setattr(dataset_module, "available_memory", lambda: 1024)
        assert (
            S2ZDataset(make_config(store_path), split="train", in_memory=True).in_memory
            is True
        )

    def test_anything_else_is_rejected(self, store_path):
        with pytest.raises(ValueError, match="in_memory"):
            S2ZDataset(make_config(store_path), split="train", in_memory="yes")


class TestS2ZDataLoader:
    """Batches are drawn the way the dataset is read, unless told otherwise."""

    def test_a_read_dataset_is_batched_in_blocks(self, store_path):
        dataset = S2ZDataset(make_config(store_path), split="train", in_memory=False)
        loader = S2ZDataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        assert loader.block_aligned is True
        assert isinstance(loader.batch_sampler, BlockAlignedBatchSampler)

    def test_a_held_dataset_is_batched_plainly(self, store_path):
        dataset = S2ZDataset(make_config(store_path), split="train", in_memory=True)
        loader = S2ZDataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        assert loader.block_aligned is False
        assert not isinstance(loader.batch_sampler, BlockAlignedBatchSampler)
        assert isinstance(loader.sampler, torch.utils.data.RandomSampler)

    def test_the_choice_can_be_overridden_either_way(self, store_path):
        config = make_config(store_path)
        held = S2ZDataset(config, split="train", in_memory=True)
        assert S2ZDataLoader(
            held, batch_size=2, block_aligned=True, num_workers=0
        ).block_aligned
        read = S2ZDataset(config, split="train", in_memory=False)
        assert not S2ZDataLoader(
            read, batch_size=2, block_aligned=False, num_workers=0
        ).block_aligned

    def test_anything_else_is_rejected(self, store_path):
        dataset = S2ZDataset(make_config(store_path), split="train", in_memory=False)
        with pytest.raises(ValueError, match="block_aligned"):
            S2ZDataLoader(
                dataset, batch_size=2, block_aligned="sometimes", num_workers=0
            )

    def test_the_group_reaches_the_block_sampler(self, store_path):
        # Held in memory there is no shard to align to, so it reaches nothing
        config = make_config(store_path)
        read = S2ZDataset(config, split="train", in_memory=False)
        loader = S2ZDataLoader(read, batch_size=2, group_size=32, num_workers=0)
        assert loader.batch_sampler.group_size == 32
        held = S2ZDataset(config, split="train", in_memory=True)
        assert not S2ZDataLoader(
            held, batch_size=2, group_size=32, num_workers=0
        ).block_aligned

    @pytest.mark.parametrize("in_memory", [True, False])
    def test_an_epoch_serves_the_split_once_either_way(self, store_path, in_memory):
        dataset = S2ZDataset(
            make_config(store_path), split="train", in_memory=in_memory
        )
        loader = S2ZDataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        seen = [int(index) for batch in loader for index in batch["source_index"]]
        assert sorted(seen) == sorted(int(i) for i in dataset.subset_indices)

    @pytest.mark.parametrize("in_memory", [True, False])
    def test_drop_last_reaches_either_sampler(self, store_path, in_memory):
        dataset = S2ZDataset(
            make_config(store_path), split="train", in_memory=in_memory
        )
        kept = S2ZDataLoader(dataset, batch_size=4, drop_last=True, num_workers=0)
        assert sum(len(batch["source_index"]) for batch in kept) == 4
        whole = S2ZDataLoader(dataset, batch_size=4, drop_last=False, num_workers=0)
        assert sum(len(batch["source_index"]) for batch in whole) == len(dataset)

    def test_omitted_workers_use_the_polite_default(self, store_path, monkeypatch):
        calls = []

        def choose(**kwargs):
            calls.append(kwargs)
            return 3

        monkeypatch.setattr(loader_module, "loader_workers", choose)
        dataset = S2ZDataset(make_config(store_path), split="train", in_memory=False)
        loader = S2ZDataLoader(dataset, batch_size=2)

        assert loader.num_workers == 3
        assert calls == [{"ceiling": len(dataset), "local_world_size": 1}]

    def test_explicit_workers_bypass_the_default(self, store_path, monkeypatch):
        def unexpected(**_):
            raise AssertionError("automatic worker selection was called")

        monkeypatch.setattr(loader_module, "loader_workers", unexpected)
        dataset = S2ZDataset(make_config(store_path), split="train", in_memory=False)
        assert S2ZDataLoader(dataset, num_workers=2).num_workers == 2

    def test_loader_auto_accounts_for_spawned_worker_copies(
        self, store_path, monkeypatch
    ):
        dataset = S2ZDataset(make_config(store_path), split="train")
        monkeypatch.setattr(
            dataset_module, "available_memory", lambda: dataset.memory_bytes * 4
        )
        monkeypatch.setattr(
            dataset_module.multiprocessing,
            "get_start_method",
            lambda allow_none=True: "spawn",
        )

        loader = S2ZDataLoader(dataset, batch_size=2, num_workers=2)

        # Half the reported memory is available to held rows: two copies fit,
        # but the main process plus two spawned workers require three.
        assert dataset._in_memory is False
        assert loader.block_aligned is True

    def test_loader_can_downgrade_a_direct_auto_decision(self, store_path, monkeypatch):
        monkeypatch.setattr(dataset_module, "available_memory", lambda: 1 << 40)
        dataset = S2ZDataset(make_config(store_path), split="train")
        assert dataset.in_memory is True

        monkeypatch.setattr(dataset_module, "available_memory", lambda: 1)
        loader = S2ZDataLoader(dataset, batch_size=2, num_workers=0)

        assert dataset._in_memory is False
        assert loader.block_aligned is True
        assert all(
            substore.cached_row(row) is None
            for substore in dataset.substores.values()
            for row in substore.index_map.values()
        )

    def test_distributed_auto_streams_without_local_topology(
        self, store_path, monkeypatch
    ):
        monkeypatch.setattr(dataset_module, "available_memory", lambda: 1 << 40)
        dataset = S2ZDataset(make_config(store_path), split="train")

        loader = S2ZDataLoader(
            dataset,
            batch_size=2,
            rank=0,
            world_size=2,
            num_workers=0,
        )

        assert dataset._in_memory is False
        assert loader.block_aligned is True

    def test_distributed_auto_holds_with_known_local_topology(
        self, store_path, monkeypatch
    ):
        monkeypatch.setattr(dataset_module, "available_memory", lambda: 1 << 40)
        dataset = S2ZDataset(make_config(store_path), split="train")

        loader = S2ZDataLoader(
            dataset,
            batch_size=2,
            rank=0,
            world_size=2,
            local_world_size=2,
            num_workers=0,
        )

        assert dataset._in_memory is True
        assert loader.block_aligned is False

    def test_explicit_workers_warn_above_node_cap(self, store_path, monkeypatch):
        messages = []
        monkeypatch.setattr(loader_module.logger, "warning", messages.append)
        monkeypatch.setattr(loader_module, "LOADER_CAP", 1)
        dataset = S2ZDataset(make_config(store_path), split="train", in_memory=False)

        loader = S2ZDataLoader(
            dataset,
            rank=0,
            world_size=2,
            local_world_size=2,
            num_workers=1,
        )

        assert loader.num_workers == 1
        assert any("above the polite node-wide cap" in message for message in messages)

    def test_spawned_workers_warn_that_held_rows_are_copied(self, monkeypatch):
        messages = []
        monkeypatch.setattr(loader_module.logger, "warning", messages.append)
        dataset = SimpleNamespace(memory_bytes=2_000_000)

        S2ZDataLoader._warn_if_workers_copy(dataset, True, 2, "spawn")

        assert any("Each is sent its own copy" in message for message in messages)
        messages.clear()
        S2ZDataLoader._warn_if_workers_copy(dataset, True, 2, "fork")
        assert messages == []
