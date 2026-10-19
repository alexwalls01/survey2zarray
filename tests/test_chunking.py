import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import zarr
from torch.utils.data import DataLoader
from zarr.codecs import BloscCodec

from survey2zarray.dataset import BlockAlignedBatchSampler, S2ZDataset, collate_views
from survey2zarray.dataset.sampler import (
    BLOCKS_PER_BATCH,
    GROUP_ROWS_PER_SAMPLE,
    MIN_BLOCKS_PER_GROUP,
)
from survey2zarray.stores.dataset_store.build import COALESCE_CHUNKS
from survey2zarray.stores.dataset_store.dataset_store import PADDING, DatasetStore
from survey2zarray.stores.dataset_store.edit import append_survey
from survey2zarray.stores.dataset_store.layout import (
    MAX_SHARD_ROWS,
    MIN_SHARD_ROWS,
    TARGET_SHARD_BYTES,
    is_power_of_two,
    shard_rows,
)
from survey2zarray.stores.store import open_store
from survey2zarray.stores.survey_store.build import build_survey_store
from tests.conftest import (
    IMAGE_DEC,
    IMAGE_RA,
    build_test_dataset_store,
    make_continuum_fits,
    split_config,
)

# A grid of sources large enough to span several chunks. The image is 200x200
# at 1"/px, so a 4" cutout (a ~12" disc) fits anywhere near the centre, and 10"
# between sources keeps them outside each other's 6" beam so the build's
# duplicate check does not collapse them.
CUTOUT_ARCSEC = 4.0
N_SOURCES = 64
SPACING_ARCSEC = 10.0


# Every fourth source filtered out, which leaves every block of source indices
# about three-quarters served: the ragged case the sampler has to handle
FILTERED_OUT = [f"src_{index:03d}" for index in range(0, N_SOURCES, 4)]


def grid_catalogue(path, n_sources=N_SOURCES, columns=8):
    rows = []
    for index in range(n_sources):
        row, column = divmod(index, columns)
        rows.append(
            {
                "ID": f"src_{index:03d}",
                "RA": IMAGE_RA + (column - columns / 2) * SPACING_ARCSEC / 3600.0,
                "DEC": IMAGE_DEC
                + (row - n_sources / columns / 2) * SPACING_ARCSEC / 3600.0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def grid_survey_store(tmp_path):
    fits_dir = tmp_path / "grid_fits"
    fits_dir.mkdir()
    make_continuum_fits(fits_dir / "wide.fits", shape_yx=(200, 200), seed=7)
    out = tmp_path / "grid_survey"
    build_survey_store(fits_dir, out, mode="copy")
    return out


@pytest.fixture
def second_survey_store(tmp_path):
    # Offset so it covers only part of the grid, which is what leaves a survey's
    # cutout indices out of step with the source indices
    fits_dir = tmp_path / "grid_fits_b"
    fits_dir.mkdir()
    make_continuum_fits(
        fits_dir / "wide_b.fits",
        shape_yx=(120, 120),
        ra=IMAGE_RA,
        dec=IMAGE_DEC + 40.0 / 3600.0,
        seed=8,
    )
    out = tmp_path / "grid_survey_b"
    build_survey_store(fits_dir, out, mode="copy")
    return out


@pytest.fixture
def column_survey_store(tmp_path):
    # Narrow in RA and tall in DEC, so it covers three of the catalogue's eight
    # columns at every row and its coverage repeats every eight sources
    fits_dir = tmp_path / "grid_fits_columns"
    fits_dir.mkdir()
    make_continuum_fits(fits_dir / "columns.fits", shape_yx=(200, 44), seed=9)
    out = tmp_path / "grid_survey_columns"
    build_survey_store(fits_dir, out, mode="copy")
    return out


@pytest.fixture
def aligned_store(
    tmp_path, grid_survey_store, column_survey_store, grid_catalogue_path
):
    # Shards of eight rows, one survey covering every source and one covering
    # three of the catalogue's eight columns
    return build_grid_store(
        tmp_path / "aligned",
        [grid_survey_store, column_survey_store],
        grid_catalogue_path,
        shard_size=8,
    )


@pytest.fixture
def grid_catalogue_path(tmp_path):
    return grid_catalogue(tmp_path / "grid_catalogue.csv")


def build_grid_store(out, survey_stores, catalogue, **kwargs):
    build_test_dataset_store([*survey_stores], catalogue, CUTOUT_ARCSEC, out, **kwargs)
    return out


@pytest.fixture
def grid_store(tmp_path, grid_survey_store, second_survey_store, grid_catalogue_path):
    return build_grid_store(
        tmp_path / "grid_dataset",
        [grid_survey_store, second_survey_store],
        grid_catalogue_path,
    )


def small_shard_store(tmp_path, survey_store, catalogue_path):
    """Build a store whose shards hold few enough rows that a split spans several."""
    return build_grid_store(
        tmp_path / "d8", [survey_store], catalogue_path, shard_size=8
    )


def group_keys(dataset, block, group_size):
    """Group each sample of a block falls in, as the sampler groups them."""
    return [int(dataset.subset_indices[index]) // group_size for index in block]


def make_config(store_path, **overrides):
    config = {
        "dataset_store": str(store_path),
        "splits": split_config(train=1.0),
        "exclude": {"sources": FILTERED_OUT},
        "preprocessing": {"linear_rescale": {}},
        "augmentation": {},
    }
    config.update(overrides)
    return config


def chunks_of(store_path, survey):
    root = open_store(store_path, mode="r")
    return root["surveys"][survey]["data"].chunks


def shards_of(store_path, survey):
    root = open_store(store_path, mode="r")
    return root["surveys"][survey]["data"].shards


def cutout_bytes_of(store_path, survey):
    data = open_store(store_path, mode="r")["surveys"][survey]["data"]
    return int(np.prod(data.shape[1:])) * data.dtype.itemsize


def coverage_of(store_path, survey):
    root = open_store(store_path, mode="r")
    return root["surveys"][survey].attrs["n_cutouts"] / root.attrs["n_sources"]


def bytes_on_disk(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def compact_copy(store, out):
    """Write each survey's cutouts packed into consecutive rows, and return its bytes.

    Parameters
    ----------
    store : DatasetStore
        Store to copy the covered rows out of.
    out : pathlib.Path
        Destination for the copy.

    Returns
    -------
    int
        Bytes the copy occupies, over every survey.
    """
    group = zarr.create_group(store=out)
    total = 0
    for survey in store.surveys:
        sub = store.survey(survey)
        rows = sorted(sub.index_map.values())
        array = group.create_array(
            survey,
            shape=(len(rows), *sub.data.shape[1:]),
            chunks=sub.data.chunks,
            shards=sub.data.shards,
            dtype=sub.data.dtype,
            fill_value=sub.data.fill_value,
            compressors=sub.data.compressors,
        )
        array[:] = sub.data.oindex[rows]
        total += bytes_on_disk(Path(out) / survey)
    return total


class TestShardSize:
    """Rows are stored several to a shard and one cutout to a chunk.

    A shard is the file a read opens, and a chunk is what it decompresses. Its
    row count is a power of two, which is what keeps every survey's shards
    dividing one another however independently they were chosen.
    """

    def test_a_shard_holds_what_the_byte_budget_affords(self, grid_store):
        cutout_bytes = cutout_bytes_of(grid_store, "grid_survey")
        wanted = shard_rows(cutout_bytes, coverage_of(grid_store, "grid_survey"))
        assert shards_of(grid_store, "grid_survey")[0] == wanted
        assert DatasetStore.open(grid_store).shard_size == wanted

    def test_every_shard_row_count_is_a_power_of_two(self, grid_store):
        root = open_store(grid_store, mode="r")
        for survey in root.attrs["surveys"]:
            assert is_power_of_two(shards_of(grid_store, survey)[0])

    def test_the_inner_chunk_is_one_cutout(self, grid_store):
        data = open_store(grid_store, mode="r")["surveys"]["grid_survey"]["data"]
        assert data.chunks == (1, *data.shape[1:])

    def test_a_survey_with_larger_cutouts_gets_fewer_to_a_shard(self):
        # Each is one cutout of the budget, so the count halves as the cutout
        # doubles, rounded down to a power of two, until the bounds take over
        assert shard_rows(TARGET_SHARD_BYTES // 128, 1.0) == 128
        assert shard_rows(TARGET_SHARD_BYTES // 200, 1.0) == 128
        assert shard_rows(TARGET_SHARD_BYTES // 256, 1.0) == 256
        assert shard_rows(TARGET_SHARD_BYTES, 1.0) == MIN_SHARD_ROWS
        assert shard_rows(1, 1.0) == MAX_SHARD_ROWS

    def test_a_survey_covering_less_gets_more_rows_to_a_shard(self):
        # Sized by the bytes a shard serves, so a sparse survey keeps its file
        # count near the budget rather than multiplying empty shards
        cutout_bytes = TARGET_SHARD_BYTES // 256
        assert shard_rows(cutout_bytes, 1.0) == 256
        assert shard_rows(cutout_bytes, 0.5) == 512
        assert shard_rows(cutout_bytes, 0.1) == 2048

    def test_an_explicit_size_is_used(
        self, tmp_path, grid_survey_store, grid_catalogue_path
    ):
        out = build_grid_store(
            tmp_path / "d8", [grid_survey_store], grid_catalogue_path, shard_size=8
        )
        assert shards_of(out, "grid_survey")[0] == 8
        assert DatasetStore.open(out).shard_size == 8

    def test_one_per_shard_is_still_available(
        self, tmp_path, grid_survey_store, grid_catalogue_path
    ):
        out = build_grid_store(
            tmp_path / "d1", [grid_survey_store], grid_catalogue_path, shard_size=1
        )
        assert shards_of(out, "grid_survey")[0] == 1
        assert DatasetStore.open(out).shard_size == 1

    def test_it_must_be_a_power_of_two(
        self, tmp_path, grid_survey_store, grid_catalogue_path
    ):
        # Anything else loses the alignment silently, so it is refused rather
        # than rounded to one
        for shard_size in (0, 3, 12, 100):
            with pytest.raises(
                ValueError, match="shard_size must be a power-of-two row count"
            ):
                build_grid_store(
                    tmp_path / f"d{shard_size}",
                    [grid_survey_store],
                    grid_catalogue_path,
                    shard_size=shard_size,
                )

    def test_the_pixels_stored_do_not_depend_on_it(
        self, tmp_path, grid_survey_store, grid_catalogue_path
    ):
        # Sharding is a layout choice, so a sharded store must hold exactly what
        # a one-per-shard store holds
        sharded = build_grid_store(
            tmp_path / "sharded",
            [grid_survey_store],
            grid_catalogue_path,
            shard_size=16,
        )
        single = build_grid_store(
            tmp_path / "single", [grid_survey_store], grid_catalogue_path, shard_size=1
        )
        a = DatasetStore.open(sharded).survey("grid_survey")
        b = DatasetStore.open(single).survey("grid_survey")
        assert a.n_cutouts == b.n_cutouts > 32  # More than one shard of them
        np.testing.assert_array_equal(a.data[:], b.data[:])
        assert a.index_map == b.index_map

    def test_an_appended_survey_is_sharded_like_the_store(
        self, tmp_path, grid_survey_store, second_survey_store, grid_catalogue_path
    ):
        out = build_grid_store(
            tmp_path / "appendable",
            [grid_survey_store],
            grid_catalogue_path,
            shard_size=8,
        )
        append_survey(out, second_survey_store)
        assert shards_of(out, "grid_survey_b")[0] == 8
        assert DatasetStore.open(out).shard_size == 8

    def test_an_appended_survey_takes_its_own_budget_where_none_was_set(
        self, tmp_path, grid_survey_store, second_survey_store, grid_catalogue_path
    ):
        out = build_grid_store(
            tmp_path / "budgeted", [grid_survey_store], grid_catalogue_path
        )
        assert open_store(out, mode="r").attrs["shard_size"] is None
        append_survey(out, second_survey_store)
        assert shards_of(out, "grid_survey_b")[0] == shard_rows(
            cutout_bytes_of(out, "grid_survey_b"),
            coverage_of(out, "grid_survey_b"),
        )

    def test_a_different_size_restarts_an_unfinished_build(
        self, tmp_path, grid_survey_store, grid_catalogue_path
    ):
        out = build_grid_store(
            tmp_path / "resumed", [grid_survey_store], grid_catalogue_path, shard_size=8
        )
        # Left as an interrupted build would leave it
        root = open_store(out, mode="r+")
        root.attrs["complete"] = False
        build_grid_store(
            out, [grid_survey_store], grid_catalogue_path, shard_size=4, resume=True
        )
        # A shard shape cannot be revised, so resuming was not an option
        assert shards_of(out, "grid_survey")[0] == 4


class TestAlignedRows:
    """Every survey holds a row per source, and row `i` holds source `i`.

    A survey covering part of the catalogue would otherwise pack its cutouts,
    which drifts its rows against the source index and makes every block a
    sampler cuts straddle two of its shards.
    """

    def test_every_survey_holds_a_row_per_source(self, grid_store):
        store = DatasetStore.open(grid_store)
        for survey in store.surveys:
            sub = store.survey(survey)
            assert sub.data.shape[0] == store.n_sources
            for name in ("crpix", "pixel_shape", "image_index", "bmaj", "bunit"):
                assert sub.group["metadata"][name].shape[0] == store.n_sources

    def test_the_index_map_is_the_identity_on_covered_sources(self, grid_store):
        store = DatasetStore.open(grid_store)
        for survey in store.surveys:
            sub = store.survey(survey)
            assert sub.index_map == {i: i for i in sub.index_map}

    def test_a_partly_covering_survey_leaves_its_uncovered_rows_as_fill(
        self, grid_store
    ):
        store = DatasetStore.open(grid_store)
        # The second survey covers only part of the grid, by construction
        sub = store.survey("grid_survey_b")
        assert 0 < sub.n_cutouts < store.n_sources
        uncovered = [i for i in range(store.n_sources) if i not in sub.index_map]
        assert np.isnan(sub.data.oindex[uncovered]).all()

    def test_n_cutouts_counts_what_a_survey_holds_not_its_rows(self, grid_store):
        store = DatasetStore.open(grid_store)
        sub = store.survey("grid_survey_b")
        assert sub.n_cutouts == len(sub.index_map) < sub.data.shape[0]

    def test_a_store_whose_rows_drift_is_refused(self, grid_store):
        # What a store written by code that packed its rows looks like: it reads
        # correctly, so only the identity check catches it
        root = open_store(grid_store, mode="r+")
        index_map = root["surveys"]["grid_survey_b"]["index_map"]
        entries = index_map[:]
        held = entries[entries != PADDING]
        packed = np.full_like(entries, PADDING)
        packed[: len(held)] = held
        index_map[:] = packed
        with pytest.raises(RuntimeError, match="index map must be the identity"):
            DatasetStore.open(grid_store)


class TestPaddingIsFree:
    """A row a survey does not cover costs shape rather than bytes.

    zarr drops a chunk that is entirely fill even where the fill value is NaN,
    which compares unequal to itself. Aligned rows rest on that, so it is
    asserted rather than trusted: a zarr release that changed it would double
    every store on disk and nothing else would say so.
    """

    def arrays(self, tmp_path, n_rows, rows, pixels, write_fill):
        group = zarr.create_group(store=tmp_path)
        array = group.create_array(
            "data",
            shape=(n_rows, 32, 32),
            chunks=(1, 32, 32),
            shards=(16, 32, 32),
            dtype=np.float32,
            fill_value=np.nan,
            compressors=[BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")],
        )
        if write_fill:
            block = np.full((n_rows, 32, 32), np.nan, dtype=np.float32)
            block[rows] = pixels
            array[:] = block
        else:
            array.oindex[rows] = pixels
        return bytes_on_disk(tmp_path)

    def test_a_padded_array_costs_what_a_compact_one_costs(self, tmp_path):
        rng = np.random.default_rng(0)
        rows = sorted(rng.choice(256, 160, replace=False).tolist())
        pixels = rng.normal(0.0, 1.0, (160, 32, 32)).astype(np.float32)
        compact = self.arrays(
            tmp_path / "compact", 160, list(range(160)), pixels, False
        )
        padded = self.arrays(tmp_path / "padded", 256, rows, pixels, False)
        written = self.arrays(tmp_path / "written", 256, rows, pixels, True)
        # The shard index of the extra files is the whole difference
        assert padded == pytest.approx(compact, rel=0.01)
        # Writing NaN explicitly costs the same as never writing the row
        assert written == padded

    def test_a_built_store_costs_what_its_cutouts_cost(
        self, tmp_path, grid_survey_store, second_survey_store, grid_catalogue_path
    ):
        out = build_grid_store(
            tmp_path / "padded_store",
            [grid_survey_store, second_survey_store],
            grid_catalogue_path,
            shard_size=8,
        )
        store = DatasetStore.open(out)
        compact = compact_copy(store, tmp_path / "compact_store")
        padded = sum(
            bytes_on_disk(Path(out) / "surveys" / survey / "data")
            for survey in store.surveys
        )
        assert padded == pytest.approx(compact, rel=0.01)


class TestCoalesceGap:
    """A read runs through a shard only as far as the store's recorded gap.

    zarr's own default is larger than a whole shard here, so the store records
    what to use and opening it applies that.
    """

    def test_the_gap_is_derived_from_the_inner_chunk(self, grid_store):
        smallest = min(
            cutout_bytes_of(grid_store, survey)
            for survey in open_store(grid_store, mode="r").attrs["surveys"]
        )
        recorded = open_store(grid_store, mode="r").attrs["coalesce_gap_bytes"]
        assert recorded == COALESCE_CHUNKS * smallest

    def test_opening_the_store_applies_it(self, grid_store):
        # Set the other way first, so passing means this call changed it
        zarr.config.set({"array.sharding_coalesce_max_gap_bytes": 1024**2})
        store = DatasetStore.open(grid_store)
        assert (
            zarr.config.get("array.sharding_coalesce_max_gap_bytes")
            == store.coalesce_gap
        )
        assert store.coalesce_gap < store.survey("grid_survey").shard_size * (
            cutout_bytes_of(grid_store, "grid_survey")
        )

    def test_an_appended_survey_brings_it_down(
        self, tmp_path, grid_survey_store, second_survey_store, grid_catalogue_path
    ):
        out = build_grid_store(
            tmp_path / "appendable_gap", [grid_survey_store], grid_catalogue_path
        )
        append_survey(out, second_survey_store)
        smallest = min(
            cutout_bytes_of(out, survey)
            for survey in open_store(out, mode="r").attrs["surveys"]
        )
        assert (
            open_store(out, mode="r").attrs["coalesce_gap_bytes"]
            == COALESCE_CHUNKS * smallest
        )

    def test_a_batched_read_applies_it_too(self, grid_store):
        # A spawned DataLoader worker unpickles the sub-store rather than
        # opening the dataset store again, so the read path sets it as well
        substore = DatasetStore.open(grid_store).survey("grid_survey")
        zarr.config.set({"array.sharding_coalesce_max_gap_bytes": 1024**2})
        substore.get_cutouts([0, 3], [None, None])
        assert (
            zarr.config.get("array.sharding_coalesce_max_gap_bytes")
            == substore.coalesce_gap
        )


class TestBatchedReads:
    """Verify batched survey reads match scalar reads."""

    def test_get_cutouts_matches_get_cutout(self, grid_store):
        substore = DatasetStore.open(grid_store).survey("grid_survey")
        store = DatasetStore.open(grid_store)
        indices = [0, 1, 5, 33, 34, 2]
        coords = [(float(store.ra[i]), float(store.dec[i])) for i in indices]
        batched = substore.get_cutouts(indices, coords, 3.0, store.disc_padding)
        for local_index, coord, (cutout, wcs) in zip(
            indices, coords, batched, strict=True
        ):
            expected, expected_wcs = substore.get_cutout(
                local_index, coord, 3.0, store.disc_padding
            )
            np.testing.assert_array_equal(cutout, expected)
            assert wcs.wcs.crpix.tolist() == expected_wcs.wcs.crpix.tolist()

    def test_whole_discs_come_back_the_same_way(self, grid_store):
        substore = DatasetStore.open(grid_store).survey("grid_survey")
        indices = [4, 5, 6]
        batched = substore.get_cutouts(indices, [None] * len(indices))
        for local_index, (cutout, _) in zip(indices, batched, strict=True):
            np.testing.assert_array_equal(cutout, substore.get_cutout(local_index)[0])

    def test_a_repeated_index_comes_back_twice(self, grid_store):
        substore = DatasetStore.open(grid_store).survey("grid_survey")
        first, second = substore.get_cutouts([3, 3], [None, None])
        np.testing.assert_array_equal(first[0], second[0])

    def test_the_order_asked_for_is_the_order_returned(self, grid_store):
        substore = DatasetStore.open(grid_store).survey("grid_survey")
        indices = [40, 2, 17]
        results = substore.get_cutouts(indices, [None] * len(indices))
        for local_index, (cutout, _) in zip(indices, results, strict=True):
            np.testing.assert_array_equal(cutout, substore.get_cutout(local_index)[0])

    def test_coordinates_must_match_the_indices(self, grid_store):
        substore = DatasetStore.open(grid_store).survey("grid_survey")
        with pytest.raises(ValueError, match="requires one coordinate"):
            substore.get_cutouts([0, 1], [None])

    def test_only_the_cutouts_asked_for_are_read(
        self, tmp_path, grid_survey_store, grid_catalogue_path, monkeypatch
    ):
        out = build_grid_store(
            tmp_path / "small_shards",
            [grid_survey_store],
            grid_catalogue_path,
            shard_size=8,
        )
        substore = DatasetStore.open(out).survey("grid_survey")
        selections = []
        original = zarr.Array.get_orthogonal_selection

        def record(self, selection, **kwargs):
            selections.append(selection)
            return original(self, selection, **kwargs)

        monkeypatch.setattr(zarr.Array, "get_orthogonal_selection", record)
        # Two indices a shard apart, so one selection each and neither asks for
        # anything between them
        substore.get_cutouts([0, 9], [None, None])
        assert [list(selection) for selection in selections] == [[0], [9]]


class TestGetItems:
    """Verify batch retrieval matches scalar retrieval."""

    def read_both_ways(self, dataset, indices):
        torch.manual_seed(0)
        dataset._rng = None
        one_at_a_time = [dataset[index] for index in indices]
        torch.manual_seed(0)
        dataset._rng = None
        return one_at_a_time, dataset.__getitems__(indices)

    def test_samples_match_reading_one_at_a_time(self, grid_store):
        dataset = S2ZDataset(make_config(grid_store, min_views=0), split="train")
        singly, batched = self.read_both_ways(dataset, [0, 1, 2, 3])
        for one, other in zip(singly, batched, strict=True):
            assert one["source_index"] == other["source_index"]
            assert one["mask"] == other["mask"]
            for survey in one["surveys"]:
                a, b = one["data"][survey], other["data"][survey]
                if a is None:
                    assert b is None
                    continue
                assert torch.equal(torch.nan_to_num(a), torch.nan_to_num(b))

    def test_any_order_of_indices_is_kept(self, grid_store):
        dataset = S2ZDataset(make_config(grid_store, min_views=0), split="train")
        indices = [7, 0, 4]
        batched = dataset.__getitems__(indices)
        assert [sample["source_index"] for sample in batched] == [
            int(dataset.subset_indices[index]) for index in indices
        ]

    def test_a_survey_that_misses_a_source_is_still_masked_out(self, grid_store):
        dataset = S2ZDataset(make_config(grid_store, min_views=0), split="train")
        masks = [sample["mask"] for sample in dataset.__getitems__(range(len(dataset)))]
        # The second survey covers only part of the grid, by construction
        partial = dataset.surveys[1]
        assert any(not mask[partial] for mask in masks)
        assert any(mask[partial] for mask in masks)

    def test_a_dataloader_uses_it_for_every_sample(self, grid_store):
        dataset = S2ZDataset(make_config(grid_store, min_views=0), split="train")
        loader = DataLoader(
            dataset, batch_size=8, shuffle=False, collate_fn=collate_views
        )
        seen = [int(index) for batch in loader for index in batch["source_index"]]
        assert seen == [int(index) for index in dataset.subset_indices]


class TestBlockAlignedBatchSampler:
    """Verify block-aligned batch construction and shard access."""

    def dataset(self, grid_store, **overrides):
        return S2ZDataset(
            make_config(grid_store, min_views=0, **overrides), split="train"
        )

    def test_a_block_is_a_fraction_of_the_group_it_is_drawn_from(
        self, tmp_path, grid_survey_store, grid_catalogue_path
    ):
        # A block as large as the group is the group's whole served set, so the
        # draw within it has nothing to choose between and the block never moves
        dataset = self.dataset(
            small_shard_store(tmp_path, grid_survey_store, grid_catalogue_path)
        )
        sampler = BlockAlignedBatchSampler(dataset, batch_size=32)
        served = sampler.served_per_group(dataset.subset_indices, sampler.group_size)
        assert sampler.block_size <= served // MIN_BLOCKS_PER_GROUP
        assert sampler.blocks_per_batch >= BLOCKS_PER_BATCH

    def test_the_block_it_picks_is_measured_against_what_a_group_serves(
        self, tmp_path, grid_survey_store, grid_catalogue_path
    ):
        # Three of every four sources are served here, so a block sized against
        # the group's rows would be one a filtered group cannot fill
        dataset = self.dataset(
            small_shard_store(tmp_path, grid_survey_store, grid_catalogue_path)
        )
        sampler = BlockAlignedBatchSampler(dataset, batch_size=32)
        served = BlockAlignedBatchSampler.served_per_group(
            dataset.subset_indices, sampler.group_size
        )
        assert served < sampler.group_size
        assert sampler.block_size == BlockAlignedBatchSampler.choose_block_size(
            served, 32
        )

    def test_a_block_holds_sources_from_one_group(
        self, tmp_path, grid_survey_store, grid_catalogue_path
    ):
        # One group to a block is what makes a block one open per survey, and it
        # is the leftovers of a group that cost a second one
        dataset = self.dataset(
            small_shard_store(tmp_path, grid_survey_store, grid_catalogue_path)
        )
        sampler = BlockAlignedBatchSampler(
            dataset, batch_size=16, block_size=4, group_size=8
        )
        for _ in range(3):
            list(sampler)
            spans = [
                len(set(group_keys(dataset, block, sampler.group_size)))
                for block in sampler.blocks
            ]
            assert max(spans) <= 2
            assert sum(span == 1 for span in spans) > len(spans) / 2

    def test_a_leftover_block_spans_neighbouring_groups(
        self, tmp_path, grid_survey_store, grid_catalogue_path
    ):
        # Leftovers are gathered in group order, so a block made of them reads
        # groups next to each other rather than across the store
        dataset = self.dataset(
            small_shard_store(tmp_path, grid_survey_store, grid_catalogue_path)
        )
        sampler = BlockAlignedBatchSampler(
            dataset, batch_size=16, block_size=4, group_size=8
        )
        list(sampler)
        for block in sampler.blocks:
            keys = sorted(set(group_keys(dataset, block, sampler.group_size)))
            assert keys == list(range(keys[0], keys[0] + len(keys)))

    def test_the_block_count_holds_whatever_the_shards_serve(
        self, tmp_path, grid_survey_store, grid_catalogue_path
    ):
        # Whole blocks and leftovers between them account for every served
        # source once, so the count is the same however the filtering falls
        dataset = self.dataset(
            small_shard_store(tmp_path, grid_survey_store, grid_catalogue_path)
        )
        for block_size in (1, 2, 4, 8):
            sampler = BlockAlignedBatchSampler(
                dataset, batch_size=16, block_size=block_size
            )
            for _ in range(3):
                list(sampler)
                assert len(sampler.blocks) == math.ceil(len(dataset) / block_size)

    def test_a_group_bigger_than_a_batch_does_not_fix_the_batch(self, grid_store):
        # One block to a batch would leave every batch a fixed set of sources
        # that only moves place in the epoch, so the block is cut below the batch
        dataset = self.dataset(grid_store)
        sampler = BlockAlignedBatchSampler(dataset, batch_size=16)
        assert sampler.group_size > sampler.batch_size
        assert sampler.block_size < sampler.batch_size
        assert sampler.blocks_per_batch > 1
        torch.manual_seed(0)
        epochs = []
        for epoch in range(2):
            sampler.set_epoch(epoch)
            epochs.append({frozenset(batch) for batch in sampler})
        assert not epochs[0] & epochs[1]

    def test_batches_are_filled_from_whole_blocks(self, grid_store):
        # Blocks of this store are partly full, so a batch takes several of
        # them - never a piece of one, which would open its shard twice
        dataset = self.dataset(grid_store)
        sampler = BlockAlignedBatchSampler(dataset, batch_size=8, block_size=8)
        batches = list(sampler)
        blocks = {frozenset(block) for block in sampler.blocks}
        for batch in batches:
            drawn = set(batch)
            covered = [block for block in blocks if block <= drawn]
            assert set().union(*covered) == drawn

    def test_batches_come_out_the_size_they_asked_for(self, grid_store):
        # Blocks are cut on position within the split, so what this store
        # filters out cannot leave one short and a batch of whole blocks is
        # full however much is filtered
        dataset = self.dataset(grid_store)
        sizes = [
            len(batch)
            for batch in BlockAlignedBatchSampler(
                dataset, batch_size=32, block_size=8, shuffle=False
            )
        ]
        assert sizes[:-1] == [32] * (len(sizes) - 1)
        assert sizes[-1] <= 32
        assert sum(sizes) == len(dataset)

    def test_a_short_batch_is_always_the_last(self, grid_store):
        # Only the split's remainder can be short, so dropping it is dropping a
        # remainder and not whichever batch the shuffle put at the end
        dataset = self.dataset(grid_store)
        sampler = BlockAlignedBatchSampler(dataset, batch_size=16, block_size=8)
        for _ in range(3):  # A new order each epoch
            sizes = [len(batch) for batch in sampler]
            assert all(size == 16 for size in sizes[:-1])

    def test_drop_last_leaves_only_full_batches(self, grid_store):
        dataset = self.dataset(grid_store)
        kept = BlockAlignedBatchSampler(dataset, batch_size=16, block_size=8)
        dropped = BlockAlignedBatchSampler(
            dataset, batch_size=16, block_size=8, drop_last=True
        )
        assert all(len(batch) == 16 for batch in dropped)
        assert len(dropped) == len(list(dropped))
        assert len(dropped) <= len(kept)
        assert sum(len(b) for b in dropped) % 16 == 0

    def test_at_most_one_batch_an_epoch_is_short(self, grid_store):
        # Blocks are one size and a batch holds a whole number of them, so the
        # only batch that can come out short is the split's remainder
        dataset = self.dataset(grid_store)
        for batch_size in (8, 16, 32):
            sampler = BlockAlignedBatchSampler(
                dataset, batch_size=batch_size, block_size=8
            )
            short = [len(b) for b in sampler if len(b) != batch_size]
            assert len(short) <= 1

    def test_every_block_is_read_once_an_epoch(self, grid_store):
        dataset = self.dataset(grid_store)
        sampler = BlockAlignedBatchSampler(dataset, batch_size=16, block_size=8)
        batches = list(sampler)
        members = [set(block) for block in sampler.blocks]
        touched = sum(
            sum(1 for block in members if block & set(batch)) for batch in batches
        )
        assert touched == len(sampler.blocks)

    def test_a_batch_of_several_blocks_holds_whole_ones(self, grid_store):
        dataset = self.dataset(grid_store)
        sampler = BlockAlignedBatchSampler(dataset, batch_size=16, block_size=8)
        batches = list(sampler)
        members = [set(block) for block in sampler.blocks]
        for batch in batches:
            drawn = set(batch)
            # No block is split across batches, which would open its shard twice
            assert all(block <= drawn or not block & drawn for block in members)

    def test_an_epoch_serves_every_sample_exactly_once(self, grid_store):
        dataset = self.dataset(grid_store)
        sampler = BlockAlignedBatchSampler(dataset, batch_size=16, block_size=8)
        drawn = sorted(index for batch in sampler for index in batch)
        assert drawn == list(range(len(dataset)))

    def test_len_is_exact_however_the_blocks_fall(self, grid_store):
        dataset = self.dataset(grid_store)
        for batch_size, block_size in ((16, 8), (8, 8), (32, 8), (8, 4)):
            sampler = BlockAlignedBatchSampler(
                dataset, batch_size=batch_size, block_size=block_size
            )
            for _ in range(2):  # Shuffled differently each time
                assert len(list(sampler)) == len(sampler)

    def test_no_batch_is_larger_than_asked_for(self, grid_store):
        dataset = self.dataset(grid_store)
        sampler = BlockAlignedBatchSampler(dataset, batch_size=16, block_size=8)
        assert all(len(batch) <= 16 for batch in sampler)

    def test_unshuffled_it_serves_the_store_order(self, grid_store):
        dataset = self.dataset(grid_store)
        sampler = BlockAlignedBatchSampler(dataset, batch_size=16, shuffle=False)
        assert [index for batch in sampler for index in batch] == list(
            range(len(dataset))
        )

    def test_shuffling_redraws_what_shares_a_block(self, grid_store):
        # Blocks fixed for the run would put the same pair in the same batch
        # every epoch of it, which a model reads as structure not in the data
        dataset = self.dataset(grid_store)
        sampler = BlockAlignedBatchSampler(dataset, batch_size=8, block_size=8)
        torch.manual_seed(0)
        drawn = []
        for epoch in range(2):
            sampler.set_epoch(epoch)
            batches = list(sampler)
            drawn.append({frozenset(block) for block in sampler.blocks})
            # Every block whole, inside exactly one batch, however they pack
            for block in sampler.blocks:
                assert sum(1 for batch in batches if set(block) <= set(batch)) == 1
        assert drawn[0] != drawn[1]

    def test_the_batch_count_does_not_move_between_epochs(self, grid_store):
        # A run that sets a schedule from len(loader) must not have it drift
        dataset = self.dataset(grid_store)
        sampler = BlockAlignedBatchSampler(dataset, batch_size=16, block_size=8)
        torch.manual_seed(0)
        assert {len(list(sampler)) for _ in range(5)} == {len(sampler)}

    def test_shuffling_can_be_pinned_to_a_generator(self, grid_store):
        dataset = self.dataset(grid_store)

        def epoch(seed):
            generator = torch.Generator().manual_seed(seed)
            sampler = BlockAlignedBatchSampler(
                dataset, batch_size=8, block_size=8, generator=generator
            )
            return list(sampler)

        assert epoch(1) == epoch(1)
        assert epoch(1) != epoch(2)

    def test_a_block_a_batch_cannot_hold_whole_is_refused(self, grid_store):
        # Every batch would fall short by the remainder, which is the one thing
        # blocks cut on position exist to prevent
        dataset = self.dataset(grid_store)
        for block_size in (3, 12, 48):
            with pytest.raises(ValueError, match="must divide batch_size"):
                BlockAlignedBatchSampler(dataset, batch_size=32, block_size=block_size)

    def test_the_block_it_picks_divides_the_batch(self, grid_store, caplog):
        dataset = self.dataset(grid_store)
        with caplog.at_level(logging.INFO):
            for batch_size in (1, 3, 8, 16, 32, 64):
                sampler = BlockAlignedBatchSampler(dataset, batch_size=batch_size)
                assert batch_size % sampler.block_size == 0
                assert sampler.block_size <= sampler.group_size

    def test_a_batch_size_must_be_positive(self, grid_store):
        with pytest.raises(ValueError, match="batch_size must be at least 1"):
            BlockAlignedBatchSampler(self.dataset(grid_store), batch_size=0)

    def test_it_refuses_a_dataset_it_cannot_map_to_sources(self, grid_store):
        with pytest.raises(TypeError, match="subset_indices"):
            BlockAlignedBatchSampler([1, 2, 3], batch_size=4)

    def test_it_drives_a_dataloader(self, grid_store):
        dataset = self.dataset(grid_store)
        sampler = BlockAlignedBatchSampler(dataset, batch_size=16, block_size=8)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_views)
        assert len(loader) == len(sampler)
        seen = [int(index) for batch in loader for index in batch["source_index"]]
        assert sorted(seen) == sorted(int(index) for index in dataset.subset_indices)

    def test_the_group_defaults_to_eight_rows_a_sample(self, grid_store):
        # Taken from the batch and not from the store, where one appended survey
        # of few rows a shard would drag every group down to that survey's
        dataset = self.dataset(grid_store)
        for batch_size, group_size in ((1, 8), (32, 256), (48, 512), (256, 2048)):
            sampler = BlockAlignedBatchSampler(dataset, batch_size=batch_size)
            assert sampler.group_size == group_size
            assert group_size == 2 ** math.ceil(
                math.log2(GROUP_ROWS_PER_SAMPLE * batch_size)
            )

    def test_a_group_that_is_not_a_power_of_two_is_refused(self, grid_store):
        # Anything else stops dividing the store's shards, and a block then
        # straddles them with nothing to say so
        dataset = self.dataset(grid_store)
        for group_size in (0, 3, 12, 100):
            with pytest.raises(ValueError, match="power-of-two row count"):
                BlockAlignedBatchSampler(dataset, batch_size=32, group_size=group_size)

    def test_it_reports_what_a_batch_opens_of_each_survey(self, grid_store, caplog):
        # A group larger than a survey's shard costs opens in proportion, so the
        # price of the group is what a run can see and lower
        dataset = self.dataset(grid_store)
        with caplog.at_level(logging.INFO):
            sampler = BlockAlignedBatchSampler(dataset, batch_size=32, group_size=8)
        for survey in dataset.surveys:
            shard = dataset.substores[survey].shard_size
            opens = sampler.blocks_per_batch * max(1, sampler.group_size // shard)
            assert f"{opens} of {survey}" in caplog.text


class CountingData:
    """Proxy over a survey's pixel array, counting the reads it serves.

    A read of one shard is one access of ``oindex``, so the count is the files a
    batch opened of that survey.
    """

    def __init__(self, data):
        self._data = data
        self.reads = 0

    def __getattr__(self, name):
        return getattr(self._data, name)

    @property
    def oindex(self):
        self.reads += 1
        return self._data.oindex


class TestOpensPerBatch:
    """What a batch opens is arithmetic.

    Row ``i`` of every survey holds source ``i`` and both the group and every
    shard row count are powers of two, so partial coverage costs nothing and no
    block straddles a shard.
    """

    def dataset(self, store):
        # Read from the store, since a split held in memory opens nothing
        dataset = S2ZDataset(
            make_config(store, min_views=0, exclude={"sources": []}),
            split="train",
            in_memory=False,
        )
        for survey in dataset.surveys:
            substore = dataset.substores[survey]
            substore.data = CountingData(substore.data)
        return dataset

    def opens(self, dataset, samples):
        """Return the shards each survey opened to read these samples."""
        for survey in dataset.surveys:
            dataset.substores[survey].data.reads = 0
        dataset.__getitems__(list(samples))
        return {
            survey: dataset.substores[survey].data.reads for survey in dataset.surveys
        }

    def test_the_store_is_aligned_and_partly_covered(self, aligned_store):
        # The counts below rest on rows being source indices, one survey covering
        # part of the catalogue, and every shard of it covering some of it
        dataset = self.dataset(aligned_store)
        assert len(dataset) == N_SOURCES
        covered = {}
        for survey in dataset.surveys:
            sub = dataset.substores[survey]
            assert sub.shard_size == 8
            assert all(source == row for source, row in sub.index_map.items())
            covered[survey] = len(sub.index_map)
            shards = {row // sub.shard_size for row in sub.index_map.values()}
            assert shards == set(range(N_SOURCES // sub.shard_size))
        assert max(covered.values()) == N_SOURCES
        assert 0 < min(covered.values()) < N_SOURCES

    @pytest.mark.parametrize("group_size", [8, 16, 32])
    def test_a_batch_opens_what_the_group_and_the_shards_say(
        self, aligned_store, group_size
    ):
        # A block is a whole group's served sources here, so what a batch opens
        # is the geometry exactly rather than a draw that came out well
        dataset = self.dataset(aligned_store)
        sampler = BlockAlignedBatchSampler(
            dataset,
            batch_size=2 * group_size,
            block_size=group_size,
            group_size=group_size,
        )
        assert sampler.blocks_per_batch == 2
        for batch in sampler:
            for survey, opened in self.opens(dataset, batch).items():
                shard = dataset.substores[survey].shard_size
                assert opened == sampler.blocks_per_batch * max(1, group_size // shard)

    def test_a_group_inside_a_shard_still_costs_one_open_a_block(
        self, tmp_path, grid_survey_store, column_survey_store, grid_catalogue_path
    ):
        # A group smaller than a survey's shard falls inside one of them, so a
        # block is a single open of every survey covering any of the group
        store = build_grid_store(
            tmp_path / "d32",
            [grid_survey_store, column_survey_store],
            grid_catalogue_path,
            shard_size=32,
        )
        dataset = self.dataset(store)
        sampler = BlockAlignedBatchSampler(
            dataset, batch_size=8, block_size=8, group_size=8
        )
        assert sampler.blocks_per_batch == 1
        for batch in sampler:
            for survey, opened in self.opens(dataset, batch).items():
                shard = dataset.substores[survey].shard_size
                assert opened == 1 == max(1, sampler.group_size // shard)
