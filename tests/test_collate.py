import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from survey2zarray.dataset import S2ZDataset, collate_views, select_covered
from survey2zarray.stores.survey_store.build import build_survey_store
from tests.conftest import (
    IMAGE_DEC,
    IMAGE_RA,
    build_test_dataset_store,
    split_config,
)

CUTOUT_ARCSEC = 16.0
FINAL_PIXELS = 12


def make_config(dataset_store_path, **overrides):
    config = {
        "dataset_store": str(dataset_store_path),
        "splits": split_config(train=1.0),
        "final_pixel_size": FINAL_PIXELS,
        "preprocessing": {},
        "augmentation": {},
    }
    config.update(overrides)
    return config


@pytest.fixture
def dataset_store_path(continuum_fits_dir, spectral_fits_dir, tmp_path):
    continuum_dir, _ = continuum_fits_dir
    spectral_dir, _ = spectral_fits_dir
    survey_cont, survey_spec = tmp_path / "cont", tmp_path / "spec"
    build_survey_store(continuum_dir, survey_cont, mode="copy")
    build_survey_store(spectral_dir, survey_spec, mode="copy")
    catalogue = tmp_path / "catalogue.csv"
    pd.DataFrame(
        [
            # Covered by both surveys
            {
                "ID": "both_0",
                "RA": IMAGE_RA,
                "DEC": IMAGE_DEC,
                "label": 1,
            },
            # Only by the continuum survey: image_b's centre, outside the cubes
            {
                "ID": "cont_only",
                "RA": IMAGE_RA + 0.02,
                "DEC": IMAGE_DEC,
                "label": 0,
            },
        ]
    ).to_csv(catalogue, index=False)
    out = tmp_path / "dataset"
    build_test_dataset_store([survey_cont, survey_spec], catalogue, CUTOUT_ARCSEC, out)
    return out


class TestBatchStructure:
    def test_surveys_stay_separate(self, dataset_store_path):
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        batch = collate_views([dataset[0], dataset[1]])
        assert batch["surveys"] == ["cont", "spec"]
        # Channel counts differ, so the two could never share a tensor
        assert batch["views"]["cont"].shape[1] == 1
        assert batch["views"]["spec"].shape[1] == 3
        # Fixed per survey, so it is the dataset that says which axis is which
        assert dataset.has_freq_axis == {"cont": False, "spec": True}

    def test_views_are_compact_and_the_mask_locates_them(self, dataset_store_path):
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        batch = collate_views([dataset[0], dataset[1]])
        # cont_only is absent from the spectral survey, so that view holds one
        # row for a batch of two
        assert batch["mask"]["cont"].tolist() == [True, True]
        assert batch["mask"]["spec"].tolist() == [True, False]
        assert batch["views"]["cont"].shape[0] == 2
        assert batch["views"]["spec"].shape[0] == 1
        rows = batch["mask"]["spec"].nonzero().flatten().tolist()
        assert rows == [0]

    def test_rows_follow_batch_order(self, dataset_store_path):
        # Reversing the samples reverses the rows, so the mask stays a valid map
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        batch = collate_views([dataset[1], dataset[0]])
        assert batch["mask"]["spec"].tolist() == [False, True]
        assert batch["source_index"].tolist() == [1, 0]

    def test_survey_covering_nothing_gives_none(self, dataset_store_path):
        config = make_config(dataset_store_path, include={"sources": ["cont_only"]})
        dataset = S2ZDataset(config, split="train")
        batch = collate_views([dataset[0]])
        assert batch["views"]["spec"] is None
        assert batch["mask"]["spec"].tolist() == [False]

    def test_scalar_fields(self, dataset_store_path):
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        batch = collate_views([dataset[0], dataset[1]])
        assert batch["source_index"].tolist() == [0, 1]
        assert tuple(batch["source_coords"].shape) == (2, 2)
        assert batch["source_coords"][0, 0] == pytest.approx(IMAGE_RA, abs=1e-3)


class TestCatalogue:
    def test_integer_labels_stay_integral(self, dataset_store_path):
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        batch = collate_views([dataset[0], dataset[1]])
        labels = batch["catalogue"]["label"]
        assert labels.dtype == torch.long
        assert labels.tolist() == [1, 0]

    def test_non_numeric_columns_stay_lists(self, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        survey = tmp_path / "survey_str"
        build_survey_store(fits_dir, survey, mode="copy")
        catalogue = tmp_path / "named.csv"
        pd.DataFrame(
            [
                {
                    "ID": "a",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC,
                    "kind": "fri",
                },
            ]
        ).to_csv(catalogue, index=False)
        out = tmp_path / "named_dataset"
        build_test_dataset_store([survey], catalogue, CUTOUT_ARCSEC, out)
        dataset = S2ZDataset(make_config(out), split="train")
        batch = collate_views([dataset[0]])
        assert batch["catalogue"]["kind"] == ["fri"]


class TestSelectCovered:
    def test_it_matches_a_partial_survey_to_the_batch(self, dataset_store_path):
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        batch = collate_views([dataset[0], dataset[1]])
        labels = batch["catalogue"]["label"]
        # The spectral survey covers only the first source, so its one row
        # belongs to that source's label rather than to the first of the batch
        assert select_covered(batch, "spec", labels).tolist() == [1]
        assert select_covered(batch, "cont", labels).tolist() == [1, 0]
        assert len(select_covered(batch, "spec", labels)) == len(batch["views"]["spec"])

    def test_non_tensor_entries_come_back_as_lists(self, dataset_store_path):
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        batch = collate_views([dataset[0], dataset[1]])
        assert select_covered(batch, "spec", ["a", "b"]) == ["a"]

    def test_an_unknown_survey_is_rejected(self, dataset_store_path):
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        batch = collate_views([dataset[0]])
        with pytest.raises(KeyError, match="no survey"):
            select_covered(batch, "nowhere", batch["source_index"])

    def test_a_wrong_length_is_rejected(self, dataset_store_path):
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        batch = collate_views([dataset[0], dataset[1]])
        # A tensor already reduced to one survey's rows is the likely mistake
        with pytest.raises(ValueError, match="one per source"):
            select_covered(batch, "cont", torch.tensor([1]))


class TestErrors:
    def test_empty_batch_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            collate_views([])

    def test_ragged_views_name_the_fix(self, dataset_store_path):
        # No final_pixel_size and a crop range: every sample a different size
        config = make_config(
            dataset_store_path,
            augmentation={
                "train": {
                    "sky_transforms": {
                        "random_crop": {"min_crop": 0.5, "max_crop": 1.0}
                    }
                }
            },
        )
        config["final_pixel_size"] = None
        dataset = S2ZDataset(config, split="train")
        samples = [dataset[0], dataset[0], dataset[0], dataset[0]]
        if len({tuple(s["data"]["cont"].shape) for s in samples}) == 1:
            pytest.skip("crop sizes happened to coincide")
        with pytest.raises(RuntimeError, match="final_pixel_size"):
            collate_views(samples)

    def test_samples_from_different_datasets_rejected(self, dataset_store_path):
        full = S2ZDataset(make_config(dataset_store_path), split="train")
        subset = S2ZDataset(
            make_config(dataset_store_path, include={"surveys": ["cont"]}),
            split="train",
        )
        with pytest.raises(ValueError, match="disagree"):
            collate_views([full[0], subset[0]])


class TestWithDataLoader:
    def test_serves_batches_through_a_dataloader(self, dataset_store_path):
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        loader = DataLoader(
            dataset, batch_size=2, shuffle=False, collate_fn=collate_views
        )
        batches = list(loader)
        assert len(batches) == 1
        assert batches[0]["views"]["cont"].shape == (2, 1, FINAL_PIXELS, FINAL_PIXELS)

    def test_batch_size_one_still_batches(self, dataset_store_path):
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        loader = DataLoader(
            dataset, batch_size=1, shuffle=False, collate_fn=collate_views
        )
        first = next(iter(loader))
        assert first["views"]["cont"].shape[0] == 1
        assert first["mask"]["cont"].tolist() == [True]
