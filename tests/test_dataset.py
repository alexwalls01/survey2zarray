from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from survey2zarray.dataset.dataset import S2ZDataset
from survey2zarray.stores.dataset_store.filters import resolve_subset
from survey2zarray.stores.survey_store.build import build_survey_store
from tests.conftest import (
    IMAGE_DEC,
    IMAGE_RA,
    build_test_dataset_store,
    split_config,
)

CUTOUT_ARCSEC = 16.0

AUGMENTATION = {
    "sky_transforms": {
        "random_crop": {"min_crop": 0.375, "max_crop": 0.5},
        "random_rotation": {"min_angle": 0.0, "max_angle": 360.0},
        "random_horizontal_flip": {"p": 0.5},
        "random_vertical_flip": {"p": 0.5},
    },
    "signal_transforms": {
        "random_gaussian_noise": {"std": 0.05, "p": 0.5},
    },
}


@pytest.fixture
def dataset_store_path(continuum_fits_dir, spectral_fits_dir, tmp_path):
    # Two surveys: continuum (images a-c) and spectral (cubes a-b), sharing
    # sky coverage around (IMAGE_RA, IMAGE_DEC)
    continuum_dir, _ = continuum_fits_dir
    spectral_dir, _ = spectral_fits_dir
    survey_cont = tmp_path / "cont"
    survey_spec = tmp_path / "spec"
    build_survey_store(continuum_dir, survey_cont, mode="copy")
    build_survey_store(spectral_dir, survey_spec, mode="copy")
    catalogue = tmp_path / "catalogue.csv"
    pd.DataFrame(
        [
            # On both surveys
            {"ID": "both_0", "RA": IMAGE_RA, "DEC": IMAGE_DEC},
            # Only on the continuum survey (image_b centre; outside spectral cubes)
            {
                "ID": "cont_only",
                "RA": IMAGE_RA + 0.02,
                "DEC": IMAGE_DEC,
            },
            # Test-split source, kept well outside the 6" beam of both_0 so the
            # build-time duplicate check does not collapse the pair
            {
                "ID": "test_0",
                "RA": IMAGE_RA,
                "DEC": IMAGE_DEC + 14.0 / 3600.0,
            },
        ]
    ).to_csv(catalogue, index=False)
    out = tmp_path / "dataset"
    build_test_dataset_store([survey_cont, survey_spec], catalogue, CUTOUT_ARCSEC, out)
    return out


# What is servable shrinks with the configured cutout size, so tests that read
# a smaller cutout need a correspondingly smaller sky crop
SMALL_CROP_AUGMENTATION = {
    **AUGMENTATION,
    "sky_transforms": {
        **AUGMENTATION["sky_transforms"],
        "random_crop": {"min_crop": 0.5, "max_crop": 1.0},
    },
}


# Three sources over three buckets, so a two-thirds share takes the first two
TRAIN_AND_TEST = split_config(train=2 / 3, test=1 / 3)


def make_config(dataset_store_path, augmentation=None, **overrides):
    # Deep-copied so a test mutating its own config cannot leak into others
    augmentation = deepcopy(augmentation or AUGMENTATION)
    config = {
        "dataset_store": str(dataset_store_path),
        # Everything served goes to train unless a test says otherwise, so a
        # test about something else is not also a test of where the cut fell
        "splits": split_config(train=1.0),
        "preprocessing": {"sigma_clip": {}, "linear_rescale": {}},
        "augmentation": {"train": augmentation, "test": deepcopy(augmentation)},
    }
    config.update(overrides)
    return config


class TestGetItem:
    def test_output_structure(self, dataset_store_path):
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        assert len(dataset) == 3
        out = dataset[0]  # both_0
        assert set(out.keys()) == {
            "data",
            "surveys",
            "mask",
            "catalogue",
            "source_coords",
            "source_index",
        }
        assert out["surveys"] == ["cont", "spec"]
        assert out["mask"] == {"cont": True, "spec": True}
        assert out["source_index"] == 0
        assert out["source_coords"] == pytest.approx((IMAGE_RA, IMAGE_DEC))
        # data: one (C, H, W) tensor per survey, keyed by name
        assert set(out["data"]) == {"cont", "spec"}
        assert torch.is_tensor(out["data"]["cont"])
        assert out["data"]["cont"].shape[0] == 1  # Continuum: one channel
        assert torch.is_tensor(out["data"]["spec"])
        assert out["data"]["spec"].shape[0] == 3  # Spectral: all channels
        # Fixed per survey, so it is held once rather than in every sample
        assert dataset.has_freq_axis == {"cont": False, "spec": True}

    def test_absent_survey_is_none_with_valid_false(self, dataset_store_path):
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        out = dataset[1]  # cont_only
        assert out["mask"] == {"cont": True, "spec": False}
        assert out["data"]["cont"] is not None
        assert out["data"]["spec"] is None

    def test_augmentation_differs_between_calls(self, dataset_store_path):
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        a = dataset[0]["data"]["cont"]
        b = dataset[0]["data"]["cont"]
        # The generator advances between calls, so the same source comes back
        # differently augmented (sizes and/or values differ)
        assert a.shape != b.shape or not torch.allclose(
            torch.nan_to_num(a), torch.nan_to_num(b)
        )

    def test_min_views_drops_uncovered_sources(self, dataset_store_path):
        # cont_only has a cutout in one survey; a source in neither would have
        # every view None, so the default threshold of one excludes it
        default = S2ZDataset(make_config(dataset_store_path), split="train")
        assert len(default) == 3
        both = S2ZDataset(make_config(dataset_store_path, min_views=2), split="train")
        assert len(both) == 1
        assert both[0]["mask"] == {"cont": True, "spec": True}

    def test_min_views_counts_selected_surveys_only(self, dataset_store_path):
        # Excluding a survey takes cont_only's only view away with it
        config = make_config(
            dataset_store_path, exclude={"surveys": ["cont"]}, min_views=1
        )
        dataset = S2ZDataset(config, split="train")
        served = [str(dataset.store.source_id[i]) for i in dataset.subset_indices]
        assert "cont_only" not in served

    def test_min_views_zero_keeps_everything(self, dataset_store_path):
        dataset = S2ZDataset(
            make_config(dataset_store_path, min_views=0), split="train"
        )
        assert len(dataset) == 3

    def test_min_views_unsatisfiable_raises(self, dataset_store_path):
        with pytest.raises(ValueError, match="at least 3 view"):
            S2ZDataset(make_config(dataset_store_path, min_views=3), split="train")

    def test_split_filtering(self, dataset_store_path):
        config = make_config(dataset_store_path, splits=TRAIN_AND_TEST)
        train = S2ZDataset(config, split="train")
        test = S2ZDataset(config, split="test")
        assert len(train) == 2
        assert len(test) == 1
        assert test[0]["source_index"] == 2

    def test_an_empty_split_raises(self, dataset_store_path):
        config = make_config(dataset_store_path, splits=TRAIN_AND_TEST)
        with pytest.raises(ValueError, match="No sources found"):
            S2ZDataset(config, split="val")

    def test_a_split_name_outside_the_three_raises(self, dataset_store_path):
        with pytest.raises(ValueError, match="split must be one of"):
            S2ZDataset(make_config(dataset_store_path), split="validation")

    def test_servable_size_matches_the_configured_size(self, dataset_store_path):
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        # The disc is padded by margin_px of the coarsest pixel scale, which
        # is exactly what the reader insets by, so the full square is reachable
        assert dataset.servable_cutout_size == pytest.approx(CUTOUT_ARCSEC, abs=0.01)


class TestCutoutSize:
    def test_defaults_to_stored_size(self, dataset_store_path):
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        assert dataset.cutout_size == CUTOUT_ARCSEC
        # Not specified -> the whole stored disc, no re-crop
        assert dataset._crop_size is None
        sub = dataset.substores["cont"]
        cutout, _ = sub.get_cutout(
            sub.index_map[0], (IMAGE_RA, IMAGE_DEC), dataset._crop_size
        )
        assert cutout.shape == (27, 27)

    def test_exact_size_returns_whole_cutout(self, dataset_store_path):
        config = make_config(dataset_store_path, cutout_size=CUTOUT_ARCSEC)
        dataset = S2ZDataset(config, split="train")
        assert dataset.cutout_size == CUTOUT_ARCSEC
        assert dataset._crop_size is None

    def test_larger_than_stored_fails(self, dataset_store_path):
        config = make_config(dataset_store_path, cutout_size=CUTOUT_ARCSEC + 4.0)
        with pytest.raises(ValueError, match="exceeds the stored"):
            S2ZDataset(config, split="train")

    def test_smaller_size_crops_about_the_source(self, dataset_store_path):
        # An 8" square needs an 11.3" disc, so 12x12 pixels at 1"/px
        config = make_config(
            dataset_store_path, SMALL_CROP_AUGMENTATION, cutout_size=8.0
        )
        dataset = S2ZDataset(config, split="train")
        assert dataset.cutout_size == 8.0
        assert dataset._crop_size == 8.0
        sub = dataset.substores["cont"]
        local = sub.index_map[0]
        full, full_wcs = sub.get_cutout(local)
        cropped, cropped_wcs = sub.get_cutout(local, (IMAGE_RA, IMAGE_DEC), 8.0, 2.0)
        assert full.shape == (27, 27)
        assert cropped.shape == (16, 16)
        # The crop is a real sub-array of the stored cutout at the origin its
        # WCS records, not a resampling. Only the smaller disc is kept, so the
        # corners of the crop window are blanked where the stored one had data.
        x0 = int(round(float(full_wcs.wcs.crpix[0] - cropped_wcs.wcs.crpix[0])))
        y0 = int(round(float(full_wcs.wcs.crpix[1] - cropped_wcs.wcs.crpix[1])))
        window = full[y0 : y0 + 16, x0 : x0 + 16]
        inside = np.isfinite(cropped)
        np.testing.assert_array_equal(cropped[inside], window[inside])
        # Re-blanked to the smaller disc rather than inheriting the stored one
        centre = (cropped.shape[-1] - 1) / 2
        yy, xx = np.mgrid[0 : cropped.shape[-2], 0 : cropped.shape[-1]]
        radius = np.hypot(xx - centre, yy - centre)  # 1"/px, so pixels = arcsec
        assert inside[radius <= 8.0 * np.sqrt(2) / 2 + 2 - 1].all()
        assert not inside[radius >= 8.0 * np.sqrt(2) / 2 + 2 + 1].any()

    def test_cropped_wcs_maps_source_to_crop_centre(self, dataset_store_path):
        from astropy import units as u
        from astropy.coordinates import ICRS, SkyCoord

        config = make_config(
            dataset_store_path, SMALL_CROP_AUGMENTATION, cutout_size=8.0
        )
        dataset = S2ZDataset(config, split="train")
        sub = dataset.substores["cont"]
        _, wcs = sub.get_cutout(sub.index_map[0], (IMAGE_RA, IMAGE_DEC), 8.0, 2.0)
        source = SkyCoord(ra=IMAGE_RA * u.deg, dec=IMAGE_DEC * u.deg, frame=ICRS)
        x, y = wcs.celestial.world_to_pixel(source)
        # Centre of a 12x12 box, within the half-pixel rounding of the origin
        assert float(x) == pytest.approx(7.5, abs=1.0)
        assert float(y) == pytest.approx(7.5, abs=1.0)

    def test_smaller_size_shrinks_what_is_served(self, dataset_store_path):
        config = make_config(
            dataset_store_path, SMALL_CROP_AUGMENTATION, cutout_size=8.0
        )
        dataset = S2ZDataset(config, split="train")
        # What is served follows the effective cutout size, not the stored one
        assert dataset.servable_cutout_size == pytest.approx(8.0, abs=0.01)

    def test_smaller_size_applies_through_getitem(self, dataset_store_path):
        config = make_config(
            dataset_store_path, SMALL_CROP_AUGMENTATION, cutout_size=8.0
        )
        dataset = S2ZDataset(config, split="train")
        out = dataset[0]
        assert out["mask"] == {"cont": True, "spec": True}
        for view in out["data"].values():
            assert view.shape[-1] > 0

    def test_spectral_crop_keeps_all_channels(self, dataset_store_path):
        config = make_config(
            dataset_store_path, SMALL_CROP_AUGMENTATION, cutout_size=8.0
        )
        dataset = S2ZDataset(config, split="train")
        sub = dataset.substores["spec"]
        cropped, _ = sub.get_cutout(sub.index_map[0], (IMAGE_RA, IMAGE_DEC), 8.0, 2.0)
        assert cropped.shape == (3, 16, 16)

    def test_per_axis_size_rejected(self, dataset_store_path):
        # Cutouts are discs, so there is one size rather than one per axis
        config = make_config(
            dataset_store_path, SMALL_CROP_AUGMENTATION, cutout_size=(8.0, 12.0)
        )
        with pytest.raises(TypeError, match="one disc diameter"):
            S2ZDataset(config, split="train")

    def test_non_positive_size_fails(self, dataset_store_path):
        with pytest.raises(ValueError, match="must be positive"):
            S2ZDataset(make_config(dataset_store_path, cutout_size=0.0), split="train")


class TestResolveCutoutSize:
    def test_not_specified_returns_stored(self):
        assert S2ZDataset.resolve_cutout_size(None, 16.0) == 16.0

    def test_integer_accepted_as_float(self):
        assert S2ZDataset.resolve_cutout_size(8, 16.0) == 8.0

    def test_exact_match_allowed(self):
        assert S2ZDataset.resolve_cutout_size(16.0, 16.0) == 16.0

    def test_larger_rejected(self):
        with pytest.raises(ValueError, match="exceeds the stored"):
            S2ZDataset.resolve_cutout_size(16.1, 16.0)

    def test_sequence_rejected(self):
        with pytest.raises(TypeError, match="one disc diameter"):
            S2ZDataset.resolve_cutout_size((8.0, 8.0), 16.0)


class TestFinalPixelSize:
    """Every view resampled to one pixel size, after preprocessing and augmentation."""

    def test_absent_leaves_each_survey_at_its_own_count(self, dataset_store_path):
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        assert dataset.final_pixel_size is None
        # Crops are drawn between 6 and 8 arcsec at 1"/px, so an unresized view
        # tracks the crop rather than settling on one size
        sizes = set()
        for _ in range(10):
            sample = dataset[0]
            sizes |= {
                tuple(view.shape[-2:])
                for view in sample["data"].values()
                if view is not None
            }
        assert sizes
        assert all(5 <= height <= 9 and height == width for height, width in sizes)

    def test_every_view_comes_back_at_that_size(self, dataset_store_path):
        config = make_config(dataset_store_path, final_pixel_size=7)
        dataset = S2ZDataset(config, split="train")
        sample = dataset[0]
        for view in sample["data"].values():
            if view is None:
                continue
            assert tuple(view.shape[-2:]) == (7, 7)
        # The channel axis is untouched: spectral surveys keep every channel
        assert sample["data"]["spec"].shape[0] == 3

    def test_absent_survey_stays_absent(self, dataset_store_path):
        config = make_config(
            dataset_store_path, include={"sources": ["cont_only"]}, final_pixel_size=7
        )
        sample = S2ZDataset(config, split="train")[0]
        assert sample["mask"] == {"cont": True, "spec": False}
        assert sample["data"]["spec"] is None

    def test_warns_when_below_the_largest_crop(self, dataset_store_path, caplog):
        # AUGMENTATION crops up to 8 arcsec at 1"/px, so 8 px is the most a
        # view will hold; 4 throws half of it away
        config = make_config(dataset_store_path, final_pixel_size=4)
        with caplog.at_level("WARNING"):
            S2ZDataset(config, split="train")
        assert any("downsamples" in r.message for r in caplog.records)

    def test_quiet_when_it_keeps_the_resolution(self, dataset_store_path, caplog):
        config = make_config(dataset_store_path, final_pixel_size=64)
        with caplog.at_level("WARNING"):
            S2ZDataset(config, split="train")
        assert not any("downsamples" in r.message for r in caplog.records)

    def test_without_random_crop_the_bound_is_the_cutout_size(
        self, dataset_store_path, caplog
    ):
        # No sky_transforms at all: the largest view is the whole served cutout.
        # Set it after the fact: make_config treats a falsy augmentation as
        # "unspecified" and substitutes its default.
        config = make_config(dataset_store_path, final_pixel_size=4)
        config["augmentation"] = {}
        with caplog.at_level("WARNING"):
            dataset = S2ZDataset(config, split="train")
        assert any("downsamples" in r.message for r in caplog.records)
        # And the bound quoted is the served size, not a crop
        assert any(
            f"{dataset.servable_cutout_size:.0f} arcsec" in r.message
            for r in caplog.records
        )

    def test_non_positive_rejected(self, dataset_store_path):
        with pytest.raises(ValueError, match="must be positive"):
            S2ZDataset(
                make_config(dataset_store_path, final_pixel_size=0), split="train"
            )


class TestCatalogueColumns:
    """Catalogue columns beyond id/ra/dec/split, returned only when asked for."""

    @pytest.fixture
    def labelled_store(self, continuum_fits_dir, spectral_fits_dir, tmp_path):
        continuum_dir, _ = continuum_fits_dir
        spectral_dir, _ = spectral_fits_dir
        survey_cont, survey_spec = tmp_path / "cont2", tmp_path / "spec2"
        build_survey_store(continuum_dir, survey_cont, mode="copy")
        build_survey_store(spectral_dir, survey_spec, mode="copy")
        catalogue = tmp_path / "labelled.csv"
        # A string column and a numeric one, which the store keeps as object
        # and float respectively
        pd.DataFrame(
            [
                {
                    "ID": "both_0",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC,
                    "label": "fri",
                    "redshift": 0.13,
                },
                {
                    "ID": "cont_only",
                    "RA": IMAGE_RA + 0.02,
                    "DEC": IMAGE_DEC,
                    "label": "frii",
                    "redshift": 0.42,
                },
            ]
        ).to_csv(catalogue, index=False)
        out = tmp_path / "labelled_dataset"
        build_test_dataset_store(
            [survey_cont, survey_spec], catalogue, CUTOUT_ARCSEC, out
        )
        return out

    def test_store_keeps_the_extra_columns(self, labelled_store):
        from survey2zarray.stores.dataset_store.dataset_store import DatasetStore

        store = DatasetStore.open(labelled_store)
        assert set(store.extra_columns) == {"label", "redshift"}
        assert [str(v) for v in store.extra_columns["label"]] == ["fri", "frii"]

    def test_served_with_every_sample(self, labelled_store):
        dataset = S2ZDataset(make_config(labelled_store), split="train")
        sample = dataset[0]
        assert sample["catalogue"] == {"label": "fri", "redshift": pytest.approx(0.13)}
        # Values follow the source, not the sample position
        assert dataset[1]["catalogue"]["label"] == "frii"

    def test_values_are_plain_python_types(self, labelled_store):
        # So a default collate can batch them without numpy scalars leaking through
        columns = S2ZDataset(make_config(labelled_store), split="train")[0]["catalogue"]
        assert type(columns["redshift"]) is float
        assert isinstance(columns["label"], str)

    def test_integer_columns_stay_integers(self, continuum_fits_dir, tmp_path):
        # A class label read back as 1.0 would need casting before it could
        # index anything, so integer columns keep their type through the store
        from survey2zarray.stores.dataset_store.dataset_store import DatasetStore

        fits_dir, _ = continuum_fits_dir
        survey = tmp_path / "int_survey"
        build_survey_store(fits_dir, survey, mode="copy")
        catalogue = tmp_path / "int_labels.csv"
        pd.DataFrame(
            [
                {
                    "ID": "a",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC,
                    "label": 1,
                    "redshift": 0.13,
                },
            ]
        ).to_csv(catalogue, index=False)
        out = tmp_path / "int_dataset"
        build_test_dataset_store([survey], catalogue, CUTOUT_ARCSEC, out)

        store = DatasetStore.open(out)
        assert store.extra_columns["label"].dtype == np.int64
        assert store.extra_columns["redshift"].dtype == np.float64

        columns = S2ZDataset(make_config(out), split="train")[0]["catalogue"]
        assert columns["label"] == 1
        assert type(columns["label"]) is int
        assert type(columns["redshift"]) is float

    def test_no_extra_columns_gives_an_empty_mapping(self, dataset_store_path):
        # The standard four only: the key is present but there is nothing in it
        config = make_config(dataset_store_path)
        assert S2ZDataset(config, split="train")[0]["catalogue"] == {}


class TestPlot:
    def test_returns_the_figure_without_showing_it(
        self, dataset_store_path, monkeypatch
    ):
        # show=False is what lets a caller embed the figure somewhere else,
        # such as an ipywidgets Output, without it being drawn twice
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shown = []
        monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(1))
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")

        figure = dataset.plot(0, show=False)
        assert isinstance(figure, plt.Figure)
        assert shown == []
        # One panel per survey, whether or not each covers the source
        assert len(figure.axes) == len(dataset.surveys)
        plt.close(figure)

        dataset.plot(0)
        assert shown == [1]
        plt.close("all")

    def test_missing_panel_matches_the_others(self, dataset_store_path):
        # cont_only is absent from the spectral survey, so its panel carries a
        # message instead of an image - and must still be the same square
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        index = next(
            i for i in range(len(dataset)) if not all(dataset[i]["mask"].values())
        )
        figure = dataset.plot(index, show=False)
        figure.canvas.draw()

        boxes = [ax.get_window_extent() for ax in figure.axes]
        assert len({(round(b.width, 3), round(b.height, 3)) for b in boxes}) == 1
        for box in boxes:
            assert box.width == pytest.approx(box.height)

        messages = [t.get_text() for ax in figure.axes for t in ax.texts]
        assert messages == ["Source not in survey"]
        plt.close(figure)

    def test_source_id_plots_the_same_source_as_its_index(self, dataset_store_path):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        source_index = int(dataset.subset_indices[1])
        source_id = str(dataset.store.source_id[source_index])

        by_index = dataset.plot(1, show=False)
        by_id = dataset.plot(source_id, show=False)
        assert by_id.get_suptitle() == by_index.get_suptitle()
        plt.close(by_index)
        plt.close(by_id)

    def test_unknown_source_id_errors(self, dataset_store_path):
        import matplotlib

        matplotlib.use("Agg")

        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        with pytest.raises(KeyError, match="not in the dataset store"):
            dataset.plot("no-such-source", show=False)

    def test_source_id_outside_the_split_errors(self, dataset_store_path):
        # Held by the store, but not served here, which is a different fault
        import matplotlib

        matplotlib.use("Agg")

        dataset = S2ZDataset(
            make_config(dataset_store_path, splits=TRAIN_AND_TEST), split="train"
        )
        served = {int(i) for i in dataset.subset_indices}
        elsewhere = next(
            str(v) for i, v in enumerate(dataset.store.source_id) if i not in served
        )
        with pytest.raises(KeyError, match="split"):
            dataset.plot(elsewhere, show=False)


class TestSubsetting:
    def test_include_surveys(self, dataset_store_path):
        config = make_config(dataset_store_path, include={"surveys": ["cont"]})
        dataset = S2ZDataset(config, split="train")
        assert dataset.surveys == ["cont"]
        out = dataset[0]
        assert out["surveys"] == ["cont"]
        assert set(out["data"]) == {"cont"}

    def test_exclude_surveys(self, dataset_store_path):
        config = make_config(dataset_store_path, exclude={"surveys": ["spec"]})
        dataset = S2ZDataset(config, split="train")
        assert dataset.surveys == ["cont"]

    def test_include_and_exclude_surveys(self, dataset_store_path):
        # Both given: includes define the universe; excludes (which may not
        # overlap the includes) remove from within it
        config = make_config(
            dataset_store_path,
            include={"surveys": ["cont"]},
            exclude={"surveys": ["spec"]},
        )
        dataset = S2ZDataset(config, split="train")
        assert dataset.surveys == ["cont"]

    def test_conflicting_lists_error(self, dataset_store_path):
        config = make_config(
            dataset_store_path,
            include={"surveys": ["cont"]},
            exclude={"surveys": ["cont"]},
        )
        with pytest.raises(ValueError, match="both include and exclude"):
            S2ZDataset(config, split="train")

    def test_unknown_include_error(self, dataset_store_path):
        config = make_config(dataset_store_path, include={"surveys": ["tpyo"]})
        with pytest.raises(ValueError, match="absent from the dataset"):
            S2ZDataset(config, split="train")

    def test_unknown_exclude_warns(self, dataset_store_path, caplog):
        config = make_config(dataset_store_path, exclude={"surveys": ["tpyo"]})
        with caplog.at_level("WARNING"):
            dataset = S2ZDataset(config, split="train")
        assert dataset.surveys == ["cont", "spec"]
        assert any("tpyo" in r.message for r in caplog.records)

    def test_source_subset_composes_with_index_map(self, dataset_store_path):
        # Subset index -> canonical source_index -> per-survey local index
        config = make_config(dataset_store_path, include={"sources": ["cont_only"]})
        dataset = S2ZDataset(config, split="train")
        assert len(dataset) == 1
        out = dataset[0]
        assert out["source_index"] == 1  # Canonical index unchanged by subsetting
        assert out["mask"] == {"cont": True, "spec": False}

    def test_exclude_sources(self, dataset_store_path):
        config = make_config(
            dataset_store_path, exclude={"sources": ["cont_only", "test_0"]}
        )
        dataset = S2ZDataset(config, split="train")
        assert len(dataset) == 1
        assert dataset[0]["source_index"] == 0


class TestLabels:
    """Turning a catalogue column into the integer a loss can use."""

    @pytest.fixture
    def classified_store(self, continuum_fits_dir, spectral_fits_dir, tmp_path):
        continuum_dir, _ = continuum_fits_dir
        spectral_dir, _ = spectral_fits_dir
        survey_cont, survey_spec = tmp_path / "cont_c", tmp_path / "spec_c"
        build_survey_store(continuum_dir, survey_cont, mode="copy")
        build_survey_store(spectral_dir, survey_spec, mode="copy")
        catalogue = tmp_path / "classified.csv"
        # FRII first, so appearance order and alphabetical order disagree
        pd.DataFrame(
            [
                {
                    "ID": "a",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC,
                    "kind": "FRII",
                },
                {
                    "ID": "b",
                    "RA": IMAGE_RA + 0.02,
                    "DEC": IMAGE_DEC,
                    "kind": "FRI",
                },
                {
                    "ID": "c",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC + 14 / 3600,
                    "kind": "FRI",
                },
            ]
        ).to_csv(catalogue, index=False)
        out = tmp_path / "classified_ds"
        build_test_dataset_store(
            [survey_cont, survey_spec], catalogue, CUTOUT_ARCSEC, out
        )
        return out

    def test_explicit_order_is_obeyed(self, classified_store):
        config = make_config(
            classified_store,
            label_by="kind",
            label_order=["FRI", "FRII"],
            splits=TRAIN_AND_TEST,
        )
        dataset = S2ZDataset(config, split="train")
        assert dataset.label_classes == ["FRI", "FRII"]
        labels = {
            str(dataset.store.source_id[dataset.subset_indices[i]]): dataset[i]["label"]
            for i in range(len(dataset))
        }
        assert labels == {"a": 1, "b": 0}

    def test_without_an_order_appearance_decides(self, classified_store, caplog):
        with caplog.at_level("WARNING"):
            dataset = S2ZDataset(
                make_config(classified_store, splits=TRAIN_AND_TEST, label_by="kind"),
                split="train",
            )
        # FRII appears first in the catalogue, so it takes zero
        assert dataset.label_classes == ["FRII", "FRI"]
        assert any(
            "Catalogue order defines label numbering" in r.message
            for r in caplog.records
        )

    def test_partial_order_puts_the_rest_after(self, classified_store):
        config = make_config(
            classified_store,
            splits=TRAIN_AND_TEST,
            label_by="kind",
            label_order=["FRI"],
        )
        dataset = S2ZDataset(config, split="train")
        assert dataset.label_classes == ["FRI", "FRII"]

    def test_named_class_keeps_its_slot_when_absent(self, classified_store):
        # Otherwise a store missing a class would number the others differently
        config = make_config(
            classified_store,
            splits=TRAIN_AND_TEST,
            label_by="kind",
            label_order=["FRI", "Hybrid", "FRII"],
        )
        dataset = S2ZDataset(config, split="train")
        assert dataset.label_classes == ["FRI", "Hybrid", "FRII"]
        labels = {
            str(dataset.store.source_id[dataset.subset_indices[i]]): dataset[i]["label"]
            for i in range(len(dataset))
        }
        assert labels == {"a": 2, "b": 0}

    def test_every_split_agrees(self, classified_store):
        # The mapping comes from every source the dataset serves, not the split
        # being read, so a class cannot mean one integer in train and another
        # in test
        config = make_config(classified_store, splits=TRAIN_AND_TEST, label_by="kind")
        train = S2ZDataset(config, split="train")
        test = S2ZDataset(config, split="test")
        assert train.label_classes == test.label_classes

    def test_a_filtered_out_class_holds_no_slot(self, classified_store):
        # 'c' is the only FRI in test, and excluding FRI should leave the model
        # two outputs rather than three, one of which nothing ever carries
        config = make_config(classified_store, label_by="kind")
        config["exclude"] = {"kind": "FRI"}
        dataset = S2ZDataset(config, split="train")
        assert dataset.label_classes == ["FRII"]
        assert [dataset[i]["label"] for i in range(len(dataset))] == [0]

    def test_a_named_class_keeps_its_slot_even_when_filtered_out(
        self, classified_store
    ):
        # label_order is the caller pinning the label space, which outranks
        # what this particular filtering happens to leave
        config = make_config(
            classified_store, label_by="kind", label_order=["FRI", "FRII"]
        )
        config["exclude"] = {"kind": "FRI"}
        dataset = S2ZDataset(config, split="train")
        assert dataset.label_classes == ["FRI", "FRII"]

    def test_a_class_only_unserved_sources_carry_is_left_out(self, classified_store):
        # 'c' is the only source in test; a class carried by nothing served is
        # not one the model has to hold an output for
        config = make_config(classified_store, splits=TRAIN_AND_TEST, label_by="kind")
        config["exclude"] = {"sources": ["a"]}
        dataset = S2ZDataset(config, split="train")
        assert dataset.label_classes == ["FRI"]

    def test_absent_unless_asked_for(self, classified_store):
        dataset = S2ZDataset(
            make_config(classified_store, splits=TRAIN_AND_TEST), split="train"
        )
        assert dataset.label_classes is None
        assert "label" not in dataset[0]

    def test_unknown_column_rejected(self, classified_store):
        with pytest.raises(ValueError, match="Cannot label by column"):
            S2ZDataset(
                make_config(classified_store, splits=TRAIN_AND_TEST, label_by="nope"),
                split="train",
            )

    def test_missing_values_rejected(self, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        survey = tmp_path / "survey_gap"
        build_survey_store(fits_dir, survey, mode="copy")
        catalogue = tmp_path / "gap.csv"
        pd.DataFrame(
            [
                {
                    "ID": "a",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC,
                    "kind": "FRI",
                },
                {
                    "ID": "b",
                    "RA": IMAGE_RA + 0.02,
                    "DEC": IMAGE_DEC,
                    "kind": "",
                },
            ]
        ).to_csv(catalogue, index=False)
        out = tmp_path / "gap_ds"
        build_test_dataset_store([survey], catalogue, CUTOUT_ARCSEC, out)
        with pytest.raises(ValueError, match="no value in the label column"):
            S2ZDataset(make_config(out, label_by="kind"), split="train")

    def test_plot_shows_the_category_not_the_integer(self, classified_store):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        config = make_config(
            classified_store,
            splits=TRAIN_AND_TEST,
            label_by="kind",
            label_order=["FRI", "FRII"],
        )
        dataset = S2ZDataset(config, split="train")
        figure = dataset.plot(0, show=False)
        title = figure._suptitle.get_text()
        assert "kind = " in title
        assert any(name in title for name in ("FRI", "FRII"))
        plt.close(figure)


class TestColumnSubsetting:
    """Selecting sources by the value of a catalogue column."""

    @pytest.fixture
    def labelled_store(self, continuum_fits_dir, spectral_fits_dir, tmp_path):
        continuum_dir, _ = continuum_fits_dir
        spectral_dir, _ = spectral_fits_dir
        survey_cont, survey_spec = tmp_path / "cont_l", tmp_path / "spec_l"
        build_survey_store(continuum_dir, survey_cont, mode="copy")
        build_survey_store(spectral_dir, survey_spec, mode="copy")
        catalogue = tmp_path / "labelled.csv"
        pd.DataFrame(
            [
                {
                    "ID": "a",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC,
                    "label": 1,
                    "kind": "fri",
                },
                {
                    "ID": "b",
                    "RA": IMAGE_RA + 0.02,
                    "DEC": IMAGE_DEC,
                    "label": 0,
                    "kind": "frii",
                },
            ]
        ).to_csv(catalogue, index=False)
        out = tmp_path / "labelled_ds"
        build_test_dataset_store(
            [survey_cont, survey_spec], catalogue, CUTOUT_ARCSEC, out
        )
        return out

    def served(self, store, **overrides):
        dataset = S2ZDataset(make_config(store, **overrides), split="train")
        return [str(dataset.store.source_id[i]) for i in dataset.subset_indices]

    def test_include_by_column(self, labelled_store):
        assert self.served(labelled_store, include={"label": 1}) == ["a"]

    def test_exclude_by_column(self, labelled_store):
        assert self.served(labelled_store, exclude={"label": [1]}) == ["b"]

    def test_scalar_and_list_are_equivalent(self, labelled_store):
        assert self.served(labelled_store, include={"label": 1}) == self.served(
            labelled_store, include={"label": [1]}
        )

    def test_string_columns_work(self, labelled_store):
        assert self.served(labelled_store, include={"kind": "frii"}) == ["b"]

    def test_criteria_combine_with_and(self, labelled_store):
        # Both satisfied by "a", so it survives; "b" fails the label test
        assert self.served(
            labelled_store, include={"label": 1, "sources": ["a", "b"]}
        ) == ["a"]
        # Nothing satisfies both, and an empty selection is an error
        with pytest.raises(ValueError, match="No sources found"):
            self.served(labelled_store, include={"label": 1, "sources": ["b"]})

    def test_include_and_exclude_compose(self, labelled_store):
        assert self.served(
            labelled_store, include={"kind": ["fri", "frii"]}, exclude={"label": 0}
        ) == ["a"]

    def test_unknown_column_rejected(self, labelled_store):
        with pytest.raises(ValueError, match="Cannot filter on column"):
            self.served(labelled_store, include={"colour": "red"})

    def test_value_in_both_lists_rejected(self, labelled_store):
        with pytest.raises(ValueError, match="both include and exclude"):
            self.served(labelled_store, include={"label": 1}, exclude={"label": [1, 0]})

    def test_value_matching_nothing_warns(self, labelled_store, caplog):
        with caplog.at_level("WARNING"):
            with pytest.raises(ValueError, match="No sources found"):
                self.served(labelled_store, include={"label": 7})
        assert any("No source has label" in r.message for r in caplog.records)

    def test_section_must_be_a_mapping(self, labelled_store):
        with pytest.raises(TypeError, match="must be a mapping"):
            self.served(labelled_store, include=["a"])

    def test_store_without_extra_columns_is_unaffected(self, dataset_store_path):
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        assert len(dataset) == 3


class TestResolveSubset:
    def test_includes_alone_define_the_set(self):
        assert resolve_subset(["a", "b", "c"], ["b", "c"], None, "items") == ["b", "c"]

    def test_excludes_alone_remove(self):
        assert resolve_subset(["a", "b", "c"], None, ["b"], "items") == ["a", "c"]

    def test_neither_gives_everything(self):
        assert resolve_subset(["a", "b"], None, None, "items") == ["a", "b"]

    def test_both_compose(self):
        # Overlap between the lists is a conflict error, so with both given
        # the excludes may only name items outside the include universe
        assert resolve_subset(["a", "b", "c"], ["a", "b"], ["c"], "items") == ["a", "b"]

    def test_overlap_is_conflict(self):
        with pytest.raises(ValueError, match="both include and exclude"):
            resolve_subset(["a", "b", "c"], ["a", "b"], ["b"], "items")


def identity_collate(x):
    return x


class TestDataLoader:
    def test_dataloader_iteration(self, dataset_store_path):
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=0, collate_fn=identity_collate
        )
        items = list(loader)
        assert len(items) == 3
        assert {item["source_index"] for item in items} == {0, 1, 2}

    def test_rng_reseeded_per_worker(self, dataset_store_path, monkeypatch):
        # In a worker process the rng derives from the worker's own seed
        # (simulated: spawning real workers is not needed to test the rule)
        class FakeWorkerInfo:
            seed = 1234567

        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        monkeypatch.setattr(
            torch.utils.data, "get_worker_info", lambda: FakeWorkerInfo()
        )
        expected = np.random.default_rng(FakeWorkerInfo.seed % 2**32).random()
        assert dataset.rng.random() == expected

    def test_rng_follows_torch_seed_in_main_process(self, dataset_store_path):
        # Outside a worker the generator descends from torch's own seed.
        torch.manual_seed(4321)
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        expected = np.random.default_rng(torch.initial_seed() % 2**32).random()
        assert dataset.rng.random() == expected

    def test_same_torch_seed_reproduces_main_process_stream(self, dataset_store_path):
        draws = []
        for _ in range(2):
            torch.manual_seed(99)
            dataset = S2ZDataset(make_config(dataset_store_path), split="train")
            draws.append(dataset.rng.random())
        assert draws[0] == draws[1]

    def test_distinct_worker_seeds_give_distinct_streams(
        self, dataset_store_path, monkeypatch
    ):
        class FakeWorkerInfo:
            seed = 101

        monkeypatch.setattr(
            torch.utils.data, "get_worker_info", lambda: FakeWorkerInfo()
        )
        first = S2ZDataset(make_config(dataset_store_path), split="train").rng.random()
        FakeWorkerInfo.seed = 102
        second = S2ZDataset(make_config(dataset_store_path), split="train").rng.random()
        assert first != second

    def test_same_worker_seed_reproduces_worker_stream(
        self, dataset_store_path, monkeypatch
    ):
        class FakeWorkerInfo:
            seed = 1234567

        monkeypatch.setattr(
            torch.utils.data, "get_worker_info", lambda: FakeWorkerInfo()
        )
        draws = [
            S2ZDataset(make_config(dataset_store_path), split="train").rng.random()
            for _ in range(2)
        ]
        assert draws[0] == draws[1]

    def test_loader_seed_resets_and_controls_main_process_stream(
        self, dataset_store_path
    ):
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        dataset.rng.random()

        dataset.set_augmentation_seed(2468)
        expected = np.random.default_rng(2468).random()

        assert dataset.rng.random() == expected

    def test_negative_augmentation_seed_is_rejected(self, dataset_store_path):
        dataset = S2ZDataset(make_config(dataset_store_path), split="train")
        with pytest.raises(ValueError, match="augmentation seed"):
            dataset.set_augmentation_seed(-1)


class TestSplits:
    """Verify bucket splits preserve fractions over served sources."""

    def test_the_fractions_are_of_the_served_population(self, dataset_store_path):
        # Two of the three sources served, so a two-thirds train share is two
        # of them and not two of the three the store was built from
        config = make_config(
            dataset_store_path,
            splits=TRAIN_AND_TEST,
            exclude={"sources": ["cont_only"]},
        )
        train = S2ZDataset(config, split="train")
        assert train.split_plan.counts == {"train": 1, "val": 0, "test": 1}

    def test_roles_occupy_contiguous_runs_of_buckets(self, dataset_store_path):
        dataset = S2ZDataset(
            make_config(dataset_store_path, splits=TRAIN_AND_TEST), split="train"
        )
        plan = dataset.split_plan
        walked = [plan.roles[bucket] for bucket in plan.order]
        # A name that starts again after another has begun would make every
        # bucket boundary a possible split boundary
        assert walked == sorted(walked, key=["train", "val", "test"].index)

    def test_the_same_seed_gives_the_same_split(self, dataset_store_path):
        config = make_config(dataset_store_path, splits=TRAIN_AND_TEST)
        first = S2ZDataset(config, split="train")
        second = S2ZDataset(config, split="train")
        assert list(first.subset_indices) == list(second.subset_indices)

    def test_the_seed_moves_the_cut_points(self, dataset_store_path):
        # Rotating the bucket sequence is what lets one store be split more
        # than one way without anything moving on disk
        plans = {
            seed: tuple(
                S2ZDataset(
                    make_config(
                        dataset_store_path,
                        splits=split_config(train=2 / 3, test=1 / 3, seed=seed),
                    ),
                    split="train",
                ).split_plan.order[0]
                for _ in (0,)
            )
            for seed in (0, 1, 2)
        }
        assert len(set(plans.values())) == 3

    def test_a_missing_splits_section_is_refused(self, dataset_store_path):
        config = make_config(dataset_store_path)
        del config["splits"]
        with pytest.raises(ValueError, match="needs a splits section"):
            S2ZDataset(config, split="train")

    def test_fractions_must_sum_to_one(self, dataset_store_path):
        config = make_config(dataset_store_path, splits=split_config(train=0.5))
        with pytest.raises(ValueError, match="must sum to 1"):
            S2ZDataset(config, split="train")

    def test_the_seed_is_required(self, dataset_store_path):
        config = make_config(dataset_store_path, splits={"train": 1.0})
        with pytest.raises(ValueError, match="needs a seed"):
            S2ZDataset(config, split="train")

    def test_an_unknown_split_name_is_refused(self, dataset_store_path):
        config = make_config(
            dataset_store_path, splits={"train": 1.0, "holdout": 0.0, "seed": 0}
        )
        with pytest.raises(ValueError, match="Unknown key"):
            S2ZDataset(config, split="train")

    def test_a_zero_share_is_allowed(self, dataset_store_path):
        # A train and test pair with no validation set stays expressible
        dataset = S2ZDataset(
            make_config(dataset_store_path, splits=TRAIN_AND_TEST), split="train"
        )
        assert dataset.split_plan.counts["val"] == 0


class TestStatisticsCoverage:
    """Measuring a scale for a survey the measured split never sees."""

    def test_a_survey_absent_from_the_split_still_gets_statistics(
        self, dataset_store_path, caplog
    ):
        # 'cont_only' is the sole train source once the other is excluded, and
        # the spectral survey does not cover it
        config = make_config(dataset_store_path)
        config["postprocessing"] = {"normalise": {}}
        config["exclude"] = {"sources": ["both_0"]}
        with caplog.at_level("WARNING"):
            dataset = S2ZDataset(config, split="train")
        statistics = dataset.postprocess.statistics
        assert set(statistics) == set(dataset.surveys)
        assert any("statistics cannot be measured" in r.message for r in caplog.records)

    def test_an_unseen_survey_is_left_unscaled(self, dataset_store_path):
        config = make_config(dataset_store_path)
        config["postprocessing"] = {"normalise": {}}
        config["exclude"] = {"sources": ["both_0"]}
        dataset = S2ZDataset(config, split="train")
        unseen = [
            s
            for s, (mean, deviation) in dataset.postprocess.statistics.items()
            if (mean, deviation) == (0.0, 1.0)
        ]
        # Subtracting 0 and dividing by 1 leaves those views exactly as they were
        assert unseen


class TestStatisticsAccumulation:
    def test_sums_match_a_direct_computation(self):
        rng = np.random.default_rng(0)
        blocks = [
            torch.tensor(rng.normal(3.0, 2.0, (1, 8, 8)), dtype=torch.float64)
            for _ in range(5)
        ]
        totals = {}
        for block in blocks:
            S2ZDataset._accumulate_statistics(totals, "A", block)
        mean, deviation = S2ZDataset._finish_statistics(totals)["A"]

        everything = torch.cat([block.reshape(-1) for block in blocks])
        assert mean == pytest.approx(float(everything.mean()), rel=1e-9)
        assert deviation == pytest.approx(
            float(everything.std(unbiased=False)), rel=1e-6
        )

    def test_non_finite_pixels_are_skipped(self):
        block = torch.ones(1, 4, 4)
        block[0, 0, 0] = float("nan")
        block[0, 1, 1] = float("inf")
        totals = {}
        S2ZDataset._accumulate_statistics(totals, "A", block)
        assert totals["A"][2] == 14  # 16 pixels, two dropped.
        assert S2ZDataset._finish_statistics(totals)["A"][0] == pytest.approx(1.0)

    def test_a_survey_with_nothing_finite_is_left_neutral(self):
        totals = {}
        S2ZDataset._accumulate_statistics(
            totals, "A", torch.full((1, 2, 2), float("nan"))
        )
        assert S2ZDataset._finish_statistics(totals) == {}

    def test_a_constant_survey_gets_a_usable_deviation(self):
        totals = {}
        S2ZDataset._accumulate_statistics(totals, "A", torch.ones(1, 4, 4))
        mean, deviation = S2ZDataset._finish_statistics(totals)["A"]
        assert mean == pytest.approx(1.0)
        assert deviation > 0


class TestStatisticsReads:
    def test_sources_are_batched_by_aligned_shard_group(
        self, dataset_store_path, monkeypatch
    ):
        dataset = S2ZDataset(
            make_config(dataset_store_path), split="train", in_memory=False
        )
        # Exercise more than one group with this small fixture. Real stores take
        # this value from the minimum selected survey shard.
        dataset.shard_size = 2
        indices = dataset._indices_for("train")
        source_groups = dataset._shard_groups(indices)
        calls = {survey: [] for survey in dataset.surveys}

        for survey, substore in dataset.substores.items():
            original = substore.get_cutouts

            def record(
                local_indices, *args, _survey=survey, _original=original, **kwargs
            ):
                calls[_survey].append(list(local_indices))
                return _original(local_indices, *args, **kwargs)

            def forbid_scalar_read(*args, _survey=survey, **kwargs):
                pytest.fail(f"statistics used a scalar {_survey} cutout read")

            monkeypatch.setattr(substore, "get_cutouts", record)
            monkeypatch.setattr(substore, "get_cutout", forbid_scalar_read)

        statistics = dataset.measure_statistics("train")

        assert set(statistics) == set(dataset.surveys)
        for survey, substore in dataset.substores.items():
            expected = [
                [
                    substore.index_map[source_index]
                    for source_index in source_group
                    if source_index in substore.index_map
                ]
                for source_group in source_groups
            ]
            assert calls[survey] == [group for group in expected if group]
            assert all(
                len({index // dataset.shard_size for index in group}) == 1
                for group in calls[survey]
            )


class TestDistributedStatistics:
    """Explicit rank partitioning and reduction of normalisation totals."""

    def test_constructor_forwards_explicit_opt_in(
        self, dataset_store_path, monkeypatch
    ):
        calls = []

        def measure(dataset, split, *, distributed=False):
            calls.append((split, distributed))
            return {survey: (0.0, 1.0) for survey in dataset.surveys}

        monkeypatch.setattr(S2ZDataset, "measure_statistics", measure)
        config = make_config(dataset_store_path)
        config["postprocessing"] = {"normalise": {}}

        S2ZDataset(config, split="train", distributed_statistics=True)

        assert calls == [("train", True)]

    def test_precomputed_statistics_never_enter_a_collective(
        self, dataset_store_path, monkeypatch
    ):
        config = make_config(dataset_store_path)
        config["postprocessing"] = {"normalise": {}}

        def forbid_measurement(*args, **kwargs):
            pytest.fail("precomputed statistics triggered measurement")

        monkeypatch.setattr(S2ZDataset, "measure_statistics", forbid_measurement)
        dataset = S2ZDataset(
            config,
            split="train",
            statistics={"cont": (1.0, 2.0), "spec": (3.0, 4.0)},
            distributed_statistics=True,
        )

        assert dataset.postprocess.statistics == {
            "cont": (1.0, 2.0),
            "spec": (3.0, 4.0),
        }

    def test_initialised_group_is_not_used_without_opt_in(
        self, dataset_store_path, monkeypatch
    ):
        dataset = S2ZDataset(
            make_config(dataset_store_path), split="train", in_memory=False
        )
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

        def forbid_collective(*args, **kwargs):
            pytest.fail("non-distributed statistics entered a collective")

        monkeypatch.setattr(torch.distributed, "all_reduce", forbid_collective)

        assert set(dataset.measure_statistics("train")) == set(dataset.surveys)

    def test_requires_an_initialised_process_group(
        self, dataset_store_path, monkeypatch
    ):
        dataset = S2ZDataset(
            make_config(dataset_store_path), split="train", in_memory=False
        )
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

        with pytest.raises(RuntimeError, match="initialised.*process group"):
            dataset.measure_statistics("train", distributed=True)

    def test_whole_shard_groups_are_partitioned_and_totals_reduced(
        self, dataset_store_path, monkeypatch
    ):
        dataset = S2ZDataset(
            make_config(dataset_store_path), split="train", in_memory=False
        )
        dataset.shard_size = 1
        groups = dataset._shard_groups(dataset._indices_for("train"))
        rank = 0
        world_size = len(groups) + 1
        calls = []
        reductions = []
        original_read = dataset._read_cutouts

        def record_read(source_indices):
            calls.append(list(source_indices))
            return original_read(source_indices)

        # Distinct rows ensure the collective keeps stable survey order.
        global_totals = torch.tensor(
            [[6.0, 20.0, 2.0], [8.0, 40.0, 2.0]], dtype=torch.float64
        )

        def all_reduce(values, op):
            reductions.append((values.detach().cpu().clone(), op))
            values.copy_(global_totals.to(values.device))

        monkeypatch.setattr(dataset, "_read_cutouts", record_read)
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
        monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")
        monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

        for current_rank in range(world_size):
            rank = current_rank
            calls.clear()
            statistics = dataset.measure_statistics("train", distributed=True)

            assert calls == groups[current_rank::world_size]
            assert statistics == {
                dataset.surveys[0]: pytest.approx((3.0, 1.0)),
                dataset.surveys[1]: pytest.approx((4.0, 2.0)),
            }

        assert len(reductions) == world_size
        assert all(
            values.shape == (len(dataset.surveys), 3) for values, _ in reductions
        )
        assert all(values.dtype == torch.float64 for values, _ in reductions)
        # The final rank owns no group but still contributes zeros and enters
        # the collective.
        assert torch.equal(reductions[-1][0], torch.zeros_like(global_totals))


class TestExclusionReporting:
    """Verify reports distinguish excluded and unsplit sources."""

    def test_each_reason_is_counted_once(self, dataset_store_path, caplog):
        config = make_config(dataset_store_path)
        config["exclude"] = {"sources": ["both_0"]}
        with caplog.at_level("INFO"):
            S2ZDataset(config, split="train")
        said = [r.message for r in caplog.records if "Excluding" in r.message]
        assert any("not among the identifiers selected" in m for m in said)

    def test_the_view_threshold_is_named_as_the_reason(
        self, dataset_store_path, caplog
    ):
        # 'cont_only' is covered by one survey, so a threshold of two drops it
        config = make_config(dataset_store_path, min_views=2)
        with caplog.at_level("INFO"):
            S2ZDataset(config, split="train")
        assert any("covered by fewer than 2" in r.message for r in caplog.records)


class TestPostprocessingIntegration:
    def test_statistics_come_from_the_named_split(self, dataset_store_path, caplog):
        config = make_config(dataset_store_path)
        config["postprocessing"] = {"normalise": {}}
        with caplog.at_level("INFO"):
            dataset = S2ZDataset(config, split="train")
        assert dataset.postprocess.statistics_split == "train"
        assert set(dataset.postprocess.statistics) == set(dataset.surveys)
        assert "Measuring normalisation statistics" in caplog.text

    def test_normalised_views_are_centred(self, dataset_store_path):
        config = make_config(dataset_store_path)
        config["postprocessing"] = {"normalise": {}}
        dataset = S2ZDataset(config, split="train")
        sample = dataset[0]
        for values in sample["data"].values():
            if values is None:
                continue
            finite = values[torch.isfinite(values)]
            # Not exactly zero for one source, but on the measured scale
            assert abs(float(finite.mean())) < 10.0

    def test_a_split_with_no_served_source_fails_clearly(self, dataset_store_path):
        config = make_config(dataset_store_path)
        config["postprocessing"] = {"normalise": {"split": "nonexistent"}}
        with pytest.raises(ValueError, match="No source this dataset serves"):
            S2ZDataset(config, split="train")

    def test_rgb_refuses_a_spectral_survey(self, dataset_store_path):
        from survey2zarray.stores.dataset_store.dataset_store import DatasetStore

        store = DatasetStore.open(dataset_store_path)
        spectral = [
            s for s in store.surveys if store.survey(s).observation_type == "spectral"
        ]
        if not spectral:
            pytest.skip("fixture has no spectral survey")
        config = make_config(dataset_store_path)
        config["postprocessing"] = {"rgb": {}}
        with pytest.raises(ValueError, match="requires one channel per survey"):
            S2ZDataset(config, split="train")

    def test_rgb_gives_three_channel_views(self, dataset_store_path):
        from survey2zarray.stores.dataset_store.dataset_store import DatasetStore

        store = DatasetStore.open(dataset_store_path)
        continuum = [
            s for s in store.surveys if store.survey(s).observation_type == "continuum"
        ]
        config = make_config(dataset_store_path)
        config["include"] = {"surveys": continuum}
        config["postprocessing"] = {"rgb": {}}
        dataset = S2ZDataset(config, split="train")
        sample = dataset[0]
        for view in sample["data"].values():
            if view is not None:
                assert view.shape[0] == 3

    def test_statistics_can_be_handed_to_another_split(self, dataset_store_path):
        # Every split of one store and config would measure the same numbers,
        # so the later ones are told rather than made to read the train split
        config = make_config(dataset_store_path, splits=TRAIN_AND_TEST)
        config["postprocessing"] = {"normalise": {}}
        train = S2ZDataset(config, split="train")
        served = S2ZDataset(
            config, split="test", statistics=train.postprocess.statistics
        )
        assert served.postprocess.statistics == train.postprocess.statistics

    def test_handing_them_over_skips_the_measurement(self, dataset_store_path, caplog):
        config = make_config(dataset_store_path, splits=TRAIN_AND_TEST)
        config["postprocessing"] = {"normalise": {}}
        train = S2ZDataset(config, split="train")
        caplog.clear()
        with caplog.at_level("INFO"):
            S2ZDataset(config, split="test", statistics=train.postprocess.statistics)
        assert "Measuring normalisation statistics" not in caplog.text


class TestRotatedSurveys:
    """A survey whose grid is rotated on the sky, served beside a north-up one.

    Reprojection has to bring the two onto one grid from their stored CD
    matrices, which are off-diagonal for the rotated survey at every step.
    """

    MARKER_ARCSEC = 3.0

    def survey_with(self, tmp_path, name, rotation):
        from astropy.io import fits as astropy_fits
        from astropy.wcs import WCS

        from tests.conftest import make_header

        fits_dir = tmp_path / f"{name}_fits"
        fits_dir.mkdir()
        header = make_header(shape_yx=(120, 120), rotation=rotation)
        data = np.zeros((120, 120), dtype=np.float32)
        # One bright pixel `MARKER_ARCSEC` north of the source, placed through
        # the tile's own WCS so both surveys mark the same piece of sky
        wcs = WCS(header).celestial
        x, y = wcs.world_to_pixel_values(
            IMAGE_RA, IMAGE_DEC + self.MARKER_ARCSEC / 3600.0
        )
        data[int(round(float(y))), int(round(float(x)))] = 1.0
        astropy_fits.PrimaryHDU(data=data, header=header).writeto(
            fits_dir / f"{name}.fits"
        )
        out = tmp_path / name
        build_survey_store(fits_dir, out, mode="copy")
        return out

    @pytest.fixture
    def mixed_store(self, tmp_path):
        north_up = self.survey_with(tmp_path, "north_up", 0.0)
        rotated = self.survey_with(tmp_path, "rotated", 33.0)
        catalogue = tmp_path / "rotated_catalogue.csv"
        pd.DataFrame([{"ID": "centre", "RA": IMAGE_RA, "DEC": IMAGE_DEC}]).to_csv(
            catalogue, index=False
        )
        out = tmp_path / "mixed_dataset"
        build_test_dataset_store([north_up, rotated], catalogue, CUTOUT_ARCSEC, out)
        return out

    def test_the_views_come_back_registered(self, mixed_store):
        # Without augmentation the canonical state still reprojects, so both
        # views arrive north up and the marker lands in the same place
        config = {
            "dataset_store": str(mixed_store),
            "splits": split_config(train=1.0),
            "preprocessing": {},
            "augmentation": {},
        }
        sample = S2ZDataset(config, split="train")[0]
        found = {}
        for survey, view in sample["data"].items():
            image = view.numpy()[0]
            iy, ix = np.unravel_index(np.nanargmax(image), image.shape)
            found[survey] = (
                float(ix) - (image.shape[1] - 1) / 2,
                float(iy) - (image.shape[0] - 1) / 2,
            )
        assert set(found) == {"north_up", "rotated"}
        for dx, dy in found.values():
            assert dx == pytest.approx(0.0, abs=1.0)
            assert dy == pytest.approx(self.MARKER_ARCSEC, abs=1.0)
