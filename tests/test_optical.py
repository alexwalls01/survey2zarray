import logging

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from click.testing import CliRunner

from survey2zarray.cli.main import cli
from survey2zarray.stores.dataset_store.dataset_store import DatasetStore
from survey2zarray.stores.store import open_store
from survey2zarray.stores.survey_store.build import (
    band_store_path,
    build_survey_store,
    resolve_fits_list,
)
from survey2zarray.stores.survey_store.edit import append_tiles, retry_failed_tiles
from survey2zarray.stores.survey_store.fits import (
    FITSOpenError,
    IncompatibleTilesError,
    check_compatibility,
    extract_metadata,
    normalise_band,
    resolve_band,
    resolve_zeropoint,
)
from survey2zarray.stores.survey_store.references import (
    UnreferenceableFITSError,
    data_offset,
)
from survey2zarray.stores.survey_store.survey_store import SurveyStore
from tests.conftest import (
    IMAGE_DEC,
    IMAGE_RA,
    build_test_dataset_store,
    make_continuum_fits,
    make_optical_fits,
)

CUTOUT_ARCSEC = 16.0
HOLE = (24, 40, 24, 40)


def make_catalogue(tmp_path, rows, name="catalogue.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def optical_dir(
    tmp_path, bands=("g", "r", "z"), per_band=2, name="fits", labelled=True, **kwargs
):
    # Files named like Legacy Surveys bricks, with the band in the header unless
    # labelled is False.
    fits_dir = tmp_path / name
    fits_dir.mkdir()
    for b, band in enumerate(bands):
        for i in range(per_band):
            make_optical_fits(
                fits_dir / f"tile{i}-image-{band}.fits",
                band=band if labelled else "",
                ra=IMAGE_RA + 0.02 * i,
                seed=10 * b + i,
                **kwargs,
            )
    return fits_dir


class TestBandHeaders:
    @pytest.mark.parametrize(
        ("value", "band"),
        [
            ("g", "g"),
            ("HSC-G", "g"),
            ("HSC-R2", "r2"),
            ("g.00000", "g"),
            ("g DECam SDSS c0001 4720.0 1520.0", "g"),
            ("NB0921", "nb0921"),
            ("NIR_Y", "nir_y"),
            ("F606W", "f606w"),
            ("", ""),
        ],
    )
    def test_filter_names_reduce_to_band_labels(self, value, band):
        assert normalise_band(value) == band

    def test_the_image_header_wins(self):
        header = fits.Header({"BAND": "r"})
        assert resolve_band(header, fits.Header({"FILTER": "g"})) == "r"

    def test_the_primary_header_is_searched_next(self):
        assert resolve_band(fits.Header(), fits.Header({"FILTER": "HSC-I"})) == "i"

    def test_an_empty_filter_position_is_skipped(self):
        header = fits.Header({"FILTER1": "CLEAR1L", "FILTER2": "F814W"})
        assert resolve_band(header) == "f814w"

    def test_a_hierarch_filter_is_read(self):
        header = fits.Header()
        header["HIERARCH FPA.FILTER"] = "y.00000"
        assert resolve_band(header) == "y"

    def test_no_filter_gives_no_band(self):
        assert resolve_band(fits.Header({"TELESCOP": "VLA"})) == ""

    def test_zeropoint_keywords(self):
        assert resolve_zeropoint(fits.Header({"MAGZERO": 22.5})) == 22.5
        assert resolve_zeropoint(fits.Header({"PHOTZP": 30.0})) == 30.0

    def test_hsc_zeropoint_comes_from_fluxmag0(self):
        header = fits.Header({"FLUXMAG0": 10**11.4})
        assert resolve_zeropoint(fits.Header(), header) == pytest.approx(28.5)

    def test_no_zeropoint_is_nan(self):
        assert np.isnan(resolve_zeropoint(fits.Header()))


class TestOpticalMetadata:
    def test_an_optical_image(self, tmp_path):
        path, _ = make_optical_fits(tmp_path / "a.fits", band="r", zeropoint=22.5)
        metadata = extract_metadata(str(path))
        assert metadata["band"] == "r"
        assert metadata["zeropoint"] == 22.5
        assert metadata["observation_type"] == "continuum"
        assert np.isnan(metadata["resolution_major"])
        assert metadata["stokes_assumed"] is False
        assert metadata["tile_compressed"] is False

    def test_a_band_in_the_primary_header(self, tmp_path):
        path, _ = make_optical_fits(
            tmp_path / "a.fits",
            band="HSC-G",
            band_keyword="FILTER",
            band_in_primary=True,
        )
        assert extract_metadata(str(path))["band"] == "g"

    def test_a_tile_compressed_image_has_no_byte_offset(self, tmp_path):
        path, _ = make_optical_fits(tmp_path / "a.fits.fz", compressed=True)
        metadata = extract_metadata(str(path))
        assert metadata["tile_compressed"] is True
        assert metadata["data_offset"] is None

    def test_data_offset_refuses_a_compressed_hdu(self, tmp_path):
        path, _ = make_optical_fits(tmp_path / "a.fits.fz", compressed=True)
        with fits.open(path) as hdul:
            with pytest.raises(UnreferenceableFITSError, match="tile compressed"):
                data_offset(hdul[1])


class TestBandCompatibility:
    def _metadata(self, path, band):
        return {
            "image_location": path,
            "observation_type": "continuum",
            "n_chan": None,
            "band": band,
        }

    def test_one_band_is_returned(self):
        catalogue = [self._metadata("a", "g"), self._metadata("b", "g")]
        assert check_compatibility(catalogue) == ("continuum", None, "g")

    def test_several_bands_are_refused(self):
        catalogue = [self._metadata("a", "g"), self._metadata("b", "r")]
        with pytest.raises(IncompatibleTilesError, match="share one band"):
            check_compatibility(catalogue)

    def test_some_images_without_a_band_are_refused(self):
        catalogue = [self._metadata("a", "g"), self._metadata("b", "")]
        with pytest.raises(IncompatibleTilesError, match="band_pattern"):
            check_compatibility(catalogue)


class TestDiscovery:
    def test_fz_files_are_found_in_a_directory(self, tmp_path):
        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        make_optical_fits(fits_dir / "a.fits.fz", compressed=True)
        assert resolve_fits_list(fits_dir) == [str(fits_dir / "a.fits.fz")]

    def test_a_single_fz_file_is_accepted(self, tmp_path):
        path, _ = make_optical_fits(tmp_path / "a.fits.fz", compressed=True)
        assert resolve_fits_list(path) == [str(path)]


class TestBandSplitting:
    def test_each_band_gets_its_own_store(self, tmp_path):
        out = tmp_path / "legacy"
        stores = build_survey_store(optical_dir(tmp_path), out)
        assert sorted(stores) == ["legacy_g", "legacy_r", "legacy_z"]
        assert not out.exists()
        for band in "grz":
            store = SurveyStore.open(tmp_path / f"legacy_{band}")
            assert store.n_images == 2
            assert store.root.attrs["band"] == band
            assert list(store.band) == [band, band]
            record = pd.read_csv(tmp_path / f"legacy_{band}" / "image_valid.csv")
            assert sorted(record["array_index"]) == [0, 1]
            assert set(record["band"]) == {band}

    def test_a_zarr_suffix_stays_last(self, tmp_path):
        assert band_store_path(tmp_path / "ls.zarr", "g") == tmp_path / "ls_g.zarr"
        assert band_store_path(tmp_path / "ls", "g") == tmp_path / "ls_g"

    def test_one_band_uses_the_given_path(self, tmp_path):
        out = tmp_path / "legacy"
        stores = build_survey_store(optical_dir(tmp_path, bands=("g",)), out)
        assert list(stores) == ["legacy"]
        assert open_store(out).attrs["band"] == "g"

    def test_radio_images_have_no_band(self, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        stores = build_survey_store(fits_dir, tmp_path / "radio")
        assert list(stores) == ["radio"]
        assert open_store(tmp_path / "radio").attrs["band"] == ""

    def test_images_without_a_band_beside_banded_ones_are_refused(self, tmp_path):
        fits_dir = optical_dir(tmp_path, bands=("g",))
        make_optical_fits(fits_dir / "unlabelled.fits", band="")
        with pytest.raises(IncompatibleTilesError, match="band_pattern"):
            build_survey_store(fits_dir, tmp_path / "legacy")
        assert not (tmp_path / "legacy_g").exists()

    def test_a_band_pattern_reads_bands_from_file_names(self, tmp_path):
        fits_dir = optical_dir(tmp_path, bands=("g", "r"), labelled=False)
        stores = build_survey_store(
            fits_dir, tmp_path / "legacy", band_pattern=r"-image-(?P<band>[a-z])\."
        )
        assert sorted(stores) == ["legacy_g", "legacy_r"]
        assert open_store(tmp_path / "legacy_r").attrs["band_pattern"]

    def test_a_band_pattern_needs_a_band_group(self, tmp_path):
        with pytest.raises(ValueError, match="group named 'band'"):
            build_survey_store(optical_dir(tmp_path), tmp_path / "x", band_pattern="-")

    def test_a_destination_that_is_not_a_store_is_kept(self, tmp_path):
        (tmp_path / "legacy_r").mkdir()
        (tmp_path / "legacy_r" / "notes.txt").write_text("keep me")
        with pytest.raises(FileExistsError, match="not a store"):
            build_survey_store(optical_dir(tmp_path), tmp_path / "legacy")
        assert (tmp_path / "legacy_r" / "notes.txt").exists()
        assert not (tmp_path / "legacy_g").exists()

    def test_optical_images_do_not_report_a_missing_stokes_axis(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            build_survey_store(optical_dir(tmp_path, bands=("g",)), tmp_path / "x")
        assert "STOKES" not in caplog.text


class TestTileCompression:
    def test_a_copy_store_holds_the_decompressed_pixels(self, tmp_path, caplog):
        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        _, data = make_optical_fits(fits_dir / "a.fits.fz", compressed=True)
        with caplog.at_level(logging.WARNING):
            build_survey_store(fits_dir, tmp_path / "store")
        assert "tile compressed" in caplog.text
        store = SurveyStore.open(tmp_path / "store")
        np.testing.assert_array_equal(store.get_cutout_region(0, 0, 64, 0, 64), data)

    def test_a_reference_store_refuses_compressed_pixels(self, tmp_path):
        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        make_optical_fits(fits_dir / "a.fits.fz", compressed=True)
        with pytest.raises(ValueError, match="Tile compression"):
            build_survey_store(fits_dir, tmp_path / "store", mode="reference")
        assert not (tmp_path / "store").exists()

    def test_a_refused_multi_band_reference_build_writes_no_store(self, tmp_path):
        fits_dir = optical_dir(tmp_path, compressed=True)
        with pytest.raises(ValueError, match="Tile compression"):
            build_survey_store(fits_dir, tmp_path / "legacy", mode="reference")
        assert not any(tmp_path.glob("legacy_*"))


class TestBlankValue:
    @pytest.mark.parametrize("mode", ["copy", "reference"])
    def test_blank_pixels_read_as_nan(self, tmp_path, mode):
        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        _, data = make_optical_fits(fits_dir / "a.fits", blank_box=HOLE)
        build_survey_store(fits_dir, tmp_path / "store", mode=mode, blank_value=0.0)
        region = SurveyStore.open(tmp_path / "store").get_cutout_region(0, 0, 64, 0, 64)
        y0, y1, x0, x1 = HOLE
        assert np.isnan(region[y0:y1, x0:x1]).all()
        outside = np.ones(data.shape, dtype=bool)
        outside[y0:y1, x0:x1] = False
        np.testing.assert_array_equal(region[outside], data[outside])

    def test_without_a_blank_value_zeros_are_data(self, tmp_path):
        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        make_optical_fits(fits_dir / "a.fits", blank_box=HOLE)
        build_survey_store(fits_dir, tmp_path / "store")
        region = SurveyStore.open(tmp_path / "store").get_cutout_region(0, 0, 64, 0, 64)
        assert np.isfinite(region).all()

    def test_a_blank_value_must_be_finite(self, tmp_path):
        with pytest.raises(ValueError, match="blank_value must be finite"):
            build_survey_store(tmp_path, tmp_path / "x", blank_value=float("nan"))


class TestResolution:
    def test_images_without_a_beam_are_given_one(self, tmp_path):
        out = tmp_path / "store"
        build_survey_store(optical_dir(tmp_path, bands=("g",)), out, resolution=1.2)
        store = SurveyStore.open(out)
        np.testing.assert_allclose(store.resolution_major, 1.2 / 3600.0)
        np.testing.assert_allclose(store.resolution_minor, 1.2 / 3600.0)
        assert store.root.attrs["resolution"] == 1.2

    def test_a_header_beam_is_kept(self, continuum_fits_dir, tmp_path, caplog):
        fits_dir, _ = continuum_fits_dir
        with caplog.at_level(logging.WARNING):
            build_survey_store(fits_dir, tmp_path / "store", resolution=1.2)
        store = SurveyStore.open(tmp_path / "store")
        np.testing.assert_allclose(store.resolution_major, 6.0 / 3600.0)
        assert "was not used" in caplog.text

    def test_a_resolution_must_be_positive(self, tmp_path):
        with pytest.raises(ValueError, match="resolution must be positive"):
            build_survey_store(
                optical_dir(tmp_path, bands=("g",)), tmp_path / "x", resolution=-1.0
            )


class TestAppend:
    @pytest.fixture
    def g_store(self, tmp_path):
        out = tmp_path / "legacy"
        build_survey_store(optical_dir(tmp_path), out, resolution=1.5)
        return tmp_path / "legacy_g"

    def test_another_band_is_refused_naming_its_store(self, g_store, tmp_path):
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        make_optical_fits(new_dir / "x.fits", band="r", ra=IMAGE_RA - 0.03)
        with pytest.raises(IncompatibleTilesError, match="legacy_r"):
            append_tiles(g_store, new_dir)
        assert SurveyStore.open(g_store).n_images == 2

    def test_the_same_band_is_appended_with_the_store_resolution(
        self, g_store, tmp_path
    ):
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        make_optical_fits(new_dir / "x.fits", band="g", ra=IMAGE_RA - 0.03)
        append_tiles(g_store, new_dir)
        store = SurveyStore.open(g_store)
        assert store.n_images == 3
        assert store.band[2] == "g"
        assert store.resolution_major[2] == pytest.approx(1.5 / 3600.0)

    def test_the_store_band_pattern_applies_to_appends(self, tmp_path):
        out = tmp_path / "legacy"
        build_survey_store(
            optical_dir(tmp_path, labelled=False),
            out,
            band_pattern=r"-image-(?P<band>[a-z])\.",
        )
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        make_optical_fits(new_dir / "t9-image-g.fits", band="", ra=IMAGE_RA - 0.03)
        append_tiles(tmp_path / "legacy_g", new_dir)
        assert SurveyStore.open(tmp_path / "legacy_g").n_images == 3

    def test_a_retry_records_a_file_of_another_band(self, tmp_path, monkeypatch):
        fits_dir = optical_dir(tmp_path, bands=("g",))
        late = fits_dir / "late.fits"
        make_optical_fits(late, band="g", ra=IMAGE_RA - 0.03)
        from survey2zarray.stores.survey_store.fits import extract_metadata as real

        def read(path, require_referenceable=False, storage_options=None):
            if path == str(late):
                raise FITSOpenError(f"Failed to open {path}")
            return real(path, require_referenceable=require_referenceable)

        monkeypatch.setattr(
            "survey2zarray.stores.survey_store.build.extract_metadata", read
        )
        out = tmp_path / "store"
        build_survey_store(fits_dir, out)
        monkeypatch.undo()
        make_optical_fits(late, band="r", ra=IMAGE_RA - 0.03)

        retry_failed_tiles(out)
        assert SurveyStore.open(out).n_images == 2
        record = pd.read_csv(out / "image_valid.csv")
        rows = record[record["image_location"] == str(late)]
        assert len(rows) == 1
        assert not rows["converted"].iloc[0]
        assert "Band r" in rows["error"].iloc[0]


class TestDatasetStores:
    def test_band_and_zeropoint_are_carried(self, tmp_path):
        out = tmp_path / "legacy"
        build_survey_store(optical_dir(tmp_path, bands=("g",), zeropoint=22.5), out)
        catalogue = make_catalogue(
            tmp_path, [{"ID": "a", "RA": IMAGE_RA, "DEC": IMAGE_DEC}]
        )
        dataset = tmp_path / "dataset"
        build_test_dataset_store([out], catalogue, CUTOUT_ARCSEC, dataset)
        sub = DatasetStore.open(dataset).survey("legacy")
        assert sub.survey_band == "g"
        assert sub.band[0] == "g"
        assert sub.zeropoint[0] == 22.5

    @pytest.mark.parametrize(("blank_value", "covered"), [(0.0, False), (None, True)])
    def test_a_disc_over_a_blank_hole_is_not_served(
        self, tmp_path, blank_value, covered
    ):
        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        make_optical_fits(fits_dir / "a.fits", blank_box=HOLE)
        out = tmp_path / "store"
        build_survey_store(fits_dir, out, blank_value=blank_value)
        catalogue = make_catalogue(
            tmp_path, [{"ID": "a", "RA": IMAGE_RA, "DEC": IMAGE_DEC}]
        )
        dataset = tmp_path / "dataset"
        build_test_dataset_store([out], catalogue, CUTOUT_ARCSEC, dataset)
        assert (0 in DatasetStore.open(dataset).survey("store").index_map) is covered

    @pytest.mark.parametrize(("resolution", "n_sources"), [(3.0, 1), (None, 2)])
    def test_duplicates_use_the_given_resolution(self, tmp_path, resolution, n_sources):
        out = tmp_path / "store"
        build_survey_store(
            optical_dir(tmp_path, bands=("g",)), out, resolution=resolution
        )
        catalogue = make_catalogue(
            tmp_path,
            [
                {"ID": "a", "RA": IMAGE_RA, "DEC": IMAGE_DEC},
                {"ID": "b", "RA": IMAGE_RA, "DEC": IMAGE_DEC + 1.0 / 3600.0},
            ],
        )
        dataset = tmp_path / "dataset"
        build_test_dataset_store([out], catalogue, CUTOUT_ARCSEC, dataset)
        assert DatasetStore.open(dataset).n_sources == n_sources


class TestCommandLine:
    def test_build_splits_bands_with_options(self, tmp_path):
        fits_dir = optical_dir(tmp_path, compressed=True, blank_box=HOLE)
        out = tmp_path / "legacy"
        result = CliRunner().invoke(
            cli,
            [
                "survey",
                "build",
                str(fits_dir),
                str(out),
                "--blank-value",
                "0",
                "--resolution",
                "1.2",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "legacy_g, legacy_r, legacy_z" in result.output
        root = open_store(tmp_path / "legacy_z")
        assert root.attrs["blank_value"] == 0.0
        assert root.attrs["resolution"] == 1.2


def test_radio_and_optical_mix_in_one_dataset(continuum_fits_dir, tmp_path):
    radio_dir, _ = continuum_fits_dir
    build_survey_store(radio_dir, tmp_path / "radio")
    build_survey_store(
        optical_dir(tmp_path, bands=("g", "r"), name="optical_fits"),
        tmp_path / "optical",
    )
    catalogue = make_catalogue(
        tmp_path, [{"ID": "a", "RA": IMAGE_RA, "DEC": IMAGE_DEC}]
    )
    dataset = tmp_path / "dataset"
    surveys = [tmp_path / "radio", tmp_path / "optical_g", tmp_path / "optical_r"]
    build_test_dataset_store(surveys, catalogue, CUTOUT_ARCSEC, dataset)
    store = DatasetStore.open(dataset)
    assert store.surveys == ["radio", "optical_g", "optical_r"]
    assert [store.survey(s).survey_band for s in store.surveys] == ["", "g", "r"]
    assert all(0 in store.survey(s).index_map for s in store.surveys)


def test_make_continuum_fits_still_reads_as_radio(tmp_path):
    path, _ = make_continuum_fits(tmp_path / "a.fits")
    metadata = extract_metadata(str(path))
    assert metadata["band"] == ""
    assert metadata["stokes_assumed"] is True
