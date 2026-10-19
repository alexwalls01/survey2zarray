import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord

from survey2zarray.stores.store import open_store
from survey2zarray.stores.survey_store.build import build_survey_store
from survey2zarray.stores.survey_store.survey_store import SurveyStore
from survey2zarray.stores.wcs import reconstruct_wcs
from tests.conftest import IMAGE_DEC, IMAGE_RA


@pytest.fixture
def local_store(continuum_fits_dir, tmp_path):
    fits_dir, images = continuum_fits_dir
    out = tmp_path / "local_store"
    build_survey_store(fits_dir, out, mode="copy")
    return SurveyStore.open(out), images


@pytest.fixture
def remote_store(continuum_fits_dir, tmp_path):
    fits_dir, images = continuum_fits_dir
    urls = sorted(str(p) for p in fits_dir.glob("*.fits"))
    out = tmp_path / "remote_store"
    build_survey_store(urls, out, mode="reference")
    return SurveyStore.open(out), images


class TestOpen:
    def test_attributes(self, local_store):
        store, _ = local_store
        assert store.observation_type == "continuum"
        assert store.access_mode == "copy"
        assert store.n_images == 3
        assert store.pixel_shape.shape == (3, 2)

    def test_incomplete_store_rejected(self, continuum_fits_dir, tmp_path):

        fits_dir, _ = continuum_fits_dir
        out = tmp_path / "store"
        build_survey_store(fits_dir, out, mode="copy")
        root = open_store(out, mode="r+")
        root.attrs["complete"] = False
        with pytest.raises(RuntimeError, match="incomplete"):
            SurveyStore.open(out)


class TestWCSReconstruction:
    def test_reconstruction_matches_source_wcs(self, local_store):
        # Reconstructed from stored numeric params, never the header string;
        # world_to_pixel must agree with the original file's WCS
        from astropy.io import fits as astropy_fits
        from astropy.wcs import WCS

        store, _ = local_store
        wcs = store.get_wcs(0)
        with astropy_fits.open(store.image_location[0]) as hdul:
            original = WCS(hdul[0].header).celestial
        target = SkyCoord(ra=IMAGE_RA * u.deg, dec=IMAGE_DEC * u.deg, frame=ICRS)
        x_r, y_r = wcs.world_to_pixel(target)
        x_o, y_o = original.world_to_pixel(target)
        assert float(x_r) == pytest.approx(float(x_o), abs=1e-6)
        assert float(y_r) == pytest.approx(float(y_o), abs=1e-6)

    def test_each_call_returns_an_independent_wcs(self, local_store):
        # Rebuilt rather than cached, so one reader cannot mutate another's
        store, _ = local_store
        first, second = store.get_wcs(0), store.get_wcs(0)
        assert first is not second
        assert np.allclose(first.wcs.crval, second.wcs.crval)
        assert np.allclose(first.wcs.crpix, second.wcs.crpix)

    def test_axis_order_non_square_shape(self, tmp_path):
        # pixel_shape is (H, W) = (row, col); world_to_pixel returns (x, y) =
        # (col, row). A non-square image catches swaps that square shapes hide.
        from tests.conftest import make_continuum_fits

        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        make_continuum_fits(fits_dir / "image.fits", shape_yx=(32, 64))
        out = tmp_path / "store"
        build_survey_store(fits_dir, out, mode="copy")
        store = SurveyStore.open(out)
        assert tuple(store.pixel_shape[0]) == (32, 64)
        wcs = store.get_wcs(0)
        centre = SkyCoord(
            ra=store.pixel_centre[0][0] * u.deg,
            dec=store.pixel_centre[0][1] * u.deg,
            frame=ICRS,
        )
        x, y = wcs.world_to_pixel(centre)
        assert float(x) == pytest.approx((64 - 1) / 2, abs=1e-6)
        assert float(y) == pytest.approx((32 - 1) / 2, abs=1e-6)


class TestGetCutoutRegion:
    def test_local_matches_tile_slice(self, local_store):
        store, images = local_store
        region = store.get_cutout_region(0, 10, 26, 12, 28)
        np.testing.assert_array_equal(region, images["image_a"][12:28, 10:26])

    def test_remote_matches_tile_slice(self, remote_store):
        store, images = remote_store
        region = store.get_cutout_region(0, 10, 26, 12, 28)
        np.testing.assert_array_equal(region, images["image_a"][12:28, 10:26])

    def test_remote_4d_stokes_slice(self, remote_store):
        # image_c is (stokes, freq, y, x) on disk; the section path must apply
        # the same Stokes-I slice as the writer
        store, images = remote_store
        region = store.get_cutout_region(2, 4, 20, 6, 22)
        np.testing.assert_array_equal(region, images["image_c"][0, 0, 6:22, 4:20])

    def test_local_and_remote_agree(self, local_store, remote_store):
        local, _ = local_store
        remote, _ = remote_store
        for index in range(3):
            np.testing.assert_array_equal(
                local.get_cutout_region(index, 5, 21, 7, 23),
                remote.get_cutout_region(index, 5, 21, 7, 23),
            )

    def test_spectral_cutout_keeps_channels(self, spectral_fits_dir, tmp_path):
        fits_dir, cubes = spectral_fits_dir
        out = tmp_path / "store"
        build_survey_store(fits_dir, out, mode="copy")
        store = SurveyStore.open(out)
        region = store.get_cutout_region(0, 8, 24, 10, 26)
        assert region.shape == (3, 16, 16)
        np.testing.assert_array_equal(region, cubes["cube_a"][0, :, 10:26, 8:24])


def test_reconstruct_wcs_spectral_axes():
    # 3-axis WCS (celestial + frequency) reconstructs and yields a celestial sub-WCS
    crval = (150.0, 2.0, 1.4e9)
    crpix = (10.0, 12.0, 1.0)
    cd = np.diag([-2.78e-4, 2.78e-4, 1e6])
    ctype = "RA---SIN|DEC--SIN|FREQ"
    wcs = reconstruct_wcs(crval, crpix, cd, ctype)
    assert wcs.naxis == 3
    celestial = wcs.celestial
    assert celestial.naxis == 2
    assert list(celestial.wcs.ctype) == ["RA---SIN", "DEC--SIN"]


def test_reconstruct_wcs_preserves_reference_frame_metadata():
    wcs = reconstruct_wcs(
        (150.0, 2.0),
        (10.0, 12.0),
        np.diag([-2.78e-4, 2.78e-4]),
        "RA---TAN|DEC--TAN",
        radesys="FK5",
        equinox=1975.0,
    )
    assert wcs.wcs.radesys == "FK5"
    assert wcs.wcs.equinox == pytest.approx(1975.0)


class TestFailedRanges:
    def test_a_failed_range_names_the_archive_and_the_cause(
        self, tmp_path, continuum_fits_dir, monkeypatch
    ):
        # cat_ranges reports failure by returning the exception as the data
        from survey2zarray.stores.survey_store.build import build_survey_store

        fits_dir, _ = continuum_fits_dir
        out = tmp_path / "referenced"
        build_survey_store(fits_dir, out, mode="reference")
        store = SurveyStore.open(out)

        class Broken:
            def cat_ranges(self, paths, starts, ends, **kwargs):
                return [OSError("connection reset") for _ in starts]

        monkeypatch.setattr(
            store, "filesystem", lambda location: Broken(), raising=False
        )
        with pytest.raises(RuntimeError, match="Could not read") as raised:
            store.read_referenced_region(0, 0, 8, 0, 8)
        assert "connection reset" in str(raised.value)
        assert isinstance(raised.value.__cause__, OSError)

    def test_a_silent_exception_still_reports_its_type(
        self, tmp_path, continuum_fits_dir, monkeypatch
    ):
        from survey2zarray.stores.survey_store.build import build_survey_store

        fits_dir, _ = continuum_fits_dir
        out = tmp_path / "referenced"
        build_survey_store(fits_dir, out, mode="reference")
        store = SurveyStore.open(out)

        class Quiet(Exception):
            pass

        class Broken:
            def cat_ranges(self, paths, starts, ends, **kwargs):
                return [Quiet() for _ in starts]

        monkeypatch.setattr(
            store, "filesystem", lambda location: Broken(), raising=False
        )
        with pytest.raises(RuntimeError, match="Quiet"):
            store.read_referenced_region(0, 0, 8, 0, 8)
