import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord

from survey2zarray.stores.dataset_store.build import get_cutout
from survey2zarray.stores.dataset_store.coverage import cutouts_inside
from survey2zarray.stores.dataset_store.dataset_store import PADDING, DatasetStore
from survey2zarray.stores.dataset_store.duplicates import (
    find_catalogue_duplicates,
    find_stored_duplicates,
)
from survey2zarray.stores.store import open_store
from survey2zarray.stores.survey_store.build import build_survey_store
from survey2zarray.stores.survey_store.survey_store import SurveyStore
from tests.conftest import IMAGE_DEC, IMAGE_RA, build_test_dataset_store

# Image A is 64x64 at 1"/px centred on (IMAGE_RA, IMAGE_DEC): half-width 32".
# With a 16" cutout (half 8") and 2 px margin, sources within ~22" of the
# centre fit; sources near the image edge are absent under Rule A.

CUTOUT_ARCSEC = 16.0


def make_catalogue(tmp_path, rows, name="catalogue.csv"):
    df = pd.DataFrame(rows)
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def test_cutout_boundary_invariant_is_a_runtime_error(survey_store_a):
    store_path, _ = survey_store_a
    store = SurveyStore.open(store_path)
    # Thirty arcseconds north is still on the 64-pixel image, but a 16-arcsec
    # cutout around it is not. The coverage matcher should prevent this call.
    with pytest.raises(RuntimeError, match="Cutout exceeds image"):
        get_cutout(
            store,
            0,
            (IMAGE_RA, IMAGE_DEC + 30 / 3600),
            CUTOUT_ARCSEC,
        )


@pytest.fixture
def survey_store_a(continuum_fits_dir, tmp_path):
    fits_dir, images = continuum_fits_dir
    out = tmp_path / "survey_a"
    build_survey_store(fits_dir, out, mode="copy")
    return out, images


@pytest.fixture
def catalogue_path(tmp_path):
    # in_a: at image A's centre. near_edge: inside image A but too close to its
    # edge for a full cutout (Rule A). outside: not on any image.
    return make_catalogue(
        tmp_path,
        [
            {"ID": "in_a", "RA": IMAGE_RA, "DEC": IMAGE_DEC},
            {
                "ID": "near_edge",
                "RA": IMAGE_RA,
                "DEC": IMAGE_DEC + 30.5 / 3600.0,
            },
            {
                "ID": "outside",
                "RA": IMAGE_RA + 5.0,
                "DEC": IMAGE_DEC - 5.0,
            },
            {"ID": "in_b", "RA": IMAGE_RA + 0.02, "DEC": IMAGE_DEC},
        ],
    )


class TestDiscPadding:
    """The disc carries padding so the full cutout_size is reachable.

    A reader insets by margin_px of its coarsest survey before interpolating.
    Without padding the square's corners sit exactly on the blanked edge, so
    the size asked for is present but cannot be read.
    """

    def test_padding_defaults_to_the_coarsest_survey_built(
        self, survey_store_a, catalogue_path, tmp_path
    ):
        out = tmp_path / "dataset"
        build_test_dataset_store(
            [store_path_of(survey_store_a)], catalogue_path, CUTOUT_ARCSEC, out
        )
        ds = DatasetStore.open(out)
        assert ds.padded_for_pixel_scale == pytest.approx(1.0)  # fixtures are 1"/px
        assert ds.disc_padding == pytest.approx(2 * 1.0)  # margin_px = 2
        assert ds.disc_diameter == pytest.approx(CUTOUT_ARCSEC * np.sqrt(2) + 4.0)
        # Which is the whole point: the full square is reachable
        assert ds.servable_square(1.0) == pytest.approx(CUTOUT_ARCSEC)

    def test_explicit_scale_pre_pays_for_a_coarser_survey(
        self, survey_store_a, catalogue_path, tmp_path
    ):
        out = tmp_path / "dataset"
        build_test_dataset_store(
            [store_path_of(survey_store_a)],
            catalogue_path,
            CUTOUT_ARCSEC,
            out,
            coarsest_pixel_scale_arcsec=4.0,
        )
        ds = DatasetStore.open(out)
        assert ds.padded_for_pixel_scale == pytest.approx(4.0)
        # A 4"/px survey added later still gets the full square
        assert ds.servable_square(4.0) == pytest.approx(CUTOUT_ARCSEC)
        # And a finer one is capped at it rather than exceeding it
        assert ds.servable_square(1.0) == pytest.approx(CUTOUT_ARCSEC)

    def test_coarser_survey_reaches_less(
        self, survey_store_a, catalogue_path, tmp_path
    ):
        out = tmp_path / "dataset"
        build_test_dataset_store(
            [store_path_of(survey_store_a)], catalogue_path, CUTOUT_ARCSEC, out
        )
        ds = DatasetStore.open(out)
        # 8"/px insets 16" rather than 2", so it reaches well short
        assert ds.servable_square(8.0) < CUTOUT_ARCSEC
        assert ds.servable_square(8.0) == pytest.approx(
            (ds.disc_diameter - 2 * 8.0 * 2) / np.sqrt(2)
        )

    def test_scale_finer_than_the_build_warns(
        self, survey_store_a, catalogue_path, tmp_path, caplog
    ):
        out = tmp_path / "dataset"
        with caplog.at_level("WARNING"):
            build_test_dataset_store(
                [store_path_of(survey_store_a)],
                catalogue_path,
                CUTOUT_ARCSEC,
                out,
                coarsest_pixel_scale_arcsec=0.25,
            )
        assert any(
            "finer than the coarsest survey" in r.message for r in caplog.records
        )


def store_path_of(fixture):
    # survey_store_a yields (path, images)
    return fixture[0]


class TestBlankTileRegions:
    """An image's bounds cannot say whether its data fills them.

    A mosaic edge, a primary-beam cutoff or a blanked corner leaves a cutout of
    NaN that passes every bounds test, and would otherwise be stored as though
    it were an observation.
    """

    @pytest.fixture
    def holed_survey(self, tmp_path):
        from astropy.io import fits

        from tests.conftest import make_header

        folder = tmp_path / "holed"
        folder.mkdir()
        n = 512
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1e-4, (n, n)).astype(np.float32)
        yy, xx = np.mgrid[0:n, 0:n]
        data[np.hypot(xx - n / 2, yy - n / 2) > n / 2 * 0.75] = np.nan
        fits.PrimaryHDU(data=data, header=make_header(shape_yx=(n, n))).writeto(
            folder / "round.fits"
        )
        store = tmp_path / "survey_holed"
        build_survey_store(folder, store, mode="copy")
        return store

    def positions(self, store_path):
        from survey2zarray.stores.survey_store.survey_store import SurveyStore

        store = SurveyStore.open(store_path)
        wcs = store.get_wcs(0).celestial
        return wcs.pixel_to_world(256, 256), wcs.pixel_to_world(70, 70)

    def build(self, store_path, tmp_path, **kwargs):
        middle, blank = self.positions(store_path)
        catalogue = tmp_path / "holed.csv"
        pd.DataFrame(
            [
                {
                    "ID": "middle",
                    "RA": middle.ra.deg,
                    "DEC": middle.dec.deg,
                },
                {
                    "ID": "in_the_blank",
                    "RA": blank.ra.deg,
                    "DEC": blank.dec.deg,
                },
            ]
        ).to_csv(catalogue, index=False)
        out = tmp_path / f"ds_{len(kwargs)}"
        build_test_dataset_store([store_path], catalogue, 40.0, out, **kwargs)
        return DatasetStore.open(out)

    def test_blank_cutout_is_not_stored(self, holed_survey, tmp_path, caplog):
        with caplog.at_level("WARNING"):
            ds = self.build(holed_survey, tmp_path)
        sub = ds.survey(list(ds.surveys)[0])
        held = {str(ds.source_id[i]) for i in sub.index_map}
        # Both sources keep their row in the table; only the real one has a cutout
        assert set(str(v) for v in ds.source_id) == {"middle", "in_the_blank"}
        assert held == {"middle"}
        assert any("discs contain blank pixels" in r.message for r in caplog.records)
        # Which sources were dropped is read from the sub-store, not the log
        assert "in_the_blank" not in held

    def test_a_single_blank_pixel_is_enough_to_drop_a_source(self, tmp_path, caplog):
        # The disc is meant to be the same fully populated patch of sky in every
        # survey, so one hole disqualifies it however small
        from astropy.io import fits

        from survey2zarray.stores.survey_store.survey_store import SurveyStore
        from tests.conftest import make_header

        folder = tmp_path / "one_hole"
        folder.mkdir()
        n = 256
        data = np.random.default_rng(0).normal(0, 1e-4, (n, n)).astype(np.float32)
        data[128, 128] = np.nan  # exactly one blank pixel, at the centre
        fits.PrimaryHDU(data=data, header=make_header(shape_yx=(n, n))).writeto(
            folder / "t.fits"
        )
        store_path = tmp_path / "survey_hole"
        build_survey_store(folder, store_path, mode="copy")

        wcs = SurveyStore.open(store_path).get_wcs(0).celestial
        on_the_hole = wcs.pixel_to_world(128, 128)
        clear = wcs.pixel_to_world(60, 60)
        catalogue = tmp_path / "one_hole.csv"
        pd.DataFrame(
            [
                {
                    "ID": "on_the_hole",
                    "RA": on_the_hole.ra.deg,
                    "DEC": on_the_hole.dec.deg,
                },
                {
                    "ID": "clear",
                    "RA": clear.ra.deg,
                    "DEC": clear.dec.deg,
                },
            ]
        ).to_csv(catalogue, index=False)

        out = tmp_path / "ds_one_hole"
        with caplog.at_level("WARNING"):
            build_test_dataset_store([store_path], catalogue, 40.0, out)
        ds = DatasetStore.open(out)
        held = {str(ds.source_id[i]) for i in ds.survey(list(ds.surveys)[0]).index_map}
        assert held == {"clear"}
        assert any("1 source(s) dropped" in r.message for r in caplog.records)

    def test_full_tile_keeps_everything(self, survey_store_a, catalogue_path, tmp_path):
        # The synthetic images have no holes, so nothing should be dropped
        store_path, _ = survey_store_a
        out = tmp_path / "full"
        build_test_dataset_store([store_path], catalogue_path, CUTOUT_ARCSEC, out)
        ds = DatasetStore.open(out)
        assert ds.survey("survey_a").n_cutouts > 0


class TestMixedTileSizes:
    def test_one_large_tile_does_not_widen_the_search(
        self, continuum_fits_dir, tmp_path
    ):
        # The search radius follows each image's own size, so a single oversized
        # image cannot make every source search as though every image were huge
        from survey2zarray.stores.dataset_store.coverage import match_coords_to_images
        from survey2zarray.stores.survey_store.survey_store import SurveyStore

        fits_dir, _ = continuum_fits_dir
        store_path = tmp_path / "survey_mixed"
        build_survey_store(fits_dir, store_path, mode="copy")
        store = SurveyStore.open(store_path)

        rng = np.random.default_rng(0)
        centre = store.pixel_centre[0]
        catalogue = pd.DataFrame(
            {
                "id": [f"s{i}" for i in range(200)],
                "ra": centre[0] + rng.uniform(-0.004, 0.004, 200),
                "dec": centre[1] + rng.uniform(-0.004, 0.004, 200),
            }
        )
        before = match_coords_to_images(catalogue, store, "survey", 8.0)

        # Pretend one image is far larger; the matches for the others must not change
        store.pixel_shape = store.pixel_shape.copy()
        store.pixel_shape[2] = store.pixel_shape[2] * 8
        after = match_coords_to_images(catalogue, store, "survey", 8.0)
        unchanged = sum(
            1
            for k in before
            if (pd.isna(before[k]) and pd.isna(after[k])) or before[k] == after[k]
        )
        assert unchanged == len(before)


class TestMatchingIsBulk:
    """Every candidate pair is tested at once, grouped by image.

    The rule is unchanged: a source takes the nearest image that can supply its
    whole disc. What changed is that this is computed rather than walked to.
    """

    def test_nearest_tile_that_fits_wins_over_a_nearer_one_that_does_not(
        self, continuum_fits_dir, tmp_path
    ):
        from survey2zarray.stores.dataset_store.coverage import match_coords_to_images
        from survey2zarray.stores.survey_store.survey_store import SurveyStore

        fits_dir, _ = continuum_fits_dir
        store_path = tmp_path / "survey"
        build_survey_store(fits_dir, store_path, mode="copy")
        store = SurveyStore.open(store_path)

        # Right at image_b's edge: its nearest image centre cannot hold the disc,
        # so an overlapping neighbour must be chosen instead
        edge = pd.DataFrame(
            {
                "id": ["edge"],
                "ra": [float(store.pixel_centre[1][0]) + 0.0055],
                "dec": [float(store.pixel_centre[1][1])],
            }
        )
        matched = match_coords_to_images(edge, store, "survey", 8.0)
        chosen = matched["edge"]
        if not pd.isna(chosen):
            # Whichever image was picked really can supply the disc
            wcs = store.get_wcs(int(chosen)).celestial
            x, y = wcs.world_to_pixel(
                SkyCoord(
                    ra=edge["ra"][0] * u.deg, dec=edge["dec"][0] * u.deg, frame=ICRS
                )
            )
            h, w = store.pixel_shape[int(chosen)]
            inside = cutouts_inside(x, y, h, w, store.pixel_scale[int(chosen)], 8.0)
            assert bool(inside)

    def test_scales_without_a_per_source_projection(self, continuum_fits_dir, tmp_path):
        # Many sources over few images: the work is per image, not per source, so
        # this stays fast rather than growing with the catalogue
        from survey2zarray.stores.dataset_store.coverage import match_coords_to_images
        from survey2zarray.stores.survey_store.survey_store import SurveyStore

        fits_dir, _ = continuum_fits_dir
        store_path = tmp_path / "survey_many"
        build_survey_store(fits_dir, store_path, mode="copy")
        store = SurveyStore.open(store_path)

        rng = np.random.default_rng(0)
        n = 4000
        centre = store.pixel_centre[0]
        catalogue = pd.DataFrame(
            {
                "id": [f"s{i}" for i in range(n)],
                "ra": centre[0] + rng.uniform(-0.004, 0.004, n),
                "dec": centre[1] + rng.uniform(-0.004, 0.004, n),
            }
        )
        matched = match_coords_to_images(catalogue, store, "survey", 8.0)
        assert len(matched) == n
        assert sum(1 for v in matched.values() if not pd.isna(v)) > 0


class TestBuildDatasetStore:
    def test_source_table_is_canonical(self, survey_store_a, catalogue_path, tmp_path):
        store_path, _ = survey_store_a
        out = tmp_path / "dataset"
        build_test_dataset_store([store_path], catalogue_path, CUTOUT_ARCSEC, out)
        ds = DatasetStore.open(out)
        # All catalogue sources are present regardless of survey coverage
        assert ds.n_sources == 4
        # Written in bucket order, which this fixture takes from the identifier
        assert list(ds.source_id) == ["in_a", "in_b", "near_edge", "outside"]
        assert ds.ra[0] == pytest.approx(IMAGE_RA)
        assert ds.max_cutout_size == CUTOUT_ARCSEC
        # The disc stored is the square's diagonal across
        assert ds.disc_diameter == pytest.approx(CUTOUT_ARCSEC * np.sqrt(2) + 2 * 2.0)
        # Padded for the coarsest survey built, so it serves the full square
        assert ds.padded_for_pixel_scale == pytest.approx(1.0)
        assert ds.servable_square(1.0) == pytest.approx(CUTOUT_ARCSEC)

    def test_index_map_rule_a_absence(self, survey_store_a, catalogue_path, tmp_path):
        store_path, _ = survey_store_a
        out = tmp_path / "dataset"
        build_test_dataset_store([store_path], catalogue_path, CUTOUT_ARCSEC, out)
        sub = DatasetStore.open(out).survey("survey_a")
        # in_a (0) and in_b (1) present; near_edge (2) absent by Rule A;
        # outside (3) matches no image. Absent sources are simply not in the map.
        assert set(sub.index_map.keys()) == {0, 1}
        assert sub.n_cutouts == 2
        # A row per source whatever the coverage, and each holds its own source
        assert sub.data.shape[0] == 4
        assert sub.index_map == {0: 0, 1: 1}

    def test_cutout_pixels_match_image(self, survey_store_a, catalogue_path, tmp_path):
        from survey2zarray.stores.survey_store.survey_store import SurveyStore

        store_path, images = survey_store_a
        out = tmp_path / "dataset"
        build_test_dataset_store([store_path], catalogue_path, CUTOUT_ARCSEC, out)
        sub = DatasetStore.open(out).survey("survey_a")
        survey = SurveyStore.open(store_path)
        local = sub.index_map[0]  # in_a, at image A's centre
        cutout, _ = sub.get_cutout(local)
        assert cutout.shape == (27, 27)
        # The stored crpix shift IS the cutout origin: crpix_image - crpix_cutout
        image_index = int(sub.image_index[local])
        x0 = int(survey.crpix[image_index][0] - sub.crpix[local][0])
        y0 = int(survey.crpix[image_index][1] - sub.crpix[local][1])
        image = images["image_a"]
        # Centred on the source to within rounding
        assert abs(x0 - 18) <= 1 and abs(y0 - 18) <= 1
        # Only the disc is kept; the blanked corners are not image pixels
        window = image[y0 : y0 + 27, x0 : x0 + 27]
        inside = np.isfinite(cutout)
        np.testing.assert_array_equal(cutout[inside], window[inside])
        # Blanking follows a disc of the stored diameter, centred on the source
        yy, xx = np.mgrid[0 : cutout.shape[-2], 0 : cutout.shape[-1]]
        centre = (cutout.shape[-1] - 1) / 2
        radius = np.hypot(xx - centre, yy - centre)  # 1"/px, so pixels = arcsec
        finite = np.isfinite(cutout if cutout.ndim == 2 else cutout[0])
        assert finite[radius <= CUTOUT_ARCSEC * np.sqrt(2) / 2 + 2 - 1].all()
        assert not finite[radius >= CUTOUT_ARCSEC * np.sqrt(2) / 2 + 2 + 1].any()

    def test_cutout_wcs_crpix_shift(self, survey_store_a, catalogue_path, tmp_path):
        store_path, _ = survey_store_a
        out = tmp_path / "dataset"
        build_test_dataset_store([store_path], catalogue_path, CUTOUT_ARCSEC, out)
        sub = DatasetStore.open(out).survey("survey_a")
        local = sub.index_map[0]
        wcs = sub.get_wcs(local).celestial
        # Same sky mapping as the image, reference shifted into the cutout
        # frame: the source must land near the cutout centre (within the
        # half-pixel rounding of the origin)
        source = SkyCoord(ra=IMAGE_RA * u.deg, dec=IMAGE_DEC * u.deg, frame=ICRS)
        x, y = wcs.world_to_pixel(source)
        assert float(x) == pytest.approx(13.0, abs=1.0)
        assert float(y) == pytest.approx(13.0, abs=1.0)

    def test_per_cutout_metadata_carried(
        self, survey_store_a, catalogue_path, tmp_path
    ):
        store_path, _ = survey_store_a
        out = tmp_path / "dataset"
        build_test_dataset_store([store_path], catalogue_path, CUTOUT_ARCSEC, out)
        sub = DatasetStore.open(out).survey("survey_a")
        assert sub.observation_type == "continuum"
        # Beam stays in degrees (raw FITS), frequency in Hz, pixel scale in arcsec
        assert sub.bmaj[0] == pytest.approx(6.0 / 3600.0)
        assert sub.frequency[0] == pytest.approx(1.4e9)
        assert tuple(sub.pixel_scale[0]) == pytest.approx((1.0, 1.0), rel=1e-6)
        assert tuple(sub.pixel_shape[0]) == (27, 27)

    def test_remote_survey_store_source(
        self, continuum_fits_dir, catalogue_path, tmp_path
    ):
        # build-dataset reads pixels through the backend interface: a remote
        # survey store must produce identical cutouts to a local one
        fits_dir, _ = continuum_fits_dir
        urls = sorted(str(p) for p in fits_dir.glob("*.fits"))
        local_survey = tmp_path / "survey_local"
        remote_survey = tmp_path / "survey_remote"
        build_survey_store(fits_dir, local_survey, mode="copy")
        build_survey_store(urls, remote_survey, mode="reference")
        build_test_dataset_store(
            [local_survey], catalogue_path, CUTOUT_ARCSEC, tmp_path / "ds_local"
        )
        build_test_dataset_store(
            [remote_survey], catalogue_path, CUTOUT_ARCSEC, tmp_path / "ds_remote"
        )
        local_sub = DatasetStore.open(tmp_path / "ds_local").survey("survey_local")
        remote_sub = DatasetStore.open(tmp_path / "ds_remote").survey("survey_remote")
        assert local_sub.index_map == remote_sub.index_map
        for source_index, local in local_sub.index_map.items():
            np.testing.assert_array_equal(
                local_sub.get_cutout(local)[0],
                remote_sub.get_cutout(remote_sub.index_map[source_index])[0],
            )

    def test_spectral_survey_cutouts(self, spectral_fits_dir, tmp_path):
        fits_dir, cubes = spectral_fits_dir
        survey_path = tmp_path / "survey_spec"
        build_survey_store(fits_dir, survey_path, mode="copy")
        catalogue = make_catalogue(
            tmp_path,
            [
                {"ID": "s0", "RA": IMAGE_RA, "DEC": IMAGE_DEC},
            ],
        )
        out = tmp_path / "dataset"
        build_test_dataset_store([survey_path], catalogue, CUTOUT_ARCSEC, out)
        sub = DatasetStore.open(out).survey("survey_spec")
        assert sub.observation_type == "spectral"
        assert sub.n_chan == 3
        cutout, _ = sub.get_cutout(sub.index_map[0])
        assert cutout.shape == (3, 27, 27)
        # All channels sliced from the same spatial box of cube_a, at the
        # origin recorded by the crpix shift
        from survey2zarray.stores.survey_store.survey_store import SurveyStore

        survey = SurveyStore.open(survey_path)
        local = sub.index_map[0]
        image_index = int(sub.image_index[local])
        x0 = int(survey.crpix[image_index][0] - sub.crpix[local][0])
        y0 = int(survey.crpix[image_index][1] - sub.crpix[local][1])
        window = cubes["cube_a"][0, :, y0 : y0 + 27, x0 : x0 + 27]
        inside = np.isfinite(cutout)
        np.testing.assert_array_equal(cutout[inside], window[inside])
        # Every channel is blanked to the same disc
        assert {
            tuple(map(tuple, np.isfinite(plane))) for plane in cutout
        }.__len__() == 1

    def test_multi_survey_store(
        self, survey_store_a, spectral_fits_dir, catalogue_path, tmp_path
    ):
        store_a, _ = survey_store_a
        fits_dir, _ = spectral_fits_dir
        store_b = tmp_path / "survey_b"
        build_survey_store(fits_dir, store_b, mode="copy")
        out = tmp_path / "dataset"
        build_test_dataset_store([store_a, store_b], catalogue_path, CUTOUT_ARCSEC, out)
        ds = DatasetStore.open(out)
        assert ds.surveys == ["survey_a", "survey_b"]
        # Canonical index space shared; per-survey coverage differs
        assert set(ds.survey("survey_a").index_map.keys()) == {0, 1}
        assert set(ds.survey("survey_b").index_map.keys()) == {0}

    def test_duplicate_ids_dropped(self, survey_store_a, tmp_path):
        store_path, _ = survey_store_a
        catalogue = make_catalogue(
            tmp_path,
            [
                {"ID": "same", "RA": IMAGE_RA, "DEC": IMAGE_DEC},
                {
                    "ID": "same",
                    "RA": IMAGE_RA + 0.02,
                    "DEC": IMAGE_DEC,
                },
            ],
        )
        out = tmp_path / "dataset"
        build_test_dataset_store([store_path], catalogue, CUTOUT_ARCSEC, out)
        ds = DatasetStore.open(out)
        assert ds.n_sources == 1
        assert list(ds.source_id) == ["same"]

    def test_missing_id_column_raises(self, survey_store_a, tmp_path):
        store_path, _ = survey_store_a
        path = tmp_path / "bad.csv"
        pd.DataFrame([{"RA": 1.0, "DEC": 2.0}]).to_csv(path, index=False)
        with pytest.raises(ValueError, match="ID column"):
            build_test_dataset_store(
                [store_path], path, CUTOUT_ARCSEC, tmp_path / "dataset"
            )


class TestBuildTimeDuplicateSources:
    # The images carry bmaj = 6", so sources closer than that are unresolvable
    # by this survey and count as one source (§12.1)

    def test_within_beam_duplicate_dropped_keeping_first(
        self, survey_store_a, tmp_path
    ):
        store_path, _ = survey_store_a
        catalogue = make_catalogue(
            tmp_path,
            [
                {"ID": "first", "RA": IMAGE_RA, "DEC": IMAGE_DEC},
                {
                    "ID": "second",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC + 2.0 / 3600.0,
                },
            ],
        )
        out = tmp_path / "dataset"
        build_test_dataset_store([store_path], catalogue, CUTOUT_ARCSEC, out)
        ds = DatasetStore.open(out)
        assert ds.n_sources == 1
        assert list(ds.source_id) == ["first"]  # Keep-first, in catalogue order

    def test_resolved_sources_both_kept(self, survey_store_a, tmp_path):
        store_path, _ = survey_store_a
        # 15" apart: the 6" beam resolves them, so they are two sources
        catalogue = make_catalogue(
            tmp_path,
            [
                {"ID": "a", "RA": IMAGE_RA, "DEC": IMAGE_DEC},
                {
                    "ID": "b",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC + 15.0 / 3600.0,
                },
            ],
        )
        out = tmp_path / "dataset"
        build_test_dataset_store([store_path], catalogue, CUTOUT_ARCSEC, out)
        assert DatasetStore.open(out).n_sources == 2

    def test_chain_collapses_to_first(self, survey_store_a, tmp_path):
        store_path, _ = survey_store_a
        # Each 2" from the last, all within 6" of the first
        catalogue = make_catalogue(
            tmp_path,
            [
                {
                    "ID": f"c{i}",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC + i * 2.0 / 3600.0,
                }
                for i in range(3)
            ],
        )
        out = tmp_path / "dataset"
        build_test_dataset_store([store_path], catalogue, CUTOUT_ARCSEC, out)
        ds = DatasetStore.open(out)
        assert list(ds.source_id) == ["c0"]

    def test_build_matches_append(self, survey_store_a, tmp_path):
        # The same rows must give the same store whether built at once or
        # built-then-appended
        from survey2zarray.stores.dataset_store.edit import append_sources

        store_path, _ = survey_store_a
        rows = [
            {"ID": "a", "RA": IMAGE_RA, "DEC": IMAGE_DEC},
            {
                "ID": "b",
                "RA": IMAGE_RA,
                "DEC": IMAGE_DEC + 2.0 / 3600.0,
            },
            {
                "ID": "c",
                "RA": IMAGE_RA,
                "DEC": IMAGE_DEC + 15.0 / 3600.0,
            },
        ]
        built = tmp_path / "ds_built"
        build_test_dataset_store(
            [store_path], make_catalogue(tmp_path, rows), CUTOUT_ARCSEC, built
        )

        appended = tmp_path / "ds_appended"
        first = make_catalogue(tmp_path, rows[:1], name="first.csv")
        rest = make_catalogue(tmp_path, rows[1:], name="rest.csv")
        build_test_dataset_store([store_path], first, CUTOUT_ARCSEC, appended)
        append_sources(appended, rest)

        assert list(DatasetStore.open(built).source_id) == list(
            DatasetStore.open(appended).source_id
        )

    def test_index_map_consistent_after_dedup(self, survey_store_a, tmp_path):
        store_path, _ = survey_store_a
        catalogue = make_catalogue(
            tmp_path,
            [
                {"ID": "a", "RA": IMAGE_RA, "DEC": IMAGE_DEC},
                {
                    "ID": "a_dup",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC + 1.0 / 3600.0,
                },
                {"ID": "b", "RA": IMAGE_RA + 0.02, "DEC": IMAGE_DEC},
            ],
        )
        out = tmp_path / "dataset"
        build_test_dataset_store([store_path], catalogue, CUTOUT_ARCSEC, out)
        # DatasetStore.open runs the index consistency check; source_index
        # values must be the compacted 0..n-1 of the surviving rows
        ds = DatasetStore.open(out)
        assert list(ds.source_id) == ["a", "b"]
        assert set(ds.survey("survey_a").index_map.keys()) <= {0, 1}


class TestFindCatalogueDuplicates:
    # Unit tests for the rule itself, with synthetic matches and beams so the
    # multi-survey cases are explicit
    def _df(self, separations_arcsec):
        return pd.DataFrame(
            {
                "id": [f"s{i}" for i in range(len(separations_arcsec))],
                "ra": [IMAGE_RA] * len(separations_arcsec),
                "dec": [IMAGE_DEC + s / 3600.0 for s in separations_arcsec],
            }
        )

    def test_finest_survey_decides(self):
        # Coarse survey (10" beam) would merge them; fine survey (2") resolves
        # them, and the finest common survey wins
        df = self._df([0.0, 5.0])
        maps = {"coarse": [0, 0], "fine": [0, 0]}
        beams = {
            "coarse": np.array([10.0, 10.0]) / 3600.0,
            "fine": np.array([2.0, 2.0]) / 3600.0,
        }
        assert not find_catalogue_duplicates(df, maps, beams).any()

    def test_coarse_only_pair_merges(self):
        # Same separation, but the fine survey does not observe them
        df = self._df([0.0, 5.0])
        maps = {"coarse": [0, 0], "fine": [np.nan, np.nan]}
        beams = {
            "coarse": np.array([10.0, 10.0]) / 3600.0,
            "fine": np.array([np.nan, np.nan]),
        }
        assert list(find_catalogue_duplicates(df, maps, beams)) == [False, True]

    def test_no_common_survey_never_duplicates(self):
        # Each source seen by a different survey: nothing can compare them
        df = self._df([0.0, 0.5])
        maps = {"one": [0, np.nan], "two": [np.nan, 0]}
        beams = {
            "one": np.array([10.0 / 3600.0, np.nan]),
            "two": np.array([np.nan, 10.0 / 3600.0]),
        }
        assert not find_catalogue_duplicates(df, maps, beams).any()

    def test_nan_beam_never_duplicates(self):
        # A survey with no beam recorded cannot declare sources unresolvable
        df = self._df([0.0, 0.5])
        maps = {"one": [0, 0]}
        beams = {"one": np.array([np.nan, np.nan])}
        assert not find_catalogue_duplicates(df, maps, beams).any()

    def test_single_row_and_empty(self):
        assert not find_catalogue_duplicates(
            self._df([0.0]), {"one": [0]}, {"one": np.array([6.0 / 3600.0])}
        ).any()

    def test_stored_check_applies_the_same_rule(self):
        # find_stored_duplicates compares new sources against stored ones, but
        # by the same finest-common-beam rule
        df_new = self._df([0.0])
        existing = SkyCoord(
            ra=[IMAGE_RA] * 2 * u.deg,
            dec=[IMAGE_DEC + 5.0 / 3600.0, IMAGE_DEC + 30.0 / 3600.0] * u.deg,
            frame=ICRS,
        )
        new_maps = {"one": [0]}
        new_beams = {"one": np.array([6.0 / 3600.0])}
        existing_presence = {"one": np.array([True, True])}
        existing_beams = {"one": np.array([6.0, 6.0]) / 3600.0}
        # 5" from the first stored source, inside the 6" beam
        assert find_stored_duplicates(
            df_new, new_maps, new_beams, existing, existing_presence, existing_beams
        )[0]
        # Same source, but the only stored neighbour is now 30" away
        far = SkyCoord(
            ra=[IMAGE_RA] * u.deg, dec=[IMAGE_DEC + 30.0 / 3600.0] * u.deg, frame=ICRS
        )
        assert not find_stored_duplicates(
            df_new,
            new_maps,
            new_beams,
            far,
            {"one": np.array([True])},
            {"one": np.array([6.0 / 3600.0])},
        )[0]

    def test_stored_check_no_common_survey(self):
        df_new = self._df([0.0])
        existing = SkyCoord(ra=[IMAGE_RA] * u.deg, dec=[IMAGE_DEC] * u.deg, frame=ICRS)
        # New source seen only by "one", stored source only by "two"
        new_maps = {"one": [0], "two": [np.nan]}
        new_beams = {"one": np.array([10.0 / 3600.0]), "two": np.array([np.nan])}
        existing_presence = {"one": np.array([False]), "two": np.array([True])}
        existing_beams = {"one": np.array([np.nan]), "two": np.array([10.0 / 3600.0])}
        assert not find_stored_duplicates(
            df_new, new_maps, new_beams, existing, existing_presence, existing_beams
        ).any()

    def test_stored_check_empty_store(self):
        df_new = self._df([0.0])
        empty = SkyCoord(ra=[] * u.deg, dec=[] * u.deg, frame=ICRS)
        assert not find_stored_duplicates(
            df_new,
            {"one": [0]},
            {"one": np.array([6.0 / 3600.0])},
            empty,
            {"one": np.array([], dtype=bool)},
            {"one": np.array([])},
        ).any()

    def test_conservative_pair_beam(self):
        # Sources with different matched beams use the larger of the two
        df = self._df([0.0, 5.0])
        maps = {"one": [0, 1]}
        beams = {"one": np.array([2.0, 8.0]) / 3600.0}  # max = 8" > 5" separation
        assert list(find_catalogue_duplicates(df, maps, beams)) == [False, True]


class TestWarningsAreConsolidated:
    def test_uncovered_sources_are_one_line_naming_a_few(self, tmp_path, caplog):
        import logging

        from survey2zarray.utils.logging import warn_many

        logger = logging.getLogger("survey2zarray.test")
        with caplog.at_level("WARNING"):
            warn_many(
                logger,
                [f"SRC{i:03d}" for i in range(460)],
                "{n} source(s) not covered by any LoTSS image",
            )
        assert len(caplog.records) == 1
        assert caplog.records[0].message == (
            "460 source(s) not covered by any LoTSS image. Examples are "
            "SRC000, SRC001, SRC002, and 457 more."
        )

    def test_a_few_are_all_named(self, caplog):
        import logging

        from survey2zarray.utils.logging import warn_many

        logger = logging.getLogger("survey2zarray.test")
        with caplog.at_level("WARNING"):
            warn_many(logger, ["SRC001", "SRC002"], "{n} source(s) missing")
        assert caplog.records[0].message == (
            "2 source(s) missing. Examples are SRC001, SRC002."
        )

    def test_nothing_is_said_when_nothing_went_wrong(self, caplog):
        import logging

        from survey2zarray.utils.logging import warn_many

        logger = logging.getLogger("survey2zarray.test")
        with caplog.at_level("WARNING"):
            warn_many(logger, [], "{n} source(s) missing")
        assert not caplog.records

    def test_a_build_reports_missing_sources_once_per_survey(
        self, survey_store_a, tmp_path, caplog
    ):
        store, _ = survey_store_a
        # Four sources well outside the survey, which used to be a line each
        catalogue = make_catalogue(
            tmp_path,
            [
                {"ID": "in_a", "RA": IMAGE_RA, "DEC": IMAGE_DEC},
            ]
            + [
                {
                    "ID": f"away_{i}",
                    "RA": IMAGE_RA + 5.0 + i,
                    "DEC": IMAGE_DEC - 5.0,
                }
                for i in range(4)
            ],
            name="spread.csv",
        )

        with caplog.at_level("WARNING"):
            build_test_dataset_store([store], catalogue, 20.0, tmp_path / "dataset")

        uncovered = [r.message for r in caplog.records if "not covered" in r.message]
        assert len(uncovered) == 1
        assert uncovered[0].startswith("4 source(s) not covered by any")
        assert "and 1 more" in uncovered[0]


class _RecordingBar:
    """A tqdm double that notes when each step was taken."""

    def __init__(self, ticks, **kwargs):
        self.ticks = ticks

    def update(self, n=1):
        import time

        self.ticks.append(time.perf_counter())

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestParallelExtraction:
    """Verify concurrent reference extraction preserves values and order."""

    def requests(self, n=6):
        return [(0, (IMAGE_RA, IMAGE_DEC)) for _ in range(n)]

    def test_a_copied_store_stays_serial(self, survey_store_a, monkeypatch):
        # Local pixels: threads would only contend on the disk
        from survey2zarray.stores.dataset_store import build as build_module

        store_path, _ = survey_store_a
        store = SurveyStore.open(store_path)
        assert store.access_mode == "copy"

        def refuse(**kwargs):
            raise AssertionError("a copied store should not start a thread pool")

        monkeypatch.setattr(build_module, "ThreadPoolExecutor", refuse)
        assert (
            len(build_module.extract_cutouts(store, self.requests(), 16.0, "test")) == 6
        )

    def test_parallel_matches_serial_exactly(self, survey_store_a, monkeypatch):
        from survey2zarray.stores.dataset_store import build as build_module

        store_path, _ = survey_store_a
        store = SurveyStore.open(store_path)
        # Vary the positions so a mis-ordered result cannot pass by luck
        requests = [
            (0, (IMAGE_RA + offset / 3600.0, IMAGE_DEC - offset / 3600.0))
            for offset in range(6)
        ]
        serial = build_module.extract_cutouts(store, requests, 16.0, "serial")

        # Ask for threads without pretending the pixels are remote, so the
        # reads stay exactly the ones the serial pass made
        monkeypatch.setattr(
            build_module, "extraction_plan", lambda store, image_index, n: (4, 4)
        )
        parallel = build_module.extract_cutouts(store, requests, 16.0, "parallel")

        assert len(parallel) == len(serial)
        for (cut_a, x_a, y_a, v_a), (cut_b, x_b, y_b, v_b) in zip(
            serial, parallel, strict=True
        ):
            assert np.array_equal(cut_a, cut_b, equal_nan=True)
            assert (x_a, y_a, v_a) == (x_b, y_b, v_b)

    def test_workers_and_ranges_share_one_connection_budget(
        self, survey_store_a, monkeypatch
    ):
        from survey2zarray.stores.dataset_store import build as build_module

        store = SurveyStore.open(survey_store_a[0])
        monkeypatch.setattr(build_module, "is_remote", lambda location: True)
        monkeypatch.setattr(store, "access_mode", "reference", raising=False)
        monkeypatch.setattr(
            store,
            "image_record",
            lambda index: {"location": "https://host/f.fits"},
            raising=False,
        )
        # However the machine splits it, the archive sees no more than the
        # budget the ladder settled on
        for count in (1, 4, 100, 10_000):
            workers, concurrency = build_module.extraction_plan(store, 0, count)
            assert workers >= 1 and concurrency >= 1
            assert workers * concurrency <= build_module.EXTRACTION_BUDGET
            assert workers <= count

    def test_concurrency_is_restored_afterwards(self, survey_store_a):
        from survey2zarray.stores.dataset_store.build import extract_cutouts

        store_path, _ = survey_store_a
        store = SurveyStore.open(store_path)
        before = store.concurrency
        extract_cutouts(store, self.requests(), 16.0, "test")
        assert store.concurrency == before

    def test_the_bar_follows_completions_not_submission_order(
        self, survey_store_a, monkeypatch
    ):
        # A slow first read must not hold the bar at zero while the rest finish
        import time

        from survey2zarray.stores.dataset_store import build as build_module

        store = SurveyStore.open(survey_store_a[0])
        real = build_module.get_cutout
        seen = []

        def slow_first(store_, index, coords, diameter):
            if not seen:
                seen.append(index)
                time.sleep(1.0)
            return real(store_, index, coords, diameter)

        monkeypatch.setattr(build_module, "get_cutout", slow_first)
        monkeypatch.setattr(
            build_module, "extraction_plan", lambda store, image_index, n: (4, 4)
        )

        ticks = []
        monkeypatch.setattr(
            build_module, "tqdm", lambda **kwargs: _RecordingBar(ticks, **kwargs)
        )
        started = time.perf_counter()
        results = build_module.extract_cutouts(store, self.requests(6), 16.0, "test")

        assert len(results) == 6 and all(r is not None for r in results)
        # Something completed well before the slow one did
        assert ticks and min(ticks) - started < 0.9

    def test_no_requests_is_not_an_error(self, survey_store_a):
        from survey2zarray.stores.dataset_store.build import extract_cutouts

        store_path, _ = survey_store_a
        assert extract_cutouts(SurveyStore.open(store_path), [], 16.0, "test") == []


class TestResumableBuild:
    """Verify resumed builds reuse completed archive reads."""

    def catalogue(self, tmp_path, n=30):
        rows = []
        for i in range(n):
            gx, gy = i % 5, (i // 5) % 5
            centre = (IMAGE_RA, IMAGE_DEC) if i < 25 else (IMAGE_RA, IMAGE_DEC + 0.02)
            rows.append(
                {
                    "id": f"S{i:03d}",
                    "ra": centre[0] + (gx - 2) * 7.5 / 3600.0,
                    "dec": centre[1] + (gy - 2) * 7.5 / 3600.0,
                }
            )
        path = tmp_path / "cat.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def build(self, survey, catalogue, out, **kwargs):
        return build_test_dataset_store(
            [survey], catalogue, 16.0, out, survey_names=["A"], **kwargs
        )

    def test_a_resumed_build_matches_one_done_in_a_single_run(
        self, survey_store_a, tmp_path, monkeypatch
    ):
        from survey2zarray.stores.dataset_store import build as build_module

        survey, _ = survey_store_a
        catalogue = self.catalogue(tmp_path)

        whole = tmp_path / "whole.zarr"
        self.build(survey, catalogue, whole)
        reference = DatasetStore.open(whole)

        # Die after the first batch of cutouts
        partial = tmp_path / "partial.zarr"
        monkeypatch.setattr(build_module, "CUTOUT_BATCH", 8)
        real = build_module.extract_cutouts
        calls = {"n": 0}

        def fail_after_two(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] > 2:
                raise KeyboardInterrupt("simulated crash")
            return real(*args, **kwargs)

        monkeypatch.setattr(build_module, "extract_cutouts", fail_after_two)
        with pytest.raises(KeyboardInterrupt):
            self.build(survey, catalogue, partial)

        monkeypatch.setattr(build_module, "extract_cutouts", real)
        self.build(survey, catalogue, partial, resume=True)

        resumed = DatasetStore.open(partial)
        assert resumed.n_sources == reference.n_sources
        a, b = reference.survey("A"), resumed.survey("A")
        assert b.n_cutouts == a.n_cutouts
        assert list(b.index_map) == list(a.index_map)
        assert np.array_equal(b.data[:], a.data[:], equal_nan=True)
        assert np.array_equal(b.crpix, a.crpix)
        assert list(b.image_index) == list(a.image_index)

    def test_an_interrupted_build_keeps_what_it_had(
        self, survey_store_a, tmp_path, monkeypatch
    ):

        from survey2zarray.stores.dataset_store import build as build_module

        survey, _ = survey_store_a
        catalogue = self.catalogue(tmp_path)
        out = tmp_path / "partial.zarr"

        monkeypatch.setattr(build_module, "CUTOUT_BATCH", 8)
        real = build_module.extract_cutouts
        calls = {"n": 0}

        def fail_after_one(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] > 1:
                raise KeyboardInterrupt("simulated crash")
            return real(*args, **kwargs)

        monkeypatch.setattr(build_module, "extract_cutouts", fail_after_one)
        with pytest.raises(KeyboardInterrupt):
            self.build(survey, catalogue, out)

        root = open_store(out, mode="r")
        assert root.attrs["complete"] is False
        stored = root["surveys"]["A"]["index_map"][:]
        held = stored[stored != PADDING]
        assert 0 < len(held) < 30  # some, not all
        # A row per source from the start, however far the build got
        assert root["surveys"]["A"]["data"].shape[0] == 30

    def test_a_finished_store_is_rebuilt_not_resumed(
        self, survey_store_a, tmp_path, caplog
    ):
        survey, _ = survey_store_a
        catalogue = self.catalogue(tmp_path)
        out = tmp_path / "done.zarr"
        self.build(survey, catalogue, out)
        with caplog.at_level("INFO"):
            self.build(survey, catalogue, out, resume=True)
        assert "already complete" in caplog.text

    def test_a_different_recipe_refuses_to_resume(
        self, survey_store_a, tmp_path, monkeypatch, caplog
    ):
        from survey2zarray.stores.dataset_store import build as build_module

        survey, _ = survey_store_a
        out = tmp_path / "partial.zarr"

        monkeypatch.setattr(build_module, "CUTOUT_BATCH", 8)
        real = build_module.extract_cutouts
        monkeypatch.setattr(
            build_module,
            "extract_cutouts",
            lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        with pytest.raises(KeyboardInterrupt):
            self.build(survey, self.catalogue(tmp_path), out)

        # Same path, fewer sources: the source indices would not line up
        monkeypatch.setattr(build_module, "extract_cutouts", real)
        with caplog.at_level("WARNING"):
            self.build(survey, self.catalogue(tmp_path, n=20), out, resume=True)
        assert "Cannot resume" in caplog.text
        assert DatasetStore.open(out).n_sources == 20

    def test_resume_is_off_by_default(self, survey_store_a, tmp_path):
        survey, _ = survey_store_a
        catalogue = self.catalogue(tmp_path)
        out = tmp_path / "store.zarr"
        self.build(survey, catalogue, out)
        first = DatasetStore.open(out).survey("A").n_cutouts
        self.build(survey, catalogue, out)  # no resume: starts again
        assert DatasetStore.open(out).survey("A").n_cutouts == first


class TestRotatedGrids:
    """Cutouts taken from a tile whose grid is rotated on the sky.

    The CD matrix is off-diagonal throughout, so anything that treats it as a
    pixel scale per axis puts the source somewhere other than the centre.
    """

    ROTATION = 33.0

    @pytest.fixture
    def rotated_survey(self, tmp_path):
        from tests.conftest import make_continuum_fits

        fits_dir = tmp_path / "rotated_fits"
        fits_dir.mkdir()
        make_continuum_fits(
            fits_dir / "rotated.fits",
            shape_yx=(120, 120),
            seed=11,
            rotation=self.ROTATION,
        )
        out = tmp_path / "rotated_survey"
        build_survey_store(fits_dir, out, mode="copy")
        return out

    @pytest.fixture
    def rotated_store(self, rotated_survey, tmp_path):
        catalogue = make_catalogue(
            tmp_path,
            [{"ID": "centre", "RA": IMAGE_RA, "DEC": IMAGE_DEC}],
            name="rotated_catalogue.csv",
        )
        out = tmp_path / "rotated_dataset"
        build_test_dataset_store(
            [rotated_survey], catalogue, CUTOUT_ARCSEC, out, bucket_by=["id"]
        )
        return out

    def test_the_survey_store_keeps_the_rotation(self, rotated_survey):
        store = SurveyStore.open(rotated_survey)
        cd = store.get_wcs(0).wcs.cd
        assert cd[0, 1] != 0.0 and cd[1, 0] != 0.0
        angle = np.degrees(np.arctan2(cd[1, 0], cd[1, 1]))
        assert angle == pytest.approx(self.ROTATION, abs=0.1)

    def test_a_cutout_carries_the_rotated_matrix(self, rotated_store):
        sub = DatasetStore.open(rotated_store).survey("rotated_survey")
        cd = sub.get_wcs(0).wcs.cd
        assert cd[0, 1] != 0.0 and cd[1, 0] != 0.0
        angle = np.degrees(np.arctan2(cd[1, 0], cd[1, 1]))
        assert angle == pytest.approx(self.ROTATION, abs=0.1)

    def test_the_source_sits_at_the_centre_of_its_cutout(self, rotated_store):
        sub = DatasetStore.open(rotated_store).survey("rotated_survey")
        cutout, wcs = sub.get_cutout(0, coords=(IMAGE_RA, IMAGE_DEC))
        target = SkyCoord(ra=IMAGE_RA * u.deg, dec=IMAGE_DEC * u.deg, frame=ICRS)
        x, y = wcs.world_to_pixel(target)
        assert float(x) == pytest.approx((cutout.shape[1] - 1) / 2, abs=0.5)
        assert float(y) == pytest.approx((cutout.shape[0] - 1) / 2, abs=0.5)

    def test_the_pixels_match_the_tile_they_came_from(
        self, rotated_store, rotated_survey
    ):
        # Read through the tile's own WCS, so a cutout taken at the wrong
        # pixels under rotation would come back holding different values
        from astropy.io import fits as astropy_fits
        from astropy.wcs import WCS

        sub = DatasetStore.open(rotated_store).survey("rotated_survey")
        cutout, wcs = sub.get_cutout(0, coords=(IMAGE_RA, IMAGE_DEC))
        location = SurveyStore.open(rotated_survey).image_location[0]
        with astropy_fits.open(location) as hdul:
            tile, tile_wcs = np.asarray(hdul[0].data), WCS(hdul[0].header).celestial
        centre = SkyCoord(
            *wcs.pixel_to_world_values(
                (cutout.shape[1] - 1) / 2, (cutout.shape[0] - 1) / 2
            ),
            unit="deg",
            frame=ICRS,
        )
        x, y = tile_wcs.world_to_pixel(centre)
        assert cutout[
            (cutout.shape[0] - 1) // 2, (cutout.shape[1] - 1) // 2
        ] == pytest.approx(tile[int(round(float(y))), int(round(float(x)))])
