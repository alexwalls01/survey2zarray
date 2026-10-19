import numpy as np
import pandas as pd
import pytest

from survey2zarray.stores.dataset_store.build import build_dataset_store
from survey2zarray.stores.dataset_store.dataset_store import DatasetStore
from survey2zarray.stores.dataset_store.edit import (
    append_sources,
    append_survey,
    read_source_table,
)
from survey2zarray.stores.dataset_store.layout import is_power_of_two
from survey2zarray.stores.store import open_store
from survey2zarray.stores.survey_store.build import build_survey_store
from survey2zarray.stores.transaction import EDIT_FLAG, begin_edit
from tests.conftest import IMAGE_DEC, IMAGE_RA, make_continuum_fits

CUTOUT_ARCSEC = 16.0


def make_catalogue(tmp_path, rows, name="catalogue.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def dataset_store_path(continuum_fits_dir, tmp_path):
    fits_dir, _ = continuum_fits_dir
    survey_path = tmp_path / "survey_a"
    build_survey_store(fits_dir, survey_path, mode="copy")
    catalogue = make_catalogue(
        tmp_path,
        [
            {"ID": "src_0", "RA": IMAGE_RA, "DEC": IMAGE_DEC},
            {"ID": "src_1", "RA": IMAGE_RA + 0.02, "DEC": IMAGE_DEC},
        ],
    )
    out = tmp_path / "dataset"
    build_dataset_store([survey_path], catalogue, CUTOUT_ARCSEC, out)
    return out


class TestAppendSources:
    def test_append_new_source(self, dataset_store_path, tmp_path):
        # 60" from src_0: well outside the 6" beam, inside image C (dec+72")
        catalogue = make_catalogue(
            tmp_path,
            [
                {
                    "ID": "src_2",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC + 60.0 / 3600.0,
                },
            ],
            name="new.csv",
        )
        append_sources(dataset_store_path, catalogue)
        ds = DatasetStore.open(dataset_store_path)
        assert ds.n_sources == 3
        # Canonical indices are never reused; the new source continues the table
        assert list(ds.source_id) == ["src_0", "src_1", "src_2"]
        sub = ds.survey("survey_a")
        assert 2 in sub.index_map
        cutout, _ = sub.get_cutout(sub.index_map[2])
        assert cutout.shape == (27, 27)
        # A disc, so the corners are blanked; everything inside it is real
        assert np.isfinite(cutout).any()
        centre = (cutout.shape[-1] - 1) / 2
        yy, xx = np.mgrid[0 : cutout.shape[-2], 0 : cutout.shape[-1]]
        inner = (
            np.hypot(xx - centre, yy - centre) <= CUTOUT_ARCSEC * np.sqrt(2) / 2 + 2 - 1
        )
        assert np.all(np.isfinite(cutout)[inner])

    def test_an_appended_source_keeps_every_survey_a_row_per_source(
        self, dataset_store_path, tmp_path
    ):
        catalogue = make_catalogue(
            tmp_path,
            [{"ID": "src_2", "RA": IMAGE_RA, "DEC": IMAGE_DEC + 60.0 / 3600.0}],
            name="new.csv",
        )
        append_sources(dataset_store_path, catalogue)
        ds = DatasetStore.open(dataset_store_path)
        sub = ds.survey("survey_a")
        assert sub.data.shape[0] == ds.n_sources == 3
        assert sub.index_map == {i: i for i in sub.index_map}

    def test_append_absent_source_only_extends_table(
        self, dataset_store_path, tmp_path
    ):
        catalogue = make_catalogue(
            tmp_path,
            [
                {
                    "ID": "far_away",
                    "RA": IMAGE_RA + 5.0,
                    "DEC": IMAGE_DEC - 5.0,
                },
            ],
            name="new.csv",
        )
        append_sources(dataset_store_path, catalogue)
        ds = DatasetStore.open(dataset_store_path)
        assert ds.n_sources == 3
        # Not covered by the survey: in the table, absent from the map
        assert 2 not in ds.survey("survey_a").index_map
        assert ds.survey("survey_a").n_cutouts == 2

    def test_append_rejects_a_matched_cutout_with_blank_pixels(
        self, dataset_store_path, tmp_path, caplog
    ):
        # Coverage matching uses image bounds. Pixel validity is checked after
        # extraction, and append must apply the same complete-disc rule as build.
        dataset = DatasetStore.open(dataset_store_path)
        survey_path = dataset.survey_store_paths[0]
        survey_root = open_store(survey_path, mode="r+")
        survey_root["data"][:] = np.nan
        catalogue = make_catalogue(
            tmp_path,
            [{"ID": "blank", "RA": IMAGE_RA, "DEC": IMAGE_DEC + 60 / 3600}],
            name="blank.csv",
        )
        with caplog.at_level("WARNING"):
            append_sources(dataset_store_path, catalogue)
        result = DatasetStore.open(dataset_store_path)
        assert result.n_sources == 3
        assert 2 not in result.survey("survey_a").index_map
        assert result.survey("survey_a").n_cutouts == 2
        assert any(
            "discs contain blank pixels" in record.message for record in caplog.records
        )

    def test_append_extracts_in_bounded_batches(
        self, dataset_store_path, tmp_path, monkeypatch
    ):
        from survey2zarray.stores.dataset_store import edit as edit_module

        catalogue = make_catalogue(
            tmp_path,
            [
                {"ID": "new_a", "RA": IMAGE_RA, "DEC": IMAGE_DEC + 60 / 3600},
                {"ID": "new_b", "RA": IMAGE_RA, "DEC": IMAGE_DEC + 75 / 3600},
            ],
            name="batch.csv",
        )
        real_extract = edit_module.extract_cutouts
        sizes = []

        def recording_extract(store, requests, *args, **kwargs):
            sizes.append(len(requests))
            return real_extract(store, requests, *args, **kwargs)

        monkeypatch.setattr(edit_module, "CUTOUT_BATCH", 1)
        monkeypatch.setattr(edit_module, "extract_cutouts", recording_extract)
        append_sources(dataset_store_path, catalogue)
        assert sizes == [1, 1]

    def test_duplicate_within_beam_skipped(self, dataset_store_path, tmp_path, caplog):
        # 2" from src_0: inside the 6" (bmaj) beam of the only survey
        # observing both -> the instrument cannot separate them -> duplicate
        catalogue = make_catalogue(
            tmp_path,
            [
                {
                    "ID": "dup_of_0",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC + 2.0 / 3600.0,
                },
            ],
            name="new.csv",
        )
        with caplog.at_level("WARNING"):
            append_sources(dataset_store_path, catalogue)
        ds = DatasetStore.open(dataset_store_path)
        assert ds.n_sources == 2
        assert any("duplicate" in r.message.lower() for r in caplog.records)

    def test_source_outside_beam_not_duplicate(self, dataset_store_path, tmp_path):
        # 15" from src_0: outside the 6" beam - the instrument resolves them
        catalogue = make_catalogue(
            tmp_path,
            [
                {
                    "ID": "near_0",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC + 15.0 / 3600.0,
                },
            ],
            name="new.csv",
        )
        append_sources(dataset_store_path, catalogue)
        assert DatasetStore.open(dataset_store_path).n_sources == 3

    def test_duplicates_within_appended_catalogue(
        self, dataset_store_path, tmp_path, caplog
    ):
        # Two rows of the appended catalogue 2" apart, both far from anything
        # stored: the pair must collapse before it reaches the store
        catalogue = make_catalogue(
            tmp_path,
            [
                {
                    "ID": "new_a",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC + 60.0 / 3600.0,
                },
                {
                    "ID": "new_b",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC + 62.0 / 3600.0,
                },
            ],
            name="new.csv",
        )
        with caplog.at_level("WARNING"):
            append_sources(dataset_store_path, catalogue)
        ds = DatasetStore.open(dataset_store_path)
        assert ds.n_sources == 3
        assert list(ds.source_id)[-1] == "new_a"  # Keep-first
        assert any("within the appended catalogue" in r.message for r in caplog.records)

    def test_stored_duplicate_cannot_suppress_a_later_new_source(
        self, dataset_store_path, tmp_path
    ):
        # new_a is 4" from stored src_0 and is rejected. new_b is 4" from
        # new_a but 8" from src_0, so it must survive. Internal-first filtering
        # incorrectly let rejected new_a suppress new_b.
        catalogue = make_catalogue(
            tmp_path,
            [
                {"ID": "new_a", "RA": IMAGE_RA, "DEC": IMAGE_DEC + 4 / 3600},
                {"ID": "new_b", "RA": IMAGE_RA, "DEC": IMAGE_DEC + 8 / 3600},
            ],
            name="chain.csv",
        )
        append_sources(dataset_store_path, catalogue)
        result = DatasetStore.open(dataset_store_path)
        assert list(result.source_id) == ["src_0", "src_1", "new_b"]

    def test_resolved_pair_within_appended_catalogue_both_kept(
        self, dataset_store_path, tmp_path
    ):
        catalogue = make_catalogue(
            tmp_path,
            [
                {
                    "ID": "new_a",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC + 45.0 / 3600.0,
                },
                {
                    "ID": "new_b",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC + 60.0 / 3600.0,
                },
            ],
            name="new.csv",
        )
        append_sources(dataset_store_path, catalogue)
        assert DatasetStore.open(dataset_store_path).n_sources == 4

    def test_duplicate_ids_within_appended_catalogue(
        self, dataset_store_path, tmp_path
    ):
        catalogue = make_catalogue(
            tmp_path,
            [
                {
                    "ID": "same",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC + 45.0 / 3600.0,
                },
                {
                    "ID": "same",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC + 60.0 / 3600.0,
                },
            ],
            name="new.csv",
        )
        append_sources(dataset_store_path, catalogue)
        ds = DatasetStore.open(dataset_store_path)
        assert ds.n_sources == 3
        assert list(ds.source_id) == ["src_0", "src_1", "same"]

    def test_existing_id_skipped(self, dataset_store_path, tmp_path, caplog):
        catalogue = make_catalogue(
            tmp_path,
            [
                {
                    "ID": "src_0",
                    "RA": IMAGE_RA + 0.01,
                    "DEC": IMAGE_DEC,
                },
            ],
            name="new.csv",
        )
        with caplog.at_level("WARNING"):
            append_sources(dataset_store_path, catalogue)
        assert DatasetStore.open(dataset_store_path).n_sources == 2

    def test_append_keeps_consistency(self, dataset_store_path, tmp_path):
        catalogue = make_catalogue(
            tmp_path,
            [
                {
                    "ID": "src_2",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC + 60.0 / 3600.0,
                },
                {
                    "ID": "src_3",
                    "RA": IMAGE_RA + 5.0,
                    "DEC": IMAGE_DEC + 5.0,
                },
            ],
            name="new.csv",
        )
        append_sources(dataset_store_path, catalogue)
        # DatasetStore.open runs the index consistency check
        ds = DatasetStore.open(dataset_store_path)
        sub = ds.survey("survey_a")
        assert set(sub.index_map.keys()) == {0, 1, 2}
        assert sub.n_cutouts == 3
        # An appended source keeps its own row, and src_3, covered by nothing,
        # gets a row too
        assert sub.index_map == {0: 0, 1: 1, 2: 2}
        assert sub.data.shape[0] == ds.n_sources == 4


class TestAppendSourceColumns:
    """A store's optional columns must be supplied for every source appended.

    There is no value that means "missing" for an integer label, so an append
    that cannot fill a column is refused rather than guessing one.
    """

    @pytest.fixture
    def labelled_store(self, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        survey_path = tmp_path / "survey_lab"
        build_survey_store(fits_dir, survey_path, mode="copy")
        catalogue = make_catalogue(
            tmp_path,
            [
                {
                    "ID": "src_0",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC,
                    "label": 1,
                },
            ],
            name="labelled.csv",
        )
        out = tmp_path / "labelled"
        build_dataset_store([survey_path], catalogue, CUTOUT_ARCSEC, out)
        return out

    def test_append_with_the_column_succeeds(self, labelled_store, tmp_path):
        catalogue = make_catalogue(
            tmp_path,
            [
                {
                    "ID": "src_1",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC + 60.0 / 3600.0,
                    "label": 0,
                },
            ],
            name="more.csv",
        )
        append_sources(labelled_store, catalogue)
        store = DatasetStore.open(labelled_store)
        assert list(store.extra_columns["label"]) == [1, 0]
        # And the column keeps its dtype through the append
        assert store.extra_columns["label"].dtype == np.int64

    def test_append_missing_the_column_is_refused(self, labelled_store, tmp_path):
        catalogue = make_catalogue(
            tmp_path,
            [
                {
                    "ID": "src_1",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC + 60.0 / 3600.0,
                },
            ],
            name="unlabelled.csv",
        )
        with pytest.raises(ValueError, match="missing column"):
            append_sources(labelled_store, catalogue)
        # Refused before anything was written
        assert DatasetStore.open(labelled_store).n_sources == 1

    def test_unknown_column_warns_and_is_dropped(
        self, labelled_store, tmp_path, caplog
    ):
        catalogue = make_catalogue(
            tmp_path,
            [
                {
                    "ID": "src_1",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC + 60.0 / 3600.0,
                    "label": 0,
                    "redshift": 0.4,
                },
            ],
            name="extra.csv",
        )
        with caplog.at_level("WARNING"):
            append_sources(labelled_store, catalogue)
        assert any("does not hold" in r.message for r in caplog.records)
        assert "redshift" not in DatasetStore.open(labelled_store).extra_columns


class TestAppendSurveyPadding:
    def test_coarser_survey_warns_that_the_square_shrinks(
        self, dataset_store_path, continuum_fits_dir, tmp_path, caplog
    ):
        # The disc was sized when the store was built and cannot grow, so a
        # survey coarser than it was padded for reaches less of it
        from tests.conftest import make_continuum_fits

        coarse_dir = tmp_path / "coarse_fits"
        coarse_dir.mkdir()
        make_continuum_fits(
            coarse_dir / "coarse.fits",
            shape_yx=(64, 64),
            ra=IMAGE_RA,
            dec=IMAGE_DEC,
            seed=7,
            pixel_scale=8.0,
        )
        coarse_store = tmp_path / "coarse"
        build_survey_store(coarse_dir, coarse_store, mode="copy")
        with caplog.at_level("WARNING"):
            append_survey(dataset_store_path, coarse_store, "coarse")
        assert any("padded for" in r.message for r in caplog.records)
        assert any("cutouts serve" in r.message for r in caplog.records)

    def test_finer_survey_does_not_warn(self, dataset_store_path, tmp_path, caplog):
        from tests.conftest import make_continuum_fits

        fine_dir = tmp_path / "fine_fits"
        fine_dir.mkdir()
        make_continuum_fits(
            fine_dir / "fine.fits",
            shape_yx=(128, 128),
            ra=IMAGE_RA,
            dec=IMAGE_DEC,
            seed=8,
            pixel_scale=0.5,
        )
        fine_store = tmp_path / "fine"
        build_survey_store(fine_dir, fine_store, mode="copy")
        with caplog.at_level("WARNING"):
            append_survey(dataset_store_path, fine_store, "fine")
        assert not any("padded for" in r.message for r in caplog.records)


class TestAppendSurvey:
    @pytest.fixture
    def spectral_survey(self, spectral_fits_dir, tmp_path):
        fits_dir, cubes = spectral_fits_dir
        out = tmp_path / "survey_spec"
        build_survey_store(fits_dir, out, mode="copy")
        return out, cubes

    def test_append_survey_adds_substore(self, dataset_store_path, spectral_survey):
        survey_path, cubes = spectral_survey
        append_survey(dataset_store_path, survey_path)
        ds = DatasetStore.open(dataset_store_path)
        # Widened across surveys; the source table is untouched
        assert ds.surveys == ["survey_a", "survey_spec"]
        assert ds.n_sources == 2
        sub = ds.survey("survey_spec")
        assert sub.observation_type == "spectral"
        assert sub.n_chan == 3
        # src_0 is on the cubes, src_1 is not
        assert set(sub.index_map.keys()) == {0}
        cutout, _ = sub.get_cutout(sub.index_map[0])
        assert cutout.shape == (3, 27, 27)

    def test_an_appended_survey_holds_a_row_per_source(
        self, dataset_store_path, spectral_survey
    ):
        survey_path, _ = spectral_survey
        append_survey(dataset_store_path, survey_path)
        ds = DatasetStore.open(dataset_store_path)
        sub = ds.survey("survey_spec")
        # src_1 is not on the cubes, so its row is there and never written
        assert sub.data.shape[0] == ds.n_sources == 2
        assert sub.index_map == {0: 0}
        assert np.isnan(sub.data[1]).all()

    def test_an_appended_survey_shards_by_a_power_of_two(
        self, dataset_store_path, tmp_path
    ):
        # Finer pixels than anything stored, so its cutouts are far larger and
        # the lower bound on rows is what settles the count
        fine_dir = tmp_path / "fine_fits"
        fine_dir.mkdir()
        make_continuum_fits(
            fine_dir / "fine.fits", shape_yx=(2048, 2048), pixel_scale=0.02
        )
        fine_survey = tmp_path / "survey_fine"
        build_survey_store(fine_dir, fine_survey, mode="copy")
        append_survey(dataset_store_path, fine_survey)
        surveys = open_store(dataset_store_path, mode="r")["surveys"]
        appended = surveys["survey_fine"]["data"].shards[0]
        existing = surveys["survey_a"]["data"].shards[0]
        # Chosen from its own cutout size and coverage, knowing nothing of what
        # is already stored, and aligned with it anyway
        assert is_power_of_two(appended) and is_power_of_two(existing)
        assert appended < existing and existing % appended == 0

    def test_existing_survey_untouched(self, dataset_store_path, spectral_survey):
        survey_path, _ = spectral_survey
        before = DatasetStore.open(dataset_store_path).survey("survey_a")
        before_cutout, _ = before.get_cutout(before.index_map[0])
        append_survey(dataset_store_path, survey_path)
        after = DatasetStore.open(dataset_store_path).survey("survey_a")
        assert after.index_map == before.index_map
        np.testing.assert_array_equal(
            after.get_cutout(after.index_map[0])[0], before_cutout
        )

    def test_custom_name(self, dataset_store_path, spectral_survey):
        survey_path, _ = spectral_survey
        append_survey(dataset_store_path, survey_path, survey_name="my_survey")
        assert DatasetStore.open(dataset_store_path).surveys == [
            "survey_a",
            "my_survey",
        ]

    def test_duplicate_name_rejected(self, dataset_store_path, spectral_survey):
        survey_path, _ = spectral_survey
        with pytest.raises(ValueError, match="already has survey"):
            append_survey(dataset_store_path, survey_path, survey_name="survey_a")
        assert DatasetStore.open(dataset_store_path).surveys == ["survey_a"]

    def test_survey_store_paths_recorded(self, dataset_store_path, spectral_survey):
        survey_path, _ = spectral_survey
        append_survey(dataset_store_path, survey_path)
        ds = DatasetStore.open(dataset_store_path)
        assert ds.survey_store_paths[-1] == str(survey_path)
        # The recorded path is what a later source append reads back
        assert len(ds.survey_store_paths) == len(ds.surveys)

    def test_sources_append_after_survey_append(
        self, dataset_store_path, spectral_survey, tmp_path
    ):
        # The two edit kinds compose: widen, then lengthen
        survey_path, _ = spectral_survey
        append_survey(dataset_store_path, survey_path)
        catalogue = make_catalogue(
            tmp_path,
            [
                {
                    "ID": "src_2",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC + 60.0 / 3600.0,
                },
            ],
            name="more.csv",
        )
        append_sources(dataset_store_path, catalogue)
        ds = DatasetStore.open(dataset_store_path)
        assert ds.n_sources == 3
        assert ds.surveys == ["survey_a", "survey_spec"]
        assert 2 in ds.survey("survey_a").index_map

    def test_empty_substore_when_no_coverage(
        self, dataset_store_path, continuum_fits_dir, tmp_path, caplog
    ):
        # A survey covering none of the sources still attaches, with an empty map
        far_dir = tmp_path / "far_fits"
        far_dir.mkdir()
        make_continuum_fits(
            far_dir / "far.fits",
            shape_yx=(32, 32),
            ra=IMAGE_RA + 40.0,
            dec=IMAGE_DEC - 20.0,
        )
        far_survey = tmp_path / "survey_far"
        build_survey_store(far_dir, far_survey, mode="copy")
        with caplog.at_level("WARNING"):
            append_survey(dataset_store_path, far_survey)
        ds = DatasetStore.open(dataset_store_path)
        sub = ds.survey("survey_far")
        assert sub.index_map == {}
        assert sub.n_cutouts == 0
        assert any("empty" in r.message for r in caplog.records)

    def test_interrupted_append_survey_removes_substore(
        self, dataset_store_path, spectral_survey
    ):
        survey_path, _ = spectral_survey
        root = open_store(dataset_store_path, mode="r+")
        begin_edit(
            root, [], attrs={"surveys": ["survey_a"]}, groups=["surveys/half_written"]
        )
        # Simulate a crash partway through writing a new sub-store
        half = root.require_group("surveys").require_group("half_written")
        half.create_array("data", shape=(1, 4, 4), dtype="f4")
        root.attrs["surveys"] = ["survey_a", "half_written"]

        # Open triggers recovery: the group is deleted, the attr restored
        ds = DatasetStore.open(dataset_store_path)
        assert ds.surveys == ["survey_a"]
        root = open_store(dataset_store_path, mode="r")
        assert "surveys/half_written" not in root
        assert not root.attrs.get(EDIT_FLAG)
        # The store is usable afterwards
        append_survey(dataset_store_path, survey_path)
        assert DatasetStore.open(dataset_store_path).surveys == [
            "survey_a",
            "survey_spec",
        ]


class TestConsolidatedMetadata:
    """A store's metadata is gathered into its root, so opening it is one read.

    What an edit adds or resizes has to reach that copy, since a reader takes it
    at its word.
    """

    def test_a_built_store_carries_it(self, dataset_store_path):
        root = open_store(dataset_store_path, mode="r")
        assert root.metadata.consolidated_metadata is not None

    def test_an_appended_survey_is_visible_to_a_reader(
        self, dataset_store_path, tmp_path, spectral_fits_dir
    ):
        fits_dir, _ = spectral_fits_dir
        survey_path = tmp_path / "survey_spec"
        build_survey_store(fits_dir, survey_path, mode="copy")
        append_survey(dataset_store_path, survey_path)
        root = open_store(dataset_store_path, mode="r")
        assert "survey_spec" in root["surveys"]
        assert root.metadata.consolidated_metadata is not None

    def test_appended_sources_are_the_length_a_reader_sees(
        self, dataset_store_path, tmp_path
    ):
        catalogue = make_catalogue(
            tmp_path,
            [
                {
                    "ID": "src_2",
                    "RA": IMAGE_RA + 0.01,
                    "DEC": IMAGE_DEC,
                }
            ],
            name="more.csv",
        )
        append_sources(dataset_store_path, catalogue)
        root = open_store(dataset_store_path, mode="r")
        assert root["sources"]["source_id"].shape[0] == 3


class TestAppendTransaction:
    def test_interrupted_append_reverts_source_table_and_substores(
        self, dataset_store_path
    ):
        root = open_store(dataset_store_path, mode="r+")
        array_paths = (
            [f"sources/{k}" for k in root["sources"].array_keys()]
            + ["surveys/survey_a/data", "surveys/survey_a/index_map"]
            + [
                f"surveys/survey_a/metadata/{k}"
                for k in root["surveys/survey_a"]["metadata"].array_keys()
            ]
        )
        begin_edit(
            root,
            array_paths,
            attrs={
                "n_sources": 2,
                "surveys/survey_a:n_cutouts": 2,
            },
        )
        # Simulate a crash: source table and data grown, index_map grown but
        # unwritten, attrs bumped, flag never cleared
        root["sources/ra"].resize((4,))
        root["surveys/survey_a/data"].resize((4, 27, 27))
        root["surveys/survey_a/index_map"].resize((3,))
        root.attrs["n_sources"] = 4
        root["surveys/survey_a"].attrs["n_cutouts"] = 4

        # Open triggers recovery; everything reverts in lockstep
        ds = DatasetStore.open(dataset_store_path)
        assert ds.n_sources == 2
        sub = ds.survey("survey_a")
        assert sub.data.shape[0] == 2
        assert len(sub.index_map) == 2
        assert sub.n_cutouts == 2
        root = open_store(dataset_store_path, mode="r")
        assert not root.attrs.get(EDIT_FLAG)
        assert root["sources/ra"].shape == (2,)


class TestStringColumns:
    """Round trip catalogue strings through the store.

    They are held as `numpy.dtypes.StringDType`, which a DataFrame column and
    an equality test both have to accommodate.
    """

    @pytest.fixture
    def classified_store(self, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        survey_path = tmp_path / "survey_a"
        build_survey_store(fits_dir, survey_path, mode="copy")
        catalogue = make_catalogue(
            tmp_path,
            [
                {"ID": "src_0", "RA": IMAGE_RA, "DEC": IMAGE_DEC, "kind": "FRI"},
                {
                    "ID": "src_1",
                    "RA": IMAGE_RA + 0.02,
                    "DEC": IMAGE_DEC,
                    "kind": "FRII",
                },
            ],
            name="classified.csv",
        )
        out = tmp_path / "classified"
        build_dataset_store([survey_path], catalogue, CUTOUT_ARCSEC, out)
        return out

    def test_the_store_holds_them_as_variable_length_strings(self, classified_store):
        sources = open_store(classified_store, mode="r")["sources"]
        assert sources["source_id"].dtype.kind == "T"
        assert sources["kind"].dtype.kind == "T"

    def test_read_source_table_gives_python_strings(self, classified_store):
        root = open_store(classified_store, mode="r")
        table = read_source_table(root)
        assert list(table["id"]) == ["src_0", "src_1"]
        assert all(isinstance(value, str) for value in table["id"])

    def test_the_matcher_can_join_on_the_identifier(self, classified_store):
        # The comparison append_sources makes, which a numpy string scalar
        # rather than a str would silently fail
        root = open_store(classified_store, mode="r")
        table = read_source_table(root)
        assert table["id"].isin({"src_1"}).tolist() == [False, True]

    def test_a_string_column_survives_an_append(self, classified_store, tmp_path):
        catalogue = make_catalogue(
            tmp_path,
            [
                {
                    "ID": "src_2",
                    "RA": IMAGE_RA,
                    "DEC": IMAGE_DEC + 60.0 / 3600.0,
                    "kind": "FRI",
                },
            ],
            name="more_classified.csv",
        )
        append_sources(classified_store, catalogue)
        store = DatasetStore.open(classified_store)
        assert [str(v) for v in store.extra_columns["kind"]] == ["FRI", "FRII", "FRI"]
        assert [str(v) for v in store.source_id] == ["src_0", "src_1", "src_2"]

    def test_a_missing_value_reads_back_empty_rather_than_nan(
        self, continuum_fits_dir, tmp_path
    ):
        fits_dir, _ = continuum_fits_dir
        survey_path = tmp_path / "survey_gap"
        build_survey_store(fits_dir, survey_path, mode="copy")
        catalogue = make_catalogue(
            tmp_path,
            [
                {"ID": "src_0", "RA": IMAGE_RA, "DEC": IMAGE_DEC, "kind": "FRI"},
                {"ID": "src_1", "RA": IMAGE_RA + 0.02, "DEC": IMAGE_DEC, "kind": None},
            ],
            name="gappy.csv",
        )
        out = tmp_path / "gappy"
        build_dataset_store([survey_path], catalogue, CUTOUT_ARCSEC, out)
        store = DatasetStore.open(out)
        assert [str(v) for v in store.extra_columns["kind"]] == ["FRI", ""]
