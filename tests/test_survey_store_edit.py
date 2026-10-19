import numpy as np
import pandas as pd
import pytest

from survey2zarray.stores.store import consolidate as consolidate_store
from survey2zarray.stores.store import open_store
from survey2zarray.stores.survey_store.build import build_survey_store
from survey2zarray.stores.survey_store.edit import append_tiles, retry_failed_tiles
from survey2zarray.stores.survey_store.fits import FITSOpenError, IncompatibleTilesError
from survey2zarray.stores.survey_store.survey_store import SurveyStore
from survey2zarray.stores.transaction import (
    COMMITTED,
    EDIT_FLAG,
    begin_edit,
    commit_edit,
    recover_pending_edit,
)
from tests.conftest import IMAGE_DEC, IMAGE_RA, make_continuum_fits, make_spectral_fits


def open_rw(path):
    return open_store(path, mode="r+")


@pytest.fixture
def built_store(continuum_fits_dir, tmp_path):
    fits_dir, images = continuum_fits_dir
    out = tmp_path / "store"
    build_survey_store(fits_dir, out, mode="copy")
    return out, images


class TestAppendTiles:
    def test_append_new_image(self, built_store, tmp_path):
        store_path, _ = built_store
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        _, new_data = make_continuum_fits(
            new_dir / "image_d.fits",
            shape_yx=(32, 32),
            ra=IMAGE_RA - 0.02,
            dec=IMAGE_DEC,
            seed=9,
        )
        append_tiles(store_path, new_dir)
        store = SurveyStore.open(store_path)
        assert store.n_images == 4
        assert len(store.crval) == 4
        assert tuple(store.pixel_shape[3]) == (32, 32)
        np.testing.assert_array_equal(
            store.get_cutout_region(3, 0, 32, 0, 32), new_data
        )

    def test_append_larger_tile_grows_padding(self, built_store, tmp_path):
        store_path, images = built_store
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        _, new_data = make_continuum_fits(
            new_dir / "big.fits",
            shape_yx=(80, 80),
            ra=IMAGE_RA + 0.04,
            dec=IMAGE_DEC,
            seed=10,
        )
        append_tiles(store_path, new_dir)
        root = open_rw(store_path)
        assert root["data"].shape == (4, 80, 80)
        # Existing images retain their pixels, new area is NaN
        np.testing.assert_array_equal(root["data"][0, :64, :64], images["image_a"])
        assert np.all(np.isnan(root["data"][0, 64:, :]))
        np.testing.assert_array_equal(root["data"][3], new_data)

    def test_duplicate_footprint_skipped(self, built_store, tmp_path, caplog):
        store_path, _ = built_store
        dup_dir = tmp_path / "dup"
        dup_dir.mkdir()
        # Identical footprint (same centre, same shape) as image_a, different
        # pixel values and different filename - still a duplicate
        make_continuum_fits(
            dup_dir / "renamed_tile_a.fits",
            shape_yx=(64, 64),
            ra=IMAGE_RA,
            dec=IMAGE_DEC,
            seed=42,
        )
        with caplog.at_level("WARNING"):
            append_tiles(store_path, dup_dir)
        assert SurveyStore.open(store_path).n_images == 3
        assert any("duplicate" in r.message.lower() for r in caplog.records)

    def test_overlapping_but_offset_tile_not_a_duplicate(self, built_store, tmp_path):
        store_path, _ = built_store
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        # Same shape as image_a but centre shifted by ~7 pixels: overlap is
        # normal for survey images and must not be rejected
        make_continuum_fits(
            new_dir / "shifted.fits",
            shape_yx=(64, 64),
            ra=IMAGE_RA + 0.002,
            dec=IMAGE_DEC,
            seed=11,
        )
        append_tiles(store_path, new_dir)
        assert SurveyStore.open(store_path).n_images == 4

    def test_incompatible_type_aborts(self, built_store, tmp_path):
        store_path, _ = built_store
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        make_spectral_fits(new_dir / "cube.fits")
        with pytest.raises(IncompatibleTilesError):
            append_tiles(store_path, new_dir)
        assert SurveyStore.open(store_path).n_images == 3


class TestTransaction:
    def test_interrupted_edit_reverts_in_lockstep(self, built_store):
        store_path, _ = built_store
        root = open_rw(store_path)
        array_paths = ["data"] + [
            f"metadata/{k}" for k in root["metadata"].array_keys()
        ]
        begin_edit(root, array_paths, attrs={"n_images": 3})
        # Simulate a crash mid-edit: data grown and partially written, one
        # metadata array grown, the rest untouched, flag never cleared
        root["data"].resize((5, 64, 64))
        root["metadata/bmaj"].resize((5,))
        root.attrs["n_images"] = 5

        recovered = recover_pending_edit(store_path)
        assert recovered
        root = open_rw(store_path)
        assert root["data"].shape[0] == 3
        assert root["metadata/bmaj"].shape[0] == 3
        assert root.attrs["n_images"] == 3
        assert not root.attrs.get(EDIT_FLAG)
        # Store still opens and reads normally
        assert SurveyStore.open(store_path).n_images == 3

    def test_recovery_is_idempotent(self, built_store):
        store_path, _ = built_store
        root = open_rw(store_path)
        begin_edit(root, ["data"], attrs={"n_images": 3})
        root["data"].resize((4, 64, 64))
        assert recover_pending_edit(store_path)
        # Second recovery finds nothing to do
        assert not recover_pending_edit(store_path)
        assert open_rw(store_path)["data"].shape[0] == 3

    def test_open_triggers_recovery(self, built_store):
        store_path, _ = built_store
        root = open_rw(store_path)
        begin_edit(root, ["data"], attrs={"n_images": 3})
        root["data"].resize((6, 64, 64))
        # SurveyStore.open performs recovery before reading
        store = SurveyStore.open(store_path)
        assert store.n_images == 3
        assert open_rw(store_path)["data"].shape[0] == 3

    def test_commit_clears_flag(self, built_store):
        store_path, _ = built_store
        root = open_rw(store_path)
        begin_edit(root, ["data"], attrs={"n_images": 3})
        commit_edit(root)
        assert not root.attrs.get(EDIT_FLAG)
        assert not recover_pending_edit(store_path)

    def test_a_second_edit_cannot_replace_the_recovery_record(self, built_store):
        store_path, _ = built_store
        root = open_rw(store_path)
        begin_edit(root, ["data"], attrs={"n_images": 3})
        with pytest.raises(RuntimeError, match="another edit"):
            begin_edit(root, ["data"], attrs={"n_images": 3})

    def test_failure_before_commit_publication_remains_recoverable(
        self, built_store, monkeypatch
    ):
        from survey2zarray.stores import transaction

        store_path, _ = built_store
        root = open_rw(store_path)
        begin_edit(root, ["data"], attrs={"n_images": 3})
        root["data"].resize((4, 64, 64))
        root.attrs["n_images"] = 4

        def fail(_root):
            raise OSError("metadata publication failed")

        monkeypatch.setattr(transaction, "consolidate", fail)
        with pytest.raises(OSError, match="publication failed"):
            commit_edit(root)
        monkeypatch.setattr(transaction, "consolidate", consolidate_store)

        assert recover_pending_edit(store_path)
        recovered = open_rw(store_path)
        assert recovered["data"].shape[0] == 3
        assert recovered.attrs["n_images"] == 3

    def test_recovery_finalises_an_already_committed_edit(self, built_store):
        store_path, _ = built_store
        root = open_rw(store_path)
        begin_edit(root, ["data"], attrs={"n_images": 3})
        root["data"].resize((4, 64, 64))
        root.attrs["n_images"] = 4
        state = dict(root.attrs[EDIT_FLAG])
        state["phase"] = COMMITTED
        root.attrs[EDIT_FLAG] = state
        consolidate_store(root)

        assert recover_pending_edit(store_path)
        recovered = open_rw(store_path)
        assert recovered["data"].shape[0] == 4
        assert recovered.attrs["n_images"] == 4
        assert not recovered.attrs.get(EDIT_FLAG)


class TestRetryFailedTiles:
    """A store built while an archive was refusing reads can be topped up."""

    def store_with_a_refused_image(self, monkeypatch, fits_dir, out):
        """Build a store in which one file could not be read."""
        from survey2zarray.stores.survey_store.fits import extract_metadata as real

        paths = sorted(str(p) for p in fits_dir.glob("*.fits"))
        refused = {paths[1]}

        def read(path, require_referenceable=False, storage_options=None):
            if path in refused:
                raise FITSOpenError(f"Failed to open {path}")
            return real(path, require_referenceable=require_referenceable)

        monkeypatch.setattr(
            "survey2zarray.stores.survey_store.build.extract_metadata", read
        )
        build_survey_store(paths, out, mode="copy")
        monkeypatch.undo()
        return paths

    def test_it_appends_what_can_now_be_read(
        self, continuum_fits_dir, tmp_path, monkeypatch
    ):
        fits_dir, _ = continuum_fits_dir
        out = tmp_path / "store"
        paths = self.store_with_a_refused_image(monkeypatch, fits_dir, out)
        assert open_rw(out).attrs["n_images"] == len(paths) - 1

        retry_failed_tiles(out)
        assert open_rw(out).attrs["n_images"] == len(paths)

    def test_the_record_stops_calling_it_a_failure(
        self, continuum_fits_dir, tmp_path, monkeypatch
    ):
        fits_dir, _ = continuum_fits_dir
        out = tmp_path / "store"
        paths = self.store_with_a_refused_image(monkeypatch, fits_dir, out)
        retry_failed_tiles(out)

        record = pd.read_csv(out / "image_valid.csv")
        assert record["converted"].all()
        # One row per file still, not one for the failure and one for the retry
        assert len(record) == len(paths)
        # And the row points at the image it actually became
        assert sorted(record["array_index"]) == list(range(len(paths)))

    def test_running_it_again_does_nothing(
        self, continuum_fits_dir, tmp_path, monkeypatch
    ):
        fits_dir, _ = continuum_fits_dir
        out = tmp_path / "store"
        self.store_with_a_refused_image(monkeypatch, fits_dir, out)
        retry_failed_tiles(out)
        before = pd.read_csv(out / "image_valid.csv")
        retry_failed_tiles(out)
        pd.testing.assert_frame_equal(before, pd.read_csv(out / "image_valid.csv"))

    def test_a_file_that_is_still_gone_stays_recorded(
        self, continuum_fits_dir, tmp_path, monkeypatch
    ):
        fits_dir, _ = continuum_fits_dir
        out = tmp_path / "store"
        paths = sorted(str(p) for p in fits_dir.glob("*.fits"))
        missing = str(fits_dir / "never_existed.fits")
        build_survey_store(paths + [missing], out, mode="copy")
        assert open_rw(out).attrs["n_images"] == len(paths)

        retry_failed_tiles(out)
        record = pd.read_csv(out / "image_valid.csv")
        assert not record.loc[record["image_location"] == missing, "converted"].any()
        assert open_rw(out).attrs["n_images"] == len(paths)

    def test_a_store_with_no_record_is_refused(self, built_store):
        out, _ = built_store
        (out / "image_valid.csv").unlink()
        with pytest.raises(FileNotFoundError):
            retry_failed_tiles(out)


class TestTheRecordMatchesTheStore:
    def test_a_duplicate_is_recorded_as_unwritten(self, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        paths = sorted(str(p) for p in fits_dir.glob("*.fits"))
        out = tmp_path / "store"
        build_survey_store(paths[:-1], out, mode="copy")

        # The last image is new; the first is already in the store
        append_tiles(out, [paths[0], paths[-1]])
        record = pd.read_csv(out / "image_valid.csv")

        stored = open_rw(out).attrs["n_images"]
        assert stored == len(paths)
        duplicate = record[record["image_location"] == paths[0]]
        # Two rows for it: the one it became, and the attempt that added nothing
        assert list(duplicate["converted"]) == [True, False]
        assert duplicate["error"].str.contains("Duplicate").any()

    def test_every_written_tile_has_its_own_row_number(
        self, continuum_fits_dir, tmp_path
    ):
        fits_dir, _ = continuum_fits_dir
        paths = sorted(str(p) for p in fits_dir.glob("*.fits"))
        out = tmp_path / "store"
        build_survey_store(paths[:-1], out, mode="copy")
        append_tiles(out, [paths[0], paths[-1]])

        record = pd.read_csv(out / "image_valid.csv")
        written = record.loc[record["converted"].astype(bool), "array_index"]
        stored = open_rw(out).attrs["n_images"]
        # Each row of the store is named once, by the file that became it
        assert sorted(written) == list(range(stored))
