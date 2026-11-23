import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from survey2zarray.cli.main import cli
from survey2zarray.stores import migrations
from survey2zarray.stores.dataset_store import dataset_store as dataset_store_module
from survey2zarray.stores.dataset_store.build import build_dataset_store
from survey2zarray.stores.dataset_store.dataset_store import DatasetStore
from survey2zarray.stores.dataset_store.edit import append_sources
from survey2zarray.stores.migrations import (
    DATASET_MIGRATIONS,
    SCHEMA_HISTORY,
    SURVEY_MIGRATIONS,
    V1_RESOLUTION_NAMES,
    Migration,
    survey_v1_to_v2,
    upgrade_dataset_store,
    upgrade_survey_store,
)
from survey2zarray.stores.store import (
    DATASET_SCHEMA_VERSION,
    SURVEY_SCHEMA_VERSION,
    StoreVersionError,
    consolidate,
    open_store,
    package_version,
    schema_version,
)
from survey2zarray.stores.survey_store import survey_store as survey_store_module
from survey2zarray.stores.survey_store.build import build_survey_store
from survey2zarray.stores.survey_store.edit import append_tiles, retry_failed_tiles
from survey2zarray.stores.survey_store.survey_store import SurveyStore
from survey2zarray.stores.transaction import (
    EDIT_FLAG,
    begin_edit,
    commit_edit,
    member_backup_root,
    recover_pending_edit,
)
from tests.conftest import IMAGE_DEC, IMAGE_RA, make_optical_fits

CUTOUT_ARCSEC = 16.0
NEXT_SURVEY = SURVEY_SCHEMA_VERSION + 1
NEXT_DATASET = DATASET_SCHEMA_VERSION + 1
VERSION_ATTRS = ("schema_version", "created_with", "upgraded_with")


def make_catalogue(tmp_path, rows, name="catalogue.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def survey_path(continuum_fits_dir, tmp_path):
    fits_dir, _ = continuum_fits_dir
    out = tmp_path / "survey_a"
    build_survey_store(fits_dir, out, mode="copy")
    return out


@pytest.fixture
def optical_path(tmp_path):
    fits_dir = tmp_path / "optical"
    fits_dir.mkdir()
    make_optical_fits(fits_dir / "tile_a.fits", band="g", zeropoint=22.5, seed=1)
    make_optical_fits(
        fits_dir / "tile_b.fits", band="g", zeropoint=22.5, ra=IMAGE_RA + 0.02, seed=2
    )
    out = tmp_path / "optical_g"
    build_survey_store(fits_dir, out, mode="copy")
    return out


def build_dataset(survey_paths, tmp_path, name="dataset"):
    catalogue = make_catalogue(
        tmp_path,
        [
            {"ID": "src_0", "RA": IMAGE_RA, "DEC": IMAGE_DEC},
            {"ID": "src_1", "RA": IMAGE_RA + 0.02, "DEC": IMAGE_DEC},
        ],
    )
    out = tmp_path / name
    build_dataset_store(survey_paths, catalogue, CUTOUT_ARCSEC, out)
    return out


@pytest.fixture
def dataset_path(survey_path, tmp_path):
    return build_dataset([survey_path], tmp_path)


def set_attrs(path, **attrs):
    root = open_store(path, mode="r+")
    for name, value in attrs.items():
        if value is None:
            if name in root.attrs:
                del root.attrs[name]
        else:
            root.attrs[name] = value
    consolidate(root)


def metadata_to_v1(metadata):
    for old, new in V1_RESOLUTION_NAMES.items():
        array = metadata[new]
        metadata.create_array(old, data=array[:], chunks=array.chunks)
        del metadata[new]
    del metadata["band"]
    del metadata["zeropoint"]


def downgrade_survey(path):
    root = open_store(path, mode="r+")
    metadata_to_v1(root["metadata"])
    for name in (*VERSION_ATTRS, "band", "blank_value", "resolution", "band_pattern"):
        if name in root.attrs:
            del root.attrs[name]
    consolidate(root)


def downgrade_dataset(path):
    root = open_store(path, mode="r+")
    for survey in root["surveys"].group_keys():
        sub = root["surveys"][survey]
        metadata_to_v1(sub["metadata"])
        del sub.attrs["band"]
    for name in VERSION_ATTRS:
        if name in root.attrs:
            del root.attrs[name]
    consolidate(root)


def rename_pa(root):
    metadata = root["metadata"]
    values = metadata["resolution_pa"][:]
    del metadata["resolution_pa"]
    metadata.create_array("beam_pa", data=values, chunks=(len(values),))


RENAME_PA = Migration(
    apply=rename_pa,
    groups=lambda root: ["metadata/beam_pa"],
    members=lambda root: ["metadata/resolution_pa"],
)


def failing(apply):
    def run(root):
        apply(root)
        raise OSError("step failed")

    return run


@pytest.fixture
def next_survey_schema(monkeypatch):
    def install(step):
        monkeypatch.setattr(migrations, "SURVEY_SCHEMA_VERSION", NEXT_SURVEY)
        monkeypatch.setattr(
            migrations,
            "SURVEY_MIGRATIONS",
            {**SURVEY_MIGRATIONS, SURVEY_SCHEMA_VERSION: step},
        )

    return install


class TestStamping:
    def test_a_new_survey_store_records_its_versions(self, survey_path):
        root = open_store(survey_path)
        assert root.attrs["schema_version"] == SURVEY_SCHEMA_VERSION
        assert root.attrs["created_with"] == package_version()

    def test_a_new_dataset_store_records_its_versions(self, dataset_path):
        root = open_store(dataset_path)
        assert root.attrs["schema_version"] == DATASET_SCHEMA_VERSION
        assert root.attrs["created_with"] == package_version()


class TestReadingVersion1:
    def test_an_untagged_store_is_version_1(self, survey_path):
        downgrade_survey(survey_path)
        assert schema_version(open_store(survey_path)) == 1

    def test_a_version_1_survey_store_reads_in_the_current_layout(self, survey_path):
        current = SurveyStore.open(survey_path)
        downgrade_survey(survey_path)
        old = SurveyStore.open(survey_path)
        assert old.schema_version == 1
        np.testing.assert_array_equal(old.resolution_major, current.resolution_major)
        np.testing.assert_array_equal(old.resolution_pa, current.resolution_pa)
        assert list(old.band) == [""] * old.n_images
        assert np.isnan(old.zeropoint).all()

    def test_a_version_1_dataset_store_reads_in_the_current_layout(self, dataset_path):
        current = DatasetStore.open(dataset_path).survey("survey_a")
        downgrade_dataset(dataset_path)
        store = DatasetStore.open(dataset_path)
        old = store.survey("survey_a")
        assert store.schema_version == 1
        np.testing.assert_array_equal(old.resolution_major, current.resolution_major)
        assert old.survey_band == ""
        assert np.isnan(old.zeropoint).all()

    def test_a_newer_survey_store_is_refused(self, survey_path):
        set_attrs(survey_path, schema_version=NEXT_SURVEY, created_with="9.9.9")
        with pytest.raises(StoreVersionError) as raised:
            SurveyStore.open(survey_path)
        message = str(raised.value)
        assert f"schema version {NEXT_SURVEY}" in message
        assert f"up to version {SURVEY_SCHEMA_VERSION}" in message
        assert "9.9.9" in message

    def test_a_newer_dataset_store_is_refused(self, dataset_path):
        set_attrs(dataset_path, schema_version=NEXT_DATASET)
        with pytest.raises(StoreVersionError, match="install that version"):
            DatasetStore.open(dataset_path)

    def test_a_current_store_skips_the_adapters(self, survey_path, monkeypatch):
        def adapter(metadata):
            raise AssertionError("adapter applied to a current store")

        monkeypatch.setattr(
            survey_store_module,
            "SURVEY_READ_ADAPTERS",
            dict.fromkeys(range(1, SURVEY_SCHEMA_VERSION + 1), adapter),
        )
        SurveyStore.open(survey_path)

    def test_the_dataset_reader_passes_its_version_to_sub_stores(
        self, dataset_path, monkeypatch
    ):
        seen = []
        original = dataset_store_module.SurveySubStore.__init__

        def record(self, group, *args, schema_version, **kwargs):
            seen.append(schema_version)
            original(self, group, *args, schema_version=schema_version, **kwargs)

        monkeypatch.setattr(dataset_store_module.SurveySubStore, "__init__", record)
        downgrade_dataset(dataset_path)
        DatasetStore.open(dataset_path)
        assert seen == [1]


class TestEditingVersion1:
    def test_a_version_1_survey_store_refuses_appends(
        self, survey_path, continuum_fits_dir
    ):
        fits_dir, _ = continuum_fits_dir
        downgrade_survey(survey_path)
        with pytest.raises(StoreVersionError, match="s2z survey upgrade"):
            append_tiles(survey_path, fits_dir)
        assert open_store(survey_path).attrs["n_images"] == 3

    def test_a_version_1_survey_store_refuses_retries(self, survey_path):
        downgrade_survey(survey_path)
        with pytest.raises(StoreVersionError, match="s2z survey upgrade"):
            retry_failed_tiles(survey_path)

    def test_a_version_1_dataset_store_refuses_appends(self, dataset_path, tmp_path):
        catalogue = make_catalogue(
            tmp_path,
            [{"ID": "src_2", "RA": IMAGE_RA, "DEC": IMAGE_DEC + 60.0 / 3600.0}],
            name="new.csv",
        )
        downgrade_dataset(dataset_path)
        with pytest.raises(StoreVersionError, match="s2z dataset upgrade"):
            append_sources(dataset_path, catalogue)
        assert open_store(dataset_path).attrs["n_sources"] == 2

    def test_a_newer_store_refuses_edits(self, survey_path, continuum_fits_dir):
        fits_dir, _ = continuum_fits_dir
        set_attrs(survey_path, schema_version=NEXT_SURVEY)
        with pytest.raises(StoreVersionError, match="install that version"):
            append_tiles(survey_path, fits_dir)


class TestUpgradeFromVersion1:
    def test_a_survey_store_is_renamed_and_stamped(self, survey_path):
        before = SurveyStore.open(survey_path)
        downgrade_survey(survey_path)

        assert upgrade_survey_store(survey_path) == SURVEY_SCHEMA_VERSION
        root = open_store(survey_path)
        assert root.attrs["schema_version"] == SURVEY_SCHEMA_VERSION
        assert root.attrs["upgraded_with"] == package_version()
        assert root.attrs["band"] == ""
        assert not set(V1_RESOLUTION_NAMES) & set(root["metadata"].array_keys())
        after = SurveyStore.open(survey_path)
        np.testing.assert_array_equal(after.resolution_major, before.resolution_major)
        np.testing.assert_array_equal(after.resolution_minor, before.resolution_minor)
        assert not root.attrs.get(EDIT_FLAG)
        assert not member_backup_root(survey_path).exists()

    def test_bands_and_zeropoints_come_from_the_stored_headers(self, optical_path):
        downgrade_survey(optical_path)
        upgrade_survey_store(optical_path)
        store = SurveyStore.open(optical_path)
        assert store.root.attrs["band"] == "g"
        assert list(store.band) == ["g", "g"]
        np.testing.assert_array_equal(store.zeropoint, [22.5, 22.5])

    def test_an_upgraded_store_accepts_appends(self, survey_path, tmp_path):
        downgrade_survey(survey_path)
        upgrade_survey_store(survey_path)
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        make_optical_fits(new_dir / "x.fits", band="", ra=IMAGE_RA - 0.03, seed=9)
        append_tiles(survey_path, new_dir)
        assert SurveyStore.open(survey_path).n_images == 4

    def test_a_dataset_store_takes_bands_from_its_survey_stores(
        self, optical_path, tmp_path
    ):
        dataset = build_dataset([optical_path], tmp_path)
        before = DatasetStore.open(dataset).survey("optical_g")
        downgrade_dataset(dataset)

        assert upgrade_dataset_store(dataset) == DATASET_SCHEMA_VERSION
        sub = DatasetStore.open(dataset).survey("optical_g")
        assert sub.survey_band == "g"
        covered = sorted(sub.index_map)
        assert list(sub.band[covered]) == ["g"] * len(covered)
        np.testing.assert_array_equal(sub.zeropoint[covered], 22.5)
        np.testing.assert_array_equal(sub.resolution_major, before.resolution_major)

    def test_a_missing_survey_store_leaves_bands_empty(
        self, dataset_path, survey_path, caplog
    ):
        downgrade_dataset(dataset_path)
        survey_path.rename(survey_path.with_name("moved"))
        upgrade_dataset_store(dataset_path)
        sub = DatasetStore.open(dataset_path).survey("survey_a")
        assert sub.survey_band == ""
        assert np.isnan(sub.zeropoint).all()
        assert "Cannot read survey store" in caplog.text

    def test_an_upgraded_dataset_store_accepts_appends(self, dataset_path, tmp_path):
        catalogue = make_catalogue(
            tmp_path,
            [{"ID": "src_2", "RA": IMAGE_RA, "DEC": IMAGE_DEC + 60.0 / 3600.0}],
            name="new.csv",
        )
        downgrade_dataset(dataset_path)
        upgrade_dataset_store(dataset_path)
        append_sources(dataset_path, catalogue)
        assert DatasetStore.open(dataset_path).n_sources == 3

    def test_a_failing_migration_leaves_the_store_as_it_was(
        self, survey_path, monkeypatch
    ):
        step = SURVEY_MIGRATIONS[1]
        monkeypatch.setattr(
            migrations,
            "SURVEY_MIGRATIONS",
            {
                1: Migration(
                    apply=failing(survey_v1_to_v2),
                    attrs=step.attrs,
                    groups=step.groups,
                    members=step.members,
                )
            },
        )
        downgrade_survey(survey_path)
        root = open_store(survey_path)
        attrs = dict(root.attrs)
        arrays = {name: root["metadata"][name][:] for name in V1_RESOLUTION_NAMES}

        with pytest.raises(OSError, match="step failed"):
            upgrade_survey_store(survey_path)
        root = open_store(survey_path)
        assert dict(root.attrs) == attrs
        assert "band" not in root["metadata"]
        for name, values in arrays.items():
            np.testing.assert_array_equal(root["metadata"][name][:], values)
        assert SurveyStore.open(survey_path).schema_version == 1


class TestUpgradeSteps:
    def test_a_current_store_is_left_alone(self, survey_path, dataset_path):
        assert upgrade_survey_store(survey_path) == SURVEY_SCHEMA_VERSION
        assert upgrade_dataset_store(dataset_path) == DATASET_SCHEMA_VERSION
        assert "upgraded_with" not in open_store(survey_path).attrs

    def test_a_step_upgrades_and_stamps_the_store(
        self, survey_path, next_survey_schema
    ):
        next_survey_schema(RENAME_PA)
        before = open_store(survey_path)["metadata/resolution_pa"][:]

        assert upgrade_survey_store(survey_path) == NEXT_SURVEY
        root = open_store(survey_path)
        assert root.attrs["schema_version"] == NEXT_SURVEY
        assert root.attrs["upgraded_with"] == package_version()
        assert "resolution_pa" not in root["metadata"]
        np.testing.assert_array_equal(root["metadata/beam_pa"][:], before)

    def test_a_second_upgrade_does_nothing(self, survey_path, next_survey_schema):
        next_survey_schema(RENAME_PA)
        upgrade_survey_store(survey_path)
        upgraded_with = open_store(survey_path).attrs["upgraded_with"]
        assert upgrade_survey_store(survey_path) == NEXT_SURVEY
        assert open_store(survey_path).attrs["upgraded_with"] == upgraded_with

    def test_a_failing_step_leaves_the_store_as_it_was(
        self, survey_path, next_survey_schema
    ):
        next_survey_schema(
            Migration(
                apply=failing(rename_pa),
                groups=RENAME_PA.groups,
                members=RENAME_PA.members,
            )
        )
        before = dict(open_store(survey_path).attrs)
        values = open_store(survey_path)["metadata/resolution_pa"][:]

        with pytest.raises(OSError, match="step failed"):
            upgrade_survey_store(survey_path)
        root = open_store(survey_path)
        assert dict(root.attrs) == before
        assert "beam_pa" not in root["metadata"]
        np.testing.assert_array_equal(root["metadata/resolution_pa"][:], values)
        assert not member_backup_root(survey_path).exists()

    def test_a_failing_step_restores_the_attributes_it_changed(
        self, dataset_path, monkeypatch
    ):
        def retitle(root):
            root.attrs["margin_px"] = -1
            root["surveys/survey_a"].attrs["added"] = True

        monkeypatch.setattr(migrations, "DATASET_SCHEMA_VERSION", NEXT_DATASET)
        monkeypatch.setattr(
            migrations,
            "DATASET_MIGRATIONS",
            {
                DATASET_SCHEMA_VERSION: Migration(
                    apply=failing(retitle),
                    attrs=lambda root: ["margin_px", "surveys/survey_a:added"],
                )
            },
        )
        before = dict(open_store(dataset_path).attrs)
        with pytest.raises(OSError, match="step failed"):
            upgrade_dataset_store(dataset_path)
        root = open_store(dataset_path)
        assert dict(root.attrs) == before
        assert "added" not in root["surveys/survey_a"].attrs

    def test_a_missing_step_is_refused(self, survey_path, monkeypatch):
        monkeypatch.setattr(migrations, "SURVEY_SCHEMA_VERSION", NEXT_SURVEY)
        with pytest.raises(
            RuntimeError, match=f"from schema version {NEXT_SURVEY - 1}"
        ):
            upgrade_survey_store(survey_path)
        assert open_store(survey_path).attrs["schema_version"] == SURVEY_SCHEMA_VERSION

    def test_a_newer_store_is_refused(self, survey_path):
        set_attrs(survey_path, schema_version=NEXT_SURVEY)
        with pytest.raises(StoreVersionError, match="newer than"):
            upgrade_survey_store(survey_path)


class TestMemberBackups:
    def test_an_interrupted_edit_restores_a_deleted_member(self, survey_path):
        root = open_store(survey_path, mode="r+")
        values = root["metadata/resolution_pa"][:]
        begin_edit(root, [], members=["metadata/resolution_pa"])
        del root["metadata/resolution_pa"]

        assert recover_pending_edit(survey_path)
        np.testing.assert_array_equal(
            open_store(survey_path)["metadata/resolution_pa"][:], values
        )
        assert not member_backup_root(survey_path).exists()

    def test_a_commit_discards_member_backups(self, survey_path):
        root = open_store(survey_path, mode="r+")
        begin_edit(root, [], members=["metadata/resolution_pa"])
        assert member_backup_root(survey_path).exists()
        commit_edit(root)
        assert not member_backup_root(survey_path).exists()

    def test_backups_are_not_members_of_the_store(self, survey_path):
        root = open_store(survey_path, mode="r+")
        names = set(root["metadata"].array_keys())
        begin_edit(root, [], members=["metadata/resolution_pa"])
        reopened = open_store(survey_path, mode="r+")
        assert set(reopened["metadata"].array_keys()) == names

    def test_an_added_attribute_is_removed_on_revert(self, survey_path):
        root = open_store(survey_path, mode="r+")
        begin_edit(root, [], added_attrs=["extra"])
        root.attrs["extra"] = 1
        recover_pending_edit(survey_path)
        assert "extra" not in open_store(survey_path).attrs


class TestHistory:
    @pytest.mark.parametrize(
        ("kind", "current", "steps", "adapters"),
        [
            (
                "survey",
                SURVEY_SCHEMA_VERSION,
                SURVEY_MIGRATIONS,
                migrations.SURVEY_READ_ADAPTERS,
            ),
            (
                "dataset",
                DATASET_SCHEMA_VERSION,
                DATASET_MIGRATIONS,
                migrations.DATASET_READ_ADAPTERS,
            ),
        ],
    )
    def test_every_version_is_recorded_reachable_and_readable(
        self, kind, current, steps, adapters
    ):
        assert set(SCHEMA_HISTORY[kind]) == set(range(1, current + 1))
        assert set(steps) == set(range(1, current))
        assert set(adapters) == set(range(1, current))


class TestCommandLine:
    def test_survey_upgrade(self, survey_path):
        downgrade_survey(survey_path)
        result = CliRunner().invoke(cli, ["survey", "upgrade", str(survey_path)])
        assert result.exit_code == 0, result.output
        assert schema_version(open_store(survey_path)) == SURVEY_SCHEMA_VERSION

    def test_dataset_upgrade(self, dataset_path):
        downgrade_dataset(dataset_path)
        result = CliRunner().invoke(cli, ["dataset", "upgrade", str(dataset_path)])
        assert result.exit_code == 0, result.output
        assert schema_version(open_store(dataset_path)) == DATASET_SCHEMA_VERSION

    def test_survey_upgrade_refuses_a_newer_store(self, survey_path):
        set_attrs(survey_path, schema_version=NEXT_SURVEY)
        result = CliRunner().invoke(cli, ["survey", "upgrade", str(survey_path)])
        assert result.exit_code != 0
        assert isinstance(result.exception, StoreVersionError)
