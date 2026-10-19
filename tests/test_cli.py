import pandas as pd
import pytest
from click.testing import CliRunner

from survey2zarray.cli.main import cli
from survey2zarray.stores.dataset_store.dataset_store import DatasetStore
from survey2zarray.stores.store import open_store
from tests.conftest import IMAGE_DEC, IMAGE_RA, make_continuum_fits


@pytest.fixture
def runner():
    return CliRunner()


class TestSurveyBuild:
    def test_local_build(self, runner, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        out = tmp_path / "store"
        result = runner.invoke(cli, ["survey", "build", str(fits_dir), str(out)])
        assert result.exit_code == 0, result.output
        root = open_store(out)
        assert root.attrs["complete"] is True
        assert root.attrs["access_mode"] == "copy"
        assert root.attrs["n_images"] == 3

    def test_reference_build_from_url_file(self, runner, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        urls = sorted(str(p) for p in fits_dir.glob("*.fits"))
        url_file = tmp_path / "urls.txt"
        url_file.write_text("\n".join(urls) + "\n")
        out = tmp_path / "store"
        result = runner.invoke(
            cli, ["survey", "build", str(url_file), str(out), "--mode", "reference"]
        )
        assert result.exit_code == 0, result.output
        root = open_store(out)
        assert root.attrs["access_mode"] == "reference"
        assert "data" not in root

    def test_missing_source_rejected(self, runner, tmp_path):
        result = runner.invoke(
            cli, ["survey", "build", str(tmp_path / "nope"), str(tmp_path / "out")]
        )
        assert result.exit_code != 0

    def test_invalid_mode_rejected(self, runner, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        result = runner.invoke(
            cli,
            ["survey", "build", str(fits_dir), str(tmp_path / "out"), "--mode", "nope"],
        )
        assert result.exit_code != 0


class TestStorageOptionsFlag:
    """Given as JSON or as a file, so a credential need not be typed."""

    def test_json_is_passed_to_the_build(self, runner, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        out = tmp_path / "store"
        result = runner.invoke(
            cli,
            [
                "survey",
                "build",
                str(fits_dir),
                str(out),
                "--storage-options",
                '{"s3": {"anon": true}}',
            ],
        )
        assert result.exit_code == 0, result.output
        assert open_store(out).attrs["storage_options"] == {"s3": {"anon": True}}

    def test_a_json_file_is_read(self, runner, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        options = tmp_path / "options.json"
        options.write_text('{"s3": {"region_name": "eu-west-1"}}')
        out = tmp_path / "store"
        result = runner.invoke(
            cli,
            [
                "survey",
                "build",
                str(fits_dir),
                str(out),
                "--storage-options",
                str(options),
            ],
        )
        assert result.exit_code == 0, result.output
        assert open_store(out).attrs["storage_options"] == {
            "s3": {"region_name": "eu-west-1"}
        }

    def test_malformed_json_is_refused(self, runner, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        result = runner.invoke(
            cli,
            [
                "survey",
                "build",
                str(fits_dir),
                str(tmp_path / "out"),
                "--storage-options",
                "{not json",
            ],
        )
        assert result.exit_code != 0
        assert "Expected JSON or a readable JSON file" in result.output

    def test_options_must_be_keyed_by_protocol(
        self, runner, continuum_fits_dir, tmp_path
    ):
        fits_dir, _ = continuum_fits_dir
        result = runner.invoke(
            cli,
            [
                "survey",
                "build",
                str(fits_dir),
                str(tmp_path / "out"),
                "--storage-options",
                '{"anon": true}',
            ],
        )
        assert result.exit_code != 0
        assert "keyed by protocol" in result.output


class TestSurveyAppend:
    def test_append(self, runner, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        out = tmp_path / "store"
        assert (
            runner.invoke(cli, ["survey", "build", str(fits_dir), str(out)]).exit_code
            == 0
        )
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        make_continuum_fits(
            new_dir / "image_d.fits",
            shape_yx=(32, 32),
            ra=IMAGE_RA - 0.02,
            dec=IMAGE_DEC,
            seed=8,
        )
        result = runner.invoke(cli, ["survey", "append", str(out), str(new_dir)])
        assert result.exit_code == 0, result.output
        assert open_store(out).attrs["n_images"] == 4


class TestDatasetBuildAndAppend:
    def test_build_and_append(self, runner, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        survey_path = tmp_path / "survey_a"
        assert (
            runner.invoke(
                cli, ["survey", "build", str(fits_dir), str(survey_path)]
            ).exit_code
            == 0
        )

        catalogue = tmp_path / "catalogue.csv"
        pd.DataFrame(
            [
                {"ID": "src_0", "RA": IMAGE_RA, "DEC": IMAGE_DEC},
            ]
        ).to_csv(catalogue, index=False)
        out = tmp_path / "dataset"
        result = runner.invoke(
            cli,
            [
                "dataset",
                "build",
                str(survey_path),
                str(catalogue),
                str(out),
                "--max-cutout-size",
                "16",
            ],
        )
        assert result.exit_code == 0, result.output
        ds = DatasetStore.open(out)
        assert ds.n_sources == 1
        assert ds.surveys == ["survey_a"]

        new_catalogue = tmp_path / "new.csv"
        pd.DataFrame(
            [
                {
                    "ID": "src_1",
                    "RA": IMAGE_RA + 0.02,
                    "DEC": IMAGE_DEC,
                },
            ]
        ).to_csv(new_catalogue, index=False)
        result = runner.invoke(
            cli, ["dataset", "append-sources", str(out), str(new_catalogue)]
        )
        assert result.exit_code == 0, result.output
        assert DatasetStore.open(out).n_sources == 2

    def test_append_survey(
        self, runner, continuum_fits_dir, spectral_fits_dir, tmp_path
    ):
        fits_dir, _ = continuum_fits_dir
        spectral_dir, _ = spectral_fits_dir
        survey_a = tmp_path / "survey_a"
        survey_b = tmp_path / "survey_b"
        assert (
            runner.invoke(
                cli, ["survey", "build", str(fits_dir), str(survey_a)]
            ).exit_code
            == 0
        )
        assert (
            runner.invoke(
                cli, ["survey", "build", str(spectral_dir), str(survey_b)]
            ).exit_code
            == 0
        )

        catalogue = tmp_path / "catalogue.csv"
        pd.DataFrame(
            [
                {"ID": "src_0", "RA": IMAGE_RA, "DEC": IMAGE_DEC},
            ]
        ).to_csv(catalogue, index=False)
        out = tmp_path / "dataset"
        assert (
            runner.invoke(
                cli,
                [
                    "dataset",
                    "build",
                    str(survey_a),
                    str(catalogue),
                    str(out),
                    "--max-cutout-size",
                    "16",
                ],
            ).exit_code
            == 0
        )
        assert DatasetStore.open(out).surveys == ["survey_a"]

        result = runner.invoke(
            cli, ["dataset", "append-survey", str(out), str(survey_b)]
        )
        assert result.exit_code == 0, result.output
        ds = DatasetStore.open(out)
        assert ds.surveys == ["survey_a", "survey_b"]
        assert ds.n_sources == 1  # Source table unchanged
        assert 0 in ds.survey("survey_b").index_map

    def test_append_survey_custom_name(
        self, runner, continuum_fits_dir, spectral_fits_dir, tmp_path
    ):
        fits_dir, _ = continuum_fits_dir
        spectral_dir, _ = spectral_fits_dir
        survey_a = tmp_path / "survey_a"
        survey_b = tmp_path / "survey_b"
        runner.invoke(cli, ["survey", "build", str(fits_dir), str(survey_a)])
        runner.invoke(cli, ["survey", "build", str(spectral_dir), str(survey_b)])
        catalogue = tmp_path / "catalogue.csv"
        pd.DataFrame([{"ID": "src_0", "RA": IMAGE_RA, "DEC": IMAGE_DEC}]).to_csv(
            catalogue, index=False
        )
        out = tmp_path / "dataset"
        runner.invoke(
            cli,
            [
                "dataset",
                "build",
                str(survey_a),
                str(catalogue),
                str(out),
                "--max-cutout-size",
                "16",
            ],
        )
        result = runner.invoke(
            cli,
            ["dataset", "append-survey", str(out), str(survey_b), "--name", "lofar"],
        )
        assert result.exit_code == 0, result.output
        assert DatasetStore.open(out).surveys == ["survey_a", "lofar"]

    def test_append_survey_missing_store_rejected(self, runner, tmp_path):
        result = runner.invoke(
            cli,
            [
                "dataset",
                "append-survey",
                str(tmp_path / "nope"),
                str(tmp_path / "also_nope"),
            ],
        )
        assert result.exit_code != 0

    def test_cutout_size_required(self, runner, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        survey_path = tmp_path / "survey_a"
        assert (
            runner.invoke(
                cli, ["survey", "build", str(fits_dir), str(survey_path)]
            ).exit_code
            == 0
        )
        catalogue = tmp_path / "catalogue.csv"
        pd.DataFrame([{"ID": "a", "RA": IMAGE_RA, "DEC": IMAGE_DEC}]).to_csv(
            catalogue, index=False
        )
        result = runner.invoke(
            cli,
            [
                "dataset",
                "build",
                str(survey_path),
                str(catalogue),
                str(tmp_path / "out"),
            ],
        )
        assert result.exit_code != 0
