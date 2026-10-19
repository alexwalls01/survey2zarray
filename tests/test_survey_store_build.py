from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits as astropy_fits

from survey2zarray.stores.store import open_store
from survey2zarray.stores.survey_store.build import (
    HEADER_ATTEMPTS,
    HEADER_WORKERS,
    PIXEL_CHUNK_SPLIT,
    build_header_index,
    build_survey_store,
    persistable_options,
    pixel_chunks,
    read_header,
    resolve_fits_list,
    resolve_header_workers,
)
from survey2zarray.stores.survey_store.fits import (
    FITSOpenError,
    IncompatibleTilesError,
    InvalidFITSError,
    extract_metadata,
)
from survey2zarray.stores.survey_store.survey_store import SurveyStore
from tests.conftest import (
    IMAGE_DEC,
    IMAGE_RA,
    make_continuum_4d_fits,
    make_continuum_fits,
    make_spectral_fits,
)


class TestLocalContinuumBuild:
    def test_build_attrs_and_layout(self, continuum_fits_dir, tmp_path):
        fits_dir, images = continuum_fits_dir
        out = tmp_path / "store"
        build_survey_store(fits_dir, out, mode="copy")
        root = open_store(out)
        assert root.attrs["complete"] is True
        assert root.attrs["observation_type"] == "continuum"
        assert root.attrs["access_mode"] == "copy"
        assert root.attrs["n_images"] == 3
        # Padded to max shape across images
        assert root["data"].shape == (3, 64, 64)
        assert (out / "image_valid.csv").exists()
        assert (out / "metadata_valid.csv").exists()

    def test_pixel_data_and_nan_padding(self, continuum_fits_dir, tmp_path):
        fits_dir, images = continuum_fits_dir
        out = tmp_path / "store"
        build_survey_store(fits_dir, out, mode="copy")
        root = open_store(out)
        # Files are sorted: image_a, image_b, image_c
        np.testing.assert_array_equal(root["data"][0], images["image_a"])
        # image_b is (48, 56): real pixels then NaN fill
        np.testing.assert_array_equal(root["data"][1, :48, :56], images["image_b"])
        assert np.all(np.isnan(root["data"][1, 48:, :]))
        assert np.all(np.isnan(root["data"][1, :, 56:]))
        # image_c is 4D on disk; stored as its Stokes-I 2D plane
        np.testing.assert_array_equal(root["data"][2], images["image_c"][0, 0])

    def test_metadata_arrays(self, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        out = tmp_path / "store"
        build_survey_store(fits_dir, out, mode="copy")
        metadata = open_store(out)["metadata"]
        np.testing.assert_array_equal(
            metadata["pixel_shape"][:], [(64, 64), (48, 56), (64, 64)]
        )
        # One row per image, single chunk
        assert metadata["crval"].chunks[0] == 3
        assert metadata["pixel_centre"][0] == pytest.approx(
            (IMAGE_RA, IMAGE_DEC), abs=1e-3
        )
        assert metadata["pixel_scale"][0] == pytest.approx((1.0, 1.0), rel=1e-6)
        assert np.all(metadata["frequency"][:] > 0)
        assert list(metadata["radesys"][:]) == ["ICRS"] * 3


class TestPixelAndHeaderLayout:
    """A shard is the file a read opens and a chunk is what it decompresses.

    A cutout therefore costs a region of one image rather than a byte budget of
    it, and a header costs one header.
    """

    def test_the_pixel_array_is_sharded_within_the_byte_budget(
        self, continuum_fits_dir, tmp_path
    ):
        fits_dir, _ = continuum_fits_dir
        out = tmp_path / "store"
        build_survey_store(fits_dir, out, mode="copy")
        data = open_store(out)["data"]
        assert data.shards[0] == 1  # One image to a shard
        assert data.chunks[1:] == tuple(
            size // PIXEL_CHUNK_SPLIT for size in data.shards[1:]
        )

    def test_the_shard_follows_the_budget_and_the_chunk_divides_it(self):
        chunks, shards = pixel_chunks("continuum", None, (4096, 4096), 4)
        assert np.prod(shards) * 4 <= 1_000_000
        assert shards[1:] == tuple(size * PIXEL_CHUNK_SPLIT for size in chunks[1:])

    def test_a_spectral_shard_keeps_its_channels_together(self):
        chunks, shards = pixel_chunks("spectral", 8, (2048, 2048), 4)
        assert chunks[1] == shards[1] == 8

    def test_one_header_to_a_chunk(self, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        out = tmp_path / "store"
        build_survey_store(fits_dir, out, mode="copy")
        header = open_store(out)["metadata"]["header"]
        assert header.chunks == (1,)
        assert header.shards == (3,)

    def test_the_metadata_is_consolidated(self, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        out = tmp_path / "store"
        build_survey_store(fits_dir, out, mode="copy")
        assert open_store(out).metadata.consolidated_metadata is not None


class TestLocalSpectralBuild:
    def test_spectral_store(self, spectral_fits_dir, tmp_path):
        fits_dir, cubes = spectral_fits_dir
        out = tmp_path / "store"
        build_survey_store(fits_dir, out, mode="copy")
        root = open_store(out)
        assert root.attrs["observation_type"] == "spectral"
        assert root.attrs["n_chan"] == 3
        assert root["data"].shape == (2, 3, 48, 48)
        np.testing.assert_array_equal(root["data"][0], cubes["cube_a"][0])
        # Smaller cube NaN-padded
        np.testing.assert_array_equal(root["data"][1, :, :40, :40], cubes["cube_b"][0])
        assert np.all(np.isnan(root["data"][1, :, 40:, :]))
        # WCS metadata includes the frequency axis
        assert open_store(out)["metadata"]["crval"].shape == (2, 3)


class TestCompatibilityAbort:
    def test_mixed_types_abort_before_writing(self, tmp_path):
        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        make_continuum_fits(fits_dir / "a.fits")
        make_spectral_fits(fits_dir / "b.fits")
        out = tmp_path / "store"
        with pytest.raises(IncompatibleTilesError):
            build_survey_store(fits_dir, out, mode="copy")
        # Nothing was written
        assert not out.exists()

    def test_mixed_n_chan_aborts(self, tmp_path):
        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        make_spectral_fits(fits_dir / "a.fits", n_chan=3)
        make_spectral_fits(fits_dir / "b.fits", n_chan=4, ra=IMAGE_RA + 0.02)
        out = tmp_path / "store"
        with pytest.raises(IncompatibleTilesError):
            build_survey_store(fits_dir, out, mode="copy")
        assert not out.exists()

    def test_empty_dir_raises(self, tmp_path):
        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            build_survey_store(fits_dir, tmp_path / "store", mode="copy")


class TestRemoteBuild:
    def test_remote_store_has_no_data_array(self, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        # Local file paths stand in for URLs: fsspec handles both
        urls = sorted(str(p) for p in fits_dir.glob("*.fits"))
        out = tmp_path / "store"
        build_survey_store(urls, out, mode="reference")
        root = open_store(out)
        assert root.attrs["access_mode"] == "reference"
        assert root.attrs["complete"] is True
        assert "data" not in root
        # URL list stored for on-demand opening
        assert list(root["metadata"]["image_location"][:]) == urls

    def test_remote_metadata_matches_local(self, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        urls = sorted(str(p) for p in fits_dir.glob("*.fits"))
        build_survey_store(fits_dir, tmp_path / "local_store", mode="copy")
        build_survey_store(urls, tmp_path / "remote_store", mode="reference")
        local = open_store(tmp_path / "local_store")["metadata"]
        remote = open_store(tmp_path / "remote_store")["metadata"]
        # Same extraction path - identical metadata either way
        for key in [
            "crval",
            "crpix",
            "cd",
            "pixel_shape",
            "pixel_centre",
            "pixel_scale",
            "bmaj",
        ]:
            np.testing.assert_array_equal(local[key][:], remote[key][:])
        assert list(local["ctype"][:]) == list(remote["ctype"][:])

    def test_remote_url_list_file(self, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        urls = sorted(str(p) for p in fits_dir.glob("*.fits"))
        url_file = tmp_path / "urls.txt"
        url_file.write_text("\n".join(urls) + "\n")
        out = tmp_path / "store"
        build_survey_store(url_file, out, mode="reference")
        assert list(open_store(out)["metadata"]["image_location"][:]) == urls


class TestConcurrentHeaderReads:
    def test_local_files_get_one_worker(self):
        # Local headers come back in a millisecond; threads would only contend
        assert resolve_header_workers(1000, remote=False) == 1

    def test_remote_is_capped_by_machine_and_by_politeness(self, monkeypatch):
        monkeypatch.setattr(
            "survey2zarray.stores.survey_store.build.os.cpu_count", lambda: 64
        )
        assert resolve_header_workers(1000, remote=True) == HEADER_WORKERS
        # A small machine never gets more threads than it has cores
        monkeypatch.setattr(
            "survey2zarray.stores.survey_store.build.os.cpu_count", lambda: 2
        )
        assert resolve_header_workers(1000, remote=True) == 2
        # Nor are there more workers than files to read
        assert resolve_header_workers(1, remote=True) == 1

    def test_explicit_workers_win_but_stay_sane(self):
        assert resolve_header_workers(1000, remote=False, max_workers=4) == 4
        assert resolve_header_workers(1000, remote=True, max_workers=0) == 1
        assert resolve_header_workers(3, remote=True, max_workers=16) == 3

    def test_concurrent_reads_match_sequential_ones(self, continuum_fits_dir):
        fits_dir, _ = continuum_fits_dir
        paths = sorted(str(p) for p in fits_dir.glob("*.fits"))
        one, one_images, one_meta = build_header_index(paths, max_workers=1)
        many, many_images, many_meta = build_header_index(paths, max_workers=4)
        assert [m["image_location"] for m in one] == [m["image_location"] for m in many]
        assert one_images.equals(many_images)
        assert one_meta.equals(many_meta)

    def test_order_follows_the_input_not_the_reads(self, continuum_fits_dir):
        fits_dir, _ = continuum_fits_dir
        # Reversed, so input order and any sorted order disagree
        paths = sorted((str(p) for p in fits_dir.glob("*.fits")), reverse=True)
        index, images, _ = build_header_index(paths, max_workers=4)
        assert [m["image_location"] for m in index] == paths
        assert list(images["image_location"]) == paths

    def test_an_unreadable_file_among_good_ones_is_still_recorded(self, tmp_path):
        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        make_continuum_fits(fits_dir / "a.fits")
        (fits_dir / "b.fits").write_bytes(b"not a FITS file")
        make_continuum_fits(fits_dir / "c.fits", ra=IMAGE_RA + 0.02)
        paths = sorted(str(p) for p in fits_dir.glob("*.fits"))
        index, images, _ = build_header_index(paths, max_workers=4)

        assert len(index) == 2
        assert list(images["converted"]) == [True, False, True]
        # The index a row points at is its position among the usable images, and
        # the unusable one has none (pandas holds the gap as NaN)
        assert images.loc[[0, 2], "array_index"].tolist() == [0, 1]
        assert np.isnan(images.loc[1, "array_index"])
        assert images.loc[1, "error"]


class TestCompressedInputs:
    def gzipped_image(self, folder, name="image.fits.gz", shape_yx=(64, 64)):
        import gzip
        import shutil

        path, data = make_continuum_fits(folder / "plain.fits", shape_yx=shape_yx)
        with open(path, "rb") as raw, gzip.open(folder / name, "wb") as packed:
            shutil.copyfileobj(raw, packed)
        Path(path).unlink()
        return folder / name, data

    def test_copy_mode_reads_compressed_files(self, tmp_path, caplog):
        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        _, data = self.gzipped_image(fits_dir)
        out = tmp_path / "store"
        with caplog.at_level("WARNING"):
            build_survey_store(fits_dir, out, mode="copy")
        assert "compressed" in caplog.text
        stored = open_store(out)["data"][0]
        np.testing.assert_allclose(stored, np.squeeze(data), rtol=1e-6)

    def test_copy_mode_finds_files_with_no_fits_suffix(self, tmp_path):
        # NVSS ships its maps as C0000M04.gz, with no .fits anywhere in the name
        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        self.gzipped_image(fits_dir, name="C0000M04.gz")
        build_survey_store(fits_dir, tmp_path / "store", mode="copy")
        assert open_store(tmp_path / "store").attrs["n_images"] == 1

    def test_reference_mode_refuses_compressed_files(self, tmp_path):
        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        path, _ = self.gzipped_image(fits_dir)
        # A byte map would address the compressed bytes and read back as noise,
        # so the file is skipped; with nothing left, the build fails
        with pytest.raises(RuntimeError, match="No usable FITS files"):
            build_survey_store([str(path)], tmp_path / "store", mode="reference")

    def test_a_compressed_file_among_readable_ones_is_skipped(self, tmp_path, caplog):
        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        good, _ = make_continuum_fits(fits_dir / "good.fits")
        packed, _ = self.gzipped_image(fits_dir, name="packed.fits.gz")
        with caplog.at_level("WARNING"):
            build_survey_store(
                [str(good), str(packed)], tmp_path / "store", mode="reference"
            )
        assert "copy mode" in caplog.text
        assert open_store(tmp_path / "store").attrs["n_images"] == 1


class TestStokesAssumption:
    def test_reported_once_for_the_survey_not_once_per_file(self, tmp_path, caplog):
        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        # Five images, none carrying a STOKES axis
        for i in range(5):
            make_continuum_fits(fits_dir / f"image{i}.fits", ra=IMAGE_RA + 0.02 * i)
        with caplog.at_level("WARNING"):
            build_survey_store(fits_dir, tmp_path / "store", mode="copy")

        said = [r.message for r in caplog.records if "STOKES" in r.message]
        assert said == ["No STOKES axis in 5 of 5 files, so Stokes I is assumed."]

    def test_nothing_said_when_the_axis_is_there(self, tmp_path, caplog):
        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        make_continuum_4d_fits(fits_dir / "image.fits")
        with caplog.at_level("WARNING"):
            build_survey_store(fits_dir, tmp_path / "store", mode="copy")
        assert not [r.message for r in caplog.records if "STOKES" in r.message]

    def test_the_count_is_of_the_files_that_lack_it(self, tmp_path, caplog):
        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        make_continuum_4d_fits(fits_dir / "with.fits")
        make_continuum_fits(fits_dir / "without.fits", ra=IMAGE_RA + 0.02)
        with caplog.at_level("WARNING"):
            build_survey_store(fits_dir, tmp_path / "store", mode="copy")
        assert ["No STOKES axis in 1 of 2 files, so Stokes I is assumed."] == [
            r.message for r in caplog.records if "STOKES" in r.message
        ]

    def test_the_stored_metadata_does_not_carry_it(self, tmp_path):
        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        make_continuum_fits(fits_dir / "image.fits")
        out = tmp_path / "store"
        build_survey_store(fits_dir, out, mode="copy")
        assert "stokes_assumed" not in list(open_store(out)["metadata"].array_keys())


class TestInputForms:
    """Either access mode accepts any way of naming the files."""

    def paths(self, fits_dir):
        return sorted(str(p) for p in fits_dir.glob("*.fits"))

    def test_a_directory_is_searched(self, continuum_fits_dir):
        fits_dir, _ = continuum_fits_dir
        assert resolve_fits_list(fits_dir) == self.paths(fits_dir)

    def test_a_list_is_taken_as_given(self, continuum_fits_dir):
        fits_dir, _ = continuum_fits_dir
        # Reversed, to show the order given is the order kept
        listed = self.paths(fits_dir)[::-1]
        assert resolve_fits_list(listed) == listed

    def test_a_text_file_lists_locations(self, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        listing = tmp_path / "images.csv"
        listing.write_text("\n".join(self.paths(fits_dir)) + "\n")
        assert resolve_fits_list(listing) == self.paths(fits_dir)

    def test_a_single_fits_file_is_one_image(self, continuum_fits_dir):
        fits_dir, _ = continuum_fits_dir
        one = self.paths(fits_dir)[0]
        # Not read as a list of locations, which is what its contents are not
        assert resolve_fits_list(one) == [one]

    def test_urls_need_no_directory(self):
        urls = ["https://host/a.fits", "https://host/b.fits"]
        assert resolve_fits_list(urls) == urls

    @pytest.mark.parametrize("mode", ["copy", "reference"])
    def test_both_modes_build_from_a_directory(
        self, continuum_fits_dir, tmp_path, mode
    ):
        fits_dir, _ = continuum_fits_dir
        out = tmp_path / f"{mode}_store"
        build_survey_store(fits_dir, out, mode=mode)
        assert open_store(out).attrs["access_mode"] == mode

    @pytest.mark.parametrize("mode", ["copy", "reference"])
    def test_both_modes_build_from_a_list_file(
        self, continuum_fits_dir, tmp_path, mode
    ):
        fits_dir, _ = continuum_fits_dir
        listing = tmp_path / "images.txt"
        listing.write_text("\n".join(self.paths(fits_dir)) + "\n")
        out = tmp_path / f"{mode}_store"
        build_survey_store(listing, out, mode=mode)
        assert open_store(out).attrs["n_images"] == len(self.paths(fits_dir))

    def test_an_unknown_mode_is_refused(self, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        with pytest.raises(ValueError, match="Expected mode"):
            build_survey_store(fits_dir, tmp_path / "store", mode="sideways")

    def test_a_missing_input_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_fits_list(tmp_path / "nothing_here")
        with pytest.raises(TypeError):
            resolve_fits_list(42)


@pytest.fixture
def bucket(continuum_fits_dir):
    """Serve continuum FITS files through an in-memory filesystem."""
    import fsspec

    fits_dir, _ = continuum_fits_dir
    filesystem = fsspec.filesystem("memory")
    for path in sorted(fits_dir.glob("*.fits")):
        filesystem.pipe_file(f"/survey/{path.name}", path.read_bytes())
    yield filesystem
    filesystem.store.clear()
    filesystem.pseudo_dirs.clear()


class TestRemoteInputForms:
    """The forms hold wherever the files are, resolved through fsspec."""

    def names(self, bucket):
        return sorted(
            bucket.unstrip_protocol(p) for p in bucket.ls("/survey", detail=False)
        )

    def test_a_prefix_is_listed(self, bucket):
        assert resolve_fits_list("memory://survey") == self.names(bucket)

    def test_a_single_served_file_is_one_image(self, bucket):
        one = self.names(bucket)[0]
        assert resolve_fits_list(one) == [one]

    def test_a_served_text_file_lists_locations(self, bucket):
        bucket.pipe_file(
            "/listing.txt", ("\n".join(self.names(bucket)) + "\n").encode()
        )
        assert resolve_fits_list("memory://listing.txt") == self.names(bucket)

    def test_a_prefix_holding_no_fits_is_refused(self, bucket):
        bucket.pipe_file("/empty/notes.md", b"nothing to see")
        with pytest.raises(FileNotFoundError, match="No FITS files"):
            resolve_fits_list("memory://empty")

    def test_a_missing_location_is_refused(self, bucket):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            resolve_fits_list("memory://survey/absent.fits.list")

    @pytest.mark.parametrize("mode", ["copy", "reference"])
    def test_both_modes_build_from_a_prefix(self, bucket, tmp_path, mode):
        build_survey_store("memory://survey", tmp_path / "store", mode=mode)
        assert open_store(tmp_path / "store").attrs["n_images"] == len(
            self.names(bucket)
        )


class TestStorageOptions:
    """Verify stores exclude authentication data from recorded access options."""

    def test_options_reach_the_filesystem_that_reads_the_headers(
        self, bucket, tmp_path
    ):
        seen = {}
        import fsspec

        real = fsspec.filesystem

        def watched(protocol, **options):
            seen.update(options)
            return real(protocol)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(fsspec, "filesystem", watched)
            build_survey_store(
                "memory://survey",
                tmp_path / "store",
                storage_options={"memory": {"anon": True}},
            )
        assert seen.get("anon") is True

    def test_options_that_name_another_protocol_are_not_applied(self, bucket, tmp_path):
        # Keyed by protocol, so a survey spanning two archives can carry both
        build_survey_store(
            "memory://survey",
            tmp_path / "store",
            storage_options={"s3": {"anon": True}},
        )
        assert open_store(tmp_path / "store").attrs["storage_options"] == {
            "s3": {"anon": True}
        }

    def test_the_store_records_how_to_reach_the_archive(self, bucket, tmp_path):
        build_survey_store(
            "memory://survey",
            tmp_path / "store",
            mode="reference",
            storage_options={"memory": {"anon": True}},
        )
        store = SurveyStore.open(tmp_path / "store")
        assert store.filesystem_options("memory") == {"anon": True}

    def test_a_credential_is_never_written_into_a_store(self, bucket, tmp_path, caplog):
        build_survey_store(
            "memory://survey",
            tmp_path / "store",
            storage_options={"memory": {"anon": True, "secret": "hunter2"}},
        )
        recorded = open_store(tmp_path / "store").attrs["storage_options"]
        assert recorded == {"memory": {"anon": True}}
        assert "hunter2" not in str(recorded)

    def test_dropping_a_credential_is_said_out_loud(self, caplog):
        with caplog.at_level("WARNING"):
            kept = persistable_options(
                {"s3": {"key": "AKIA", "region_name": "eu-west-1"}}
            )
        assert kept == {"s3": {"region_name": "eu-west-1"}}
        assert "s3.key" in caplog.text

    def test_a_reader_supplies_what_the_store_could_not_hold(self, bucket, tmp_path):
        build_survey_store(
            "memory://survey",
            tmp_path / "store",
            mode="reference",
            storage_options={"memory": {"anon": True}},
        )
        store = SurveyStore.open(
            tmp_path / "store", storage_options={"memory": {"secret": "hunter2"}}
        )
        # The reader's own options join what the build recorded
        assert store.filesystem_options("memory") == {"anon": True, "secret": "hunter2"}

    def test_a_reader_overrides_what_the_store_recorded(self, bucket, tmp_path):
        build_survey_store(
            "memory://survey",
            tmp_path / "store",
            mode="reference",
            storage_options={"memory": {"anon": True}},
        )
        store = SurveyStore.open(
            tmp_path / "store", storage_options={"memory": {"anon": False}}
        )
        assert store.filesystem_options("memory") == {"anon": False}

    def test_a_store_built_without_options_records_none(
        self, continuum_fits_dir, tmp_path
    ):
        fits_dir, _ = continuum_fits_dir
        build_survey_store(fits_dir, tmp_path / "store")
        assert "storage_options" not in open_store(tmp_path / "store").attrs

    def test_pixels_read_back_from_a_served_reference_store(
        self, bucket, tmp_path, continuum_fits_dir
    ):
        _, images = continuum_fits_dir
        build_survey_store("memory://survey", tmp_path / "store", mode="reference")
        store = SurveyStore.open(tmp_path / "store")
        first = str(store.image_location[0]).rsplit("/", 1)[-1].removesuffix(".fits")
        assert np.allclose(
            store.get_cutout_region(0, 10, 20, 5, 15), images[first][5:15, 10:20]
        )


class TestOneBadFileDoesNotEndABuild:
    def test_a_failed_open_is_retried_for_served_files(
        self, monkeypatch, continuum_fits_dir
    ):
        fits_dir, _ = continuum_fits_dir
        real = str(sorted(fits_dir.glob("*.fits"))[0])
        url = "https://host/image.fits"
        calls = []

        def flaky(path, require_referenceable=False, storage_options=None):
            calls.append(path)
            if len(calls) < 3:  # busy archive, then it serves
                raise FITSOpenError(f"Failed to open {path}")
            return extract_metadata(real)

        monkeypatch.setattr(
            "survey2zarray.stores.survey_store.build.extract_metadata", flaky
        )
        monkeypatch.setattr("survey2zarray.stores.survey_store.build.RETRY_PAUSE", 0.0)
        location, metadata, error = read_header(url, require_referenceable=True)
        assert error is None and metadata is not None
        assert len(calls) == 3

    def test_retries_give_up_and_report(self, monkeypatch):
        def always_busy(path, require_referenceable=False, storage_options=None):
            raise FITSOpenError(f"Failed to open {path}")

        monkeypatch.setattr(
            "survey2zarray.stores.survey_store.build.extract_metadata", always_busy
        )
        monkeypatch.setattr("survey2zarray.stores.survey_store.build.RETRY_PAUSE", 0.0)
        _, metadata, error = read_header(
            "https://host/image.fits", require_referenceable=True
        )
        assert metadata is None and isinstance(error, FITSOpenError)

    def test_local_files_are_not_retried(self, monkeypatch, tmp_path):
        calls = []

        def missing(path, require_referenceable=False, storage_options=None):
            calls.append(path)
            raise FITSOpenError(f"Failed to open {path}")

        monkeypatch.setattr(
            "survey2zarray.stores.survey_store.build.extract_metadata", missing
        )
        read_header(str(tmp_path / "gone.fits"), require_referenceable=False)
        # Nothing about a file on disk will have changed a second later
        assert len(calls) == 1

    def test_a_file_that_cannot_be_read_is_not_retried(self, monkeypatch):
        calls = []

        def unreadable(path, require_referenceable=False, storage_options=None):
            calls.append(path)
            raise InvalidFITSError("No image HDU found")

        monkeypatch.setattr(
            "survey2zarray.stores.survey_store.build.extract_metadata", unreadable
        )
        read_header("https://host/image.fits", require_referenceable=True)
        assert len(calls) == 1

    def test_an_unexpected_error_skips_the_file_rather_than_the_build(
        self, monkeypatch, continuum_fits_dir, tmp_path
    ):
        fits_dir, _ = continuum_fits_dir
        paths = sorted(str(p) for p in fits_dir.glob("*.fits"))
        real = extract_metadata

        def explodes_on_the_second(
            path, require_referenceable=False, storage_options=None
        ):
            if path == paths[1]:
                raise ZeroDivisionError("something a library did")
            return real(path, require_referenceable=require_referenceable)

        monkeypatch.setattr(
            "survey2zarray.stores.survey_store.build.extract_metadata",
            explodes_on_the_second,
        )
        out = tmp_path / "store"
        build_survey_store(paths, out, mode="reference")
        # The rest of the survey survives the one file that misbehaved
        assert open_store(out).attrs["n_images"] == len(paths) - 1


class TestQuietProgress:
    def test_failures_are_summarised_after_the_bar_not_through_it(
        self, tmp_path, caplog
    ):
        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        good, _ = make_continuum_fits(fits_dir / "good.fits")
        for name in ("bad1.fits", "bad2.fits"):
            (fits_dir / name).write_bytes(b"not a FITS file")
        with caplog.at_level("WARNING"):
            build_survey_store(fits_dir, tmp_path / "store", mode="copy")

        said = [
            r.message
            for r in caplog.records
            if "Skip" in r.message or "FITSOpenError" in r.message
        ]
        # One summary and one line per kind of failure, rather than one per file
        assert said[0] == "Skipped 2 of 3 files, which image_valid.csv records."
        assert said[1].startswith("2 FITSOpenError. Example ")
        assert not any(r.message.startswith("Skipping file") for r in caplog.records)

    def test_astropy_header_warnings_stay_quiet_across_the_pool(
        self, continuum_fits_dir, tmp_path, recwarn
    ):
        from astropy.io.fits.verify import VerifyWarning
        from astropy.wcs import FITSFixedWarning

        fits_dir, _ = continuum_fits_dir
        paths = sorted(str(p) for p in fits_dir.glob("*.fits"))
        build_header_index(paths, max_workers=4)
        leaked = [
            w
            for w in recwarn
            if issubclass(w.category, (VerifyWarning, FITSFixedWarning))
        ]
        assert not leaked

    def test_the_global_filters_are_left_as_they_were(
        self, continuum_fits_dir, tmp_path
    ):
        import warnings as warnings_module

        fits_dir, _ = continuum_fits_dir
        before = list(warnings_module.filters)
        build_survey_store(fits_dir, tmp_path / "store", mode="copy")
        assert list(warnings_module.filters) == before


class TestFailuresAreSweptUp:
    def flaky_reader(self, real, failing, fail_times):
        """Return a reader with temporary per-file failures."""
        seen = {}

        def read(path, require_referenceable=False, storage_options=None):
            if path in failing:
                seen[path] = seen.get(path, 0) + 1
                if seen[path] <= fail_times:
                    raise FITSOpenError(f"Failed to open {path}")
            return real(path, require_referenceable=require_referenceable)

        return read

    def test_a_burst_of_refusals_is_recovered_after_the_pass(
        self, monkeypatch, continuum_fits_dir, tmp_path
    ):
        fits_dir, _ = continuum_fits_dir
        paths = sorted(str(p) for p in fits_dir.glob("*.fits"))
        urls = ["https://host/" + Path(p).name for p in paths]
        by_url = dict(zip(urls, paths, strict=True))

        # The middle file is refused for longer than its attempts last, so only
        # the sweep after the pass recovers it
        refused = {urls[1]}
        seen = {}

        def read(path, require_referenceable=False, storage_options=None):
            seen[path] = seen.get(path, 0) + 1
            if path in refused and seen[path] <= HEADER_ATTEMPTS:
                raise FITSOpenError(f"Failed to open {path}")
            return extract_metadata(by_url[path], require_referenceable=False)

        monkeypatch.setattr(
            "survey2zarray.stores.survey_store.build.extract_metadata", read
        )
        monkeypatch.setattr("survey2zarray.stores.survey_store.build.RETRY_PAUSE", 0.0)
        index, images, _ = build_header_index(urls, max_workers=2)

        assert len(index) == len(urls)
        assert list(images["converted"]) == [True] * len(urls)

    def test_order_survives_the_sweep(self, monkeypatch, continuum_fits_dir):
        fits_dir, _ = continuum_fits_dir
        paths = sorted(str(p) for p in fits_dir.glob("*.fits"))
        urls = ["https://host/" + Path(p).name for p in paths]
        by_url = dict(zip(urls, paths, strict=True))
        seen = {}

        def read(path, require_referenceable=False, storage_options=None):
            seen[path] = seen.get(path, 0) + 1
            if path == urls[0] and seen[path] <= HEADER_ATTEMPTS:
                raise FITSOpenError(f"Failed to open {path}")
            return extract_metadata(by_url[path], require_referenceable=False)

        monkeypatch.setattr(
            "survey2zarray.stores.survey_store.build.extract_metadata", read
        )
        monkeypatch.setattr("survey2zarray.stores.survey_store.build.RETRY_PAUSE", 0.0)
        index, images, _ = build_header_index(urls, max_workers=2)
        # The recovered file keeps its place rather than landing at the end
        assert [m["image_location"] for m in index] == paths
        assert list(images["image_location"]) == urls

    def test_a_file_that_stays_broken_is_still_skipped(
        self, monkeypatch, continuum_fits_dir
    ):
        fits_dir, _ = continuum_fits_dir
        paths = sorted(str(p) for p in fits_dir.glob("*.fits"))
        urls = ["https://host/" + Path(p).name for p in paths]
        by_url = dict(zip(urls, paths, strict=True))

        def read(path, require_referenceable=False, storage_options=None):
            if path == urls[1]:
                raise FITSOpenError(f"Failed to open {path}")
            return extract_metadata(by_url[path], require_referenceable=False)

        monkeypatch.setattr(
            "survey2zarray.stores.survey_store.build.extract_metadata", read
        )
        monkeypatch.setattr("survey2zarray.stores.survey_store.build.RETRY_PAUSE", 0.0)
        index, images, _ = build_header_index(urls, max_workers=2)
        assert len(index) == len(urls) - 1
        assert list(images["converted"]) == [True, False] + [True] * (len(urls) - 2)
        assert (
            list(images["array_index"])[0] == 0 and list(images["array_index"])[2] == 1
        )


class TestTruncationWarnings:
    def test_a_genuinely_short_file_still_says_so_when_pixels_are_read(
        self, tmp_path, recwarn
    ):
        from astropy.utils.exceptions import AstropyUserWarning

        from survey2zarray.stores.survey_store.fits import quiet_fits_warnings

        path, _ = make_continuum_fits(tmp_path / "image.fits", shape_yx=(64, 64))
        cut = tmp_path / "cut.fits"
        cut.write_bytes(
            path.read_bytes()[:6000]
            if hasattr(path, "read_bytes")
            else open(path, "rb").read(6000)
        )
        # The pixel pass makes no such promise, so a short file is still flagged
        with quiet_fits_warnings():
            with astropy_fits.open(cut) as hdul:
                hdul[0].header["NAXIS1"]
        assert [
            w
            for w in recwarn
            if issubclass(w.category, AstropyUserWarning)
            and "truncated" in str(w.message)
        ]


class TestResumableHeaderPass:
    """Verify resumed header passes reuse completed archive reads."""

    def locations(self, continuum_fits_dir):
        fits_dir, _ = continuum_fits_dir
        return sorted(str(p) for p in fits_dir.glob("*.fits"))

    def test_headers_are_recorded_as_they_are_read(self, continuum_fits_dir, tmp_path):
        from survey2zarray.stores.survey_store.build import (
            build_header_index,
            read_header_checkpoint,
        )

        checkpoint = tmp_path / "headers.jsonl"
        locations = self.locations(continuum_fits_dir)
        build_header_index(locations, checkpoint=checkpoint)

        recorded = read_header_checkpoint(checkpoint)
        assert set(recorded) == set(locations)

    def test_a_resumed_pass_reads_only_what_is_missing(
        self, continuum_fits_dir, tmp_path, monkeypatch
    ):
        from survey2zarray.stores.survey_store import build as build_module

        checkpoint = tmp_path / "headers.jsonl"
        locations = self.locations(continuum_fits_dir)
        build_module.build_header_index(locations[:1], checkpoint=checkpoint)

        asked = []
        real = build_module.read_header
        monkeypatch.setattr(
            build_module,
            "read_header",
            lambda loc, req, opts=None: (asked.append(loc), real(loc, req, opts))[1],
        )
        index, _, _ = build_module.build_header_index(locations, checkpoint=checkpoint)

        assert asked == locations[1:]  # the first was not re-read
        assert len(index) == len(locations)  # but it is in the result
        assert [m["image_location"] for m in index] == locations

    def test_a_half_written_record_is_dropped_not_fatal(
        self, continuum_fits_dir, tmp_path
    ):
        from survey2zarray.stores.survey_store.build import (
            build_header_index,
            read_header_checkpoint,
        )

        checkpoint = tmp_path / "headers.jsonl"
        locations = self.locations(continuum_fits_dir)
        build_header_index(locations, checkpoint=checkpoint)

        # As a run killed mid-write would leave it
        text = checkpoint.read_text()
        checkpoint.write_text(text + '{"image_location": "half-writ')
        recovered = read_header_checkpoint(checkpoint)
        assert set(recovered) == set(locations)

    def test_the_checkpoint_is_cleared_once_the_store_exists(
        self, continuum_fits_dir, tmp_path
    ):
        from survey2zarray.stores.survey_store.build import (
            HEADER_CHECKPOINT_SUFFIX,
            build_survey_store,
        )

        fits_dir, _ = continuum_fits_dir
        out = tmp_path / "survey.zarr"
        build_survey_store(fits_dir, out, mode="copy")
        assert not Path(str(out) + HEADER_CHECKPOINT_SUFFIX).exists()

    def test_without_resume_the_checkpoint_is_discarded(
        self, continuum_fits_dir, tmp_path, monkeypatch
    ):
        from survey2zarray.stores.survey_store import build as build_module

        fits_dir, _ = continuum_fits_dir
        out = tmp_path / "survey.zarr"
        checkpoint = Path(str(out) + build_module.HEADER_CHECKPOINT_SUFFIX)
        checkpoint.write_text('{"image_location": "/gone.fits"}\n')

        build_module.build_survey_store(fits_dir, out, mode="copy")
        # The stale record was not folded into the store
        store = SurveyStore.open(out)
        assert "/gone.fits" not in [str(v) for v in store.image_location]
