import json
import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from survey2zarray.stores.store import open_store
from survey2zarray.stores.survey_store.build import (
    build_header_index,
    build_survey_store,
)
from survey2zarray.stores.survey_store.edit import append_tiles
from survey2zarray.stores.survey_store.fits import (
    extract_metadata,
    extract_stokes_i,
    stokes_i_planes,
)
from survey2zarray.stores.survey_store.references import (
    HEADER_BYTES,
    REFERENCE_MAP_PATH,
    UnreferenceableFITSError,
    check_referenceable,
    compression_suffix,
    probe_size,
    raw_array_spec,
    region_ranges,
    remote_header_buffer,
    storage_protocol,
    write_json_atomically,
)
from survey2zarray.stores.survey_store.survey_store import SurveyStore
from survey2zarray.stores.transaction import begin_edit
from tests.conftest import (
    IMAGE_DEC,
    IMAGE_RA,
    make_continuum_4d_fits,
    make_continuum_fits,
    make_spectral_fits,
)


def expected_region(path, y0, y1, x0, x1):
    """Read the expected region with Astropy."""
    with fits.open(path) as hdul:
        full = extract_stokes_i(np.asarray(hdul[0].data), hdul[0].header)
    return full[..., y0:y1, x0:x1]


@pytest.fixture
def range_server(tmp_path):
    """Serve local HTTP byte ranges and count transferred bytes."""
    served = {"bytes": 0, "requests": 0}

    class Handler(SimpleHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def end_headers(self):
            self.send_header("Accept-Ranges", "bytes")
            super().end_headers()

        def do_GET(self):
            header = self.headers.get("Range")
            path = self.translate_path(self.path)
            if header is None or not os.path.isfile(path):
                return super().do_GET()
            size = os.path.getsize(path)
            _, _, span = header.partition("=")
            first, _, last = span.partition("-")
            if first:
                start, end = int(first), int(last) if last else size - 1
            else:
                start, end = size - int(last), size - 1
            end = min(end, size - 1)
            length = end - start + 1
            served["bytes"] += length
            served["requests"] += 1
            self.send_response(206)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            with open(path, "rb") as handle:
                handle.seek(start)
                self.wfile.write(handle.read(length))

    class Quiet(ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):
            pass

    httpd = Quiet(("127.0.0.1", 0), partial(Handler, directory=str(tmp_path)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", served
    httpd.shutdown()


class TestReferenceBuilding:
    def test_rows_of_a_plane_are_one_range(self):
        # 3 planes of 64 x 48 float32, pixels starting at 2880
        ranges = region_ranges(2880, (1, 3, 64, 48), 4, planes=[1], y0=10, y1=20).fetch
        row, plane = 48 * 4, 64 * 48 * 4
        assert ranges == [(2880 + plane + 10 * row, 2880 + plane + 20 * row)]

    def test_a_range_is_split_only_to_bound_memory(self):
        row = 48 * 4
        # 10 rows at a time, so 25 rows is three pieces, adjacent and in order
        ranges = region_ranges(
            0, (64, 48), 4, planes=[0], y0=0, y1=25, max_bytes=10 * row
        ).fetch
        assert ranges == [(0, 10 * row), (10 * row, 20 * row), (20 * row, 25 * row)]
        assert ranges[-1][1] == 25 * row

    def test_ranges_cover_every_plane_asked_for(self):
        ranges = region_ranges(0, (2, 3, 8, 4), 4, planes=[0, 4], y0=0, y1=8).fetch
        plane = 8 * 4 * 4
        assert ranges == [(0, plane), (4 * plane, 5 * plane)]

    def test_stokes_planes_selected_from_the_header(self):
        # STOKES on FITS axis 3 of a 3D image: I, Q, U, V in order
        header = fits.Header(
            {
                "NAXIS": 3,
                "NAXIS1": 8,
                "NAXIS2": 8,
                "NAXIS3": 4,
                "CTYPE3": "STOKES",
                "CRVAL3": 1.0,
                "CRPIX3": 1.0,
                "CDELT3": 1.0,
            }
        )
        planes, kept = stokes_i_planes((4, 8, 8), header)
        assert planes == [0] and kept == ()

        # Same, but stored V, U, Q, I so Stokes I is the last plane
        header["CRVAL3"], header["CDELT3"] = 4.0, -1.0
        planes, _ = stokes_i_planes((4, 8, 8), header)
        assert planes == [3]

    def test_every_channel_of_the_stokes_i_plane_is_kept(self):
        # (stokes, freq, y, x) with 2 polarisations and 3 channels
        header = fits.Header(
            {
                "NAXIS": 4,
                "NAXIS1": 8,
                "NAXIS2": 8,
                "NAXIS3": 3,
                "NAXIS4": 2,
                "CTYPE3": "FREQ",
                "CTYPE4": "STOKES",
                "CRVAL4": 1.0,
                "CRPIX4": 1.0,
                "CDELT4": 1.0,
            }
        )
        planes, kept = stokes_i_planes((2, 3, 8, 8), header)
        # Stokes I is the first of the two, so its three channels are planes 0-2
        assert planes == [0, 1, 2] and kept == (3,)

    def test_all_planes_kept_when_there_is_no_stokes_axis(self):
        header = fits.Header(
            {"NAXIS": 3, "NAXIS1": 8, "NAXIS2": 8, "NAXIS3": 3, "CTYPE3": "FREQ"}
        )
        planes, kept = stokes_i_planes((3, 8, 8), header)
        assert planes == [0, 1, 2] and kept == (3,)

    def test_compressed_files_are_refused(self):
        assert compression_suffix("/data/image.fits") is None
        assert compression_suffix("ftp://host/nvss/MAPS/C0000M04.gz") == ".gz"
        assert compression_suffix("https://host/image.FITS.GZ") == ".gz"
        with pytest.raises(UnreferenceableFITSError, match="compression"):
            check_referenceable("ftp://host/nvss/MAPS/C0000M04.gz")
        # Uncompressed files pass through
        check_referenceable("https://host/image.fits")

    def test_scaled_pixels_are_refused(self, tmp_path):
        path = tmp_path / "scaled.fits"
        data = np.random.default_rng(0).normal(size=(32, 32)).astype(np.float32)
        hdu = fits.PrimaryHDU(data=data)
        hdu.scale("int16", bscale=1e-5, bzero=0.0)
        hdu.writeto(path)
        with fits.open(path) as hdul:
            with pytest.raises(UnreferenceableFITSError, match="scaling"):
                raw_array_spec(hdul[0].header)

    def test_protocol_inferred_from_location(self):
        assert storage_protocol("/data/image.fits") == "file"
        assert storage_protocol("http://host/image.fits") == "http"
        assert storage_protocol("s3://bucket/image.fits") == "s3"


class TestReferenceReads:
    @pytest.mark.parametrize(
        "maker,kwargs",
        [
            (make_continuum_fits, {}),
            (make_continuum_4d_fits, {}),
            (make_continuum_4d_fits, {"n_stokes": 4}),
            (make_spectral_fits, {}),
        ],
    )
    def test_matches_astropy_for_every_shape(self, maker, kwargs, tmp_path):
        # Degenerate axes and a real Stokes axis are where a byte map is most
        # likely to slice the wrong axis
        folder = tmp_path / "fits"
        folder.mkdir()
        path, _ = maker(folder / "image.fits", ra=IMAGE_RA, dec=IMAGE_DEC, **kwargs)
        store = tmp_path / "store"
        build_survey_store([str(path)], store, mode="reference")
        region = np.asarray(SurveyStore.open(store).get_cutout_region(0, 5, 25, 5, 25))
        np.testing.assert_array_equal(region, expected_region(path, 5, 25, 5, 25))

    def test_reads_the_bottom_of_a_tile_that_divides_awkwardly(self, tmp_path):
        # 66 rows against a 10 row budget: the chunk has to be 11 rows, since
        # blocks of 10 leave a 6 row remainder, and that final block reads back
        # as a buffer zarr cannot reshape. Only a cutout reaching the bottom of
        # the image ever touches it.
        path, _ = make_continuum_fits(tmp_path / "image.fits", shape_yx=(66, 48))
        store = tmp_path / "ref"
        build_survey_store([str(path)], store, mode="reference", chunk_bytes=2000)
        opened = SurveyStore.open(store)
        np.testing.assert_array_equal(
            np.asarray(opened.get_cutout_region(0, 4, 40, 50, 66)),
            expected_region(path, 50, 66, 4, 40),
        )
        # And the whole image, which crosses every block boundary
        np.testing.assert_array_equal(
            np.asarray(opened.get_cutout_region(0, 0, 48, 0, 66)),
            expected_region(path, 0, 66, 0, 48),
        )

    def test_copy_and_reference_agree(self, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        paths = sorted(str(p) for p in fits_dir.glob("*.fits"))
        copied, referenced = tmp_path / "copy", tmp_path / "ref"
        build_survey_store(fits_dir, copied, mode="copy")
        build_survey_store(paths, referenced, mode="reference")
        a, b = SurveyStore.open(copied), SurveyStore.open(referenced)
        assert a.access_mode == "copy" and b.access_mode == "reference"
        for index in range(a.n_images):
            np.testing.assert_array_equal(
                np.asarray(a.get_cutout_region(index, 4, 20, 4, 20)),
                np.asarray(b.get_cutout_region(index, 4, 20, 4, 20)),
            )

    def test_a_one_pixel_wide_region_keeps_both_celestial_axes(self, tmp_path):
        # Squeezing every degenerate axis would drop a column of one pixel and
        # hand back a 1D region, which the copy path never does
        fits_dir = tmp_path / "fits"
        fits_dir.mkdir()
        path, _ = make_continuum_fits(fits_dir / "image.fits")
        copied, referenced = tmp_path / "copy", tmp_path / "ref"
        build_survey_store([str(path)], copied, mode="copy")
        build_survey_store([str(path)], referenced, mode="reference")
        for x0, x1, y0, y1 in ((5, 6, 5, 25), (5, 25, 5, 6), (5, 6, 5, 6)):
            region = np.asarray(
                SurveyStore.open(referenced).get_cutout_region(0, x0, x1, y0, y1)
            )
            assert region.shape == (y1 - y0, x1 - x0)
            np.testing.assert_array_equal(
                region,
                np.asarray(
                    SurveyStore.open(copied).get_cutout_region(0, x0, x1, y0, y1)
                ),
            )

    def test_reference_store_holds_no_pixels(self, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        paths = sorted(str(p) for p in fits_dir.glob("*.fits"))
        store = tmp_path / "ref"
        build_survey_store(paths, store, mode="reference")
        assert (store / "data" / "references.json").exists()
        assert "data" not in SurveyStore.open(store).root
        # The map is a small fraction of the pixels it addresses
        mapped = sum(p.stat().st_size for p in fits_dir.glob("*.fits"))
        assert (store / "data" / "references.json").stat().st_size < mapped / 10

    def test_filesystems_are_kept_between_reads(self, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        store = tmp_path / "ref"
        build_survey_store(
            sorted(str(p) for p in fits_dir.glob("*.fits")), store, mode="reference"
        )
        survey = SurveyStore.open(store)
        survey.get_cutout_region(0, 0, 8, 0, 8)
        location = str(survey.image_location[0])
        first = survey.filesystem(location)
        survey.get_cutout_region(0, 8, 16, 8, 16)
        # One filesystem per protocol, holding the connections its files share
        assert survey.filesystem(location) is first
        assert survey.filesystem(str(survey.image_location[1])) is first

    def test_copy_store_has_no_byte_map(self, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        store = tmp_path / "copy"
        build_survey_store(fits_dir, store, mode="copy")
        with pytest.raises(RuntimeError, match="no byte map"):
            SurveyStore.open(store).image_record(0)


class TestOverHTTP:
    def test_reads_byte_ranges_not_whole_files(self, tmp_path, range_server):
        # Big enough that one chunk is a fraction of the file; below about
        # 1024 square the whole image fits in a single chunk and the test would
        # be measuring the chunk budget rather than the byte map
        base, served = range_server
        path, _ = make_continuum_fits(tmp_path / "image.fits", shape_yx=(1024, 1024))
        store = tmp_path / "store"
        build_survey_store([f"{base}/image.fits"], store, mode="reference")

        served["bytes"] = served["requests"] = 0
        region = np.asarray(
            SurveyStore.open(store).get_cutout_region(0, 400, 440, 400, 440)
        )
        np.testing.assert_array_equal(region, expected_region(path, 400, 440, 400, 440))

        # Full-width rows are unavoidable, but the rest of the image is not read,
        # and the 40 rows wanted come back in one range
        assert served["requests"] == 1
        assert served["bytes"] == 40 * 1024 * 4

    def test_only_the_stokes_i_plane_is_fetched(self, tmp_path, range_server):
        base, served = range_server
        path, _ = make_continuum_4d_fits(
            tmp_path / "quad.fits", shape_yx=(256, 256), n_stokes=4
        )
        store = tmp_path / "store"
        build_survey_store([f"{base}/quad.fits"], store, mode="reference")

        served["bytes"] = served["requests"] = 0
        region = np.asarray(
            SurveyStore.open(store).get_cutout_region(0, 10, 40, 10, 40)
        )
        np.testing.assert_array_equal(region, expected_region(path, 10, 40, 10, 40))

        # 30 rows of one plane, not of all four polarisations
        assert served["requests"] == 1
        assert served["bytes"] == 30 * 256 * 4

    def test_a_large_read_is_split_into_adjacent_pieces(self, tmp_path, range_server):
        base, served = range_server
        path, _ = make_continuum_fits(tmp_path / "image.fits", shape_yx=(256, 256))
        store = tmp_path / "store"
        build_survey_store([f"{base}/image.fits"], store, mode="reference")

        served["bytes"] = served["requests"] = 0
        # 10 rows per request, so 45 rows is five pieces holding the same bytes
        survey = SurveyStore.open(store, max_request_bytes=10 * 256 * 4)
        region = np.asarray(survey.get_cutout_region(0, 0, 256, 5, 50))
        np.testing.assert_array_equal(region, expected_region(path, 5, 50, 0, 256))
        assert served["requests"] == 5
        assert served["bytes"] == 45 * 256 * 4

    def test_size_is_taken_from_a_range_request(self, tmp_path, range_server):
        base, _ = range_server
        path, _ = make_continuum_fits(tmp_path / "image.fits", shape_yx=(64, 64))
        assert probe_size(f"{base}/image.fits") == os.path.getsize(path)
        assert probe_size(f"{base}/missing.fits") is None


class TestByteMapIsCrashSafe:
    """The byte map takes part in the edit transaction.

    Rolling the metadata back while leaving the map extended would describe
    images nothing indexes, and the store would read past its own source table.
    """

    @pytest.fixture
    def reference_store(self, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        store = tmp_path / "ref"
        build_survey_store(
            sorted(str(p) for p in fits_dir.glob("*.fits")), store, mode="reference"
        )
        return store

    def test_interrupted_append_reverts_the_map(
        self, reference_store, tmp_path, monkeypatch
    ):
        import survey2zarray.stores.survey_store.edit as edit_module

        before = json.loads((reference_store / REFERENCE_MAP_PATH).read_text())
        extra, _ = make_continuum_fits(
            tmp_path / "extra.fits",
            shape_yx=(32, 32),
            ra=IMAGE_RA + 0.05,
            dec=IMAGE_DEC,
            seed=11,
        )

        # Fail after the map has been rewritten but before the edit commits
        original = edit_module.append_metadata_arrays

        def explode(*args, **kwargs):
            raise RuntimeError("interrupted")

        monkeypatch.setattr(edit_module, "append_metadata_arrays", explode)
        with pytest.raises(RuntimeError, match="interrupted"):
            append_tiles(reference_store, [str(extra)])
        monkeypatch.setattr(edit_module, "append_metadata_arrays", original)

        # The map on disk now describes four images, and the flag is still set
        during = json.loads((reference_store / REFERENCE_MAP_PATH).read_text())
        assert len(during["images"]) == 4

        # Opening the store reverts it, map included
        survey = SurveyStore.open(reference_store)
        assert survey.n_images == 3
        after = json.loads((reference_store / REFERENCE_MAP_PATH).read_text())
        assert after == before
        assert not (reference_store / (REFERENCE_MAP_PATH + ".orig")).exists()
        # And it still reads
        np.testing.assert_array_equal(
            np.asarray(survey.get_cutout_region(0, 2, 12, 2, 12)),
            expected_region(
                sorted(Path(survey.image_location[0]).parent.glob("*.fits"))[0],
                2,
                12,
                2,
                12,
            ),
        )

    def test_successful_append_leaves_no_copy(self, reference_store, tmp_path):
        extra, _ = make_continuum_fits(
            tmp_path / "extra2.fits",
            shape_yx=(32, 32),
            ra=IMAGE_RA + 0.05,
            dec=IMAGE_DEC,
            seed=12,
        )
        append_tiles(reference_store, [str(extra)])
        assert not (reference_store / (REFERENCE_MAP_PATH + ".orig")).exists()
        assert (
            len(
                json.loads((reference_store / REFERENCE_MAP_PATH).read_text())["images"]
            )
            == 4
        )

    def test_revert_is_idempotent(self, reference_store, tmp_path):
        # A revert interrupted partway must be safe to re-run

        from survey2zarray.stores.transaction import recover_edit

        root = open_store(reference_store, mode="r+")
        before = (reference_store / REFERENCE_MAP_PATH).read_text()
        begin_edit(root, [], files=[REFERENCE_MAP_PATH])
        (reference_store / REFERENCE_MAP_PATH).write_text('{"images": []}')
        assert recover_edit(root) is True
        assert (reference_store / REFERENCE_MAP_PATH).read_text() == before
        assert recover_edit(root) is False

    def test_atomic_write_leaves_no_partial_file(self, tmp_path):
        target = tmp_path / "nested" / "map.json"
        write_json_atomically(target, {"images": [1, 2, 3]})
        assert json.loads(target.read_text()) == {"images": [1, 2, 3]}
        assert not (target.parent / "map.json.tmp").exists()


class TestAppendKeepsTheMap:
    def test_appended_tile_is_readable(self, continuum_fits_dir, tmp_path):
        fits_dir, _ = continuum_fits_dir
        store = tmp_path / "ref"
        build_survey_store(
            sorted(str(p) for p in fits_dir.glob("*.fits")), store, mode="reference"
        )
        extra, _ = make_continuum_fits(
            tmp_path / "extra.fits",
            shape_yx=(32, 32),
            ra=IMAGE_RA + 0.05,
            dec=IMAGE_DEC,
            seed=9,
        )
        append_tiles(store, [str(extra)])
        survey = SurveyStore.open(store)
        assert survey.n_images == 4
        np.testing.assert_array_equal(
            np.asarray(survey.get_cutout_region(3, 2, 20, 2, 20)),
            expected_region(extra, 2, 20, 2, 20),
        )


class TestHeaderFetching:
    def test_a_served_header_costs_one_request(self, tmp_path, range_server):
        base, served = range_server
        make_continuum_fits(tmp_path / "image.fits", shape_yx=(512, 512))
        served["bytes"] = served["requests"] = 0
        metadata = extract_metadata(f"{base}/image.fits", require_referenceable=True)
        assert metadata["pixel_shape"] == (512, 512)
        # The front of the file, once - not a size probe and a block read
        assert served["requests"] == 1
        assert served["bytes"] <= HEADER_BYTES

    def test_it_falls_back_when_the_front_is_not_enough(
        self, tmp_path, range_server, monkeypatch
    ):
        base, served = range_server
        path, _ = make_continuum_fits(tmp_path / "image.fits", shape_yx=(512, 512))
        # Too little to hold a header, so the cheap source cannot work
        monkeypatch.setattr(
            "survey2zarray.stores.survey_store.references.HEADER_BYTES", 64
        )
        served["bytes"] = served["requests"] = 0
        metadata = extract_metadata(f"{base}/image.fits", require_referenceable=True)
        assert metadata["pixel_shape"] == (512, 512)
        assert served["requests"] > 1

    def test_the_two_sources_agree(self, tmp_path, range_server):
        from survey2zarray.stores.survey_store.references import open_remote

        base, _ = range_server
        make_continuum_4d_fits(tmp_path / "quad.fits", shape_yx=(256, 256), n_stokes=4)
        url = f"{base}/quad.fits"
        from astropy.io import fits as astropy_fits

        with astropy_fits.open(remote_header_buffer(url)) as buffered:
            with astropy_fits.open(open_remote(url)) as opened:
                assert buffered[0].header == opened[0].header
                assert (
                    buffered[0].fileinfo()["datLoc"] == opened[0].fileinfo()["datLoc"]
                )


class TestTruncationWarnings:
    def test_the_header_pass_expects_a_partial_file(
        self, tmp_path, range_server, recwarn
    ):
        from astropy.utils.exceptions import AstropyUserWarning

        base, _ = range_server
        make_continuum_fits(tmp_path / "image.fits", shape_yx=(512, 512))
        # Only the front of the file is fetched, so astropy noticing the rest is
        # absent says nothing about the file
        build_header_index([f"{base}/image.fits"], require_referenceable=True)
        truncation = [
            w
            for w in recwarn
            if issubclass(w.category, AstropyUserWarning)
            and "truncated" in str(w.message)
        ]
        assert not truncation


class TestReadPlanning:
    """Verify row-range planning across image planes and size bounds."""

    SHAPE = (100, 1000)
    ITEM = 4
    ROW = 1000 * 4

    def plan(self, **kwargs):
        from survey2zarray.stores.survey_store.references import region_ranges

        options = dict(
            offset=0, shape=self.SHAPE, itemsize=self.ITEM, planes=[0], y0=10, y1=20
        )
        options.update(kwargs)
        return region_ranges(**options)

    def test_a_run_of_rows_is_one_request(self):
        plan = self.plan()
        assert plan.fetch == [(10 * self.ROW, 20 * self.ROW)]
        assert plan.fetched_bytes == 10 * self.ROW

    def test_the_ranges_reassemble_the_region(self):
        source = np.arange(100 * 1000, dtype=">f4")
        raw = source.tobytes()
        plan = self.plan()
        joined = b"".join(raw[start:end] for start, end in plan.fetch)
        got = np.frombuffer(joined, dtype=">f4").reshape(10, 1000)
        assert np.array_equal(got, source.reshape(100, 1000)[10:20])

    def test_planes_are_never_read_through(self):
        # The planes between two wanted ones are the polarisations being skipped
        plan = self.plan(shape=(4, 100, 1000), planes=[0, 3])
        assert len(plan.fetch) == 2

    def test_a_request_is_split_at_the_size_bound(self):
        plan = self.plan(y0=0, y1=100, max_bytes=10 * self.ROW)
        assert len(plan.fetch) == 10
        assert plan.fetched_bytes == 100 * self.ROW
