import numpy as np
import pytest
from astropy.io import fits as astropy_fits
from astropy.wcs import WCS

from survey2zarray.stores.survey_store.fits import (
    IncompatibleTilesError,
    InvalidFITSError,
    check_compatibility,
    extract_metadata,
    extract_stokes_i,
    resolve_observation_type,
    resolve_wcs_params,
    stokes_i_shape,
)
from tests.conftest import (
    image_pixels,
    make_continuum_4d_fits,
    make_continuum_fits,
    make_header,
    make_spectral_fits,
)


class TestObservationType:
    def test_degenerate_freq_axis_is_continuum(self):
        header = make_header(naxis_extra=[("FREQ", 1), ("STOKES", 1)])
        assert resolve_observation_type(header) == "continuum"

    def test_non_degenerate_freq_axis_is_spectral(self):
        header = make_header(naxis_extra=[("FREQ", 4), ("STOKES", 1)])
        assert resolve_observation_type(header) == "spectral"

    def test_no_freq_axis_is_continuum(self):
        header = make_header()
        assert resolve_observation_type(header) == "continuum"

    def test_no_image_axes_is_ambiguous(self):
        header = astropy_fits.Header()
        header["NAXIS"] = 0
        assert resolve_observation_type(header) is None


class TestStokesExtraction:
    def test_stokes_i_not_first_plane(self):
        # STOKES axis running [Q=2, I=1]: I is index 1, not 0
        header = make_header(
            shape_yx=(8, 8),
            naxis_extra=[("FREQ", 1), ("STOKES", 2)],
            stokes_crval=2.0,
            stokes_crpix=1.0,
            stokes_cdelt=-1.0,
        )
        data = np.zeros((2, 1, 8, 8), dtype=np.float32)
        data[1] = 7.0  # The I plane
        out = extract_stokes_i(data, header)
        assert out.shape == (8, 8)
        assert np.all(out == 7.0)

    def test_stokes_i_absent_raises(self):
        # STOKES axis holding only Q (value 2)
        header = make_header(
            shape_yx=(8, 8),
            naxis_extra=[("FREQ", 1), ("STOKES", 1)],
            stokes_crval=2.0,
        )
        data = np.zeros((1, 1, 8, 8), dtype=np.float32)
        with pytest.raises(InvalidFITSError):
            extract_stokes_i(data, header)

    def test_no_stokes_axis_passes_through_quietly(self, caplog):
        # Reported once per survey by the build, not once per call
        header = make_header(shape_yx=(8, 8))
        data = np.ones((8, 8), dtype=np.float32)
        with caplog.at_level("WARNING"):
            out = extract_stokes_i(data, header)
        assert out.shape == (8, 8)
        assert not any("STOKES" in r.message for r in caplog.records)

    def test_a_missing_stokes_axis_is_reported_by_extract_metadata(self, tmp_path):
        with_stokes, _ = make_continuum_4d_fits(tmp_path / "with.fits")
        without, _ = make_continuum_fits(tmp_path / "without.fits")
        assert extract_metadata(str(with_stokes))["stokes_assumed"] is False
        assert extract_metadata(str(without))["stokes_assumed"] is True

    def test_spectral_cube_keeps_frequency_axis(self):
        # (stokes, freq, y, x) -> (freq, y, x)
        header = make_header(shape_yx=(8, 8), naxis_extra=[("FREQ", 3), ("STOKES", 1)])
        data = image_pixels((1, 3, 8, 8))
        out = extract_stokes_i(data, header)
        assert out.shape == (3, 8, 8)
        assert np.array_equal(out, data[0])

    def test_header_only_shape_matches_data_extraction(self):
        for naxis_extra, shape in [
            ([("FREQ", 1), ("STOKES", 1)], (1, 1, 10, 12)),
            ([("FREQ", 3), ("STOKES", 1)], (1, 3, 10, 12)),
            ([("STOKES", 1), ("FREQ", 3)], (3, 1, 10, 12)),
            ([], (10, 12)),
        ]:
            header = make_header(shape_yx=(10, 12), naxis_extra=naxis_extra)
            data = np.zeros(shape, dtype=np.float32)
            assert stokes_i_shape(header) == extract_stokes_i(data, header).shape


class TestExtractMetadata:
    def test_continuum_metadata(self, tmp_path):
        path, _ = make_continuum_fits(tmp_path / "image.fits", shape_yx=(32, 40))
        metadata = extract_metadata(str(path))
        assert metadata["observation_type"] == "continuum"
        assert metadata["pixel_shape"] == (32, 40)
        assert metadata["n_chan"] is None
        # Pixel scale in arcsec, per axis
        assert metadata["pixel_scale"] == pytest.approx((1.0, 1.0), rel=1e-6)
        # Beam kept in degrees (raw FITS)
        assert metadata["bmaj"] == pytest.approx(6.0 / 3600.0)
        assert metadata["bpa"] == pytest.approx(10.0)
        assert metadata["radesys"] == "ICRS"
        assert metadata["telescope"] == "TESTSCOPE"
        assert len(metadata["crval"]) == 2
        assert np.asarray(metadata["cd"]).shape == (2, 2)
        # header string round-trips via Header.fromstring
        header = astropy_fits.Header.fromstring(metadata["header"])
        assert header["TELESCOP"] == "TESTSCOPE"

    def test_4d_continuum_uses_post_extraction_shape(self, tmp_path):
        path, _ = make_continuum_4d_fits(tmp_path / "image.fits", shape_yx=(24, 24))
        metadata = extract_metadata(str(path))
        assert metadata["observation_type"] == "continuum"
        assert metadata["pixel_shape"] == (24, 24)
        # The removed Stokes axis is not represented in the stored WCS
        assert len(metadata["crval"]) == 2
        assert "STOKES" not in metadata["ctype"]

    def test_spectral_metadata_keeps_frequency_axis_wcs(self, tmp_path):
        path, _ = make_spectral_fits(
            tmp_path / "cube.fits", shape_yx=(16, 16), n_chan=3
        )
        metadata = extract_metadata(str(path))
        assert metadata["observation_type"] == "spectral"
        assert metadata["n_chan"] == 3
        assert metadata["pixel_shape"] == (16, 16)
        assert len(metadata["crval"]) == 3
        assert np.asarray(metadata["cd"]).shape == (3, 3)
        assert "FREQ" in metadata["ctype"]
        assert "STOKES" not in metadata["ctype"]

    def test_non_tan_projection_preserved(self, tmp_path):
        # Input WCS projection is not always TAN (LoTSS is RA---SIN)
        path, _ = make_continuum_fits(tmp_path / "image.fits", projection="SIN")
        metadata = extract_metadata(str(path))
        assert "RA---SIN" in metadata["ctype"]

    def test_unrepresentable_projection_parameters_are_rejected(self):
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [16.5, 16.5]
        wcs.wcs.crval = [150.0, 2.0]
        wcs.wcs.cd = np.diag([-1 / 3600, 1 / 3600])
        wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
        wcs.wcs.set_pv([(2, 1, 0.1)])
        with pytest.raises(InvalidFITSError, match="PV projection parameters"):
            resolve_wcs_params(wcs, "distorted.fits")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(InvalidFITSError):
            extract_metadata(str(tmp_path / "nope.fits"))


class TestCompatibilityCheck:
    def _metadata(self, path="x.fits", observation_type="continuum", n_chan=None):
        return {
            "image_location": path,
            "observation_type": observation_type,
            "n_chan": n_chan,
        }

    def test_uniform_continuum_passes(self):
        catalogue = [self._metadata(f"{i}.fits") for i in range(3)]
        assert check_compatibility(catalogue) == ("continuum", None)

    def test_uniform_spectral_passes(self):
        catalogue = [self._metadata(f"{i}.fits", "spectral", 4) for i in range(2)]
        assert check_compatibility(catalogue) == ("spectral", 4)

    def test_mixed_types_abort_identifying_tiles(self):
        catalogue = [self._metadata("a.fits"), self._metadata("b.fits", "spectral", 4)]
        with pytest.raises(IncompatibleTilesError) as excinfo:
            check_compatibility(catalogue)
        assert "a.fits" in str(excinfo.value)
        assert "b.fits" in str(excinfo.value)

    def test_mixed_n_chan_aborts(self):
        catalogue = [
            self._metadata("a.fits", "spectral", 4),
            self._metadata("b.fits", "spectral", 5),
        ]
        with pytest.raises(IncompatibleTilesError) as excinfo:
            check_compatibility(catalogue)
        assert "n_chan" in str(excinfo.value)

    def test_ambiguous_type_aborts(self):
        catalogue = [self._metadata("a.fits", None)]
        with pytest.raises(IncompatibleTilesError) as excinfo:
            check_compatibility(catalogue)
        assert "cannot determine observation_type" in str(excinfo.value)


class TestOpenFailures:
    def test_a_timeout_is_an_open_failure_not_a_bad_file(self, monkeypatch, tmp_path):
        from fsspec.exceptions import FSTimeoutError

        from survey2zarray.stores.survey_store.fits import FITSOpenError

        path, _ = make_continuum_fits(tmp_path / "image.fits")

        def times_out(*args, **kwargs):
            raise FSTimeoutError()

        monkeypatch.setattr(
            "survey2zarray.stores.survey_store.fits.fits.open", times_out
        )
        # FSTimeoutError derives from asyncio.TimeoutError, which is not an
        # OSError on Python 3.10, so it needs catching in its own right
        with pytest.raises(FITSOpenError):
            extract_metadata(str(path))

    def test_a_missing_file_is_an_open_failure(self, tmp_path):
        from survey2zarray.stores.survey_store.fits import FITSOpenError

        with pytest.raises(FITSOpenError):
            extract_metadata(str(tmp_path / "not_here.fits"))
