import warnings

import numpy as np
import pytest
import torch
from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord
from astropy.wcs import WCS
from dfreproject import calculate_reprojection

from survey2zarray.dataset.augmentation import S2ZAugmentation
from survey2zarray.dataset.signal_transforms import SignalTransform
from survey2zarray.dataset.sky_transforms import SkyTransform

MAX_CROP = 8.0  # arcsec, the largest square servable at any rotation

SKY_CONFIG = {
    "random_crop": {"min_crop": 0.5, "max_crop": 1.0},
    "random_rotation": {"min_angle": 0.0, "max_angle": 360.0},
    "random_horizontal_flip": {"p": 0.5},
    "random_vertical_flip": {"p": 0.5},
}


def make_tile_wcs(ra=150.0, dec=2.0, nx=16, ny=16, scale_arcsec=1.0):
    scale = scale_arcsec / 3600.0
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.cd = np.array([[-scale, 0.0], [0.0, scale]])
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


class TestValidation:
    def test_unknown_augmentation_section_is_rejected(self):
        with pytest.raises(ValueError, match="Unrecognised augmentation section"):
            S2ZAugmentation({"sky_transform": {}}, MAX_CROP)

    def test_unknown_sky_transform_and_argument_are_rejected(self):
        with pytest.raises(ValueError, match="Unrecognised sky transform"):
            S2ZAugmentation({"sky_transforms": {"random_translate": {}}}, MAX_CROP)
        with pytest.raises(ValueError, match="takes only"):
            S2ZAugmentation(
                {"sky_transforms": {"random_crop": {"low": 0.5, "high": 1.0}}},
                MAX_CROP,
            )

    def test_a_crop_above_the_whole_cutout_is_rejected(self):
        # Crops are fractions of the largest servable square, checked once at
        # construction, so anything above 1.0 asks for more than there is
        config = {
            "sky_transforms": {
                **SKY_CONFIG,
                "random_crop": {"min_crop": 0.5, "max_crop": 1.5},
            }
        }
        with pytest.raises(ValueError, match="fractions of the largest servable"):
            S2ZAugmentation(config, MAX_CROP)

    def test_a_crop_naming_one_bound_is_rejected(self):
        # Half a random_crop passes every range check and then fails on the
        # first sample, with the store open and the workers already running
        for crop in ({"min_crop": 0.5}, {"max_crop": 0.5}, {}):
            config = {"sky_transforms": {**SKY_CONFIG, "random_crop": crop}}
            with pytest.raises(
                ValueError, match=r"random_crop requires (min_crop|max_crop)"
            ):
                S2ZAugmentation(config, MAX_CROP)

    def test_a_rotation_naming_one_bound_is_rejected(self):
        for rotation in ({"min_angle": 0.0}, {"max_angle": 90.0}, {}):
            with pytest.raises(
                ValueError,
                match=r"random_rotation requires (min_angle|max_angle)",
            ):
                S2ZAugmentation(
                    {"sky_transforms": {"random_rotation": rotation}}, MAX_CROP
                )

    def test_probabilities_outside_unit_interval_are_rejected(self):
        for name in ("random_horizontal_flip", "random_vertical_flip"):
            with pytest.raises(ValueError, match="between 0 and 1"):
                S2ZAugmentation({"sky_transforms": {name: {"p": 1.01}}}, MAX_CROP)
        with pytest.raises(ValueError, match="between 0 and 1"):
            S2ZAugmentation(
                {"signal_transforms": {"random_gaussian_noise": {"p": -0.1}}},
                MAX_CROP,
            )

    def test_an_unrecognised_signal_transform_is_refused(self):
        # Skipping it would serve views that were never noised, silently
        config = {"signal_transforms": {"gaussian_noise": {"p": 1.0}}}
        with pytest.raises(ValueError, match="Unrecognised signal transform"):
            S2ZAugmentation(config, MAX_CROP)

    def test_removed_fourier_transform_is_refused(self):
        config = {"signal_transforms": {"fourier_transform": {}}}
        with pytest.raises(ValueError, match="Unrecognised signal transform"):
            S2ZAugmentation(config, MAX_CROP)

    def test_a_crop_of_the_whole_cutout_is_accepted(self):
        # The bound is now the square side itself, not a radius to convert
        config = {
            "sky_transforms": {
                **SKY_CONFIG,
                "random_crop": {"min_crop": 0.5, "max_crop": 1.0},
            }
        }
        S2ZAugmentation(config, MAX_CROP)


class TestSkyTransform:
    @pytest.mark.parametrize("shape", [(8, 8), (9, 7)])
    def test_target_reference_pixel_is_the_geometric_centre(self, shape):
        ny, nx = shape
        transform = SkyTransform({}, MAX_CROP)
        centre = SkyCoord(ra=150.0 * u.deg, dec=2.0 * u.deg, frame=ICRS)
        wcs = transform._build_skybox_wcs(centre, nx, ny, (1.0, 1.0), 0.0)
        assert tuple(wcs.wcs.crpix) == pytest.approx(((nx + 1) / 2, (ny + 1) / 2))
        x, y = wcs.world_to_pixel(centre)
        assert float(x) == pytest.approx((nx - 1) / 2)
        assert float(y) == pytest.approx((ny - 1) / 2)

    def test_subpixel_angular_crop_still_produces_one_pixel(self):
        config = {"random_crop": {"min_crop": 1.0, "max_crop": 1.0}}
        transform = SkyTransform(config, max_crop_size=0.1)
        centre = SkyCoord(ra=150.0 * u.deg, dec=2.0 * u.deg, frame=ICRS)
        data = np.ones((3, 5, 5), dtype=np.float32)
        out = transform(
            [[(data, False)]],
            [make_tile_wcs(nx=5, ny=5)],
            [(1.0, 1.0)],
            [True],
            centre,
            np.random.default_rng(0),
        )
        assert out[0][0][0].shape == (3, 1, 1)

    def test_batches_all_groups_per_survey_into_one_reprojection(self, monkeypatch):
        calls = []

        def fake_reprojection(source_hdus, _target_wcs, shape_out, **_kwargs):
            calls.append(source_hdus)
            return np.stack(
                [
                    np.full(shape_out, np.asarray(source[0]).mean(), dtype=np.float32)
                    for source in source_hdus
                ]
            )

        monkeypatch.setattr(
            "survey2zarray.dataset.sky_transforms.calculate_reprojection",
            fake_reprojection,
        )
        transform = SkyTransform({}, max_crop_size=4.0)
        centre = SkyCoord(ra=150.0 * u.deg, dec=2.0 * u.deg, frame=ICRS)
        groups_a = [
            (np.stack([np.ones((8, 8)), np.full((8, 8), 2.0)]), True),
            (np.full((1, 8, 8), 3.0), False),
        ]
        groups_b = [(np.full((2, 16, 16), 4.0), True)]

        out = transform(
            [groups_a, groups_b],
            [make_tile_wcs(nx=8, ny=8), make_tile_wcs(nx=16, ny=16, scale_arcsec=0.5)],
            [(1.0, 1.0), (0.5, 0.5)],
            [True, True],
            centre,
            np.random.default_rng(0),
        )

        assert [len(source_hdus) for source_hdus in calls] == [3, 2]
        assert [group[1] for group in out[0]] == [True, False]
        assert out[0][0][0].shape == (2, 4, 4)
        assert out[0][1][0].shape == (1, 4, 4)
        assert out[1][0][0].shape == (2, 8, 8)
        np.testing.assert_array_equal(out[0][0][0][:, 0, 0], [1.0, 2.0])
        np.testing.assert_array_equal(out[0][1][0][:, 0, 0], [3.0])
        np.testing.assert_array_equal(out[1][0][0][:, 0, 0], [4.0, 4.0])

    def test_batched_planes_match_scalar_reprojection(self):
        transform = SkyTransform({}, MAX_CROP)
        centre = SkyCoord(ra=150.0 * u.deg, dec=2.0 * u.deg, frame=ICRS)
        wcs_in = make_tile_wcs()
        data = np.random.default_rng(1).normal(size=(3, 16, 16))

        out = transform(
            [[(data, True)]],
            [wcs_in],
            [(1.0, 1.0)],
            [True],
            centre,
            np.random.default_rng(0),
        )[0][0][0]

        wcs_out = transform._build_skybox_wcs(centre, 8, 8, (1.0, 1.0), 0.0)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="cdelt will be ignored", category=RuntimeWarning
            )
            expected = np.stack(
                [
                    np.asarray(
                        calculate_reprojection(
                            (plane, wcs_in),
                            wcs_out,
                            shape_out=(8, 8),
                            order="bilinear",
                            device="cpu",
                            conserve_flux=False,
                        )
                    ).reshape(8, 8)
                    for plane in data
                ]
            )
        np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6, equal_nan=True)

    def test_output_shapes_and_state_sharing(self):
        # One sampled state applied to all surveys: same angular crop size,
        # per-survey pixel counts follow each survey's pixel scale
        transform = SkyTransform(SKY_CONFIG, MAX_CROP)
        rng = np.random.default_rng(0)
        groups_a = [(np.random.default_rng(1).normal(size=(1, 16, 16)), False)]
        groups_b = [(np.random.default_rng(2).normal(size=(1, 32, 32)), False)]
        wcs_a = make_tile_wcs(nx=16, ny=16, scale_arcsec=1.0)
        wcs_b = make_tile_wcs(nx=32, ny=32, scale_arcsec=0.5)
        centre = SkyCoord(ra=150.0 * u.deg, dec=2.0 * u.deg, frame=ICRS)
        out = transform(
            [groups_a, groups_b],
            [wcs_a, wcs_b],
            [(1.0, 1.0), (0.5, 0.5)],
            [True, True],
            centre,
            rng,
        )
        (array_a, _), (array_b, _) = out[0][0], out[1][0]
        # Same sky box: survey B at half the pixel scale gets twice the pixels
        assert array_b.shape[-1] == pytest.approx(2 * array_a.shape[-1], abs=1)

    def test_absent_survey_passes_through(self):
        transform = SkyTransform(SKY_CONFIG, MAX_CROP)
        rng = np.random.default_rng(0)
        centre = SkyCoord(ra=150.0 * u.deg, dec=2.0 * u.deg, frame=ICRS)
        out = transform([None], [None], [None], [False], centre, rng)
        assert out == [None]

    def test_zero_rotation_no_offset_is_aligned_centre_crop(self):
        # With rotation 0, no flips, no offset headroom and a fixed crop size,
        # the crop must be centred on the source and keep the image's
        # orientation (values interpolate, so alignment is checked via a
        # gradient field rather than exact pixel equality)
        config = {
            "random_crop": {"min_crop": 1.0, "max_crop": 1.0},
            "random_rotation": {"min_angle": 0.0, "max_angle": 0.0},
            "random_horizontal_flip": {"p": 0.0},
            "random_vertical_flip": {"p": 0.0},
        }
        transform = SkyTransform(config, max_crop_size=8.0)
        rng = np.random.default_rng(0)
        # array value = image x index (a pure x-gradient), on a second group a
        # pure y-gradient
        x_gradient = np.tile(np.arange(16.0), (16, 1))[np.newaxis]
        y_gradient = np.tile(np.arange(16.0)[:, np.newaxis], (1, 16))[np.newaxis]
        wcs = make_tile_wcs(nx=16, ny=16)
        centre = SkyCoord(ra=150.0 * u.deg, dec=2.0 * u.deg, frame=ICRS)
        # crop size == max_crop_size -> max_offset == 0: deterministic state
        out = transform(
            [[(x_gradient, False), (y_gradient, False)]],
            [wcs],
            [(1.0, 1.0)],
            [True],
            centre,
            rng,
        )
        result_x = out[0][0][0]
        result_y = out[0][1][0]
        assert result_x.shape == (1, 8, 8)
        # The crop centre sits on the source, i.e. image pixel (7.5, 7.5)
        centre_value_x = np.nanmean(result_x[0, 3:5, 3:5])
        centre_value_y = np.nanmean(result_y[0, 3:5, 3:5])
        assert centre_value_x == pytest.approx(7.5, abs=0.75)
        assert centre_value_y == pytest.approx(7.5, abs=0.75)
        # Orientation preserved: gradients increase in the same direction
        finite_col = np.isfinite(result_x[0, 4, :])
        assert np.all(np.diff(result_x[0, 4, finite_col]) > 0)
        finite_row = np.isfinite(result_y[0, :, 4])
        assert np.all(np.diff(result_y[0, finite_row, 4]) > 0)


class TestCropContainment:
    """The crop must stay inside every survey's stored disc.

    Containment is radial: a crop of side s reaches s*sqrt(2)/2 from its own
    centre at any rotation, so |offset| + s*sqrt(2)/2 must stay within what the
    largest servable square reaches. Drawing offset_x and offset_y
    independently satisfies that per axis but lets the offset vector reach
    sqrt(2) times the bound along the diagonals, which is enough to leave the
    disc of a survey whose grid is rotated relative to the offset frame.
    """

    def test_offsets_stay_within_the_radius(self):
        transform = SkyTransform(SKY_CONFIG, MAX_CROP)
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(5000):
            state = transform.sample_state(rng)
            reach = (
                np.hypot(state["offset_x"], state["offset_y"])
                + np.sqrt(2) * state["size"] / 2
            )
            worst = max(worst, reach)
        assert worst <= MAX_CROP * np.sqrt(2) / 2 + 1e-9

    def test_offsets_are_not_confined_to_a_small_disc(self):
        # A radial draw should still use most of the room available, so this
        # guards against fixing containment by over-shrinking the offset
        transform = SkyTransform(SKY_CONFIG, MAX_CROP)
        rng = np.random.default_rng(0)
        reaches = []
        for _ in range(5000):
            state = transform.sample_state(rng)
            reaches.append(
                np.hypot(state["offset_x"], state["offset_y"])
                + np.sqrt(2) * state["size"] / 2
            )
        assert max(reaches) > 0.95 * MAX_CROP * np.sqrt(2) / 2

    def test_reprojected_crop_has_no_blank_pixels(self):
        # The end of the chain: a disc in, a fully populated crop out, for a
        # survey whose grid is rotated well away from the offset frame
        config = {**SKY_CONFIG, "random_crop": {"min_crop": 1.0, "max_crop": 1.0}}
        transform = SkyTransform(config, MAX_CROP)
        rng = np.random.default_rng(0)
        n = 41
        wcs = make_tile_wcs(nx=n, ny=n)
        wcs.wcs.cd = wcs.wcs.cd @ np.array(
            [
                [np.cos(np.radians(22.0)), -np.sin(np.radians(22.0))],
                [np.sin(np.radians(22.0)), np.cos(np.radians(22.0))],
            ]
        )
        centre = SkyCoord(ra=150.0 * u.deg, dec=2.0 * u.deg, frame=ICRS)
        # The disc a dataset store would hold: the largest crop's corners,
        # plus the margin its padding provides, blanked outside
        yy, xx = np.mgrid[0:n, 0:n]
        radius = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2)
        for _ in range(50):
            data = np.ones((1, n, n), dtype=np.float32)
            data[0][radius > MAX_CROP * np.sqrt(2) / 2 + 2.0] = np.nan
            out = transform([[(data, False)]], [wcs], [(1.0, 1.0)], [True], centre, rng)
            assert np.all(np.isfinite(out[0][0][0]))


class TestAlignmentWithoutAugmentation:
    def test_empty_config_still_reprojects_to_a_shared_grid(self):
        # No sky_transforms configured: the views must still be brought onto
        # one grid, or a val split would return each survey on its own
        transform = SkyTransform({}, MAX_CROP)
        assert transform.identity
        state = transform.sample_state(np.random.default_rng(0))
        assert state["rotation"] == 0.0
        assert state["offset_x"] == 0.0 and state["offset_y"] == 0.0
        assert not state["hflip"] and not state["vflip"]
        assert state["size"] == pytest.approx(MAX_CROP)

    def test_two_rotated_surveys_come_back_registered(self):
        # Same sky, two grids: north-up at 1"/px and rotated 22 deg at 0.5"/px,
        # each carrying a marker 2 arcsec north-east of the source
        max_crop = 12.0
        transform = SkyTransform({}, max_crop)
        centre = SkyCoord(ra=150.0 * u.deg, dec=2.0 * u.deg, frame=ICRS)
        marker = SkyCoord(
            ra=(150.0 + 2.0 / 3600 / np.cos(np.radians(2.0))) * u.deg,
            dec=(2.0 + 2.0 / 3600) * u.deg,
            frame=ICRS,
        )
        groups, wcs_list, scales = [], [], []
        for scale, rotation in ((1.0, 0.0), (0.5, 22.0)):
            n = int(2 * 20.0 / scale) + 1  # image spans +/- 20 arcsec
            wcs = make_tile_wcs(nx=n, ny=n, scale_arcsec=scale)
            if rotation:
                t = np.radians(rotation)
                wcs.wcs.cd = wcs.wcs.cd @ np.array(
                    [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]
                )
            data = np.zeros((1, n, n), dtype=np.float32)
            mx, my = wcs.world_to_pixel(marker)
            data[0, int(round(float(my))), int(round(float(mx)))] = 1.0
            groups.append([(data, False)])
            wcs_list.append(wcs)
            scales.append((scale, scale))

        out = transform(
            groups, wcs_list, scales, [True, True], centre, np.random.default_rng(0)
        )

        # Measure where the marker landed in angular terms, which is what the
        # two views must agree on; they differ in pixel count by design
        found = []
        for view, (scale, _) in zip(out, scales, strict=True):
            a = view[0][0][0]
            iy, ix = np.unravel_index(np.nanargmax(a), a.shape)
            found.append(
                (
                    (ix - (a.shape[1] - 1) / 2) * scale,
                    (iy - (a.shape[0] - 1) / 2) * scale,
                )
            )
        # East is -x under this CD matrix, so the marker sits at (-2, +2)
        for dx, dy in found:
            assert dx == pytest.approx(-2.0, abs=1.0)
            assert dy == pytest.approx(+2.0, abs=1.0)
        assert found[0][0] == pytest.approx(found[1][0], abs=1.0)
        assert found[0][1] == pytest.approx(found[1][1], abs=1.0)


class TestSignalTransform:
    def test_random_gaussian_noise_gated_by_probability(self):
        transform = SignalTransform({"random_gaussian_noise": {"std": 1.0, "p": 1.0}})
        image = torch.zeros(1, 8, 8)
        out = transform.apply(image, np.random.default_rng(0), has_freq_axis=False)
        assert not torch.equal(out, image)
        transform_off = SignalTransform(
            {"random_gaussian_noise": {"std": 1.0, "p": 0.0}}
        )
        out_off = transform_off.apply(
            image, np.random.default_rng(0), has_freq_axis=False
        )
        assert torch.equal(out_off, image)

    def test_random_gaussian_blur_smooths(self):
        transform = SignalTransform(
            {"random_gaussian_blur": {"kernel_size": 3, "sigma": (1.0, 1.0), "p": 1.0}}
        )
        image = torch.zeros(1, 9, 9)
        image[0, 4, 4] = 1.0
        out = transform.apply(image, np.random.default_rng(0), has_freq_axis=False)
        assert out[0, 4, 4] < 1.0
        assert out[0, 4, 3] > 0.0

    def test_nan_fill_replaces_nan_with_configured_value(self):
        transform = SignalTransform({"nan_fill": {"value": -3.0}})
        image = torch.ones(1, 8, 8)
        image[0, 0, 0] = float("nan")
        out = transform.apply(image, np.random.default_rng(0), has_freq_axis=False)
        expected = torch.ones_like(image)
        expected[0, 0, 0] = -3.0
        assert torch.equal(out, expected)

    def test_absent_survey_becomes_none(self):
        transform = SignalTransform({})
        out = transform(
            [None, [(np.ones((1, 4, 4)), False)]],
            [False, True],
            np.random.default_rng(0),
        )
        assert out[0] is None
        tensor, has_freq_axis = out[1][0]
        assert torch.is_tensor(tensor)
        assert has_freq_axis is False
