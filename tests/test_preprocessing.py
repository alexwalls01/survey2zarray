import numpy as np
import pytest

from survey2zarray.dataset.preprocessing import S2ZPreprocessing


def make_preprocessing(config):
    return S2ZPreprocessing(config, pixel_scale={})


class TestSigmaClip:
    def test_zeroes_low_outliers_and_keeps_bright_pixels(self):
        # Clipping is one-sided: what is well below the noise is background and
        # goes to zero, while a bright pixel is the source and must survive
        rng = np.random.default_rng(0)
        plane = rng.normal(0.0, 1.0, size=(1, 32, 32)).astype(np.float64)
        plane[0, 3, 3] = -50.0  # Low outlier
        plane[0, 5, 5] = +50.0  # High outlier
        pre = make_preprocessing({})
        out = pre.sigma_clip(plane, has_freq_axis=False, n=3.0)
        assert out[0, 3, 3] == 0.0
        assert out[0, 5, 5] == 50.0

    def test_nan_aware(self):
        plane = np.full((1, 4, 4), np.nan)
        pre = make_preprocessing({})
        out = pre.sigma_clip(plane, has_freq_axis=False)
        assert np.all(np.isnan(out))


class TestLinearRescale:
    def test_explicit_range(self):
        array = np.linspace(0, 10, 16).reshape(1, 4, 4)
        out = make_preprocessing({}).linear_rescale(
            array, has_freq_axis=False, in_min=0.0, in_max=10.0
        )
        assert out.min() == pytest.approx(0.0, abs=1e-4)
        assert out.max() == pytest.approx(1.0, abs=1e-4)

    def test_percentile_range(self):
        rng = np.random.default_rng(6)
        array = rng.normal(0, 1, (1, 32, 32))
        array[0, 0, 0] = 1e6  # Would wreck a plain min-max
        out = make_preprocessing({}).linear_rescale(
            array, has_freq_axis=False, in_percentiles=(1, 99)
        )
        # Bulk of the data lands inside [0, 1]
        inside = (out >= 0) & (out <= 1)
        assert inside.mean() > 0.9

    def test_each_explicit_input_bound_is_honoured_independently(self):
        array = np.arange(11.0).reshape(1, 1, 11)
        lower_fixed = make_preprocessing({}).linear_rescale(
            array, has_freq_axis=False, in_min=5.0
        )
        upper_fixed = make_preprocessing({}).linear_rescale(
            array, has_freq_axis=False, in_max=5.0
        )
        assert lower_fixed[0, 0, 0] == pytest.approx(-1.0, abs=1e-5)
        assert lower_fixed[0, 0, -1] == pytest.approx(1.0, abs=1e-5)
        assert upper_fixed[0, 0, 0] == pytest.approx(0.0, abs=1e-5)
        assert upper_fixed[0, 0, -1] == pytest.approx(2.0, abs=1e-5)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"in_min": 2.0, "in_max": 1.0}, "in_min"),
            ({"in_percentiles": (90, 10)}, "in_percentiles"),
            ({"in_percentiles": (10,)}, "exactly two"),
            ({"epsilon": 0.0}, "epsilon"),
        ],
    )
    def test_invalid_bounds_are_rejected(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            make_preprocessing({}).linear_rescale(
                np.arange(4.0).reshape(1, 2, 2),
                has_freq_axis=False,
                **kwargs,
            )

    def test_output_range_offset(self):
        array = np.linspace(0, 1, 16).reshape(1, 4, 4)
        out = make_preprocessing({}).linear_rescale(
            array,
            has_freq_axis=False,
            in_min=0.0,
            in_max=1.0,
            out_min=-1.0,
            out_max=1.0,
        )
        assert out.min() == pytest.approx(-1.0, abs=1e-4)
        assert out.max() == pytest.approx(1.0, abs=1e-4)


class TestCall:
    def test_applies_config_in_order_and_wraps_groups(self):
        config = {"sigma_clip": {"n": 3.0}, "linear_rescale": {}}
        pre = make_preprocessing(config)
        cutout = np.random.default_rng(7).normal(0, 1, (16, 16))
        groups = pre(cutout, has_freq_axis=False)
        # One group per survey (list structure kept even at length 1)
        assert isinstance(groups, list) and len(groups) == 1
        array, has_freq_axis = groups[0]
        assert array.shape == (1, 16, 16)
        assert has_freq_axis is False

    def test_unknown_step_is_refused(self):
        # A misspelled step doing nothing quietly is how a run is lost
        import pytest

        with pytest.raises(ValueError, match="Unrecognised preprocessing step"):
            make_preprocessing({"sigma_clipp": {}})

    def test_a_step_name_that_is_another_attribute_is_refused(self):
        # getattr would resolve these, and the config would then be called
        import pytest

        for name in ("config", "pixel_scale", "sigma_clipped_stats"):
            with pytest.raises(ValueError, match="Unrecognised preprocessing step"):
                make_preprocessing({name: {}})

    def test_default_config_is_robust(self):
        # No config -> sigma clip then a linear rescale, not a min-max stretch
        pre = S2ZPreprocessing(None, pixel_scale={})
        assert "linear_rescale" in pre.config
