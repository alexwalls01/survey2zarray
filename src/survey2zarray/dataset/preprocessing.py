"""Prepare cutout intensities deterministically."""

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from astropy.stats import sigma_clipped_stats

from survey2zarray.utils.logging import get_logger

logger = get_logger(__name__)

# Clipped before rescaling, so the brightest pixel does not set the stretch.
DEFAULT_PREPROCESSING_CONFIG: dict[str, dict] = {"sigma_clip": {}, "linear_rescale": {}}


class S2ZPreprocessing:
    """Apply a configured sequence of intensity steps to each cutout.

    Deterministic steps preserve agreement between views of one source. All
    steps preserve NaNs. Configuration selects the steps for each survey.

    Parameters
    ----------
    preprocessing_config
        Step names mapped to their keyword arguments, applied in order. None
        selects `DEFAULT_PREPROCESSING_CONFIG`.
    pixel_scale
        Per survey, the pixel scale of each cutout in arcsec per pixel.

    Raises
    ------
    ValueError
        If a step is unrecognised.
    """

    STEPS: tuple[str, ...] = ("sigma_clip", "linear_rescale")

    def __init__(
        self, preprocessing_config: dict | None, pixel_scale: dict[str, np.ndarray]
    ) -> None:
        if preprocessing_config is None:
            preprocessing_config = DEFAULT_PREPROCESSING_CONFIG
        unknown = [name for name in preprocessing_config if name not in self.STEPS]
        if unknown:
            raise ValueError(
                f"Unrecognised preprocessing step(s) {unknown}. The steps are "
                f"{list(self.STEPS)}."
            )
        self.config = preprocessing_config
        self.pixel_scale = pixel_scale

    def _apply_per_group(
        self,
        array: np.ndarray,
        has_freq_axis: bool,
        fn: Callable[[np.ndarray], tuple[Any, Any]],
    ) -> np.ndarray:
        # fn supplies a and b for (array - a) / b over finite values.
        if has_freq_axis:
            finite = array[np.isfinite(array)]
            if finite.size == 0:
                return array
            a, b = fn(finite)
            return (array - a) / b
        out = np.empty_like(array)
        for c in range(array.shape[0]):
            plane = array[c]
            finite = plane[np.isfinite(plane)]
            if finite.size == 0:
                out[c] = plane
                continue
            a, b = fn(finite)
            out[c] = (plane - a) / b
        return out

    def sigma_clip(
        self,
        array: np.ndarray,
        has_freq_axis: bool,
        n: float = 3.0,
        max_passes: int = 5,
    ) -> np.ndarray:
        """Zero low-outlier pixels using astropy sigma-clipped statistics.

        Parameters
        ----------
        array
            Cutout as ``(C, H, W)``.
        has_freq_axis
            Whether the channel axis is frequency.
        n
            Standard deviations below the mean to clip at.
        max_passes
            Maximum iterations per plane.

        Returns
        -------
        result
            Cutout with low outliers set to zero.
        """
        array = array.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for c in range(array.shape[0]):
                plane = array[c]
                for _ in range(int(max_passes)):
                    mean, _median, std = sigma_clipped_stats(
                        plane,
                        sigma_lower=n,
                        sigma_upper=np.inf,
                        maxiters=1,
                        cenfunc="mean",
                        stdfunc="std",
                    )
                    threshold = mean - n * std
                    below = plane < threshold
                    if not np.any(below):
                        break
                    plane[below] = 0.0
        return array

    def linear_rescale(
        self,
        array: np.ndarray,
        has_freq_axis: bool,
        in_min: float | None = None,
        in_max: float | None = None,
        out_min: float = 0.0,
        out_max: float = 1.0,
        in_percentiles: Sequence[float] | None = None,
        epsilon: float = 1e-6,
    ) -> np.ndarray:
        """Map an input range linearly onto an output range.

        Parameters
        ----------
        array
            Cutout as ``(C, H, W)``.
        has_freq_axis
            Whether the channel axis is frequency.
        in_min, in_max
            Fixed input bounds. Taken from the data when omitted.
        out_min, out_max
            Target range.
        in_percentiles
            Percentiles defining outlier-resistant input bounds.
        epsilon
            Guard against division by zero.

        Returns
        -------
        result
            Values outside the input range fall outside the output range.
        """

        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be above zero, got {epsilon}.")
        if in_min is not None and in_max is not None and in_min >= in_max:
            raise ValueError(f"in_min must be below in_max, got {in_min} and {in_max}.")
        if in_percentiles is not None:
            if len(in_percentiles) != 2:
                raise ValueError("in_percentiles must contain exactly two values.")
            lower, upper = in_percentiles
            if not 0.0 <= lower < upper <= 100.0:
                raise ValueError(
                    "in_percentiles must be increasing values between 0 and 100, "
                    f"got {tuple(in_percentiles)}."
                )

        def fn(finite: np.ndarray) -> tuple[Any, Any]:
            """Resolve linear-rescaling terms from finite pixels.

            Parameters
            ----------
            finite
                Finite pixel values.

            Returns
            -------
            result
                Offset and divisor for the linear transformation.
            """
            if in_percentiles is not None:
                measured_lo, measured_hi = np.percentile(finite, in_percentiles)
            else:
                measured_lo, measured_hi = finite.min(), finite.max()
            lo = measured_lo if in_min is None else in_min
            hi = measured_hi if in_max is None else in_max
            if hi < lo:
                raise ValueError(
                    f"Resolved input range is reversed, with minimum {lo} and "
                    f"maximum {hi}."
                )
            scale = (out_max - out_min) / (hi - lo + epsilon)
            return lo, (1.0 / scale)

        # _apply_per_group omits the additive out_min term.
        scaled = self._apply_per_group(array, has_freq_axis, fn)
        return scaled + out_min

    def __call__(
        self, cutout: np.ndarray, has_freq_axis: bool
    ) -> list[tuple[np.ndarray, bool]]:
        """Run the configured steps over one cutout.

        Parameters
        ----------
        cutout
            ``(H, W)`` or ``(C, H, W)``.
        has_freq_axis
            Whether the channel axis is frequency.

        Returns
        -------
        result
            One ``(array, has_freq_axis)`` group.
        """
        array = cutout[np.newaxis, ...] if cutout.ndim == 2 else cutout
        for fn_name, params in self.config.items():
            array = getattr(self, fn_name)(array, has_freq_axis, **(params or {}))
        return [(array, has_freq_axis)]
