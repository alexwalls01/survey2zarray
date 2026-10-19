"""Apply per-view signal augmentation."""

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torchvision.transforms import v2 as T

from survey2zarray.utils.logging import get_logger

logger = get_logger(__name__)


class SignalTransform:
    """Apply a configured sequence of signal steps to each view.

    Stochastic steps gate on their own probability and draw from the generator
    passed in, so a view's randomness stays reproducible from that generator
    alone.

    Parameters
    ----------
    signal_transform_config
        Step names mapped to their keyword arguments, applied in order.

    Raises
    ------
    ValueError
        Where a step is unrecognised.
    """

    STEPS: tuple[str, ...] = (
        "random_gaussian_blur",
        "random_gaussian_noise",
        "nan_fill",
    )
    PARAMETERS: dict[str, frozenset[str]] = {
        "random_gaussian_blur": frozenset({"kernel_size", "sigma", "p"}),
        "random_gaussian_noise": frozenset({"std", "p"}),
        "nan_fill": frozenset({"value"}),
    }

    def __init__(self, signal_transform_config: dict) -> None:
        if not isinstance(signal_transform_config, Mapping):
            raise ValueError("signal_transforms must be a mapping.")
        unknown = [name for name in signal_transform_config if name not in self.STEPS]
        if unknown:
            raise ValueError(
                f"Unrecognised signal transform(s) {unknown}. The transforms are "
                f"{list(self.STEPS)}."
            )
        for name, arguments in signal_transform_config.items():
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, Mapping):
                raise ValueError(f"{name} arguments must be a mapping or None.")
            extra = [key for key in arguments if key not in self.PARAMETERS[name]]
            if extra:
                raise ValueError(
                    f"{name} takes only {sorted(self.PARAMETERS[name])}, got {extra}."
                )
            if name in ("random_gaussian_blur", "random_gaussian_noise"):
                probability = arguments.get("p", 0.1)
                if not 0.0 <= probability <= 1.0:
                    raise ValueError(
                        f"{name} p must be between 0 and 1, got {probability}."
                    )
        self.config = dict(signal_transform_config)

    def random_gaussian_blur(
        self,
        image: torch.Tensor,
        rng: np.random.Generator,
        has_freq_axis: bool,
        kernel_size: int = 5,
        sigma: Sequence[float] = (0.1, 2.0),
        p: float = 0.1,
    ) -> torch.Tensor:
        """Blur with a Gaussian kernel of randomly chosen width.

        Blurring spreads NaNs across their neighbourhood.

        Parameters
        ----------
        image
            View as ``(C, H, W)``.
        rng
            Random source.
        has_freq_axis
            Whether the channel axis is frequency.
        kernel_size
            Kernel extent in pixels.
        sigma
            Range to draw the kernel width from.
        p
            Probability of applying at all.

        Returns
        -------
        result
            Blurred view, or the original view when skipped.
        """
        if rng.random() >= p:
            return image
        s = float(rng.uniform(sigma[0], sigma[1]))
        return T.functional.gaussian_blur(image, kernel_size=kernel_size, sigma=s)

    def random_gaussian_noise(
        self,
        image: torch.Tensor,
        rng: np.random.Generator,
        has_freq_axis: bool,
        std: float = 0.1,
        p: float = 0.1,
    ) -> torch.Tensor:
        """Add uncorrelated Gaussian noise.

        Parameters
        ----------
        image
            View as ``(C, H, W)``.
        rng
            Random source.
        has_freq_axis
            Whether the channel axis is frequency.
        std
            Noise standard deviation.
        p
            Probability of applying at all.

        Returns
        -------
        result
            View with sampled noise, or the original view when skipped.
        """
        if rng.random() >= p:
            return image
        noise = torch.from_numpy(rng.normal(0.0, std, size=tuple(image.shape))).to(
            image.dtype
        )
        return image + noise

    def nan_fill(
        self,
        image: torch.Tensor,
        rng: np.random.Generator,
        has_freq_axis: bool,
        value: float = 0.0,
    ) -> torch.Tensor:
        """Replace NaNs with a constant.

        Steps that mix neighbouring pixels spread NaNs, so place this before
        them if the input may still contain any.

        Parameters
        ----------
        image
            View as ``(C, H, W)``.
        rng
            Unused, accepted for call symmetry.
        has_freq_axis
            Whether the channel axis is frequency.
        value
            Replacement value.

        Returns
        -------
        result
            View with NaNs replaced by ``value``.
        """
        return torch.nan_to_num(image, nan=value)

    def apply(
        self, image: torch.Tensor, rng: np.random.Generator, has_freq_axis: bool
    ) -> torch.Tensor:
        """Run the configured steps over one array.

        Parameters
        ----------
        image
            View as ``(C, H, W)``.
        rng
            Random source.
        has_freq_axis
            Whether the channel axis is frequency.

        Returns
        -------
        result
            Transformed view.
        """
        for fn_name, params in self.config.items():
            image = getattr(self, fn_name)(image, rng, has_freq_axis, **(params or {}))
        return image

    def __call__(
        self,
        groups_per_survey: list,
        valid: list[bool],
        rng: np.random.Generator,
    ) -> list:
        """Convert every group to a tensor and run the configured steps.

        Parameters
        ----------
        groups_per_survey
            Per survey, a list of ``(array, has_freq_axis)`` groups, or None
            where the survey does not cover the source.
        valid
            Per survey, whether it covers the source.
        rng
            Random source.

        Returns
        -------
        result
            Same structure, with tensors in place of arrays and None for any
            survey that does not cover the source.
        """
        output = []
        for i, groups in enumerate(groups_per_survey):
            if not valid[i]:
                output.append(None)
                continue
            new_groups = []
            for array, has_freq_axis in groups:
                image = torch.from_numpy(np.ascontiguousarray(array)).float()
                image = self.apply(image, rng, has_freq_axis)
                new_groups.append((image, has_freq_axis))
            output.append(new_groups)
        return output
