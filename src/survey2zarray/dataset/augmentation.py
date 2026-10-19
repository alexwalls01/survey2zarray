"""Compose the geometric and signal augmentation stages."""

from collections.abc import Mapping, Sequence

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from survey2zarray.dataset.signal_transforms import SignalTransform
from survey2zarray.dataset.sky_transforms import SkyTransform
from survey2zarray.utils.logging import get_logger

logger = get_logger(__name__)


class S2ZAugmentation:
    """Produce one augmented view.

    Apply sky transforms first, followed by signal transforms, according to a
    configuration.

    Parameters
    ----------
    augmentation_config
        Configuration with ``sky_transforms`` and ``signal_transforms``
        sections.
    max_crop_size
        Side in arcsec of the largest crop that can be served, at any rotation.

    Raises
    ------
    ValueError
        Where ``random_crop`` names one bound and not both, or where the largest
        configured crop could leave that limit under rotation.
    """

    SECTIONS: tuple[str, ...] = ("sky_transforms", "signal_transforms")

    def __init__(self, augmentation_config: dict, max_crop_size: float) -> None:
        if not isinstance(augmentation_config, Mapping):
            raise ValueError("augmentation must be a mapping.")
        unknown = [name for name in augmentation_config if name not in self.SECTIONS]
        if unknown:
            raise ValueError(
                f"Unrecognised augmentation section(s) {unknown}. The sections are "
                f"{list(self.SECTIONS)}."
            )
        self.sky_transform_config = augmentation_config.get("sky_transforms", {})
        self.signal_transform_config = augmentation_config.get("signal_transforms", {})
        self.max_crop_size = max_crop_size
        self._validate_config()
        self.sky_transform = SkyTransform(self.sky_transform_config, max_crop_size)
        self.signal_transform = SignalTransform(self.signal_transform_config)

    def _validate_config(self) -> None:
        for name, section in (
            ("sky_transforms", self.sky_transform_config),
            ("signal_transforms", self.signal_transform_config),
        ):
            if not isinstance(section, Mapping):
                raise ValueError(f"augmentation.{name} must be a mapping.")

    def __call__(
        self,
        groups_per_survey: list,
        wcs: list[WCS | None],
        pixel_scales: list[Sequence[float] | None],
        valid: list[bool],
        centre: SkyCoord,
        rng: np.random.Generator,
    ) -> list:
        """Augment every survey's groups into one view.

        Parameters
        ----------
        groups_per_survey
            Per survey, a list of ``(array, has_freq_axis)`` groups, or None
            where the survey does not cover the source.
        wcs
            Per survey, the WCS of its cutout, or None.
        pixel_scales
            Per survey, the pixel scale in arcsec per pixel as (x, y), or None.
        valid
            Per survey, whether it covers the source.
        centre
            Source position.
        rng
            Random source.

        Returns
        -------
        result
            Per survey, a list of ``(tensor, has_freq_axis)`` groups, or None.
        """
        groups_per_survey = self.sky_transform(
            groups_per_survey, wcs, pixel_scales, valid, centre, rng
        )
        groups_per_survey = self.signal_transform(groups_per_survey, valid, rng)
        return groups_per_survey
