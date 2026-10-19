"""Apply identical geometric augmentation across surveys."""

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from astropy.wcs import WCS
from dfreproject import calculate_reprojection

from survey2zarray.utils.logging import get_logger

logger = get_logger(__name__)

# dfreproject queries cdelt even when the stored cd matrix defines reprojection.
# Suppress the resulting Astropy warning for every plane.
CDELT_IGNORED = "cdelt will be ignored since cd is present"


class SkyTransform:
    """Resample every survey onto one randomly placed patch of sky.

    Sample one state per source and apply it to every survey. This aligns views
    across differing pixel grids and projections. An empty configuration uses
    a fixed state on the shared grid.

    Parameters
    ----------
    sky_transform_config
        Configuration with fractional ``random_crop`` bounds ``min_crop`` and
        ``max_crop``. ``random_rotation`` contains degree bounds ``min_angle``
        and ``max_angle``. ``random_horizontal_flip`` and
        ``random_vertical_flip`` contain probability ``p``.
    max_crop_size
        Side in arcsec of the largest crop that can be served, at any rotation.
    """

    STEPS: tuple[str, ...] = (
        "random_crop",
        "random_rotation",
        "random_horizontal_flip",
        "random_vertical_flip",
    )
    PARAMETERS: dict[str, frozenset[str]] = {
        "random_crop": frozenset({"min_crop", "max_crop"}),
        "random_rotation": frozenset({"min_angle", "max_angle"}),
        "random_horizontal_flip": frozenset({"p"}),
        "random_vertical_flip": frozenset({"p"}),
    }

    def __init__(self, sky_transform_config: dict, max_crop_size: float) -> None:
        if not isinstance(sky_transform_config, Mapping):
            raise ValueError("sky_transforms must be a mapping.")
        self.config = dict(sky_transform_config)
        self.max_crop_size = float(max_crop_size)
        self._validate_config()
        # Every configuration reprojects views onto a shared grid.
        self.identity = not sky_transform_config

    def _validate_config(self) -> None:
        """Reject unusable geometry before a worker attempts its first sample."""
        if not np.isfinite(self.max_crop_size) or self.max_crop_size <= 0.0:
            raise ValueError(
                f"max_crop_size must be finite and above zero, got "
                f"{self.max_crop_size}."
            )
        unknown = [name for name in self.config if name not in self.STEPS]
        if unknown:
            raise ValueError(
                f"Unrecognised sky transform(s) {unknown}. The transforms are "
                f"{list(self.STEPS)}."
            )
        for name, arguments in self.config.items():
            if not isinstance(arguments, Mapping):
                raise ValueError(f"{name} arguments must be a mapping.")
            extra = [key for key in arguments if key not in self.PARAMETERS[name]]
            if extra:
                raise ValueError(
                    f"{name} takes only {sorted(self.PARAMETERS[name])}, got {extra}."
                )

        crop = self.config.get("random_crop")
        if crop is not None:
            self._validate_pair(crop, "random_crop", "min_crop", "max_crop")
            for name in ("min_crop", "max_crop"):
                value = crop[name]
                if not 0.0 < value <= 1.0:
                    raise ValueError(
                        f"random_crop {name} must be above 0 and at most 1, got "
                        f"{value}. Crops are fractions of the largest servable "
                        f"square ({self.max_crop_size:.1f} arcsec)."
                    )

        rotation = self.config.get("random_rotation")
        if rotation is not None:
            self._validate_pair(rotation, "random_rotation", "min_angle", "max_angle")
            if not all(
                np.isfinite(rotation[name]) for name in ("min_angle", "max_angle")
            ):
                raise ValueError("random_rotation angles must be finite.")

        for name in ("random_horizontal_flip", "random_vertical_flip"):
            flip = self.config.get(name)
            if flip is None:
                continue
            if "p" not in flip:
                raise ValueError(f"{name} requires p.")
            if not 0.0 <= flip["p"] <= 1.0:
                raise ValueError(f"{name} p must be between 0 and 1, got {flip['p']}.")

    @staticmethod
    def _validate_pair(
        arguments: Mapping, step: str, minimum: str, maximum: str
    ) -> None:
        absent = [name for name in (minimum, maximum) if arguments.get(name) is None]
        if absent:
            raise ValueError(f"{step} requires {' and '.join(absent)}.")
        if arguments[minimum] > arguments[maximum]:
            raise ValueError(
                f"{step} {minimum} ({arguments[minimum]}) is above {maximum} "
                f"({arguments[maximum]})."
            )

    def canonical_state(self) -> dict[str, Any]:
        """Return the fixed state for shared-grid reprojection.

        Returns
        -------
        result
            Same keys as :meth:`sample_state`, north up, centred on the source,
            unflipped, and at the full size served.
        """
        return {
            "size": self.max_crop_size,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "rotation": 0.0,
            "hflip": False,
            "vflip": False,
        }

    def sample_state(self, rng: np.random.Generator) -> dict[str, Any]:
        """Draw one geometric state.

        Draw the offset uniformly by area within a disc. The radius bounds all
        rotated crop corners inside the stored sky region.

        Parameters
        ----------
        rng
            Random source.

        Returns
        -------
        result
            Crop ``size`` in arcsec, ``offset_x`` and ``offset_y`` from the
            source in arcsec, ``rotation`` in degrees, and ``hflip`` and
            ``vflip`` flags. The crop is resolved from its configured fraction
            to arcsec here, once, so every survey's view is cropped alike.
        """
        if self.identity:
            return self.canonical_state()

        crop = self.config.get("random_crop")
        if crop is None:
            size = self.max_crop_size
        else:
            # random_crop values are fractions of the largest servable crop.
            size = rng.uniform(crop["min_crop"], crop["max_crop"]) * self.max_crop_size

        # The difference between square corners bounds the centre offset radius.
        max_offset = (self.max_crop_size - size) * np.sqrt(2) / 2
        # The square root of a uniform draw distributes area uniformly over the disc.
        radius = max_offset * np.sqrt(rng.random())
        angle = rng.uniform(0.0, 2.0 * np.pi)
        offset_x = radius * np.cos(angle)
        offset_y = radius * np.sin(angle)

        rotation_config = self.config.get("random_rotation")
        rotation = (
            0.0
            if rotation_config is None
            else rng.uniform(rotation_config["min_angle"], rotation_config["max_angle"])
        )
        hflip = rng.random() < self.config.get("random_horizontal_flip", {}).get(
            "p", 0.0
        )
        vflip = rng.random() < self.config.get("random_vertical_flip", {}).get("p", 0.0)

        # Enforce containment at runtime because Python can strip assertions.
        limit = self.max_crop_size * np.sqrt(2) / 2
        reach = radius + np.sqrt(2) * size / 2
        if reach > limit + 1e-9:
            raise RuntimeError(
                f"Crop of {size:.1f} arcsec offset by {radius:.1f} reaches "
                f"{reach:.1f} arcsec from the source, outside the {limit:.1f} "
                f"arcsec a {self.max_crop_size:.1f} arcsec crop allows."
            )
        return {
            "size": size,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "rotation": rotation,
            "hflip": hflip,
            "vflip": vflip,
        }

    def _offset_to_sky(
        self, centre: SkyCoord, offset_x: float, offset_y: float
    ) -> SkyCoord:
        # Convert the source-centred tangent-plane offset to a sky coordinate.
        frame = SkyOffsetFrame(origin=centre)
        p = SkyCoord(lon=offset_x * u.arcsec, lat=offset_y * u.arcsec, frame=frame)
        return p.transform_to("icrs")

    def _build_skybox_wcs(
        self,
        skybox_centre: SkyCoord,
        nx: int,
        ny: int,
        pixel_scale: Sequence[float],
        rotation_deg: float,
    ) -> WCS:
        # A TAN target grid provides one projection across all input surveys.
        scale_x = pixel_scale[0] / 3600.0
        scale_y = pixel_scale[1] / 3600.0
        w = WCS(naxis=2)
        # FITS one-based coordinates place the geometric centre at (n + 1) / 2.
        w.wcs.crpix = [(nx + 1) / 2, (ny + 1) / 2]
        w.wcs.crval = [skybox_centre.ra.deg, skybox_centre.dec.deg]
        theta = np.deg2rad(rotation_deg)
        # Coupled axis signs preserve crop alignment.
        w.wcs.cd = np.array(
            [
                [-scale_x * np.cos(theta), scale_y * np.sin(theta)],
                [scale_x * np.sin(theta), scale_y * np.cos(theta)],
            ]
        )
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        return w

    def _reproject_planes(
        self,
        planes: Sequence[np.ndarray],
        wcs_in: WCS,
        wcs_out: WCS,
        shape_out: tuple[int, int],
    ) -> np.ndarray:
        """Reproject one survey's planes together onto a shared target grid."""
        if not planes:
            return np.empty((0, *shape_out), dtype=np.float32)

        # Disabling flux conservation removes crop size as a brightness cue.
        # List input lets dfreproject batch planes that share spatial dimensions.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=CDELT_IGNORED, category=RuntimeWarning
            )
            result = calculate_reprojection(
                [(plane, wcs_in) for plane in planes],
                wcs_out,
                shape_out=shape_out,
                order="bilinear",
                device="cpu",
                conserve_flux=False,
            )
        # dfreproject squeezes singleton batch and spatial dimensions.
        return np.asarray(result).reshape(len(planes), *shape_out)

    def __call__(
        self,
        groups_per_survey: list,
        wcs: list[WCS | None],
        pixel_scales: list[Sequence[float] | None],
        valid: list[bool],
        centre: SkyCoord,
        rng: np.random.Generator,
    ) -> list:
        """Reproject every survey onto one shared, randomly placed grid.

        Each survey keeps its own pixel scale, so a finer survey yields more
        pixels for the same patch of sky. Centre and orientation are shared,
        which is what makes the outputs comparable pixel for pixel after
        accounting for that scale.

        The crop is inside every survey's stored disc by construction, so
        nothing here reads past the edge of the data.

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
            Same structure as the input, with each array resampled.
        """
        state = self.sample_state(rng)
        skybox_centre = self._offset_to_sky(
            centre, state["offset_x"], state["offset_y"]
        )

        output = []
        for i, groups in enumerate(groups_per_survey):
            if not valid[i]:
                output.append(groups)
                continue

            px, py = pixel_scales[i]
            # Positive angular crops always occupy at least one coarse pixel.
            nx = max(1, int(round(state["size"] / px)))
            ny = max(1, int(round(state["size"] / py)))
            skybox_wcs = self._build_skybox_wcs(
                skybox_centre, nx, ny, pixel_scales[i], state["rotation"]
            )

            counts = [array.shape[0] for array, _ in groups]
            planes = [
                array[channel]
                for array, _ in groups
                for channel in range(array.shape[0])
            ]
            reprojected = self._reproject_planes(planes, wcs[i], skybox_wcs, (ny, nx))

            new_groups = []
            start = 0
            for (_, semantics), count in zip(groups, counts, strict=True):
                out = reprojected[start : start + count]
                start += count
                if state["hflip"]:
                    out = out[:, :, ::-1]
                if state["vflip"]:
                    out = out[:, ::-1, :]
                new_groups.append((np.ascontiguousarray(out), semantics))
            output.append(new_groups)
        return output
