"""Apply steps after augmentation."""

from collections.abc import Mapping, Sequence
from typing import Any

from survey2zarray.utils.logging import get_logger

logger = get_logger(__name__)

# Statistics belong to the data the model trains on.
TRAIN_SPLIT: str = "train"

RGB_CHANNELS: int = 3


class S2ZPostprocessing:
    """Apply configured final steps to every source view.

    Validate complete step requirements during construction.

    Parameters
    ----------
    config
        Step name to its arguments, in the order to apply them.
    surveys
        Survey names, in the order views arrive in.
    channels
        Channels each survey's cutouts hold, needed to check a step that only
        makes sense for one. None represents a single channel.

    Attributes
    ----------
    statistics_split
        Split the normalisation statistics are to be measured over, or None
        where nothing needs them.
    statistics
        Per survey, the ``(mean, standard deviation)`` to normalise by, once
        `set_statistics` has been told them.

    Raises
    ------
    ValueError
        Where a step is unrecognised, an argument is not one the step takes, or
        a step needs something the surveys cannot give it.
    """

    STEPS: tuple[str, ...] = ("normalise", "rgb")

    def __init__(
        self,
        config: Mapping[str, Any] | None,
        surveys: Sequence[str],
        channels: Mapping[str, int | None] | None = None,
    ) -> None:
        self.config = dict(config or {})
        self.surveys = list(surveys)
        self.statistics: dict[str, tuple[float, float]] | None = None
        self.statistics_split: str | None = None
        self.repeats: bool = False

        unknown = [name for name in self.config if name not in self.STEPS]
        if unknown:
            raise ValueError(
                f"Unrecognised postprocessing step(s) {unknown}. The steps are "
                f"{list(self.STEPS)}."
            )

        if "normalise" in self.config:
            self.statistics_split = self._resolve_normalise(self.config["normalise"])
        if "rgb" in self.config:
            arguments = dict(self.config["rgb"] or {})
            if arguments:
                raise ValueError(
                    f"rgb takes no arguments, got {sorted(arguments)}. It copies "
                    f"one channel into {RGB_CHANNELS}."
                )
            self.repeats = True
            self._require_single_channel(channels or {})

    def _resolve_normalise(self, arguments: Mapping[str, Any] | None) -> str:
        arguments = dict(arguments or {})
        split = arguments.pop("split", None)
        if arguments:
            raise ValueError(f"normalise takes only 'split', got {sorted(arguments)}.")
        if split is None:
            return TRAIN_SPLIT
        split = str(split)
        if split != TRAIN_SPLIT:
            logger.warning(
                f"Normalisation statistics will use '{split}'. Training data should "
                f"supply these values. Validation or test data leaks split contents "
                f"into input scaling."
            )
        return split

    def _require_single_channel(self, channels: Mapping[str, int | None]) -> None:
        many = {
            survey: channels[survey]
            for survey in self.surveys
            if channels.get(survey) not in (None, 1)
        }
        if many:
            raise ValueError(
                f"rgb requires one channel per survey, but got {many}. Drop the step "
                f"or exclude these surveys."
            )

    def set_statistics(self, statistics: Mapping[str, tuple[float, float]]) -> None:
        """Record the mean and standard deviation to normalise each survey by.

        Parameters
        ----------
        statistics
            Per survey, its ``(mean, standard deviation)``.

        Raises
        ------
        ValueError
            Where a selected survey lacks statistics.
        """
        missing = [survey for survey in self.surveys if survey not in statistics]
        if missing:
            raise ValueError(f"No normalisation statistics for survey(s) {missing}.")
        self.statistics = {
            survey: (float(mean), float(deviation))
            for survey, (mean, deviation) in statistics.items()
        }

    def normalise(self, views: list, valid: Sequence[bool]) -> list:
        """Put every view on the scale its survey was measured to have.

        Use one fixed pair per survey for batch-independent scaling. Per-survey
        values preserve differences in frequency, units, and depth.

        Parameters
        ----------
        views
            Per survey, its ``(tensor, has_freq_axis)`` groups, or None.
        valid
            Whether each survey covers this source.

        Returns
        -------
        result
            Normalised view groups.

        Raises
        ------
        RuntimeError
            Where the statistics were never measured.
        """
        if self.statistics is None:
            raise RuntimeError("normalise requires measured statistics.")
        output = []
        for survey, groups, covered in zip(self.surveys, views, valid, strict=True):
            if not covered or groups is None:
                output.append(groups)
                continue
            mean, deviation = self.statistics[survey]
            output.append(
                [
                    ((tensor - mean) / deviation, has_freq_axis)
                    for tensor, has_freq_axis in groups
                ]
            )
        return output

    def rgb(self, views: list, valid: Sequence[bool]) -> list:
        """Copy each view's single channel into three.

        For a backbone that expects three channels because it was trained on
        colour photographs.

        The expanded channels share memory. Later steps treat each sample as
        read-only.

        Parameters
        ----------
        views
            Per survey, its ``(tensor, has_freq_axis)`` groups, or None.
        valid
            Whether each survey covers this source.

        Returns
        -------
        result
            View groups with three-channel tensors.

        Raises
        ------
        RuntimeError
            Where a view holds more than one channel.
        """
        output = []
        for groups, covered in zip(views, valid, strict=True):
            if not covered or groups is None:
                output.append(groups)
                continue
            repeated = []
            for tensor, has_freq_axis in groups:
                if tensor.shape[0] != 1:
                    raise RuntimeError(
                        f"rgb expects one channel, got {tensor.shape[0]}."
                    )
                repeated.append((tensor.expand(RGB_CHANNELS, -1, -1), has_freq_axis))
            output.append(repeated)
        return output

    def __call__(self, views: list, valid: Sequence[bool]) -> list:
        """Run the configured steps over one sample's views.

        Configuration order controls normalisation and channel repetition.

        Parameters
        ----------
        views
            Per survey, its ``(tensor, has_freq_axis)`` groups, or None.
        valid
            Whether each survey covers this source.

        Returns
        -------
        result
            Postprocessed view groups.
        """
        for name in self.config:
            views = getattr(self, name)(views, valid)
        return views
