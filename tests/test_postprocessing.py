import pytest
import torch

from survey2zarray.dataset.postprocessing import S2ZPostprocessing

SURVEYS = ["A", "B"]


def views(*channel_counts):
    """One group per survey, or None where a survey does not cover the source."""
    out = []
    for count in channel_counts:
        out.append(
            None
            if count is None
            else [(torch.ones(count, 4, 4, dtype=torch.float32), False)]
        )
    return out


class TestConfiguration:
    def test_an_unknown_step_is_refused(self):
        with pytest.raises(ValueError, match="Unrecognised postprocessing step"):
            S2ZPostprocessing({"normalize": {}}, SURVEYS)

    def test_an_unknown_argument_is_refused(self):
        with pytest.raises(ValueError, match="takes only 'split'"):
            S2ZPostprocessing({"normalise": {"splits": "train"}}, SURVEYS)

    def test_rgb_takes_no_arguments(self):
        # Three is what a pretrained backbone wants; any other number is nothing
        with pytest.raises(ValueError, match="takes no arguments"):
            S2ZPostprocessing({"rgb": {"count": 4}}, SURVEYS, channels={"A": 1, "B": 1})

    def test_statistics_default_to_the_training_split(self):
        assert S2ZPostprocessing({"normalise": {}}, SURVEYS).statistics_split == "train"

    def test_another_split_is_allowed_but_warned_about(self, caplog):
        with caplog.at_level("WARNING"):
            step = S2ZPostprocessing({"normalise": {"split": "val"}}, SURVEYS)
        assert step.statistics_split == "val"
        assert "Training data should supply these values" in caplog.text

    def test_nothing_configured_needs_no_statistics(self):
        assert S2ZPostprocessing(None, SURVEYS).statistics_split is None


class TestSingleChannelCheck:
    def test_multi_channel_surveys_are_refused_up_front(self):
        with pytest.raises(ValueError, match="requires one channel per survey"):
            S2ZPostprocessing({"rgb": {}}, SURVEYS, channels={"A": 1, "B": 8})

    def test_the_offending_survey_is_named(self):
        with pytest.raises(ValueError, match="'B'"):
            S2ZPostprocessing({"rgb": {}}, SURVEYS, channels={"A": None, "B": 8})

    def test_a_one_channel_cube_passes(self):
        S2ZPostprocessing({"rgb": {}}, SURVEYS, channels={"A": 1, "B": 1})

    def test_only_checked_when_the_step_is_configured(self):
        # A spectral survey is perfectly valid without it
        S2ZPostprocessing({"normalise": {}}, SURVEYS, channels={"A": 1, "B": 8})


class TestSteps:
    def statistics(self):
        return {"A": (1.0, 2.0), "B": (0.0, 1.0)}

    def test_normalise_uses_each_survey_s_own_numbers(self):
        step = S2ZPostprocessing({"normalise": {}}, SURVEYS)
        step.set_statistics(self.statistics())
        out = step(views(1, 1), [True, True])
        assert torch.allclose(out[0][0][0], torch.zeros(1, 4, 4))  # (1-1)/2
        assert torch.allclose(out[1][0][0], torch.ones(1, 4, 4))  # (1-0)/1

    def test_normalise_needs_statistics_for_every_survey(self):
        step = S2ZPostprocessing({"normalise": {}}, SURVEYS)
        with pytest.raises(ValueError, match=r"statistics for survey\(s\) \['B'\]"):
            step.set_statistics({"A": (0.0, 1.0)})

    def test_normalise_without_statistics_is_an_error_not_a_no_op(self):
        step = S2ZPostprocessing({"normalise": {}}, SURVEYS)
        with pytest.raises(RuntimeError, match="requires measured statistics"):
            step(views(1, 1), [True, True])

    def test_rgb_copies_the_single_channel(self):
        step = S2ZPostprocessing({"rgb": {}}, SURVEYS, channels={"A": 1, "B": 1})
        out = step(views(1, 1), [True, True])
        for survey_views in out:
            assert survey_views[0][0].shape == (3, 4, 4)

    def test_uncovered_surveys_are_left_alone(self):
        step = S2ZPostprocessing({"rgb": {}}, SURVEYS, channels={"A": 1, "B": 1})
        step_n = S2ZPostprocessing({"normalise": {}}, SURVEYS)
        step_n.set_statistics(self.statistics())
        for configured in (step, step_n):
            out = configured(views(1, None), [True, False])
            assert out[1] is None

    def test_steps_run_in_the_order_given(self):
        # Normalising after repeating would scale three channels instead of one;
        # the result is the same, the order is what the config says
        config = {"normalise": {}, "rgb": {}}
        step = S2ZPostprocessing(config, SURVEYS, channels={"A": 1, "B": 1})
        step.set_statistics(self.statistics())
        out = step(views(1, 1), [True, True])
        assert out[0][0][0].shape == (3, 4, 4)
        assert torch.allclose(out[0][0][0], torch.zeros(3, 4, 4))
