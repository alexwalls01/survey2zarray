"""Gather samples into a batch, keeping the surveys apart."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from survey2zarray.utils.logging import get_logger

logger = get_logger(__name__)


def collate_views(samples: Sequence[dict]) -> dict[str, Any]:
    """Collate samples from a :class:`S2ZDataset` into one batch.

    Stack each survey separately because channel counts differ. Store covered
    rows only and use ``mask[survey]`` to align them with the full batch.

    Parameters
    ----------
    samples
        Samples as returned by ``S2ZDataset.__getitem__``.

    Returns
    -------
    result
        Batch with covered view tensors, coverage masks, survey names,
        catalogue columns, source indices, coordinates, and optional labels.

    Raises
    ------
    ValueError
        Where no samples are given, or they disagree about which surveys they
        hold.
    RuntimeError
        Where a survey's views differ in shape across the batch.
    """
    if len(samples) == 0:
        raise ValueError("Cannot collate an empty list of samples.")

    surveys = list(samples[0]["surveys"])
    if any(list(sample["surveys"]) != surveys for sample in samples):
        raise ValueError(
            "Samples disagree about their surveys. They must all come from one dataset."
        )

    views: dict[str, torch.Tensor | None] = {}
    mask: dict[str, torch.Tensor] = {}

    for survey in surveys:
        covered = [bool(sample["mask"][survey]) for sample in samples]
        mask[survey] = torch.tensor(covered, dtype=torch.bool)

        tensors = [
            sample["data"][survey]
            for sample, is_covered in zip(samples, covered, strict=True)
            if is_covered
        ]
        if not tensors:
            # Partial survey footprints can leave a batch with zero covered sources.
            views[survey] = None
            continue

        shapes = {tuple(tensor.shape) for tensor in tensors}
        if len(shapes) > 1:
            raise RuntimeError(
                f"Views of {survey!r} differ in shape across the batch "
                f"({sorted(shapes)}). Set final_pixel_size in the dataset config, or "
                f"give random_crop the same min_crop and max_crop, so every view "
                f"comes back at one size."
            )
        views[survey] = torch.stack(tensors)

    batch: dict[str, Any] = {
        "views": views,
        "mask": mask,
        "surveys": surveys,
        "catalogue": collate_catalogue(samples),
        "source_index": torch.tensor([int(s["source_index"]) for s in samples]),
        "source_coords": torch.tensor(
            [tuple(s["source_coords"]) for s in samples], dtype=torch.float64
        ),
    }

    if "label" in samples[0]:
        batch["label"] = torch.tensor(
            [int(s["label"]) for s in samples], dtype=torch.long
        )
    return batch


def collate_catalogue(samples: Sequence[dict]) -> dict[str, Any]:
    """Gather the catalogue columns of a batch.

    Parameters
    ----------
    samples
        Samples holding a ``catalogue`` entry.

    Returns
    -------
    result
        Column name to a tensor for numeric columns, or a list for anything
        else. Integer columns retain an integer dtype for class indexing. Empty
        when the catalogue has no additional columns.
    """
    columns: dict[str, Any] = {}
    for name in samples[0]["catalogue"]:
        values = [sample["catalogue"][name] for sample in samples]
        # bool is an int subclass, but True cannot serve as a loss target.
        numeric = all(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            for value in values
        )
        if not numeric:
            columns[name] = values
            continue
        integral = all(isinstance(value, (int, np.integer)) for value in values)
        columns[name] = torch.tensor(
            values, dtype=torch.long if integral else torch.float32
        )
    return columns


def select_covered(batch: dict, survey: str, values: Any) -> Any:
    """Reduce a batch-length entry to the rows one survey's views hold.

    Apply ``mask[survey]`` to align full-batch values with covered view rows.

    >>> logits = model(batch["views"]["survey_a"])
    >>> loss = criterion(logits, select_covered(batch, "survey_a", batch["label"]))

    Parameters
    ----------
    batch
        A batch as `collate_views` returns.
    survey
        Survey whose views the result is to line up with.
    values
        A batch-length entry, such as ``batch["label"]`` or a column of
        ``batch["catalogue"]``.

    Returns
    -------
    result
        Covered entries in view order. Tensor inputs produce tensors. Other
        inputs produce lists.

    Raises
    ------
    KeyError
        Where the batch holds no such survey.
    ValueError
        Where ``values`` is not batch-length.
    """
    if survey not in batch["mask"]:
        raise KeyError(
            f"Batch holds no survey {survey!r}. It holds {list(batch['mask'])}."
        )
    mask = batch["mask"][survey]
    if len(values) != len(mask):
        raise ValueError(
            f"Expected {len(mask)} value(s), one per source in the batch, but got "
            f"{len(values)}."
        )
    if torch.is_tensor(values):
        return values[mask]
    return [
        value for value, covered in zip(values, mask.tolist(), strict=True) if covered
    ]
