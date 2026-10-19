"""Select sources by the value of their catalogue columns."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from survey2zarray.utils.logging import get_logger

logger = get_logger(__name__)

RESERVED_SUBSET_KEYS: tuple[str, ...] = ("sources", "surveys")

# Inclusive and exclusive bounds express exact cuts at meaningful values.
RANGE_KEYS: tuple[str, ...] = ("min", "max", "greater_than", "less_than")


def as_list(values: Any) -> list:
    """Return a configuration value as a list.

    Parameters
    ----------
    values
        A single value, or a sequence of them.

    Returns
    -------
    result
        Empty for None. A string forms one value.
    """
    if values is None:
        return []
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        return [values]
    return list(values)


def resolve_subset(
    available: Sequence[str],
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
    kind: str,
) -> list[str]:
    """Reduce a set of available items to the selected subset.

    Includes alone define the set, and excludes alone remove from everything.
    Given both, the includes form the universe the excludes then reduce. A missing
    include is an error because the request cannot be satisfied, while a
    missing exclude only warns, being harmless but likely a typo.

    Parameters
    ----------
    available
        Everything on offer, in the order to preserve.
    include
        Items to select. None selects everything.
    exclude
        Items to drop.
    kind
        What is being selected, used in messages.

    Returns
    -------
    result
        The selection, in the order given by ``available``.

    Raises
    ------
    ValueError
        Where an item appears in both lists, or an included item does not exist.
    """
    include = list(include) if include is not None else None
    exclude = list(exclude) if exclude is not None else []
    conflict = set(include or []) & set(exclude)
    if conflict:
        raise ValueError(
            f"{kind} appear in both include and exclude lists. Found "
            f"{sorted(conflict)}."
        )
    available_set = set(available)
    if include is not None:
        missing = [item for item in include if item not in available_set]
        if missing:
            raise ValueError(
                f"Included {kind} are absent from the dataset. Found {missing}."
            )
        universe = [item for item in available if item in set(include)]
    else:
        universe = list(available)
    missing = [item for item in exclude if item not in available_set]
    if missing:
        logger.warning(f"Excluded {kind} are absent from the dataset. Found {missing}.")
    exclude_set = set(exclude)
    return [item for item in universe if item not in exclude_set]


def is_range_filter(spec: Any) -> bool:
    """Report whether a column filter specifies a range.

    Parameters
    ----------
    spec
        One column's entry in an include or exclude section.

    Returns
    -------
    result
        Whether the specification contains a range bound.
    """
    return isinstance(spec, Mapping) and any(key in spec for key in RANGE_KEYS)


def range_mask(values: np.ndarray, spec: Mapping[str, Any], name: str) -> np.ndarray:
    """Select the values inside a numerical range.

    A missing value compares false against every bound and so falls outside any
    range. That is the useful reading, since a source whose measurement is
    absent has not been shown to be inside one.

    Parameters
    ----------
    values
        One catalogue column.
    spec
        Any of ``min`` and ``max``, which include their bound, and
        ``greater_than`` and ``less_than``, which do not.
    name
        Column name, used in messages.

    Returns
    -------
    result
        Boolean, True inside the range.

    Raises
    ------
    ValueError
        Where the column is not numerical, a bound is unrecognised, or the
        bounds cannot contain anything.
    """
    unknown = sorted(set(spec) - set(RANGE_KEYS))
    if unknown:
        raise ValueError(
            f"Unrecognised bound(s) {unknown} for column {name!r}. Use any of "
            f"{list(RANGE_KEYS)}."
        )
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(
            f"Cannot apply a range to column {name!r} with dtype {values.dtype}. "
            f"Filter it by value."
        )

    lower = max(
        [float(spec[key]) for key in ("min", "greater_than") if key in spec]
        or [-np.inf]
    )
    upper = min(
        [float(spec[key]) for key in ("max", "less_than") if key in spec] or [np.inf]
    )
    if lower > upper:
        raise ValueError(
            f"Range for column {name!r} is empty, since nothing is both above "
            f"{lower} and below {upper}."
        )

    keep = np.ones(len(values), dtype=bool)
    with np.errstate(invalid="ignore"):
        if "min" in spec:
            keep &= values >= float(spec["min"])
        if "max" in spec:
            keep &= values <= float(spec["max"])
        if "greater_than" in spec:
            keep &= values > float(spec["greater_than"])
        if "less_than" in spec:
            keep &= values < float(spec["less_than"])
    return keep


def describe_range(spec: Mapping[str, Any]) -> str:
    """Render a range filter for a message.

    Parameters
    ----------
    spec
        As `range_mask` takes.

    Returns
    -------
    result
        Human-readable range expression.
    """
    parts = []
    for key, symbol in (
        ("min", ">="),
        ("greater_than", ">"),
        ("max", "<="),
        ("less_than", "<"),
    ):
        if key in spec:
            parts.append(f"{symbol} {spec[key]}")
    return " and ".join(parts)


def resolve_column_filters(
    extra_columns: dict[str, np.ndarray],
    include: dict,
    exclude: dict,
    n_sources: int,
) -> np.ndarray:
    """Select sources by the value of their catalogue columns.

    Filters combine with AND, so a source must match every included column and
    avoid every excluded one. Values are compared as they were stored, so an
    integer label is matched by an integer.

    A column may be given either a set of values or a range, and the two read
    the same way round, where including keeps what matches and excluding drops
    it.

    Parameters
    ----------
    extra_columns
        Catalogue columns beyond the standard four, in source index order.
    include
        Column name to the value or values to keep, or to a range as
        `range_mask` takes. Subset-kind keys are ignored here.
    exclude
        Column name to the value or values to drop, or to a range to drop.
    n_sources
        Rows in the source table and length of the result.

    Returns
    -------
    result
        Boolean, one entry per source, True where every filter is satisfied.

    Raises
    ------
    ValueError
        Where a column is unknown, a range is unusable, or a value appears in
        both lists for one column.
    """
    keep = np.ones(int(n_sources), dtype=bool)

    named = {name for name in (*include, *exclude) if name not in RESERVED_SUBSET_KEYS}
    unknown = sorted(name for name in named if name not in extra_columns)
    if unknown:
        raise ValueError(
            f"Cannot filter on column(s) {unknown}. The dataset store holds "
            f"{sorted(extra_columns) or 'no extra columns'}."
        )

    for name in sorted(named):
        values = extra_columns[name]
        wanted = include.get(name)
        unwanted = exclude.get(name)

        # Validate range mappings early to expose mistyped bounds.
        for section, spec in (("include", wanted), ("exclude", unwanted)):
            if isinstance(spec, Mapping) and not is_range_filter(spec):
                raise ValueError(
                    f"Filter for column {name!r} in {section} names no bound. A "
                    f"range uses any of {list(RANGE_KEYS)}, got {sorted(spec)}."
                )

        if is_range_filter(wanted):
            matched = range_mask(values, wanted, name)
            if not matched.any():
                logger.warning(f"No source has {name} {describe_range(wanted)}.")
            keep &= matched
        elif wanted is not None:
            wanted = as_list(wanted)
            if is_range_filter(unwanted):
                pass
            elif unwanted is not None:
                conflict = [value for value in wanted if value in as_list(unwanted)]
                if conflict:
                    raise ValueError(
                        f"Values {conflict} appear in both include and exclude for "
                        f"column {name!r}."
                    )
            if wanted:
                matched = np.isin(values, wanted)
                if not matched.any():
                    logger.warning(
                        f"No source has {name} in {wanted}. The present values are "
                        f"{sorted(set(values.tolist()))}."
                    )
                keep &= matched

        if is_range_filter(unwanted):
            keep &= ~range_mask(values, unwanted, name)
        elif unwanted is not None:
            unwanted = as_list(unwanted)
            if unwanted:
                keep &= ~np.isin(values, unwanted)
    return keep


@dataclass(frozen=True)
class SourceSelection:
    """Record served sources and exclusion reasons.

    Attributes
    ----------
    eligible
        Boolean, one entry per source, True where every filter is satisfied and
        the view threshold is met. Eligibility is independent of split.
    selected_ids
        Source identifiers the subsetting keeps.
    column_keep
        Boolean, True where a source passes every column filter.
    reasons
        Excluded sources counted by the first applicable reason.
    """

    eligible: np.ndarray
    selected_ids: set
    column_keep: np.ndarray
    reasons: Counter


def count_views(index_maps: Mapping[str, Any], n_sources: int) -> np.ndarray:
    """Count how many of a set of surveys covers each source.

    Count coverage across the supplied survey set.

    Parameters
    ----------
    index_maps
        Per survey, the source indices it holds cutouts for.
    n_sources
        Rows in the source table, which is the length of the result.

    Returns
    -------
    result
        Views per source, as int64.
    """
    views = np.zeros(int(n_sources), dtype=np.int64)
    for indices in index_maps.values():
        indices = np.asarray(list(indices), dtype=np.int64)
        if indices.size:
            views[indices] += 1
    return views


def select_sources(
    source_ids: Sequence[str],
    extra_columns: dict[str, np.ndarray],
    views: np.ndarray,
    include: Mapping[str, Any] | None,
    exclude: Mapping[str, Any] | None,
    min_views: int,
    n_surveys: int,
) -> SourceSelection:
    """Reduce a source table to the sources one configuration serves.

    This is the one place the served population is worked out, so the split
    cut over the store's buckets and the samples a reader is served cannot
    come to different answers about which sources those are.

    Parameters
    ----------
    source_ids
        Source identifiers, in source index order.
    extra_columns
        Catalogue columns beyond the standard four, in source index order.
    views
        Views per source among the selected surveys, as `count_views` returns.
    include, exclude
        Subsetting, keyed by ``sources`` or by a catalogue column. A
        ``surveys`` key is ignored here, the survey set having already been
        resolved into ``views``.
    min_views
        Views a source needs to be served. Zero serves a source no selected
        survey covers, whose every view would be None.
    n_surveys
        Surveys selected, used only in messages.

    Returns
    -------
    result
        Resolved source selection.
    """
    include = dict(include or {})
    exclude = dict(exclude or {})
    selected_ids = set(
        resolve_subset(
            list(source_ids),
            include.get("sources"),
            exclude.get("sources"),
            "sources",
        )
    )
    column_keep = resolve_column_filters(
        extra_columns, include, exclude, len(source_ids)
    )

    reasons: Counter = Counter()
    eligible = np.zeros(len(source_ids), dtype=bool)
    for source_index, source_id in enumerate(source_ids):
        if source_id not in selected_ids:
            reasons["not among the identifiers selected"] += 1
        elif not column_keep[source_index]:
            reasons["removed by a column filter"] += 1
        elif views[source_index] < min_views:
            reasons[
                f"covered by fewer than {min_views} of {n_surveys} selected survey(s)"
            ] += 1
        else:
            eligible[source_index] = True
    return SourceSelection(eligible, selected_ids, column_keep, reasons)
