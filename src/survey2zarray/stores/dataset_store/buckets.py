"""Deal sources into stratified buckets, and cut splits across them."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import zarr
from scipy import stats

from survey2zarray.utils.logging import get_logger

logger = get_logger(__name__)

# Persist the rule and assignments so appends reproduce existing placement.
BUCKETING_RULE: str = "bucketing"
BUCKET_ARRAY: str = "bucket"

RESERVED_COLUMNS: tuple[str, ...] = ("id", "ra", "dec", "source_id", BUCKET_ARRAY)

# Fixed split names give normalisation one unambiguous training split.
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")

# Require this many sources from each discrete stratum in every bucket.
SOURCES_PER_STRATUM: int = 10
# Limit split granularity and the number of shard-crossing cut points.
MAX_BUCKETS: int = 500
# Preserve requested split fractions for catalogues too small to stratify fully.
MIN_BUCKETS: int = 10

CONTINUOUS_BINS: int = 10

# Treat high-cardinality discrete columns as identifiers.
MAX_DISCRETE_VALUES: int = 32

# Report split distributions below this population-consistency probability.
BALANCE_THRESHOLD: float = 0.01
# Require this sample size before testing a split distribution.
BALANCE_MINIMUM: int = 20


def canonical_dtype(values: np.ndarray) -> str:
    """Return the name a column's dtype is recorded and compared under.

    Parameters
    ----------
    values
        The column's values.

    Returns
    -------
    result
        Stable dtype name.
    """
    kind = np.asarray(values).dtype.kind
    if kind in "TUSO":
        return "str"
    if kind == "b":
        return "bool"
    if kind in "iu":
        return "int64"
    return "float64"


def column_kind(values: np.ndarray) -> str:
    """Classify a column as discrete or continuous.

    Parameters
    ----------
    values
        The column's values.

    Returns
    -------
    result
        Strings, booleans, and integers are discrete; anything else is
        continuous.
    """
    return "continuous" if canonical_dtype(values) == "float64" else "discrete"


def quantile_boundaries(values: np.ndarray, bins: int = CONTINUOUS_BINS) -> list[float]:
    """Cut a continuous column at the quantiles its bins fall on.

    Parameters
    ----------
    values
        The column's values.
    bins
        Bins to divide the column into.

    Returns
    -------
    result
        Interior boundaries, ascending and distinct, so a column holding fewer
        distinct values than bins gives fewer than ``bins - 1`` of them.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or bins < 2:
        return []
    quantiles = np.arange(1, bins) / bins
    edges = np.nanquantile(values, quantiles)
    return [float(edge) for edge in np.unique(edges[np.isfinite(edges)])]


def distinct_strata(frame: pd.DataFrame, columns: Sequence[str]) -> int:
    """Count the combinations the discrete columns take between them.

    Parameters
    ----------
    frame
        Catalogue.
    columns
        Discrete columns of the bucketing rule.

    Returns
    -------
    result
        One where no column is named, since the whole catalogue is then a
        single stratum.
    """
    if not columns:
        return 1
    crossed = frame.loc[:, list(columns)].astype(str).drop_duplicates()
    return int(max(1, len(crossed)))


def choose_bucket_count(n_sources: int, n_strata: int) -> int:
    """Choose how many buckets a catalogue of a given shape is dealt into.

    ``B`` is squeezed from below by stratification, since a bucket needs
    `SOURCES_PER_STRATUM` sources of every stratum before it is a miniature of
    the catalogue, and from above by the granularity of a split fraction, which
    is one bucket in ``B``. Where the two conflict on a small catalogue,
    stratification gives way, since a bucket count too low to express the
    fractions serves nothing at all.

    Parameters
    ----------
    n_sources
        Sources in the catalogue.
    n_strata
        Combinations the discrete columns take.

    Returns
    -------
    result
        Buckets, between one and `MAX_BUCKETS`.
    """
    target = SOURCES_PER_STRATUM * max(1, int(n_strata))
    count = min(int(n_sources) // target, MAX_BUCKETS)
    return int(max(1, min(int(n_sources), max(count, MIN_BUCKETS))))


@dataclass(frozen=True)
class BucketingRule:
    """Define bucket placement from catalogue values.

    Attributes
    ----------
    columns
        Catalogue columns forming the sort key, discrete ones first.
    dtypes
        Per column, its `canonical_dtype`.
    kinds
        Per column, ``'discrete'`` or ``'continuous'``.
    boundaries
        Per continuous column, the quantiles it is cut at. Persisted boundaries
        place appended sources against the original distribution.
    n_buckets
        Buckets the sorted catalogue is dealt into.
    seed
        Seed for the tie-break within the sort and the shuffle within each
        bucket.
    """

    columns: tuple[str, ...]
    dtypes: tuple[str, ...]
    kinds: tuple[str, ...]
    boundaries: dict[str, tuple[float, ...]]
    n_buckets: int
    seed: int

    @property
    def discrete(self) -> tuple[str, ...]:
        """Return columns naming a stratum.

        Returns
        -------
        result
            Discrete catalogue column names.
        """
        return tuple(
            column
            for column, kind in zip(self.columns, self.kinds, strict=True)
            if kind == "discrete"
        )

    @property
    def continuous(self) -> tuple[str, ...]:
        """Return columns measuring a quantity.

        Returns
        -------
        result
            Continuous catalogue column names.
        """
        return tuple(
            column
            for column, kind in zip(self.columns, self.kinds, strict=True)
            if kind == "continuous"
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the rule as JSON-serialisable store metadata.

        Returns
        -------
        result
            Rule fields represented by JSON-compatible values.
        """
        return {
            "columns": list(self.columns),
            "dtypes": list(self.dtypes),
            "kinds": list(self.kinds),
            "boundaries": {
                column: list(edges) for column, edges in self.boundaries.items()
            },
            "n_buckets": int(self.n_buckets),
            "seed": int(self.seed),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "BucketingRule":
        """Reconstruct a rule from stored values.

        Parameters
        ----------
        record
            As `as_dict` wrote it.

        Returns
        -------
        result
            Reconstructed rule.
        """
        return cls(
            columns=tuple(record["columns"]),
            dtypes=tuple(record["dtypes"]),
            kinds=tuple(record["kinds"]),
            boundaries={
                column: tuple(float(edge) for edge in edges)
                for column, edges in dict(record.get("boundaries", {})).items()
            },
            n_buckets=int(record["n_buckets"]),
            seed=int(record["seed"]),
        )

    def sort_keys(self, frame: pd.DataFrame) -> list[np.ndarray]:
        """Turn a catalogue into the integer keys the sort orders it by.

        Parameters
        ----------
        frame
            Catalogue holding every column of the rule.

        Returns
        -------
        result
            One key per column, most important first. A discrete column keys on
            its own values and a continuous one on the bin its value falls in,
            so both mean the same thing to a build and to a later append.
        """
        keys: list[np.ndarray] = []
        for column, kind in zip(self.columns, self.kinds, strict=True):
            values = frame[column].to_numpy()
            if kind == "continuous":
                edges = np.asarray(self.boundaries.get(column, ()), dtype=np.float64)
                keys.append(
                    np.searchsorted(
                        edges, np.asarray(values, dtype=np.float64), side="right"
                    )
                )
            elif canonical_dtype(values) == "str":
                # Sorted category codes preserve value order.
                _, codes = np.unique(np.asarray(values, dtype=str), return_inverse=True)
                keys.append(codes)
            else:
                keys.append(np.asarray(values, dtype=np.int64))
        return keys


def bucketing_columns(frame: pd.DataFrame) -> list[str]:
    """Choose which catalogue columns a store buckets on.

    Parameters
    ----------
    frame
        Normalised catalogue.

    Returns
    -------
    result
        Discrete columns first, then continuous ones, each group in catalogue
        order. A discrete column holding more than `MAX_DISCRETE_VALUES`
        distinct values is dropped with a warning.
    """
    discrete: list[str] = []
    continuous: list[str] = []
    for column in frame.columns:
        if column in RESERVED_COLUMNS:
            continue
        values = frame[column].to_numpy()
        if column_kind(values) == "continuous":
            continuous.append(column)
            continue
        distinct = len(np.unique(np.asarray(values, dtype=str)))
        if distinct > MAX_DISCRETE_VALUES:
            logger.warning(
                f"Column '{column}' takes {distinct} distinct values and appears "
                f"source-specific. It is excluded from bucketing. Set bucket_by "
                f"to choose the columns."
            )
            continue
        discrete.append(column)
    return discrete + continuous


def build_rule(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    seed: int = 0,
    n_buckets: int | None = None,
) -> BucketingRule:
    """Build a bucketing rule from catalogue columns.

    Parameters
    ----------
    frame
        Normalised catalogue.
    columns
        Columns to bucket on, most important first. Defaults to what
        `bucketing_columns` chooses.
    seed
        Seed for the tie-break within the sort and the shuffle within each
        bucket.
    n_buckets
        Buckets to deal into. Defaults to what `choose_bucket_count` derives
        from the catalogue's size and its stratum count.

    Returns
    -------
    result
        Derived bucketing rule.

    Raises
    ------
    KeyError
        Where a named column is not in the catalogue.
    ValueError
        Where ``n_buckets`` is not positive.
    """
    chosen = bucketing_columns(frame) if columns is None else [str(c) for c in columns]
    missing = [column for column in chosen if column not in frame.columns]
    if missing:
        raise KeyError(
            f"Cannot bucket on column(s) {missing}. The catalogue holds "
            f"{sorted(c for c in frame.columns if c not in RESERVED_COLUMNS)}."
        )
    kinds, dtypes = [], []
    boundaries: dict[str, tuple[float, ...]] = {}
    for column in chosen:
        values = frame[column].to_numpy()
        kind = column_kind(values)
        kinds.append(kind)
        dtypes.append(canonical_dtype(values))
        if kind == "continuous":
            boundaries[column] = tuple(quantile_boundaries(values))
    # Discrete keys must outrank continuous measurements in each stratum.
    order = [i for i, kind in enumerate(kinds) if kind == "discrete"]
    order += [i for i, kind in enumerate(kinds) if kind == "continuous"]
    chosen = [chosen[i] for i in order]
    kinds = [kinds[i] for i in order]
    dtypes = [dtypes[i] for i in order]

    if n_buckets is None:
        strata = distinct_strata(
            frame, [c for c, k in zip(chosen, kinds, strict=True) if k == "discrete"]
        )
        n_buckets = choose_bucket_count(len(frame), strata)
    n_buckets = int(n_buckets)
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be at least 1, got {n_buckets}.")
    return BucketingRule(
        columns=tuple(chosen),
        dtypes=tuple(dtypes),
        kinds=tuple(kinds),
        boundaries=boundaries,
        n_buckets=n_buckets,
        seed=int(seed),
    )


def validate_columns(frame: pd.DataFrame, rule: BucketingRule) -> None:
    """Validate a catalogue against a stored bucketing rule.

    Parameters
    ----------
    frame
        Catalogue being appended.
    rule
        What the store recorded.

    Raises
    ------
    ValueError
        Where a column of the rule is absent, holds a different dtype, or has
        changed between discrete and continuous.
    """
    missing = [column for column in rule.columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Appended catalogue is missing column(s) {missing} that the store "
            f"buckets on, so its sources cannot be placed. Add them to the "
            f"catalogue or rebuild with a reduced bucketing rule."
        )
    for column, dtype, kind in zip(rule.columns, rule.dtypes, rule.kinds, strict=True):
        values = frame[column].to_numpy()
        found_dtype, found_kind = canonical_dtype(values), column_kind(values)
        if found_dtype != dtype or found_kind != kind:
            raise ValueError(
                f"Column '{column}' arrives as {found_dtype} and {found_kind} but "
                f"the store buckets on it as {dtype} and {kind}, so the same "
                f"source would fall in a different bucket."
            )


def bucket_order(
    frame: pd.DataFrame,
    rule: BucketingRule,
    start_rank: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Deal a catalogue into buckets and put it in the order a store holds it.

    Sources are sorted by the rule's composite key, dealt round robin so that
    sorted rank ``r`` goes to bucket ``r mod B``, and then shuffled within each
    bucket. The shuffle is what stops a shard holding one classification and a
    narrow range of every continuous column, which the sort would otherwise
    leave it with.

    Parameters
    ----------
    frame
        Catalogue holding every column of the rule.
    rule
        Rule to deal by.
    start_rank
        Rank the deal continues from, which is the sources already stored.

    Returns
    -------
    order
        Row of ``frame`` per position in the store, so that buckets ascend.
    buckets
        Bucket per position, in the same order.
    """
    n = len(frame)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng([rule.seed, int(start_rank)])

    # lexsort consumes the primary key last, requiring reversed rule order.
    keys = rule.sort_keys(frame)
    ranked = np.lexsort([rng.permutation(n), *reversed(keys)])
    buckets = np.empty(n, dtype=np.int64)
    buckets[ranked] = (np.arange(n) + int(start_rank)) % rule.n_buckets

    order = np.concatenate(
        [
            rng.permutation(np.flatnonzero(buckets == bucket))
            for bucket in range(rule.n_buckets)
        ]
    ).astype(np.int64)
    return order, buckets[order]


def write_rule(root: zarr.Group, rule: BucketingRule, buckets: np.ndarray) -> None:
    """Record a store's bucketing rule and what it assigned.

    Parameters
    ----------
    root
        Root group of the store, opened for writing.
    rule
        Rule the sources were dealt by.
    buckets
        Bucket per source, in source index order.
    """
    root.attrs[BUCKETING_RULE] = rule.as_dict()
    sources = root.require_group("sources")
    if BUCKET_ARRAY in sources:
        del sources[BUCKET_ARRAY]
    sources.create_array(
        BUCKET_ARRAY,
        data=np.asarray(buckets, dtype=np.int64),
        chunks=(max(len(buckets), 1),),
    )


def read_rule(root: zarr.Group) -> BucketingRule:
    """Read the bucketing rule a store was built with.

    Parameters
    ----------
    root
        Root group of the store.

    Returns
    -------
    result
        Stored bucketing rule.

    Raises
    ------
    KeyError
        Where the store records no rule, which means it predates bucketing and
        has to be rebuilt.
    """
    if BUCKETING_RULE not in root.attrs:
        raise KeyError(
            "This dataset store records no bucketing rule, so its sources "
            "cannot be split. Rebuild it."
        )
    return BucketingRule.from_dict(dict(root.attrs[BUCKETING_RULE]))


def read_split_config(section: Any) -> tuple[dict[str, float], int]:
    """Read the fractions and seed a config's ``splits`` section holds.

    Parameters
    ----------
    section
        The config's ``splits`` value.

    Returns
    -------
    fractions
        One entry per name in `SPLIT_NAMES`, a name the section leaves out
        taking zero.
    seed
        Seed for the rotation the cut points start from.

    Raises
    ------
    TypeError
        Where the section is not a mapping.
    ValueError
        Where a name is not one of `SPLIT_NAMES`, the seed is absent, a
        fraction is negative, or the fractions do not sum to one.
    """
    if not isinstance(section, Mapping):
        raise TypeError(
            f"Expected a mapping for the splits config, but got "
            f"{type(section).__name__}."
        )
    unknown = [key for key in section if key not in (*SPLIT_NAMES, "seed")]
    if unknown:
        raise ValueError(
            f"Unknown key(s) {unknown} in the splits config. The split names are "
            f"fixed at {list(SPLIT_NAMES)}, alongside a seed."
        )
    if "seed" not in section:
        raise ValueError(
            "The splits config needs a seed. A split that cannot be reproduced "
            "is worse than no split."
        )
    fractions = {name: float(section.get(name, 0.0)) for name in SPLIT_NAMES}
    negative = [name for name, value in fractions.items() if value < 0]
    if negative:
        raise ValueError(f"Split fraction(s) {negative} must not be negative.")
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Split fractions must sum to 1, got {fractions} summing to {total}."
        )
    return fractions, int(section["seed"])


@dataclass(frozen=True)
class SplitPlan:
    """Map each bucket's sources to a split.

    Attributes
    ----------
    roles
        Split name per bucket. Every name occupies a contiguous run of
        `order`, so a split spans two shards of the store at its ends and none
        in the middle.
    order
        Buckets in the order they were walked, rotated by the seed.
    survivors
        Sources surviving the configuration's filtering, per bucket.
    counts
        Survivors per split.
    """

    roles: np.ndarray
    order: np.ndarray
    survivors: np.ndarray
    counts: dict[str, int]

    def of_source(self, buckets: np.ndarray) -> np.ndarray:
        """Return the split each source falls in.

        Parameters
        ----------
        buckets
            Bucket per source, in source index order.

        Returns
        -------
        result
            Split name per source.
        """
        return self.roles[np.asarray(buckets, dtype=np.int64)]


def plan_splits(
    buckets: np.ndarray,
    eligible: np.ndarray,
    fractions: Mapping[str, float],
    seed: int,
    n_buckets: int,
) -> SplitPlan:
    """Cut the bucket sequence into splits of the fractions asked for.

    Rotate buckets by a seed-derived offset and place boundaries by cumulative
    eligible-source counts. Fractions are accurate to one bucket over the
    served population.

    Parameters
    ----------
    buckets
        Bucket per source, in source index order.
    eligible
        Boolean, True where the configuration serves the source.
    fractions
        Share per split, over `SPLIT_NAMES`.
    seed
        Seed for the rotation, which is it modulo ``n_buckets``.
    n_buckets
        Buckets the store was dealt into.

    Returns
    -------
    result
        Split assignment plan.
    """
    buckets = np.asarray(buckets, dtype=np.int64)
    eligible = np.asarray(eligible, dtype=bool)
    survivors = np.bincount(buckets[eligible], minlength=n_buckets)[:n_buckets]
    total = int(survivors.sum())

    # Modular rotation keeps split assignment independent of NumPy RNG streams.
    offset = int(seed) % n_buckets
    order = (offset + np.arange(n_buckets)) % n_buckets
    cuts = np.cumsum([fractions[name] for name in SPLIT_NAMES]) * total

    roles = np.empty(n_buckets, dtype=f"<U{max(len(name) for name in SPLIT_NAMES)}")
    running = 0
    for bucket in order:
        running += int(survivors[bucket])
        crossed = np.flatnonzero(running <= cuts + 1e-9)
        roles[bucket] = (
            SPLIT_NAMES[int(crossed[0])] if len(crossed) else SPLIT_NAMES[-1]
        )

    counts = {name: int(survivors[roles == name].sum()) for name in SPLIT_NAMES}
    return SplitPlan(roles=roles, order=order, survivors=survivors, counts=counts)


def balance_warnings(
    columns: Mapping[str, np.ndarray],
    rule: BucketingRule,
    served: np.ndarray,
    split_of_source: np.ndarray,
    threshold: float = BALANCE_THRESHOLD,
) -> list[str]:
    """Report where a split's distribution departs from the population it came out of.

    Compare each split with the served population. Use chi-square tests for
    discrete columns and Kolmogorov-Smirnov tests for continuous columns.

    Parameters
    ----------
    columns
        Catalogue columns, in source index order.
    rule
        Rule the store was bucketed by, whose columns are the ones tested.
    served
        Boolean, True where the configuration serves the source.
    split_of_source
        Split name per source, in source index order.
    threshold
        Probability below which a difference is reported.

    Returns
    -------
    result
        One line per split and column that differs, empty where none does.
    """
    reports: list[str] = []
    for name in SPLIT_NAMES:
        inside = served & (split_of_source == name)
        if int(inside.sum()) < BALANCE_MINIMUM:
            continue
        for column, kind in zip(rule.columns, rule.kinds, strict=True):
            values = columns.get(column)
            if values is None:
                continue
            values = np.asarray(values)
            if kind == "continuous":
                sample = np.asarray(values[inside], dtype=np.float64)
                population = np.asarray(values[served], dtype=np.float64)
                probability = float(stats.ks_2samp(sample, population).pvalue)
            else:
                # Native dtype comparison supports variable-length store strings.
                categories, population = np.unique(values[served], return_counts=True)
                if len(categories) < 2:
                    continue
                observed = np.array(
                    [
                        int((values[inside] == category).sum())
                        for category in categories
                    ],
                    dtype=np.float64,
                )
                expected = population / population.sum() * observed.sum()
                probability = float(stats.chisquare(observed, expected).pvalue)
            if probability < threshold:
                reports.append(
                    f"'{name}' differs from the served population in '{column}' "
                    f"(p={probability:.3g})"
                )
    return reports
