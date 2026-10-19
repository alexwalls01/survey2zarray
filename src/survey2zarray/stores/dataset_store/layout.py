"""Choose and validate dataset store row layouts."""

# A byte budget gives differently sized cutouts comparable shard allocations.
# Any grouping within the resulting row count preserves survey alignment.
TARGET_SHARD_BYTES: int = 512_000_000
# Bound per-file overhead and keep each shard index within 64 kB.
MIN_SHARD_ROWS: int = 8
MAX_SHARD_ROWS: int = 4096


def is_power_of_two(rows: int) -> bool:
    """Report whether a row count is a power of two.

    Power-of-two row counts keep independently chosen store shards and loader
    sampling groups aligned.

    Parameters
    ----------
    rows
        Rows to test.

    Returns
    -------
    result
        Whether ``rows`` is a positive power of two.
    """
    rows = int(rows)
    return rows >= 1 and rows & (rows - 1) == 0


def shard_rows(
    cutout_bytes: int, coverage: float, target_bytes: int = TARGET_SHARD_BYTES
) -> int:
    """Choose how many rows a shard of the pixel array holds.

    Powers of two are the one set closed under being chosen independently, so a
    survey appended later picks its own from its own cutout size and is aligned
    with every survey already stored, with nothing recorded and nothing to agree
    on. In exchange, a shard uses between half and all of the budget.

    The budget is divided by coverage because padding is free in bytes but not
    in file count. A survey covering 1% of the catalogue would otherwise get a
    hundred times the shard files, nearly all of them empty. Sizing by served
    bytes keeps every survey near the budget regardless of its coverage.

    Parameters
    ----------
    cutout_bytes
        Bytes one cutout occupies uncompressed, which is one inner chunk.
    coverage
        Fraction of the store's sources this survey covers. Its own, never
        another survey's or the store's mean.
    target_bytes
        Bytes a shard aims at.

    Returns
    -------
    result
        Rows to a shard, a power of two between `MIN_SHARD_ROWS` and
        `MAX_SHARD_ROWS`.
    """
    served = max(1, int(round(float(cutout_bytes) * float(coverage))))
    ratio = int(target_bytes) // served
    rows = 1 << max(0, ratio.bit_length() - 1)
    return max(MIN_SHARD_ROWS, min(rows, MAX_SHARD_ROWS))
