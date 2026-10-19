"""Allocate bounded machine and server resources."""

import os
import subprocess
import sys
from pathlib import Path

from survey2zarray.utils.logging import get_logger

logger = get_logger(__name__)

# Reserve capacity for other work during long builds.
MACHINE_FRACTION: float = 0.75


def machine_workers(fraction: float = MACHINE_FRACTION, minimum: int = 1) -> int:
    """Return how many threads this machine can spare.

    Prefer the process affinity mask because containers can expose fewer CPUs
    than the host count.

    Parameters
    ----------
    fraction
        Share of the processors to take.
    minimum
        Minimum worker count.

    Returns
    -------
    result
        Available worker threads.
    """
    if hasattr(os, "sched_getaffinity"):
        available = len(os.sched_getaffinity(0))
    else:
        available = os.cpu_count() or 1
    return max(int(minimum), int(available * float(fraction)))


def polite_workers(
    cap: int,
    requested: int | None = None,
    ceiling: int | None = None,
    minimum: int = 1,
) -> int:
    """Choose a worker count from capacity limits and an optional override.

    Parameters
    ----------
    cap
        Maximum concurrency for shared services.
    requested
        Explicit worker count, bounded by ``ceiling``.
    ceiling
        Maximum useful workers for the available tasks.
    minimum
        Minimum capacity estimate before other limits apply.

    Returns
    -------
    result
        At least one.
    """
    if requested is not None:
        workers = max(1, int(requested))
    else:
        workers = min(int(cap), machine_workers(minimum=minimum))
    if ceiling is not None:
        workers = min(workers, max(int(ceiling), 1))
    return max(1, workers)


# Reserve the final cores for the batch consumer.
LOADER_MINIMUM_CORES: int = 4
# Cap concurrent readers against shared storage.
LOADER_CAP: int = 16


def loader_workers(
    reserved: int | None = None,
    requested: int | None = None,
    ceiling: int | None = None,
    cap: int = LOADER_CAP,
    local_world_size: int = 1,
) -> int:
    """Choose reader processes while reserving cores for batch consumption.

    Bound concurrent readers to prevent CPU oversubscription and shared-storage
    saturation. Spawned readers also incur interpreter startup overhead.

    Parameters
    ----------
    reserved
        Cores to leave for what consumes the batches. Defaults to the
        threads torch is configured to use, since that is what the model will
        occupy while the readers work.
    requested
        Explicit worker count.
    ceiling
        Maximum useful workers for the available batches.
    cap
        Maximum concurrent readers against shared storage.
    local_world_size
        Training processes sharing this node and its reader budget. Automatic
        capacity is divided evenly among them. An explicit ``requested`` value
        remains a per-process override.

    Returns
    -------
    result
        Worker count. Zero selects main-process loading.
    """
    local_world_size = int(local_world_size)
    if local_world_size < 1:
        raise ValueError(f"local_world_size must be positive, got {local_world_size}.")

    if requested is not None:
        return max(0, int(requested))

    if hasattr(os, "sched_getaffinity"):
        available = len(os.sched_getaffinity(0))
    else:
        available = os.cpu_count() or 1

    if reserved is None:
        try:
            import torch

            reserved = int(torch.get_num_threads())
        except Exception:
            reserved = 1

    if available < LOADER_MINIMUM_CORES:
        return 0
    # Every local rank consumes model threads and readers concurrently. Build
    # one node-wide allowance, then give every rank the same integer share so
    # independently constructed loaders cannot each claim the full machine.
    workers = min(
        max(0, available - max(1, int(reserved)) * local_world_size),
        max(int(cap), 0),
    )
    workers //= local_world_size
    if ceiling is not None:
        workers = min(workers, max(int(ceiling), 0))
    return workers


# Reserve memory for the model, in-flight batches, and reader processes.
MEMORY_FRACTION: float = 0.5


def _cgroup_memory() -> int | None:
    """Return what this process's cgroup still allows it, in bytes."""
    try:
        entries = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None

    # Each ancestor cgroup constrains this process, so use the smallest limit.
    paths = []
    for entry in entries:
        fields = entry.split(":")
        if len(fields) != 3:
            continue
        controllers, path = fields[1], fields[2].lstrip("/")
        if controllers in ("", "memory") or "memory" in controllers.split(","):
            root = Path("/sys/fs/cgroup") / ("" if controllers == "" else "memory")
            node = root / path
            while node != root.parent:
                paths.append(node)
                node = node.parent

    limits = []
    for node in paths:
        for limit_name, usage_name in (
            ("memory.max", "memory.current"),
            ("memory.limit_in_bytes", "memory.usage_in_bytes"),
        ):
            try:
                limit = (node / limit_name).read_text().strip()
                usage = (node / usage_name).read_text().strip()
            except OSError:
                continue
            if limit == "max":
                continue
            # Treat word-size-scale cgroup v1 values as unlimited.
            if int(limit) >= 1 << 60:
                continue
            limits.append(max(0, int(limit) - int(usage)))
    return min(limits) if limits else None


def available_memory() -> int | None:
    """Return bytes the current process can allocate.

    Prefer cgroup limits for scheduled jobs, then Linux ``MemAvailable``,
    POSIX available pages, and the macOS virtual-memory report.

    Returns
    -------
    result
        Available bytes, or None when the platform exposes no estimate.
    """
    cgroup = _cgroup_memory()
    if cgroup is not None:
        return cgroup

    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass

    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        pass

    return _darwin_memory()


def _darwin_memory() -> int | None:
    """Return macOS memory obtainable before paging, in bytes."""
    if sys.platform != "darwin":
        return None
    try:
        report = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None

    # Include reclaimable inactive page-cache memory on macOS.
    counts = {}
    for line in report.splitlines():
        name, _, value = line.partition(":")
        counts[name.strip()] = (
            int(value.strip().rstrip("."), 10)
            if value.strip().rstrip(".").isdigit()
            else 0
        )
    pages = sum(
        counts.get(name, 0)
        for name in (
            "Pages free",
            "Pages inactive",
            "Pages speculative",
            "Pages purgeable",
        )
    )
    if not pages:
        return None
    try:
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return None
