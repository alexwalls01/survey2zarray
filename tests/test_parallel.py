"""How much of the machine each kind of work is allowed to take."""

import pytest

from survey2zarray.utils import parallel


@pytest.fixture
def machine(monkeypatch):
    """Report a fixed processor count, whichever way the platform is asked."""

    def with_cores(n):
        monkeypatch.setattr(parallel.os, "cpu_count", lambda: n)
        monkeypatch.setattr(
            parallel.os, "sched_getaffinity", lambda _: set(range(n)), raising=False
        )

    return with_cores


class TestLoaderWorkers:
    """Reader processes run alongside the consumer they feed.

    The machine is shared rather than taken.
    """

    def test_leaves_the_consumer_its_cores(self, machine):
        machine(16)
        assert parallel.loader_workers(reserved=4) == 12
        assert parallel.loader_workers(reserved=1) == 15

    def test_a_small_machine_reads_in_the_main_process(self, machine):
        machine(2)
        # A reader here would be taking a core from the model it feeds, and
        # zero is what a DataLoader takes to mean no worker processes
        assert parallel.loader_workers(reserved=1) == 0

    def test_never_more_workers_than_there_is_work(self, machine):
        machine(32)
        assert parallel.loader_workers(reserved=1, ceiling=3) == 3

    def test_a_large_allocation_is_capped(self, machine):
        # Every worker is another concurrent reader against a shared filer
        machine(128)
        assert parallel.loader_workers(reserved=1) == parallel.LOADER_CAP
        assert parallel.loader_workers(reserved=1, cap=8) == 8

    def test_the_cap_does_not_raise_a_small_machine(self, machine):
        machine(8)
        assert parallel.loader_workers(reserved=1, cap=32) == 7

    def test_an_explicit_number_is_honoured(self, machine):
        machine(2)
        # Even where the machine, or the cap, would have said otherwise
        assert parallel.loader_workers(requested=4) == 4
        assert parallel.loader_workers(requested=0) == 0
        machine(128)
        assert parallel.loader_workers(requested=64) == 64

    def test_reserves_torch_threads_by_default(self, machine):
        import torch

        machine(10)
        assert parallel.loader_workers() == 10 - torch.get_num_threads()

    def test_reserves_at_least_one_core(self, machine):
        machine(8)
        # Reserving nothing would hand the consumer no core of its own
        assert parallel.loader_workers(reserved=0) == 7

    def test_shares_the_node_budget_across_local_ranks(self, machine):
        machine(16)
        assert parallel.loader_workers(reserved=2, local_world_size=2) == 6

    def test_divides_the_storage_cap_across_local_ranks(self, machine):
        machine(64)
        assert parallel.loader_workers(reserved=1, cap=5, local_world_size=2) == 2

    def test_rejects_an_invalid_local_world_size(self):
        with pytest.raises(ValueError, match="local_world_size"):
            parallel.loader_workers(local_world_size=0)


class TestMachineWorkers:
    """Threads within this process, where nothing else is competing for the cores."""

    def test_takes_the_configured_share(self, machine):
        machine(16)
        assert parallel.machine_workers(fraction=0.5) == 8

    def test_never_returns_fewer_than_the_minimum(self, machine):
        machine(1)
        assert parallel.machine_workers(fraction=0.75, minimum=2) == 2
