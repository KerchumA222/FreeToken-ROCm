"""The speculative rollback snapshot must reclaim a GDN slot from the prefix tree.

Finished requests donate GDN snapshots to the hybrid radix tree, so the pool's free list
drains even though the tree could give slots back. The snapshot path used to check only
the free list, which switched speculation off for every request after the first few."""

from types import SimpleNamespace

from freetoken.scheduler.scheduler import Scheduler


class _Pool:
    def __init__(self, free):
        self.free_list = list(free)
        self.copies = []

    @property
    def num_free_slots(self):
        return len(self.free_list)

    def alloc(self, n):
        out, self.free_list = self.free_list[:n], self.free_list[n:]
        return out

    def copy_from(self, src, dst):
        self.copies.append((src, dst))


def test_snapshot_evicts_a_tree_slot_when_the_free_list_is_empty():
    pool = _Pool([])
    tree = [7]

    def ensure_mamba_slots(n):
        while pool.num_free_slots < n and tree:
            pool.free_list.append(tree.pop())

    stub = SimpleNamespace(cache_manager=SimpleNamespace(ensure_mamba_slots=ensure_mamba_slots))
    req = SimpleNamespace(linear_slot_idx=1, spec_state_slot=None)
    assert Scheduler._snapshot_linear_state(stub, req, pool)
    assert req.spec_state_slot == 7 and pool.copies == [(1, 7)]


def test_snapshot_reports_failure_when_nothing_can_be_reclaimed():
    pool = _Pool([])
    stub = SimpleNamespace(cache_manager=SimpleNamespace(ensure_mamba_slots=lambda n: None))
    req = SimpleNamespace(linear_slot_idx=1, spec_state_slot=None)
    assert not Scheduler._snapshot_linear_state(stub, req, pool)
