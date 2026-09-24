"""KV pages across speculative verify rounds.

``allocate_paged`` hands out fresh pages for ``[cached_len, device_len)`` on every forward,
so between forwards a request must own pages for exactly ``[0, cached_len)``. A verify
forward allocates through its staged drafts; anything at or past the post-acceptance
``cached_len`` has to go back, or the next forward allocates over it and the old page
leaks. It used to leak on every reject-and-restage (the hybrid GDN path, where cached_len
does not move) and on every partial accept (the correction's slot), which surfaced as
``free_pages + cache_pages != num_pages`` once the request finished."""

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.scheduler.cache import CacheManager

NUM_PAGES = 64


def _setup(page_size):
    pt = torch.zeros(1, NUM_PAGES * page_size, dtype=torch.int32)
    cm = CacheManager(NUM_PAGES, page_size, pt, "naive")
    prompt = torch.arange(5, dtype=torch.int32)
    req = Req(input_ids=prompt, table_idx=0, cached_len=0, output_len=40, uid=1,
              sampling_params=SamplingParams(), cache_handle=None)
    cm.allocate_paged([req])  # prefill
    req.complete_one()
    return cm, req


def _owned_pages(cm, page_size):
    return NUM_PAGES - len(cm.free_slots)


def _expect_owned(cm, req, page_size):
    assert _owned_pages(cm, page_size) == -(-req.cached_len // page_size)


def _verify(cm, req, draft):
    req.reserve_drafts(1)
    req.write_draft(0, torch.tensor(draft))
    cm.allocate_paged([req])
    return req.device_len


@pytest.mark.parametrize("page_size", [1, 2])
def test_reject_and_restage_returns_the_verify_pages(page_size):
    cm, req = _setup(page_size)
    for _ in range(5):
        staged = _verify(cm, req, draft=7)
        req.reject_and_restage(torch.tensor(9))
        cm.rollback_speculative(req, staged)
        _expect_owned(cm, req, page_size)
        # The restaged draft is already in place; the next verify forwards it and accepts.
        cm.allocate_paged([req])
        staged = req.device_len
        req.accept(1, torch.tensor(3))
        cm.rollback_speculative(req, staged)
        _expect_owned(cm, req, page_size)


@pytest.mark.parametrize("page_size", [1, 2])
def test_partial_accept_returns_the_correction_slot(page_size):
    cm, req = _setup(page_size)
    for _ in range(5):
        staged = _verify(cm, req, draft=7)
        req.accept(0, torch.tensor(9))
        cm.rollback_speculative(req, staged)
        _expect_owned(cm, req, page_size)


@pytest.mark.parametrize("page_size", [1, 2])
def test_full_accept_keeps_every_page(page_size):
    cm, req = _setup(page_size)
    for _ in range(5):
        staged = _verify(cm, req, draft=7)
        req.accept(1, torch.tensor(9))
        cm.rollback_speculative(req, staged)
        _expect_owned(cm, req, page_size)


@pytest.mark.parametrize("page_size", [1, 2])
@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_draft_head_scratch_slots_are_returned(page_size, accepted):
    """k drafts allocate k - 1 scratch slots past the staged rows for the head's chain;
    whatever acceptance commits, the rollback returns them with the rejected drafts."""
    cm, req = _setup(page_size)
    for _ in range(4):
        req.reserve_drafts(2)
        req.write_draft(0, torch.tensor(7))
        req.write_draft(1, torch.tensor(8))
        req.spec_scratch = 1
        cm.allocate_paged([req])
        staged = req.device_len + req.spec_scratch
        req.spec_scratch = 0
        req.accept(accepted, torch.tensor(9))
        cm.rollback_speculative(req, staged)
        _expect_owned(cm, req, page_size)


def test_scratch_slots_of_a_dropped_request_are_freed():
    cm, req = _setup(1)
    req.reserve_drafts(2)
    req.spec_scratch = 1
    cm.allocate_paged([req])
    before = len(cm.free_slots)
    cm.free_spec_scratch(req)
    assert len(cm.free_slots) == before + 1 and req.spec_scratch == 0
