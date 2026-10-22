"""Speculative decoding: MTP (nextn) draft heads, verify and rollback (``--speculative-algorithm mtp``)."""

from .accept import accept_length, batch_draft_depth, committed_tokens, draft_depth

__all__ = ["accept_length", "batch_draft_depth", "committed_tokens", "draft_depth"]
