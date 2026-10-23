from __future__ import annotations

import ctypes
import functools
import gc
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Sequence

import torch
from freetoken.core import Batch, MtpPlan, Req, SpecVerify, VerifyRow, get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.env import ENV
from freetoken.utils import init_logger, mem_GB
from freetoken.utils.progress import emit_progress
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.models import BaseLLMModel
    from freetoken.moe.offload_cache import OffloadMoeCache
    from freetoken.speculative.mtp import MtpDrafter

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    # [3, bs] t/h/w rope positions; allocated only for mrope models (else None).
    mrope_positions: torch.Tensor | None
    logits: torch.Tensor
    table_idx: torch.Tensor  # per-request slot id for GatedDeltaNet state gather/scatter
    # Decode GDN query indptr = arange(bs+1); a constant per captured bs, filled once.
    fla_cu_seqlens: torch.Tensor

    @classmethod
    def init(
        cls, bs: int, vocab_size: int, device: torch.device, mrope: bool = False
    ) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            mrope_positions=(
                torch.zeros(3, bs, dtype=torch.int32, device=device) if mrope else None
            ),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
            table_idx=torch.zeros(bs, dtype=torch.int32, device=device),
            fla_cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        from freetoken.attention.linear import FLAMetadata

        _slice = slice(batch.padded_size)
        bs = batch.padded_size
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]
        if self.mrope_positions is not None:
            batch.mrope_positions = self.mrope_positions[:, _slice]
        batch.linear_table_idx = self.table_idx[_slice]
        # Decode GDN metadata reads the persistent cu_seqlens (constant arange) and the
        # persistent table_idx slot map, so the captured kernels see stable addresses.
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.fla_cu_seqlens[: bs + 1], cache_indices=self.table_idx[_slice]
        )

    def copy_from(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        if batch.out_loc is not None:
            self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions
        if self.mrope_positions is not None:
            self.mrope_positions[:, _slice] = batch.mrope_positions
        if batch.linear_table_idx is not None:
            self.table_idx[_slice] = batch.linear_table_idx


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    candidates = [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
    return [bs for bs in candidates if bs <= cuda_graph_max_bs]


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


@functools.cache
def _cu_graph_upload():
    if sys.platform != "linux" or torch.version.hip is not None:
        return None
    fn = ctypes.CDLL("libcuda.so.1").cuGraphUpload
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    fn.restype = ctypes.c_int
    return fn


def _upload_graph(graph: torch.cuda.CUDAGraph, stream: torch.cuda.Stream) -> None:
    """Upload the graph at capture time.

    Otherwise the first replay pushes every node behind the graph's own stream WAIT, and a large graph then blocks cudaGraphLaunch until the WAIT is released. PLE wait-sync releases it only after the launch returns, so that deadlocks.
    """
    upload = _cu_graph_upload()
    if upload is None:
        return
    rc = upload(graph.raw_cuda_graph_exec(), stream.cuda_stream)
    if rc != 0:
        raise RuntimeError(f"cuGraphUpload failed with CUresult {rc}")
    stream.synchronize()


@dataclass
class VerifyGraph:
    """The captured verify of one draft depth: the target forward, and the greedy sample + draft head after it."""

    forward: torch.cuda.CUDAGraph
    tail: torch.cuda.CUDAGraph
    hidden: torch.Tensor
    # (sampled tokens, drafts, draft confidence or None), written by the tail graph
    tail_out: tuple


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        moe_offload_cache: OffloadMoeCache | None = None,
        mrope: bool = False,
        verify_depths: Sequence[int] = (),
        verify_tail: Callable[[Batch, torch.Tensor], tuple] | None = None,
        mtp: MtpDrafter | None = None,
    ) -> None:
        # MTP verify graphs, one per draft depth: verify_tail(batch, logits) is the greedy sample and draft head
        # captured after the forward, and the chained step of ``mtp``'s head is captured with them
        assert not verify_depths or (verify_tail is not None and mtp is not None)
        self.verify_depths = tuple(verify_depths)
        self.verify_tail = verify_tail
        self.mtp = mtp
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.moe_offload_cache = moe_offload_cache
        self.mrope = mrope
        self.stream = stream
        self.device = device
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _note_step(self) -> None:
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.note_graph_step()

    def _reset_moe_offload_cache(self) -> None:
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.reset()

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        # Mark the post-weights "warmup" phase for /health: this stretch (graph capture — or the
        # remaining readiness work when graphs are disabled) moves no bytes, so without this the
        # loader would sit at 100% (last byte bar) until the ready ack. total=0 ⇒ the desktop
        # reads it as an indeterminate phase and animates the bar. Must precede the
        # graphs-disabled early return so that config gets the phase too.
        emit_progress("Capturing CUDA graphs / warming up", 0, 0)
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        self.verify_graphs: Dict[int, VerifyGraph] = {}
        self.chain_graph: torch.cuda.CUDAGraph | None = None
        if self.max_graph_bs == 0 and not self.verify_depths:
            return logger.info_rank0("CUDA graph is disabled.")

        # a verify is a decode batch of 1 + depth rows
        attn_bs = self.graph_bs_list + [k + 1 for k in self.verify_depths] + ([1] if self._chain_depth() else [])
        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=attn_bs)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(
            self.max_graph_bs, vocab_size, self.device, mrope=self.mrope
        )
        self._reset_moe_offload_cache()

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            # capture on the dummy linear-state slot so GatedDeltaNet gather/scatter
            # touches scratch (real slot indices are written by copy_from on replay). Hybrid-
            # radix decouples the GDN slot from table_idx -> use the GDN padding slot.
            dummy_slot = (self.dummy_req.linear_slot_idx
                          if self.dummy_req.linear_slot_idx is not None
                          else self.dummy_req.table_idx)
            self.buffer.table_idx[:bs].fill_(dummy_slot)
            with get_global_ctx().forward_batch(batch):
                self.buffer.logits[:bs] = model.forward()
                # Keep the offload cache warmed for capture. Resetting here forces
                # CUDA graph capture to replay cold-cache expert copies.
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
                _upload_graph(graph, self.stream)
                self._reset_moe_offload_cache()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph

        if self.verify_depths:
            pool = self._capture_verify_graphs(vocab_size, model, pool)
            if self._chain_depth():
                self._capture_chain_graph(vocab_size, pool)

        self._reset_moe_offload_cache()
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def _capture_verify_graphs(self, vocab_size: int, model: BaseLLMModel, pool):
        """One request's verify at each draft depth, as a decode batch of ``1 + depth`` rows (returns the graph pool).

        Its rows are decode rows and the GDN, PLE and QSA ops step through them with their decode kernels, so it
        captures like a decode step. One request's snapshot slots are the same on every verify, so they are baked
        in; the live slot, store rows and attention addressing come from the static buffers."""
        rows_max = max(self.verify_depths) + 1
        self.vbuffer = GraphCaptureBuffer.init(rows_max, vocab_size, self.device, mrope=self.mrope)
        dummy_slot = (self.dummy_req.linear_slot_idx
                      if self.dummy_req.linear_slot_idx is not None
                      else self.dummy_req.table_idx)
        self.vbuffer.table_idx.fill_(dummy_slot)
        store_rows = [self.mtp.store_index(self.dummy_req.table_idx, s) for s in range(rows_max)]
        self.v_store_idx = torch.tensor(store_rows, dtype=torch.int64, device=self.device)
        linear_pool = get_global_ctx().linear_state_pool
        self.v_snap_slots = (
            [torch.tensor([linear_pool.spec_slot(s)], dtype=torch.int64, device=self.device) for s in range(rows_max - 1)]
            if linear_pool is not None else None
        )
        for k in sorted(self.verify_depths, reverse=True):
            rows = k + 1
            batch = Batch(reqs=[VerifyRow(self.dummy_req, s) for s in range(rows)], phase="decode")
            batch.padded_reqs = batch.reqs
            snap_slots = self.v_snap_slots[:k] if self.v_snap_slots is not None else None
            batch.spec = SpecVerify(reqs=[self.dummy_req], k=k, first_drafts=[0], snap_slots=snap_slots)
            batch.capture_hidden = True
            batch.mtp_plan = MtpPlan(store_idx=self.v_store_idx[:rows])
            self.attn_backend.prepare_for_capture(batch)
            self.vbuffer.set_batch(batch)
            logits = self.vbuffer.logits[:rows]
            forward, tail = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with get_global_ctx().forward_batch(batch):
                logits[:] = model.forward()
                self.verify_tail(batch, logits)
                with torch.cuda.graph(forward, pool=pool, stream=self.stream):
                    logits[:] = model.forward()
                _upload_graph(forward, self.stream)
                if pool is None:
                    pool = forward.pool()
                hidden = batch.hidden_states
                with torch.cuda.graph(tail, pool=pool, stream=self.stream):
                    tail_out = self.verify_tail(batch, logits)
                _upload_graph(tail, self.stream)
                self._reset_moe_offload_cache()
            self.verify_graphs[k] = VerifyGraph(forward, tail, hidden, tail_out)
        logger.info_rank0(f"Captured MTP verify graphs for draft depths {sorted(self.verify_graphs)}")
        return pool

    def _chain_depth(self) -> int:
        """Chained drafts a verify can take at most (0 when there is no chain to capture)."""
        return max(self.verify_depths, default=1) - 1 if self.mtp is not None else 0

    def _capture_chain_graph(self, vocab_size: int, pool) -> None:
        """One chained draft step of a single request, replayed once per chained draft.

        The step feeds its state and token back into its own inputs and appends its draft (and confidence) at a
        device counter, so the host only restages the step's position and attention addressing."""
        mtp, min_p = self.mtp, ENV.SPEC_DRAFT_MIN_P.value
        steps = self._chain_depth()
        self.cbuffer = GraphCaptureBuffer.init(1, vocab_size, self.device, mrope=self.mrope)
        self.c_state = torch.zeros(1, mtp.store.shape[-1], dtype=mtp.store.dtype, device=self.device)
        self.c_drafts = torch.zeros(steps, dtype=torch.int32, device=self.device)
        self.c_conf = torch.zeros(steps, dtype=torch.float32, device=self.device)
        self.c_next = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.c_done = 0
        batch = Batch(reqs=[VerifyRow(self.dummy_req, 0)], phase="decode")
        batch.padded_reqs = batch.reqs
        self.attn_backend.prepare_for_capture(batch)
        self.cbuffer.set_batch(batch)
        token = self.cbuffer.input_ids

        def step() -> None:
            state, drafted, conf = mtp.chain_step(self.c_state, token, min_p is not None)
            self.c_state.copy_(state)
            token.copy_(drafted)
            self.c_drafts.index_copy_(0, self.c_next, drafted)
            if conf is not None:
                self.c_conf.index_copy_(0, self.c_next, conf)
            self.c_next += 1

        graph = torch.cuda.CUDAGraph()
        with get_global_ctx().forward_batch(batch):
            step()
            # a head has no trunk layer to rebuild the per-forward plans, so the warmup's would be baked into the graph
            self.attn_backend.prepare_for_capture(batch)
            with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                step()
            _upload_graph(graph, self.stream)
        self.c_next.zero_()
        self.chain_graph = graph
        self._reset_moe_offload_cache()
        logger.info_rank0(f"Captured the MTP chained draft step ({steps} per chain at most)")

    def can_replay_chain(self, spec: SpecVerify) -> bool:
        return self.chain_graph is not None and spec.num_reqs == 1

    def begin_chain(self, state: torch.Tensor, token: torch.Tensor) -> None:
        """Load the first chained step's head state and the first draft."""
        self.c_state.copy_(state)
        self.cbuffer.input_ids.copy_(token)
        self.c_next.zero_()
        self.c_done = 0

    def replay_chain_step(self, step: Batch) -> torch.Tensor | None:
        """Run the next chained step on ``step`` (a one-row decode batch with its attention metadata); returns its
        confidence when FREETOKEN_SPEC_DRAFT_MIN_P tracks it."""
        buf = self.cbuffer
        buf.positions.copy_(step.positions)
        buf.out_loc.copy_(step.out_loc)
        if buf.mrope_positions is not None:
            buf.mrope_positions.copy_(step.mrope_positions)
        self.attn_backend.prepare_for_replay(step)
        self.chain_graph.replay()
        self.c_done += 1
        return self.c_conf[self.c_done - 1 : self.c_done] if ENV.SPEC_DRAFT_MIN_P.value is not None else None

    def finish_chain(self, token_pool: torch.Tensor, req: Req, steps: int) -> torch.Tensor:
        """Put the chain's drafts in the token pool where the verify reads them; returns them as ``[1, steps]``."""
        drafts = self.c_drafts[:steps]
        token_pool[req.table_idx, req.cached_len + 2 : req.cached_len + 2 + steps] = drafts
        return drafts.view(1, steps)

    def can_replay_verify(self, batch: Batch) -> bool:
        """A one-request verify of a captured depth."""
        spec = batch.spec
        return spec is not None and spec.num_reqs == 1 and spec.step is None and spec.k in self.verify_graphs

    def replay_verify(self, batch: Batch, with_tail: bool) -> tuple[torch.Tensor, tuple | None]:
        """Replay the verify's graph: (logits of its rows, the tail's (tokens, drafts, confidence) or None).

        ``batch.hidden_states`` is the graph's LM-head input for an eager tail."""
        assert self.can_replay_verify(batch)
        graph = self.verify_graphs[batch.spec.k]
        rows = batch.padded_size
        self.vbuffer.copy_from(batch)
        self.v_store_idx[:rows].copy_(batch.mtp_plan.store_idx)
        self.attn_backend.prepare_for_replay(batch)
        graph.forward.replay()
        self._note_step()
        batch.hidden_states = graph.hidden
        if not with_tail:
            return self.vbuffer.logits[:rows], None
        graph.tail.replay()
        batch.draft_conf = graph.tail_out[2]
        return self.vbuffer.logits[:rows], graph.tail_out

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        # the MTP draft head reads every row's hidden state after the forward, outside any graph
        if getattr(batch, "capture_hidden", False) or getattr(batch, "spec", None) is not None:
            return False
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        self._note_step()
        return self.buffer.logits[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        # Drop the CUDAGraph objects (and the shared mempool they hold) AND the static
        # GraphCaptureBuffer tensors ([max_bs, vocab] logits + input/out_loc/positions/...).
        # Dropping the references is the load-bearing step; without it a runtime rebuild's
        # free-before-alloc cannot reclaim this GPU memory. empty_cache() is left to the
        # caller / next capture (GraphRunner._capture_graphs already runs it).
        self.graph_map = {}
        self.buffer = None
        self.verify_graphs = {}
        self.vbuffer = self.cbuffer = None
        self.v_snap_slots = None
        self.chain_graph = None
        gc.collect()
