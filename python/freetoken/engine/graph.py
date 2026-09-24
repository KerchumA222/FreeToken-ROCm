from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from freetoken.core import Batch, Req, get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.utils import init_logger, mem_GB
from freetoken.utils.progress import emit_progress
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.models import BaseLLMModel
    from freetoken.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor
    table_idx: torch.Tensor  # per-request slot id for GatedDeltaNet state gather/scatter
    # Decode GDN query indptr = arange(bs+1); a constant per captured bs, filled once.
    fla_cu_seqlens: torch.Tensor

    @classmethod
    def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
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
        capture_hidden: bool = False,
        verify_rows: tuple[int, ...] = (),
        verify_tail=None,
    ) -> None:
        # ``verify_tail(batch, logits) -> (next_tokens, draft_tokens)``: the greedy sample and
        # the draft head, captured into the verify graph after the target forward.
        self.verify_tail = verify_tail
        # Rows per request of the speculative verify forwards (1 + drafts) to capture
        # graphs for; empty captures decode graphs only.
        self.verify_rows = tuple(verify_rows)
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
        self.capture_hidden = capture_hidden
        self.stream = stream
        self.device = device
        self._capture_graphs(max_seq_len, vocab_size, model)

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
        self.hidden_map: Dict[int, torch.Tensor] = {}
        self.verify_graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        self.verify_hidden_map: Dict[int, torch.Tensor] = {}
        self.verify_tail_out: Dict[int, tuple] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        # Verify graphs run bs * verify_rows query rows, so the attention capture buffers
        # are sized for the larger of the two.
        attn_bs = list(self.graph_bs_list)
        if self.verify_rows:
            attn_bs.append(self.max_graph_bs * max(self.verify_rows))
        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=attn_bs)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)
        self._reset_moe_offload_cache()
        # getattr: the duck-typed cache doubles in tests carry neither attribute.
        if getattr(self.moe_offload_cache, "host_tier", None) is not None:
            # Pinned staging for the disk tier's admission host nodes: allocating it is
            # illegal once capture opens, so every layer's has to exist beforehand.
            self.moe_offload_cache.prepare_graph_admission(
                range(self.moe_offload_cache.num_layers)
            )

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
            batch.capture_hidden = self.capture_hidden
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
                # The model stashes the hidden states only when unset; drop the warmup's
                # eager tensor so the captured forward records the graph's own output.
                batch.hidden_states = None
                # Keep the offload cache warmed for capture. Resetting here forces
                # CUDA graph capture to replay cold-cache expert copies.
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
                self._reset_moe_offload_cache()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph
            if self.capture_hidden:
                assert batch.hidden_states is not None
                self.hidden_map[bs] = batch.hidden_states

        if self.verify_rows:
            self._capture_verify_graphs(vocab_size, model, pool)

        self._reset_moe_offload_cache()
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def _capture_verify_graphs(self, vocab_size: int, model: BaseLLMModel, pool) -> None:
        """Graphs for the speculative verify forward: ``bs`` requests of ``R`` rows each,
        for every ``R`` in ``verify_rows``. The verify is decode-shaped throughout --
        attention runs each row as a decode query, the GDN layers run the recurrent decode
        kernel over each request's rows, the MoE takes its decode path -- so it captures
        like a decode step. Eager, it spends most of its wall time launching kernels."""
        from freetoken.attention.linear import FLAMetadata

        rows_max = self.max_graph_bs * max(self.verify_rows)
        vb = self.vbuffer = GraphCaptureBuffer.init(rows_max, vocab_size, self.device)
        self.v_page_row = torch.zeros(self.max_graph_bs, dtype=torch.int64, device=self.device)
        self.v_cu_seqlens = {
            R: torch.arange(0, (self.max_graph_bs + 1) * R, R, dtype=torch.int32, device=self.device)
            for R in self.verify_rows
        }
        self.v_logits_indices = torch.arange(rows_max, device=self.device)
        dummy_slot = (self.dummy_req.linear_slot_idx
                      if self.dummy_req.linear_slot_idx is not None
                      else self.dummy_req.table_idx)
        for R in self.verify_rows:
            for bs in sorted(self.graph_bs_list, reverse=True):
                rows = bs * R
                graph = torch.cuda.CUDAGraph()
                batch = Batch(reqs=[self.dummy_req] * bs, phase="prefill")
                batch.is_spec_verify = True
                batch.spec_uniform_rows = R
                batch.capture_hidden = self.capture_hidden
                batch.padded_reqs = batch.reqs
                batch.logits_indices = self.v_logits_indices[:rows]
                self.attn_backend.prepare_for_capture_rows(batch, rows)
                batch.input_ids = vb.input_ids[:rows]
                batch.out_loc = vb.out_loc[:rows]
                batch.positions = vb.positions[:rows]
                vb.table_idx[:bs].fill_(dummy_slot)
                self.v_page_row[:bs].fill_(self.dummy_req.table_idx)
                batch.spec_page_row = self.v_page_row[:bs]
                batch.fla_metadata = FLAMetadata(
                    cu_seqlens=self.v_cu_seqlens[R][: bs + 1], cache_indices=vb.table_idx[:bs])
                with get_global_ctx().forward_batch(batch):
                    vb.logits[:rows] = model.forward()
                    if self.verify_tail is not None:
                        self.verify_tail(batch, vb.logits[:rows])
                    batch.hidden_states = None
                    with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                        vb.logits[:rows] = model.forward()
                        if self.verify_tail is not None:
                            nt, dt = self.verify_tail(batch, vb.logits[:rows])
                            self.verify_tail_out[(bs, R)] = (nt, dt)
                    self._reset_moe_offload_cache()
                self.verify_graph_map[(bs, R)] = graph
                if self.capture_hidden:
                    assert batch.hidden_states is not None
                    self.verify_hidden_map[(bs, R)] = batch.hidden_states
        logger.info_rank0(f"Captured speculative verify graphs (bs, rows/request): "
                          f"{sorted(self.verify_graph_map)}")

    def _is_verify_graph(self, batch: Batch) -> bool:
        return (
            batch.is_spec_verify
            and (batch.size, getattr(batch, "spec_uniform_rows", 0)) in getattr(
                self, "verify_graph_map", {})
            and getattr(getattr(batch, "attn_metadata", None), "rows_as_decode", False)
        )

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        if batch.is_spec_verify:
            return self._is_verify_graph(batch)
        return batch.is_decode and batch.size <= self.max_graph_bs

    def _replay_verify(self, batch: Batch) -> torch.Tensor:
        R = batch.spec_uniform_rows
        bs, rows = batch.size, batch.size * R
        vb = self.vbuffer
        vb.input_ids[:rows] = batch.input_ids
        vb.out_loc[:rows] = batch.out_loc
        vb.positions[:rows] = batch.positions
        vb.table_idx[:bs] = batch.fla_metadata.cache_indices
        if getattr(batch, "spec_page_row", None) is not None:
            self.v_page_row[:bs] = batch.spec_page_row
        self.attn_backend.prepare_for_replay_rows(batch, rows)
        self.verify_graph_map[(bs, R)].replay()
        if self.capture_hidden:
            batch.hidden_states = self.verify_hidden_map[(bs, R)][:rows]
        tail = self.verify_tail_out.get((bs, R))
        if tail is not None:
            batch.graph_next_tokens, batch.graph_draft_tokens = tail
        return vb.logits[:rows]

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        if batch.is_spec_verify:
            return self._replay_verify(batch)
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        if self.capture_hidden:
            batch.hidden_states = self.hidden_map[batch.padded_size][: batch.size]
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
        self.hidden_map = {}
        self.verify_graph_map = {}
        self.verify_hidden_map = {}
        self.verify_tail_out = {}
        self.buffer = None
        self.vbuffer = None
        gc.collect()
