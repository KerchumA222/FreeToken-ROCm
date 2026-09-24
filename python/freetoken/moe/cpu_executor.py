"""Python wrapper around the ``_cpu_moe`` C++ executor (--moe-strategy cpu).

Owns the persistent CPU worker pool, the per-batch-size pinned IO buffers, and
the per-(layer, batch-size) host-func task descriptors. ``decode`` issues the
whole CUDA-graph-capturable sequence on the current stream:

    D2H (hidden, topk_ids, topk_weights -> pinned)
      -> submit host node (cudaLaunchHostFunc: enqueue MoE task to the pool)
      -> sync host node   (cudaLaunchHostFunc: spin until the pool drains)
      -> H2D (pinned expert output -> GPU)

Buffers and tasks are allocated lazily per batch size. GraphRunner runs an eager
``model.forward()`` at each batch size immediately before capturing it, so the
first (eager) call materializes the stable pinned buffers + task pointers that
the subsequent capture embeds in its host/memcpy nodes.
"""

from __future__ import annotations

import os
import threading
import time
import weakref

import torch

from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.utils import init_logger

logger = init_logger(__name__)

# Flag-based GPU<->CPU handshake for hybrid/cpu decode. The default host-func path
# (cudaLaunchHostFunc submit+sync per layer) pays ~30-50us of callback dispatch latency
# per call with the GPU stream idle -- 2 calls per MoE layer per decode step (~6 ms/step
# on a 75-layer model). Instead the GPU raises a mapped-pinned "ready" flag at submit; a
# persistent CPU coordinator (in _cpu_moe) polls it, runs the layer, and sets a "done"
# flag the GPU waits on at sync -- no host-func round-trip. Both GPU-side operations are
# STREAM MEMORY OPERATIONS (cuStreamWriteValue64 / cuStreamWaitValue64, resolved from the
# driver at runtime): they execute on the GPU front end with no SM-resident kernel, so
# GPU utilization stays truthful during the CPU compute window. (The first cut used a
# spin-wait kernel; that pinned reported utilization at 99% and laptop CPU/GPU dynamic
# power schedulers responded by clamping the CPU frequency -- a net decode regression on
# power-coupled edge devices.) Each (layer, decode batch size) pair gets its own flag
# slot, so every captured decode graph rides the handshake. On ROCm the host-func path is
# selected directly because HIP stream memops are not reliable graph dependencies. Where
# CUDA memops are unavailable, rejected during graph capture, or the slot capacity is
# exceeded, decode keeps the host-func path (functional, slower). A Python watchdog
# thread turns a wedged coordinator into a loud RuntimeError (via err[] +
# raise_if_unhealthy) instead of an indefinite stall.
# Caveat: the coordinator busy-polls one core while decode traffic flows (idle backoff
# otherwise); FREETOKEN_CPU_MOE_FLAG_SYNC=0 opts out entirely.
_FLAG_SYNC = os.getenv("FREETOKEN_CPU_MOE_FLAG_SYNC", "1") != "0"
# Flag slots per MoE layer: covers this many distinct decode batch sizes (captured graph
# sizes plus any eager padded sizes); more than that is unheard of, and the overflow
# just keeps the host-func path for the extra combos.
_FLAG_SLOTS_PER_LAYER = 16


def _flag_sync_platform_enabled(enabled: bool, device_type: str, is_rocm: bool) -> bool:
    """HIP stream memops are not reliable graph dependencies on current ROCm."""
    return enabled and device_type == "cuda" and not is_rocm

# Activation ids must match ActKind in csrc/cpu_moe/cpu_moe_ext.cpp. Id 3 is the
# clamped (up + 1) swiglu: "swigluoai" runs it in the generic GEMV epilogue,
# "gpt_oss_swiglu" is the same math fused inside the mxfp4 kernel.
_ACT_IDS = {
    "silu": 0,
    "swish": 0,
    "gelu": 1,
    "gelu_tanh": 2,
    "gelu_pytorch_tanh": 2,
    "gpt_oss_swiglu": 3,
    "swigluoai": 3,
    "swiglu_clamp": 4,
}

# Weight-format ids must match WFmt in csrc/cpu_moe/cpu_moe_ext.cpp.
_WFMT_IDS = {"bf16": 0, "nvfp4": 1, "mxfp4_triton": 2, "ds_fp4": 3, "q4_0": 4}


def _remap_host_ids(
    ids: torch.Tensor, host_tier, layer_id: int, num_experts: int
) -> None:
    """Admit unique raw routes and rewrite the pinned route buffer in place."""
    flat = ids.reshape(-1)
    raw_ids = flat.tolist()
    invalid = [int(raw) for raw in raw_ids if int(raw) >= num_experts]
    if invalid:
        raise ValueError(
            f"CPU MoE route id {invalid[0]} is outside [0, {num_experts})"
        )
    unique: list[int] = []
    seen: set[int] = set()
    for raw in raw_ids:
        raw = int(raw)
        if 0 <= raw < num_experts and raw not in seen:
            seen.add(raw)
            unique.append(raw)
    if not unique:
        return
    slots = host_tier.ensure(layer_id, unique)
    if len(slots) != len(unique):
        raise RuntimeError("host tier returned a slot list with the wrong length")
    remap = dict(zip(unique, (int(slot) for slot in slots)))
    for i, raw in enumerate(raw_ids):
        if 0 <= raw < num_experts:
            flat[i] = remap[raw]


def _native_num_experts(cache) -> int:
    """Return the row bound used by the native executor for this cache."""
    host_tier = getattr(cache, "host_tier", None)
    return int(host_tier.capacity) if host_tier is not None else int(cache.num_experts)


def compiled_extension_supports(activation: str) -> bool:
    """Whether the compiled ``_cpu_moe`` extension can serve ``activation``
    through its generic epilogue. A stale prebuilt .so accepts newer act ids
    while silently computing the wrong math; the executor hard-errors on that,
    but the engine's auto offload->hybrid upgrade consults this first so a
    default boot degrades to offload instead of crashing after weight load."""
    if activation not in _ACT_IDS:
        return False
    if _ACT_IDS[activation] < 3:
        return True
    try:
        from freetoken.kernel import _cpu_moe
    except ImportError:
        return False
    return _ACT_IDS[activation] <= getattr(_cpu_moe, "max_generic_act_id", lambda: 2)()


def _windows_physical_core_cpus() -> list[int]:
    """One logical CPU per physical core on Windows.

    ``GetLogicalProcessorInformationEx(RelationProcessorCore)`` returns one record
    per physical core, each carrying a group affinity mask over that core's SMT
    siblings; the lowest set bit is the core's representative CPU. Processor
    groups are 64 CPUs wide, which is how the ids flatten into the same numbering
    the Linux branch uses. The whole machine is reported: Windows affinity is
    per-group and ``_cpu_moe`` does not pin threads there anyway
    (``CPU_MOE_HAS_AFFINITY`` is Linux-only).
    """
    import ctypes

    class GroupAffinity(ctypes.Structure):
        _fields_ = [
            ("Mask", ctypes.c_size_t),
            ("Group", ctypes.c_uint16),
            ("Reserved", ctypes.c_uint16 * 3),
        ]

    class ProcessorRelationship(ctypes.Structure):
        _fields_ = [
            ("Flags", ctypes.c_uint8),
            ("EfficiencyClass", ctypes.c_uint8),
            ("Reserved", ctypes.c_uint8 * 20),
            ("GroupCount", ctypes.c_uint16),
            ("GroupMask", GroupAffinity * 1),
        ]

    class ProcessorInfoEx(ctypes.Structure):
        _fields_ = [
            ("Relationship", ctypes.c_uint32),
            ("Size", ctypes.c_uint32),
            ("Processor", ProcessorRelationship),
        ]

    relation_processor_core = 0
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    size = ctypes.c_uint32(0)
    kernel32.GetLogicalProcessorInformationEx(relation_processor_core, None, ctypes.byref(size))
    buffer = (ctypes.c_char * size.value)()
    if not kernel32.GetLogicalProcessorInformationEx(
        relation_processor_core, buffer, ctypes.byref(size)
    ):
        raise ctypes.WinError(ctypes.get_last_error())

    cpus: list[int] = []
    offset = 0
    while offset < size.value:
        record = ProcessorInfoEx.from_buffer(buffer, offset)
        affinity = record.Processor.GroupMask[0]
        if affinity.Mask:
            lowest = (affinity.Mask & -affinity.Mask).bit_length() - 1
            cpus.append(affinity.Group * 64 + lowest)
        offset += record.Size
    return sorted(cpus)


def physical_core_cpus() -> list[int]:
    """One logical CPU per physical core, restricted to this process's affinity.

    MoE decode is memory-bandwidth-bound, so SMT siblings only contend for the
    same core's load ports without adding bandwidth. Picking one logical CPU per
    physical core (and pinning to it) gives the best, most stable bandwidth.
    Falls back to the full affinity set when the topology is unavailable.
    """
    if os.name == "nt":
        return _windows_physical_core_cpus()
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except AttributeError:
        allowed = list(range(os.cpu_count() or 1))
    reps: list[int] = []
    seen: set[str] = set()
    for cpu in allowed:
        try:
            with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as f:
                key = f.read().strip()
        except OSError:
            reps.append(cpu)
            continue
        if key not in seen:
            seen.add(key)
            reps.append(cpu)
    return reps or allowed or [0]


def resolve_threads_and_affinity(requested: int) -> tuple[int, list[int]]:
    """Return (num_threads, core_ids) for the worker pool.

    ``requested == 0`` -> one thread per physical core, pinned to it (best for the
    bandwidth-bound GEMV: SMT siblings only contend for a core's load ports and the
    spin-barrier degrades badly when oversubscribed). An explicit count is honored,
    spreading first across physical cores, then across the remaining logical CPUs
    (so distinct hardware threads are used before any core is doubled up).
    """
    reps = physical_core_cpus()
    if requested and requested > 0:
        n = int(requested)
        try:
            allowed = sorted(os.sched_getaffinity(0))
        except AttributeError:
            allowed = list(range(os.cpu_count() or 1))
        # physical-core reps first, then the rest of the logical CPUs.
        order = reps + [c for c in allowed if c not in set(reps)]
        if not order:
            order = [0]
        core_ids = [order[i % len(order)] for i in range(n)]
        return n, core_ids
    return len(reps), list(reps)


class CpuMoeExecutor:
    """Decode-time CPU expert compute over an ``OffloadMoeCache``'s host banks
    (bf16, nvfp4, mxfp4_triton, ds_fp4 or q4_0 — see ``_WFMT_IDS`` / ``_resolve_banks``)."""

    def __init__(
        self,
        cache,
        *,
        top_k: int,
        activation: str,
        apply_router_weight_on_input: bool,
        num_threads: int,
        max_tokens: int,
        device: torch.device,
        swiglu_alpha: float = 1.702,
        swiglu_limit: float | None = None,
        fmt: str | None = None,
        ggml_types: tuple[int, int] | None = None,
    ) -> None:
        from freetoken.kernel import _cpu_moe
        from freetoken.moe.legacy_format import canonical_role

        fmt = fmt or cache.quant_format
        if fmt not in _WFMT_IDS:
            raise NotImplementedError(
                f"--moe-strategy cpu/hybrid computes experts on the CPU and supports "
                f"{sorted(_WFMT_IDS)} formats, but this checkpoint's experts are "
                f"{fmt!r}; use --moe-strategy offload (GPU-side dequant) instead."
            )
        if activation not in _ACT_IDS:
            raise NotImplementedError(f"CPU MoE backend: unsupported activation {activation!r}")
        # ABI probe: a stale prebuilt _cpu_moe.so accepts newer act ids without
        # error and silently computes the wrong activation in the generic
        # epilogue -- fail loudly with the rebuild instruction instead. (mxfp4
        # handles its act inside the kernel and predates the marker.)
        if _ACT_IDS[activation] >= 3 and fmt != "mxfp4_triton":
            supported = getattr(_cpu_moe, "max_generic_act_id", lambda: 2)()
            if _ACT_IDS[activation] > supported:
                raise RuntimeError(
                    f"the compiled _cpu_moe extension predates activation "
                    f"{activation!r} (max generic act id {supported}); rebuild it "
                    "with `python setup.py build_ext --inplace` (or reinstall the "
                    "wheel) before serving this model on the cpu/hybrid backend."
                )

        self.num_layers = int(cache.num_layers)
        self.num_experts = int(cache.num_experts)
        self.host_tier = getattr(cache, "host_tier", None)
        self._cache_ref = weakref.ref(cache) if self.host_tier is not None else None
        self.native_num_experts = _native_num_experts(cache)
        if self.host_tier is not None:
            required_slots = min(self.num_experts, int(top_k) * int(max_tokens))
            if self.native_num_experts < required_slots:
                raise ValueError(
                    f"bounded host tier has {self.native_num_experts} slots, but CPU MoE "
                    f"may route {required_slots} distinct experts in one layer"
                )
            from freetoken.moe import graph_host

            if not graph_host.available():
                raise RuntimeError(
                    "CPU MoE with a bounded host tier requires graph_host host-function "
                    "support in the active GPU runtime"
                )
        self.top_k = int(top_k)
        self.quant_format = fmt
        self.device = device
        self.max_tokens = int(max_tokens)
        self.apply_router_weight_on_input = bool(apply_router_weight_on_input)
        self.ggml_types = tuple(ggml_types) if ggml_types is not None else (2, 2)
        if fmt == "q4_0" and self.ggml_types not in ((2, 2), (12, 7)):
            raise NotImplementedError(
                "CPU GGUF MoE currently supports Q4_0/Q4_0 or Q4_K/Q5_1 banks, "
                f"got GGML types {self.ggml_types}"
            )
        if fmt == "q4_0" and self.ggml_types != (2, 2):
            supported = getattr(_cpu_moe, "supports_gguf_q4k_q5_1", lambda: False)()
            if not supported:
                raise RuntimeError(
                    "the compiled _cpu_moe extension lacks Q4_K/Q5_1 CPU support; "
                    "rebuild it with `python setup.py build_ext --inplace` before "
                    "serving this GGUF on the cpu/hybrid backend."
                )
        # The per-layer tensors and their pointer tables must outlive the executor
        # (C++ holds raw addresses into both).
        self._banks: list[torch.Tensor] = []
        ptrs, (self.H, self.I) = self._resolve_banks(
            {canonical_role(name): per_layer for name, per_layer in cache.bank_sources.items()},
            fmt,
            self.ggml_types,
        )

        # Decide the flag handshake up front (env + device + a functional stream-memop
        # probe): its coordinator needs a core of its own, which the auto thread sizing
        # below reserves (a coordinator time-slicing against the GEMV workers measurably
        # destabilizes throughput on fully-subscribed boxes).
        is_rocm = torch.version.hip is not None
        self._flag_sync = _flag_sync_platform_enabled(
            _FLAG_SYNC, device.type, is_rocm
        )
        self._cpu_moe = _cpu_moe  # module ref for the decode-path memop calls
        if _FLAG_SYNC and device.type == "cuda" and is_rocm:
            logger.info_rank0(
                "cpu-moe flag handshake is disabled on ROCm because HIP stream memops "
                "are not reliable graph dependencies; using the cudaLaunchHostFunc sync"
            )
        if self._flag_sync:
            probe_scratch = alloc_pinned_tensor(1, dtype=torch.int64)
            probe_scratch.zero_()
            probe_stream = torch.cuda.current_stream(device)
            memops_ok = _cpu_moe.memops_probe(
                probe_stream.cuda_stream, probe_scratch.data_ptr()
            )
            if memops_ok:
                memops_ok = self._probe_memops_capture(_cpu_moe, probe_scratch, device)
            if not memops_ok:
                logger.info_rank0(
                    "cpu-moe flag handshake unavailable or not graph-capturable on this "
                    "device; using the cudaLaunchHostFunc sync"
                )
                self._flag_sync = False

        nthreads, core_ids = resolve_threads_and_affinity(num_threads)
        coord_core = -1
        if self._flag_sync and num_threads == 0 and nthreads > 2:
            # Auto sizing: give the coordinator the last physical core instead of
            # oversubscribing (workers drop from N to N-1).
            coord_core = core_ids[-1]
            nthreads -= 1
            core_ids = core_ids[:-1]
        self._coord_core = coord_core
        self._ext = _cpu_moe.CpuMoeExecutor(
            num_threads=nthreads,
            num_layers=self.num_layers,
            num_experts=self.native_num_experts,
            top_k=self.top_k,
            hidden_size=self.H,
            inter_size=self.I,
            max_tokens=self.max_tokens,
            activation_id=_ACT_IDS[activation],
            apply_router_weight_on_input=1 if apply_router_weight_on_input else 0,
            weight_format=_WFMT_IDS[fmt],
            swiglu_alpha=float(swiglu_alpha),
            swiglu_limit=float(swiglu_limit) if swiglu_limit is not None else float("inf"),
            core_ids=core_ids,
            **(
                {"ggml_gate_up_type": self.ggml_types[0], "ggml_down_type": self.ggml_types[1]}
                if fmt == "q4_0" and self.ggml_types != (2, 2)
                else {}
            ),
            **ptrs,
        )
        self.num_threads = nthreads
        self.core_ids = core_ids
        self.isa = self._ext.isa_name()

        spare = len(physical_core_cpus()) - nthreads - (1 if coord_core >= 0 else 0) - 1
        clamp = max(1, min(torch.get_num_threads(), spare))
        if clamp < torch.get_num_threads():
            logger.info_rank0(
                f"torch intra-op threads: {torch.get_num_threads()} -> {clamp} "
                "(cores reserved for the pinned CPU MoE pool)"
            )
            torch.set_num_threads(clamp)

        self._io: dict[int, dict[str, torch.Tensor]] = {}
        self._tasks: dict[tuple[int, int], int] = {}
        # ctypes host-function trampolines must outlive every captured graph replay.
        # Keep one callback per task's layer and batch size, alongside the task itself.
        self._host_callbacks: dict[tuple[int, int], object] = {}

        # Flag-based handshake: mapped-pinned ready/done/err int64 arrays (one slot per
        # (MoE layer, decode batch size) pair, allocated as tasks are created) + a
        # persistent CPU coordinator that polls ready[], runs the slot's task on the
        # pool, and sets done[]. Binary per-step protocol (GPU memops: done=0, ready=1;
        # coordinator: consume ready, run, done=1; GPU waits done>=1 -- the WAIT
        # immediate is constant, so CUDA-graph replays are safe). err[] is raised by the
        # WATCHDOG thread when a ready flag stays unanswered (dead coordinator): it
        # poisons done to unblock the stream and raise_if_unhealthy() turns the step
        # into a loud error instead of silent stale output. Buffers are kept alive on
        # self so the coordinator's pinned pointers stay valid for the executor's
        # lifetime (flag_sync itself was decided above, before thread sizing).
        self._ready = self._done = self._err = None
        self._flag_slots: dict[tuple[int, int], int] = {}  # (layer_id, bs) -> slot
        self._flag_capacity = self.num_layers * _FLAG_SLOTS_PER_LAYER
        if self._flag_sync:
            self._ready = alloc_pinned_tensor(self._flag_capacity, dtype=torch.int64)
            self._done = alloc_pinned_tensor(self._flag_capacity, dtype=torch.int64)
            self._err = alloc_pinned_tensor(self._flag_capacity, dtype=torch.int64)
            self._ready.zero_()
            self._done.zero_()
            self._err.zero_()
            self._ext.start_flag_coordinator(
                self._ready.data_ptr(), self._done.data_ptr(), self._flag_capacity,
                self._coord_core,
            )
            self._watchdog_stop = False
            # The thread target holds a WEAKREF and re-derefs it each tick: a bound
            # method would strong-reference the executor forever (the loop never ends
            # on its own), pinning the C++ worker pool and the pinned banks against GC
            # in build-many-engines scenarios. NB: the stop flag / weakref death is
            # observed only between 2 s sleeps, so teardown of the THREAD can lag up to
            # ~2 s -- it is a daemon, so neither GC of the executor (weakref breaks the
            # cycle) nor process exit waits on it.
            self._watchdog = threading.Thread(
                target=_watchdog_main,
                args=(weakref.ref(self),),
                name="cpu-moe-flag-watchdog",
                daemon=True,
            )
            self._watchdog.start()

        # ds_fp4: the reference FP8 activation round-trip is a scalar per-element chain
        # that the C++ side runs single-threaded on the CUDA host-callback thread --
        # straight on the decode critical path (~0.3ms/layer at H=4096, every worker and
        # the GPU waiting on it). When a GPU is present we run the numerically identical
        # round-trip as a captured GPU kernel BEFORE the D2H (see decode_submit) and tell
        # the C++ side to skip its own. Measured on DeepSeek-V4-Flash bs=1 decode:
        # 12.85 -> 15.65 tok/s, output bit-identical (tests/moe/test_dsfp4_prequant.py).
        self._gpu_prequant = fmt == "ds_fp4" and device.type == "cuda"
        if self._gpu_prequant:
            self._ext.set_input_prequant(True)
            logger.info_rank0(
                "ds_fp4: input FP8 round-trip moved to the GPU "
                "(bit-identical grid; the CPU-side scalar round-trip is skipped)"
            )

        logger.info_rank0(
            f"CPU MoE executor ready: threads={nthreads} (pinned to cores "
            f"{core_ids[0]}..{core_ids[-1]}) isa={self.isa} fmt={fmt} "
            f"H={self.H} I={self.I} experts={self.num_experts} "
            f"native_rows={self.native_num_experts} layers={self.num_layers} "
            f"top_k={self.top_k} act={activation} max_tokens={self.max_tokens}"
        )

    @staticmethod
    def _probe_memops_capture(
        cpu_moe, scratch: torch.Tensor, device: torch.device
    ) -> bool:
        """Require stream writes to be graph nodes, not merely accepted eagerly."""
        scratch.zero_()
        try:
            with torch.cuda.device(device):
                graph = torch.cuda.CUDAGraph()
                sink = torch.zeros(1, device=device)
                stream = torch.cuda.current_stream(device)
                torch.cuda.synchronize(device)
                with torch.cuda.graph(graph, stream=stream):
                    sink.add_(1.0)
                    # Point both writes at one scratch word: submit leaves it at 1.
                    cpu_moe.memop_submit(
                        stream.cuda_stream, scratch.data_ptr(), scratch.data_ptr(), 0
                    )
                torch.cuda.synchronize(device)
                if int(scratch[0]) != 0:
                    return False  # the memops ran eagerly while capture was open
                graph.replay()
                torch.cuda.synchronize(device)
                return int(scratch[0]) == 1
        except Exception:
            return False
        finally:
            scratch.zero_()

    def _make_table(self, layers: list[torch.Tensor]) -> torch.Tensor:
        """Build a CPU int64 tensor of per-layer base addresses for one bank.

        ``layers`` is one ``[num_experts, ...]`` tensor per layer (the per-layer host
        bank contract). The C++ side stores this table's pointer and indexes
        ``tbl[layer_id]`` at call time; both the table and the layer tensors are kept
        on ``self._banks`` (GC guard) so the raw pointers stay valid for the
        executor's lifetime.
        """
        assert len(layers) == self.num_layers, (len(layers), self.num_layers)
        table = torch.tensor([t.data_ptr() for t in layers], dtype=torch.int64)
        self._banks.append(table)
        self._banks.extend(layers)
        return table

    def _resolve_banks(
        self, banks: dict, fmt: str, ggml_types: tuple[int, int] = (2, 2)
    ) -> tuple[dict, tuple[int, int]]:
        """Return (pointer kwargs for the C++ ctor, (H, I)) for the given format.

        ``banks[name]`` is a list of ``num_layers`` ``[num_experts, ...]`` tensors
        (the per-layer host bank contract); shapes are read from the first layer so
        per-partition (TP) sizes are exact. Unused pointers are 0. Every pointer kwarg
        is actually a per-layer table's address (see ``_make_table``), not a single
        bank's -- the C++ ctor resolves ``tbl[layer_id]`` per task.
        """
        if fmt == "bf16":
            gate_up = banks["gate_up"]
            down = banks["down"]
            if gate_up[0].dtype != torch.bfloat16 or down[0].dtype != torch.bfloat16:
                raise NotImplementedError(
                    f"bf16 CPU MoE requires bf16 banks, got {gate_up[0].dtype}/{down[0].dtype}"
                )
            H = int(gate_up[0].shape[2])
            I = int(gate_up[0].shape[1] // 2)
            assert gate_up[0].shape[1] == 2 * I
            assert tuple(down[0].shape[1:]) == (H, I), (down[0].shape, H, I)
            ptrs = dict(
                gate_up_ptr=self._make_table(gate_up).data_ptr(),
                down_ptr=self._make_table(down).data_ptr(),
                gate_up_scale_ptr=0,
                gate_up_global_ptr=0,
                down_scale_ptr=0,
                down_global_ptr=0,
                gate_up_bias_ptr=0,
                down_bias_ptr=0,
            )
            return ptrs, (H, I)

        if fmt == "q4_0":
            return self._resolve_gguf_banks(banks, ggml_types)

        if fmt == "mxfp4_triton":
            return self._resolve_mxfp4_banks(banks)

        if fmt == "ds_fp4":
            return self._resolve_dsfp4_banks(banks)

        # nvfp4: packed e2m1 (2/byte) + fp8-e4m3 per-16 block scales + fp16 row globals.
        gup, gus, gug = banks["gate_up"], banks["gate_up_scale"], banks["gate_up_global"]
        dnp, dns, dng = banks["down"], banks["down_scale"], banks["down_global"]
        assert gup[0].dtype == torch.uint8 and dnp[0].dtype == torch.uint8, (gup[0].dtype, dnp[0].dtype)
        assert gus[0].element_size() == 1 and dns[0].element_size() == 1, "block scales must be 1 byte"
        assert gug[0].dtype == torch.float16 and dng[0].dtype == torch.float16, (gug[0].dtype, dng[0].dtype)
        I = int(gup[0].shape[1] // 2)
        H = int(gup[0].shape[2] * 2)
        assert gup[0].shape[1] == 2 * I
        assert H % 16 == 0 and I % 16 == 0, (H, I)
        assert tuple(dnp[0].shape[1:]) == (H, I // 2), (dnp[0].shape, H, I)
        assert tuple(gus[0].shape[1:]) == (2 * I, H // 16), (gus[0].shape, I, H)
        assert tuple(dns[0].shape[1:]) == (H, I // 16), (dns[0].shape, H, I)
        assert tuple(gug[0].shape[1:]) == (2 * I,) and tuple(dng[0].shape[1:]) == (H,)
        ptrs = dict(
            gate_up_ptr=self._make_table(gup).data_ptr(),
            down_ptr=self._make_table(dnp).data_ptr(),
            gate_up_scale_ptr=self._make_table(gus).data_ptr(),
            gate_up_global_ptr=self._make_table(gug).data_ptr(),
            down_scale_ptr=self._make_table(dns).data_ptr(),
            down_global_ptr=self._make_table(dng).data_ptr(),
            gate_up_bias_ptr=0,
            down_bias_ptr=0,
        )
        return ptrs, (H, I)

    def _resolve_gguf_banks(
        self, banks: dict, ggml_types: tuple[int, int]
    ) -> tuple[dict, tuple[int, int]]:
        """Validate native GGUF rows using each bank's own GGML type geometry."""
        from freetoken.models.gguf.dequant import row_bytes

        gate_up, down = banks["gate_up"], banks["down"]
        assert gate_up[0].dtype == torch.uint8 and down[0].dtype == torch.uint8, (
            gate_up[0].dtype, down[0].dtype,
        )
        I = int(gate_up[0].shape[1] // 2)
        H = int(down[0].shape[1])
        assert gate_up[0].shape[1] == 2 * I
        gu_type, dn_type = ggml_types
        assert int(gate_up[0].shape[2]) == row_bytes(H, gu_type), (
            gate_up[0].shape, H, gu_type
        )
        assert int(down[0].shape[2]) == row_bytes(I, dn_type), (
            down[0].shape, I, dn_type
        )
        ptrs = dict(
            gate_up_ptr=self._make_table(gate_up).data_ptr(),
            down_ptr=self._make_table(down).data_ptr(),
            gate_up_scale_ptr=0,
            gate_up_global_ptr=0,
            down_scale_ptr=0,
            down_global_ptr=0,
            gate_up_bias_ptr=0,
            down_bias_ptr=0,
        )
        return ptrs, (H, I)

    def _resolve_mxfp4_banks(self, banks: dict) -> tuple[dict, tuple[int, int]]:
        """gpt-oss mxfp4 ``mxfp4_triton`` schema: transposed split-K blocks/scales
        (N innermost) + per-output-row biases. The C++ kernel streams K and
        accumulates a contiguous N-block, so the GPU-tiled layout is read in place
        (no repack, no extra host memory). Block scales are e8m0 (1 byte / 32 K)."""
        gub, gus, gob = banks["gate_up"], banks["gate_up_scale"], banks["gate_up_bias"]
        dnb, dns, dob = banks["down"], banks["down_scale"], banks["down_bias"]
        assert gub[0].dtype == torch.uint8 and dnb[0].dtype == torch.uint8, (gub[0].dtype, dnb[0].dtype)
        assert gus[0].dtype == torch.uint8 and dns[0].dtype == torch.uint8, (gus[0].dtype, dns[0].dtype)
        assert gob[0].dtype == torch.bfloat16 and dob[0].dtype == torch.bfloat16, (gob[0].dtype, dob[0].dtype)
        # gate_up_blocks [E, H//2, 2I]; down_blocks [E, I//2, H]
        H = int(gub[0].shape[1] * 2)
        I = int(gub[0].shape[2] // 2)
        assert gub[0].shape[2] == 2 * I
        assert H % 32 == 0 and I % 32 == 0, (H, I)
        assert tuple(dnb[0].shape[1:]) == (I // 2, H), (dnb[0].shape, H, I)
        assert tuple(gus[0].shape[1:]) == (H // 32, 2 * I), (gus[0].shape, H, I)
        assert tuple(dns[0].shape[1:]) == (I // 32, H), (dns[0].shape, H, I)
        assert tuple(gob[0].shape[1:]) == (2 * I,) and tuple(dob[0].shape[1:]) == (H,)
        ptrs = dict(
            gate_up_ptr=self._make_table(gub).data_ptr(),
            down_ptr=self._make_table(dnb).data_ptr(),
            gate_up_scale_ptr=self._make_table(gus).data_ptr(),
            gate_up_global_ptr=0,
            down_scale_ptr=self._make_table(dns).data_ptr(),
            down_global_ptr=0,
            gate_up_bias_ptr=self._make_table(gob).data_ptr(),
            down_bias_ptr=self._make_table(dob).data_ptr(),
        )
        return ptrs, (H, I)

    def _resolve_dsfp4_banks(self, banks: dict) -> tuple[dict, tuple[int, int]]:
        """DeepSeek-V4 ``ds_fp4`` schema: row-major e2m1 (2/byte) + e8m0 per-32 block
        scales, no global, no bias. Layout matches nvfp4 (K contiguous per output row),
        so the C++ GEMV reads it in place. The kernel additionally FP8-round-trips the
        activations (block 128) to match DSV4's W4A8 reference, hence the %128 dims."""
        gup, gus = banks["gate_up"], banks["gate_up_scale"]
        dnp, dns = banks["down"], banks["down_scale"]
        assert gup[0].dtype == torch.uint8 and dnp[0].dtype == torch.uint8, (gup[0].dtype, dnp[0].dtype)
        assert gus[0].element_size() == 1 and dns[0].element_size() == 1, "block scales must be 1 byte"
        I = int(gup[0].shape[1] // 2)
        H = int(gup[0].shape[2] * 2)
        assert gup[0].shape[1] == 2 * I
        assert H % 128 == 0 and I % 128 == 0, (H, I)  # FP8 activation round-trip block=128
        assert tuple(dnp[0].shape[1:]) == (H, I // 2), (dnp[0].shape, H, I)
        assert tuple(gus[0].shape[1:]) == (2 * I, H // 32), (gus[0].shape, I, H)
        assert tuple(dns[0].shape[1:]) == (H, I // 32), (dns[0].shape, H, I)
        ptrs = dict(
            gate_up_ptr=self._make_table(gup).data_ptr(),
            down_ptr=self._make_table(dnp).data_ptr(),
            gate_up_scale_ptr=self._make_table(gus).data_ptr(),
            gate_up_global_ptr=0,
            down_scale_ptr=self._make_table(dns).data_ptr(),
            down_global_ptr=0,
            gate_up_bias_ptr=0,
            down_bias_ptr=0,
        )
        return ptrs, (H, I)

    def _io_for(self, bs: int) -> dict[str, torch.Tensor]:
        io = self._io.get(bs)
        if io is None:
            io = {
                "x": alloc_pinned_tensor(bs, self.H, dtype=torch.bfloat16),
                "ids": alloc_pinned_tensor(bs, self.top_k, dtype=torch.int32),
                "w": alloc_pinned_tensor(bs, self.top_k, dtype=torch.float32),
                "y": alloc_pinned_tensor(bs, self.H, dtype=torch.bfloat16),
            }
            self._io[bs] = io
        return io

    def _task_for(self, layer_id: int, bs: int) -> int:
        key = (layer_id, bs)
        task = self._tasks.get(key)
        if task is None:
            io = self._io_for(bs)
            task = self._ext.create_task(
                layer_id,
                bs,
                io["x"].data_ptr(),
                io["ids"].data_ptr(),
                io["w"].data_ptr(),
                io["y"].data_ptr(),
            )
            self._tasks[key] = task
            if self.host_tier is not None:
                from freetoken.moe import graph_host

                self._host_callbacks[key] = graph_host.make_host_func(
                    self._host_admit_callback(layer_id, io)
                )
            # Allocate this (layer, bs) combo a flag slot and register its task with the
            # coordinator. Combos past the slot capacity keep the host-func path.
            if self._flag_sync and key not in self._flag_slots:
                slot = len(self._flag_slots)
                if slot < self._flag_capacity:
                    self._flag_slots[key] = slot
                    self._ext.register_flag_task(slot, task)
        return task

    def _host_admit_callback(self, layer_id: int, io: dict[str, torch.Tensor]):
        """Return a graph-host callback that remaps raw routes to pooled host slots."""
        cache_ref = self._cache_ref
        assert cache_ref is not None
        host_tier = self.host_tier
        num_experts = self.num_experts

        def _run() -> None:
            try:
                # This callback runs on a driver thread. It may only touch the already
                # D2H'd pinned ids and the host tier; in particular, do not call CUDA.
                with torch.inference_mode():
                    _remap_host_ids(io["ids"], host_tier, layer_id, num_experts)
            except BaseException as exc:  # never unwind into the GPU driver
                with torch.inference_mode():
                    io["ids"].fill_(-1)
                cache = cache_ref()
                if cache is not None and getattr(cache, "_admit_error", None) is None:
                    cache._admit_error = exc

        return _run

    def decode(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """One MoE layer of decode on the CPU. Returns a GPU [bs, H] tensor.

        All ops go on the current CUDA stream so the whole thing is captured into
        the active CUDA graph (the two host nodes carry the data dependency on the
        pinned buffers, which hold this step's real routing on replay)."""
        pending = self.decode_submit(layer_id, hidden_states, topk_weights, topk_ids)
        return self.decode_sync(pending)

    def decode_submit(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple:
        """Issue the D2H copies + the CPU-pool submit host node, then return without
        waiting. Lets a caller (the hybrid backend) enqueue GPU work between this and
        :meth:`decode_sync` so the CPU compute overlaps the GPU GEMM / PCIe fetch.

        ``topk_ids`` may carry ``-1`` entries (the C++ kernel skips them), so the CPU
        computes only the routes assigned to it. Returns an opaque handle to pass to
        :meth:`decode_sync`. The output tensor is allocated here so it stays live (and
        distinct from the interleaved GPU work) across the overlap window."""
        bs = hidden_states.shape[0]
        io = self._io_for(bs)

        if self._gpu_prequant:
            # DSV4: apply the reference FP8 round-trip on the GPU (the same kernel the
            # GPU W4A8 path uses -> bit-identical grid) so the CPU side reads
            # pre-quantized activations and skips its serial scalar pass.
            from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_roundtrip

            hidden_states = act_quant_fp8_roundtrip(hidden_states, block=128)

        # D2H: ship this step's activations + routing to pinned host memory.
        io["x"].copy_(hidden_states, non_blocking=True)
        io["ids"].copy_(topk_ids.to(torch.int32), non_blocking=True)
        io["w"].copy_(topk_weights.to(torch.float32), non_blocking=True)

        task = self._task_for(layer_id, bs)
        out = torch.empty_like(hidden_states)
        stream = torch.cuda.current_stream().cuda_stream
        if self.host_tier is not None:
            from freetoken.moe import graph_host

            graph_host.launch_host_func(stream, self._host_callbacks[(layer_id, bs)])
        slot = self._flag_slots.get((layer_id, bs)) if self._flag_sync else None
        if slot is not None:
            # Front-end memops: done[slot]=0 then ready[slot]=1 (the coordinator's
            # doorbell). No kernel launched; no host-func round trip.
            self._cpu_moe.memop_submit(
                stream,
                self._done.data_ptr(), self._ready.data_ptr(), slot,
            )
        else:
            self._ext.submit_with_cuda_stream(stream, task)
        return (bs, task, out, slot)

    def decode_sync(self, pending: tuple) -> torch.Tensor:
        """Issue the CPU-pool sync + the H2D result copy for a prior :meth:`decode_submit`,
        and return the GPU output tensor. With flag-sync the wait is a front-end stream
        memop on done[slot] (set by the CPU coordinator); otherwise a cudaLaunchHostFunc."""
        bs, task, out, slot = pending
        if slot is not None:
            # Front-end WAIT(done[slot] >= 1): blocks this stream's later nodes without
            # occupying an SM, so GPU utilization stays truthful during the CPU window.
            self._cpu_moe.memop_sync(
                torch.cuda.current_stream().cuda_stream, self._done.data_ptr(), slot,
            )
        else:
            stream = torch.cuda.current_stream().cuda_stream
            self._ext.sync_with_cuda_stream(stream, task)
        io = self._io[bs]
        out.copy_(io["y"], non_blocking=True)
        return out

    def _watchdog_tick(self, suspects: dict) -> None:
        """One watchdog sampling round (called every 2 s by ``_watchdog_main``).

        A slot is only declared dead when THREE things hold across >=10 s: its doorbell
        is still pending (ready==1 && done==0), it was already pending when first
        suspected, and the coordinator has served NOTHING on it since (flag_served_count
        unchanged). The served-count criterion kills the false-positive window: two
        point samples can land on the same slot's (independent, us-scale) pending
        windows under heavy external load, but a coordinator that made progress in
        between is alive by definition. ``suspects`` maps slot -> (first_seen,
        served_at_first_sight) and persists across ticks."""
        stuck = (self._ready == 1) & (self._done == 0)
        if not bool(stuck.any()):
            suspects.clear()
            return
        now = time.monotonic()
        pending = set(stuck.nonzero().flatten().tolist())
        for slot in list(suspects):
            if slot not in pending:
                del suspects[slot]
        dead = []
        for slot in pending:
            served = self._ext.flag_served_count(slot)
            first_seen, served_then = suspects.get(slot, (None, None))
            if first_seen is None or served != served_then:
                suspects[slot] = (now, served)  # new suspect, or alive-but-loaded: rearm
                continue
            if now - first_seen >= 10.0:
                dead.append(slot)
        if not dead:
            return
        logger.error(
            f"cpu-moe flag watchdog: slots {dead} unanswered for >10s with no coordinator "
            "progress (wedged/dead); poisoning done[] and failing the next step"
        )
        for i in dead:
            self._err[i] = 1
        for i in dead:
            self._done[i] = 1  # after err: unblock the stream into a checked failure
            suspects.pop(i, None)

    def raise_if_unhealthy(self) -> None:
        """Raise if the flag watchdog fired (a doorbell stayed unanswered because the
        coordinator never responded). Called by the engine once per forward -- a single
        pinned read -- so a dead coordinator surfaces as a loud error on the next step
        instead of silently shipping stale expert outputs."""
        if self._err is not None and bool((self._err != 0).any()):
            raise RuntimeError(
                "CPU MoE flag-handshake watchdog fired: a decode step's doorbell was "
                "never answered by the coordinator thread (its outputs cannot be "
                "trusted). This indicates a wedged/killed coordinator; restart the "
                "engine, or set FREETOKEN_CPU_MOE_FLAG_SYNC=0 to use the "
                "cudaLaunchHostFunc sync."
            )


def _watchdog_main(executor_ref) -> None:
    """Watchdog daemon body: weakref-deref per tick so the thread never keeps a dead
    executor alive (see the start site in ``CpuMoeExecutor.__init__``)."""
    suspects: dict = {}
    while True:
        time.sleep(2.0)
        executor = executor_ref()
        if executor is None or executor._watchdog_stop or executor._ready is None:
            return
        try:
            executor._watchdog_tick(suspects)
        finally:
            del executor  # drop the strong ref before the next sleep
