from __future__ import annotations

import os
from functools import partial
from typing import Callable, Generic, TypeVar


class BaseEnv:
    def _init(self, name: str) -> None:
        raise NotImplementedError


T = TypeVar("T")


class EnvVar(BaseEnv, Generic[T]):
    def __init__(self, default_value: T, fn: Callable[[str], T]):
        self.value = default_value
        self.fn = fn
        super().__init__()

    def _init(self, name: str) -> None:
        env_value = os.getenv(name)
        if env_value is not None:
            try:
                self.value = self.fn(env_value)
            except Exception:
                pass

    def __bool__(self):
        return self.value

    def __str__(self):
        return str(self.value)


_TO_BOOL = lambda x: x.lower() in ("1", "true", "yes")


def _PARSE_MEM_BYTES(mem: str) -> int:
    mem = mem.strip().upper()
    if not mem[-1].isalpha():
        return int(mem)
    if mem.endswith("B"):
        mem = mem[:-1]
    UNIT_MAP = {"K": 1024, "M": 1024**2, "G": 1024**3}
    return int(float(mem[:-1]) * UNIT_MAP[mem[-1]])


ENV_PREFIX = "FREETOKEN_"
EnvInt = partial(EnvVar[int], fn=int)
EnvFloat = partial(EnvVar[float], fn=float)
EnvBool = partial(EnvVar[bool], fn=_TO_BOOL)
EnvOption = partial(EnvVar[bool | None], fn=_TO_BOOL, default_value=None)
EnvMem = partial(EnvVar[int], fn=_PARSE_MEM_BYTES)
EnvStr = partial(EnvVar[str], fn=str)


class EnvClassSingleton:
    _instance: EnvClassSingleton | None = None

    # shell
    SHELL_MAX_TOKENS = EnvInt(2048)
    # None = unset -> resolved from the model's generation_config.json sampling defaults
    # (sglang's sampling_defaults='model'); set the env var to override.
    SHELL_TOP_K = EnvInt(None)
    SHELL_TOP_P = EnvFloat(None)
    SHELL_TEMPERATURE = EnvFloat(None)

    # backend runtime
    FLASHINFER_USE_TENSOR_CORES = EnvOption()
    DISABLE_OVERLAP_SCHEDULING = EnvBool(False)
    PYNCCL_MAX_BUFFER_SIZE = EnvMem(1024**3)
    # GatedDeltaNet recurrent (SSM) state dtype: float32 (default) | bfloat16 | float16.
    # fp32 matches the Qwen3.x configs (mamba_ssm_dtype); fp16/bf16 halves the GDN state
    # pool at some precision cost on the long recurrence (mirrors SGLang's mamba_ssm_dtype).
    MAMBA_SSM_DTYPE = EnvStr("float32")
    # MTP verification controls used for correctness checks.
    SPEC_SEQUENTIAL_VERIFY = EnvBool(False)
    SPEC_FORCE_REJECT = EnvBool(False)
    # Speculate on sampled (non-greedy) requests too: each verify row samples on its own, so a draft
    # survives with the target's probability for it and a rejection draws from the rest (lossless).
    SPEC_SAMPLED = EnvBool(False)
    # Lossy, opt-in: a verify row keeps the draft whenever the target gives it at least this probability.
    SPEC_ACCEPT_MIN_P = EnvFloat(None)
    # Opt-in adaptive draft depth: a chain stops growing after the first draft the head gave less than this probability.
    SPEC_DRAFT_MIN_P = EnvFloat(None)
    # Opt-in: capture the MTP verify (forward, greedy sample, draft head) into CUDA graphs per draft depth; batch size 1, else eager.
    SPEC_VERIFY_GRAPH = EnvBool(False)
    # Opt-in: --moe-disk-tier decode (and the MTP verify) runs under CUDA graphs; each MoE layer's expert misses are admitted by a host thread through stream memops.
    DISK_TIER_GRAPH = EnvBool(False)
    # Opt-in: on HIP, decode-sized (<= 4 rows) dense linears run as a Triton GEMV instead of rocBLAS GEMM tiles.
    SMALL_GEMV = EnvBool(False)
    # Resolve simultaneous stop-string matches by earliest output position; off preserves upstream list order.
    EARLIEST_STOP_MATCH = EnvBool(False)
    QWEN35_ROUTER_GEMV = EnvBool(False)
    # Opt-in: on HIP, decode-sized (<= 4 rows) Qwen3.5/3.6 and Flash-Next MoE layers compute the router logits and the shared-expert gate in one Triton GEMV launch.
    FUSED_ROUTER = EnvBool(False)
    # Opt-in: a single-request MTP verify runs each GDN layer's rows through one multi-token conv update and one
    # recurrent kernel that writes the rollback snapshots itself, instead of a decode step and two copies per row.
    GDN_FUSED_VERIFY = EnvBool(False)
    # Opt-in: requests that all set a top_k (<= 256) sample over torch.topk's candidates instead of the fused
    # cooperative-launch kernels, which on ROCm wait for every other stream's work (~10% of a decode step).
    TOPK_SAMPLER = EnvBool(False)
    # Opt-in: QSA prefill (>= 256 rows) runs as tiled batched GEMMs over the union of each tile's selected pages instead of
    # one sparse-kernel program per row; same attention, ~3x faster on RDNA2, with a byte-capped scratch.
    QSA_GEMM_PREFILL = EnvBool(False)
    # Opt-in: the GDN prefill conv reads its input and writes its output token-major (a strided view) instead of
    # transposing [total, conv_dim] to channels-first and back; the transposed copies ran at ~2.5 GB/s on RDNA2.
    GDN_CONV_CHANNEL_LAST = EnvBool(False)

    def __new__(cls):
        # single instance
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr_value = getattr(self, attr_name)
            assert isinstance(attr_value, BaseEnv)
            attr_value._init(f"{ENV_PREFIX}{attr_name}")


ENV = EnvClassSingleton()
