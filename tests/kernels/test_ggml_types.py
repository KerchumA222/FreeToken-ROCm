"""Every ggml type our GGUFs use, through every vendored ggml-cuda op, against gguf-py's CPU dequantization.

Weights are random blocks with their fp16 scale fields pinned to small exact values (so no block decodes to inf/nan), plus, when the
GGUFs are on disk (``FT_GGUF_MODELS``, default ``~/models``), real rows sliced from them. Activations are integers / 64 with one
+-127 per 32-block, which q8_1 represents exactly, so the only rounding left is fp32 accumulation and the q8_1 block sums.
"""

from __future__ import annotations

import functools
import glob
import os

import numpy as np
import pytest
import torch

from freetoken.layers.quantization.ggml_types import MATRIX_ROW_PADDING, Q2_0_SYM, dequantize_cpu, pad_rows, row_bytes, type_name
from freetoken.third_party.gguf import GGML_QUANT_SIZES
from freetoken.third_party.gguf import GGMLQuantizationType as Q

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")

# fp16 fields of each block that hold a scale or a min, by byte offset; IQ1_M spreads its scale over four nibbles instead
_SCALE_FIELDS = {
    Q.Q4_0: (0,), Q.Q4_1: (0, 2), Q.Q5_0: (0,), Q.Q5_1: (0, 2), Q.Q8_0: (0,), Q.IQ4_NL: (0,), Q.Q2_0: (0,), Q2_0_SYM: (0,),
    Q.Q2_K: (80, 82), Q.Q3_K: (108,), Q.Q4_K: (0, 2), Q.Q5_K: (0, 2), Q.Q6_K: (208,),
    Q.IQ2_XXS: (0,), Q.IQ2_XS: (0,), Q.IQ2_S: (0,), Q.IQ3_XXS: (0,), Q.IQ3_S: (0,), Q.IQ4_XS: (0,), Q.IQ1_S: (0,), Q.IQ1_M: (),
}
# every quantized type in the Qwen3.5/3.6/3.8 and Flash-Next GGUFs we serve, and the 32-wide ones whose partial MMQ tiles once over-read
TYPES = list(_SCALE_FIELDS)
_D = np.float16(2.0**-6).view(np.uint16)
_M = np.float16(2.0**-7).view(np.uint16)


def _block(ggml_type: int) -> tuple[int, int]:
    return (64, 18) if ggml_type == Q2_0_SYM else GGML_QUANT_SIZES[Q(ggml_type)]


def _random_rows(ggml_type: int, rows: int, k: int, seed: int) -> np.ndarray:
    block, size = _block(ggml_type)
    raw = np.random.default_rng(seed).integers(0, 256, (rows, k // block, size), dtype=np.uint8)
    for j, off in enumerate(_SCALE_FIELDS[ggml_type]):
        raw[..., off : off + 2] = np.array([_D if j == 0 else _M], dtype=np.uint16).view(np.uint8)
    if ggml_type == Q.IQ1_M:
        # the fp16 scale is the top nibble of each of the four u16 scale words at byte 48
        for i in range(4):
            hi = raw[..., 48 + 2 * i + 1]
            raw[..., 48 + 2 * i + 1] = (hi & 0x0F) | ((((_D >> (4 * i)) & 0xF) << 4).astype(np.uint8))
    return raw.reshape(rows, -1)


def _activation(shape, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    v = rng.integers(-127, 128, shape)
    v.reshape(*shape[:-1], -1, 32)[..., 0] = rng.choice([-127, 127], (*shape[:-1], shape[-1] // 32))
    return torch.from_numpy(v.astype(np.float32) / 64)


def _device_weight(raw: np.ndarray, k: int, ggml_type: int) -> torch.Tensor:
    """``raw`` on the GPU followed by the zero row(s) MMQ's tail reads land in, as the GGUF loaders lay it out."""
    rows, rb = raw.shape
    storage = torch.zeros(rows + pad_rows(k), rb, dtype=torch.uint8, device="cuda")
    storage[:rows].copy_(torch.from_numpy(raw))
    return storage[:rows]


def _close(actual: torch.Tensor, expected: torch.Tensor, rtol: float, what: str) -> None:
    actual, expected = actual.float().cpu(), expected.float()
    assert torch.isfinite(actual).all(), f"{what}: non-finite output"
    err = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
    worst = (actual - expected).abs().max() / expected.abs().max().clamp_min(1e-12)
    assert err < rtol and worst < 10 * rtol, f"{what}: relative error {err:.2e} (worst element {worst:.2e})"


def _cases():
    out = []
    for t in TYPES:
        block = _block(t)[0]
        ks = (640, 320, 160) if block == 32 else (1280, 2048) if block == 256 else (640, 1280, 192)
        out += [pytest.param(t, k, id=f"{type_name(t)}-K{k}") for k in ks]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# The ops on one weight


# the IQ1 grids carry a per-block delta that the kernels apply through the q8_1 block sums, stored as fp16
_RTOL = {Q.IQ1_S: 1e-3, Q.IQ1_M: 1e-3, Q.Q2_K: 1e-3, Q.Q3_K: 1e-3}


def _check_ops(ggml_type: int, raw: np.ndarray, k: int, what: str) -> None:
    from freetoken.kernel import ggml

    rows = raw.shape[0]
    ref = torch.from_numpy(dequantize_cpu(raw, ggml_type, k))
    w = _device_weight(raw, k, ggml_type)

    deq = ggml.dequantize(w, ggml_type, k, torch.float32)
    torch.testing.assert_close(deq.cpu(), ref, rtol=1e-6, atol=1e-7, msg=lambda m: f"{what} dequantize: {m}")
    ids = torch.tensor([rows - 1, 0, rows // 2, 0], dtype=torch.int32, device="cuda")
    got = ggml.get_rows(w, ggml_type, k, ids, torch.float32)
    torch.testing.assert_close(got.cpu(), ref[ids.long().cpu()], rtol=1e-6, atol=1e-7, msg=lambda m: f"{what} get_rows: {m}")

    for m in (1, 3, 8, 64):  # MMVQ, batched MMVQ, MMQ
        x = _activation((m, k), seed=m)
        expected = x @ ref.T
        kernel = "mmvq" if ggml.uses_mmvq(ggml_type, m) else "mmq" if ggml.uses_mmq(ggml_type, m, 0) else "dequant"
        _close(ggml.mul_mat(w, ggml_type, x.cuda()), expected, _RTOL.get(ggml_type, 2e-4), f"{what} mul_mat M={m} ({kernel})")
        for dtype in (torch.bfloat16, torch.float16):
            _close(ggml.mul_mat(w, ggml_type, x.to("cuda", dtype)), expected, 1e-2, f"{what} mul_mat M={m} {dtype}")
    x = _activation((1, k), seed=5).cuda()
    _close(ggml.mul_mat(w, ggml_type, x, xq=ggml.quantize_q8_1(x)), x.cpu() @ ref.T, _RTOL.get(ggml_type, 2e-4), f"{what} mul_mat with shared q8_1")


@pytest.mark.parametrize(("ggml_type", "k"), _cases())
def test_ops_match_cpu_dequant(ggml_type, k):
    _check_ops(ggml_type, _random_rows(ggml_type, 96, k, seed=k), k, type_name(ggml_type))


@pytest.mark.parametrize(("ggml_type", "k"), _cases())
def test_expert_ops_match_cpu_dequant(ggml_type, k):
    """``mul_mat_id`` (small-batch MMVQ and the MoE MMQ grid) and the fused GLU over a [E, 2I, row_bytes] bank."""
    from freetoken.kernel import ggml

    experts, n = 6, 64
    raw = _random_rows(ggml_type, experts * n, k, seed=k + 1)
    ref = torch.from_numpy(dequantize_cpu(raw, ggml_type, k)).view(experts, n, k)
    w = _device_weight(raw, k, ggml_type).view(experts, n, -1)
    for t in (1, 4, 48):
        x = _activation((t, k), seed=t)
        # a router's top-k never repeats an expert within a token; the MoE MMQ id helper relies on that
        rng = np.random.default_rng(t)
        ids = torch.from_numpy(np.stack([rng.permutation(experts)[:2] for _ in range(t)])).int()
        expected = torch.einsum("tk,tjnk->tjn", x, ref[ids.long()])
        got = ggml.mul_mat_id(w, ggml_type, x.cuda(), ids.cuda())
        tol = _RTOL.get(ggml_type, 2e-4)
        _close(got, expected, tol, f"{type_name(ggml_type)} mul_mat_id T={t}")
        per_route = _activation((t, 2, k), seed=t + 100)
        got = ggml.mul_mat_id(w, ggml_type, per_route.cuda(), ids.cuda())
        _close(got, torch.einsum("tjk,tjnk->tjn", per_route, ref[ids.long()]), tol, f"{type_name(ggml_type)} mul_mat_id T={t} per-route x")
        if t <= ggml.mmvq_max_tokens(ggml_type):
            h = torch.einsum("tk,tjnk->tjn", x, ref[ids.long()])
            gate, up = h[..., : n // 2], h[..., n // 2 :]
            got = ggml.mul_mat_id_glu(w, ggml_type, x.cuda(), ids.cuda(), "swiglu")
            _close(got, torch.nn.functional.silu(gate) * up, tol, f"{type_name(ggml_type)} mul_mat_id_glu T={t}")


# ---------------------------------------------------------------------------------------------------------------------
# Real rows from the GGUFs on disk


@functools.cache
def _real_files() -> list[str]:
    root = os.path.expanduser(os.environ.get("FT_GGUF_MODELS", "~/models"))
    files = glob.glob(f"{root}/*/*.gguf") + glob.glob(f"{root}/*/*/*.gguf")
    # whole models only (draft heads and imatrix files are not), smallest first, each split set once
    files = [p for p in files if not os.path.basename(p).startswith(("mtp-", "dflash", "qwen36-35b-a3b-imatrix")) and "-of-0" not in p or "-00001-of-" in p]
    return sorted(files, key=os.path.getsize)


@functools.cache
def _real_tensor(ggml_type: int):
    from freetoken.models.gguf.reader import gguf_tensor_index

    for path in _real_files():
        try:
            index = gguf_tensor_index(path)
        except Exception:
            continue
        for name, info in index.items():
            if info.ggml_type == ggml_type and len(info.ne) >= 2 and name.endswith(".weight"):
                return f"{os.path.basename(path)}:{name}", info
    return None


@pytest.mark.parametrize("ggml_type", [pytest.param(t, id=type_name(t)) for t in TYPES])
def test_real_rows_match_cpu_dequant(ggml_type):
    found = _real_tensor(ggml_type)
    if found is None:
        pytest.skip(f"no {type_name(ggml_type)} tensor in the GGUFs under FT_GGUF_MODELS")
    where, info = found
    k = info.ne[0]
    rows = min(96, int(np.prod(info.ne[1:])))
    rb = row_bytes(ggml_type, k)
    raw = np.array(info.data()[: rows * rb]).reshape(rows, rb)
    _check_ops(ggml_type, raw, k, where)


def test_mmq_tail_fits_in_pad_rows():
    """The loaders pad a packed matrix with ``pad_rows(K)`` zero rows; MMQ's tail read must fit in them, small K included."""
    for t in TYPES:
        for k in (32, 64, 160, 192, 320, 640, 1280, 2048, 4096, 6144):
            if k % _block(t)[0] == 0 and k % MATRIX_ROW_PADDING:
                from freetoken.layers.quantization.ggml_types import mmq_tail_bytes

                assert mmq_tail_bytes(t, k) <= pad_rows(k) * row_bytes(t, k), (type_name(t), k)


# ---------------------------------------------------------------------------------------------------------------------
# Tail reads against an unmapped page


_GUARD: dict[str, int] = {}
_PAGE = 4096
_GUARD_BYTES = 1 << 20


def _guarded_region() -> int:
    """Device address of ``_GUARD_BYTES`` mapped bytes followed by a reserved, unmapped page, so a read past them faults.

    One per process: on ROCm (gfx1030) only the first hipMem* VMM mapping of a process is usable."""
    if "va" in _GUARD:
        return _GUARD["va"]
    if getattr(torch.version, "hip", None) is None:
        pytest.skip("the guard page uses the HIP VMM API")
    import ctypes

    try:
        lib = ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("libamdhip64.so not available")

    class Prop(ctypes.Structure):
        _fields_ = [("type", ctypes.c_int), ("requestedHandleType", ctypes.c_int), ("location_type", ctypes.c_int), ("location_id", ctypes.c_int),
                    ("win32HandleMetaData", ctypes.c_void_p), ("allocFlags", ctypes.c_uint64)]

    class Access(ctypes.Structure):
        _fields_ = [("location_type", ctypes.c_int), ("location_id", ctypes.c_int), ("flags", ctypes.c_int)]

    dev = torch.cuda.current_device()
    ptr, handle = ctypes.c_void_p(), ctypes.c_void_p()
    assert lib.hipMemAddressReserve(ctypes.byref(ptr), ctypes.c_size_t(_GUARD_BYTES + _PAGE), ctypes.c_size_t(0), ctypes.c_void_p(0), ctypes.c_uint64(0)) == 0
    prop = Prop(type=1, location_type=1, location_id=dev)
    assert lib.hipMemCreate(ctypes.byref(handle), ctypes.c_size_t(_GUARD_BYTES), ctypes.byref(prop), ctypes.c_uint64(0)) == 0
    assert lib.hipMemMap(ptr, ctypes.c_size_t(_GUARD_BYTES), ctypes.c_size_t(0), handle, ctypes.c_uint64(0)) == 0
    access = Access(location_type=1, location_id=dev, flags=3)
    assert lib.hipMemSetAccess(ptr, ctypes.c_size_t(_GUARD_BYTES), ctypes.byref(access), ctypes.c_size_t(1)) == 0
    _GUARD["va"] = ptr.value
    return ptr.value


def _at_guard(raw: np.ndarray, k: int) -> torch.Tensor:
    """``raw`` plus its ``pad_rows(K)`` zero rows, ending (to 16 bytes) at the guard page; returns the ``raw`` rows."""
    rows, rb = raw.shape
    total = (rows + pad_rows(k)) * rb
    start = (_guarded_region() + _GUARD_BYTES - total) // 16 * 16

    class Buf:
        __cuda_array_interface__ = {"shape": (rows + pad_rows(k), rb), "typestr": "|u1", "data": (start, False), "version": 3}

    storage = torch.as_tensor(Buf(), device="cuda")
    storage.zero_()
    storage[:rows].copy_(torch.from_numpy(raw))
    return storage[:rows]


@pytest.mark.parametrize(("ggml_type", "k"), [pytest.param(t, k, id=f"{type_name(t)}-K{k}") for t in (Q.Q4_0, Q.Q4_1, Q.Q5_0, Q.Q5_1, Q.Q8_0, Q.IQ4_NL, Q2_0_SYM) for k in (640, 320, 192, 160) if k % _block(t)[0] == 0])
def test_tail_reads_stay_in_pad_rows(ggml_type, k):
    """The 32-wide types tile K in steps that do not divide these K: MMQ, MMVQ and the MoE kernels must read no further than the pad rows
    the loaders add (the old GGUF kernels read past the row on a partial K tile and faulted when that crossed into an unmapped page)."""
    from freetoken.kernel import ggml

    experts, n = 4, 64
    raw = _random_rows(ggml_type, experts * n, k, seed=3)
    ref = torch.from_numpy(dequantize_cpu(raw, ggml_type, k))
    w = _at_guard(raw, k)
    for m in (1, 64):
        x = _activation((m, k), seed=m)
        _close(ggml.mul_mat(w, ggml_type, x.cuda()), x @ ref.T, 2e-4, f"guarded mul_mat M={m}")
    banks = w.view(experts, n, -1)
    for t in (1, 32):
        x = _activation((t, k), seed=t)
        rng = np.random.default_rng(t)
        ids = torch.from_numpy(np.stack([rng.permutation(experts)[:2] for _ in range(t)])).int()
        expected = torch.einsum("tk,tjnk->tjn", x, ref.view(experts, n, k)[ids.long()])
        _close(ggml.mul_mat_id(banks, ggml_type, x.cuda(), ids.cuda()), expected, 2e-4, f"guarded mul_mat_id T={t}")
    torch.cuda.synchronize()
