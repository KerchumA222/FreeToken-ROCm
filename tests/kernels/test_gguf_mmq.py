"""Regression tests for GGUF MMQ partial-tile loads and memory safety.

Guards against out-of-bounds reads on partial K-tiles (e.g. in_features=640)
where unconditional warp-wide loads reach beyond the allocated row / tensor into
unmapped pages.
"""

from __future__ import annotations

import ctypes
import pytest
import torch
from freetoken.kernel.gguf import ggml_dequantize, ggml_mul_mat_a8

PAGE_SIZE = 4096
IN_FEATURES = 640
OUT_FEATURES = 2560
_GUARDED = {}


def _guarded_region(nbytes: int) -> int:
    """Device address of a mapped region of at least ``nbytes`` followed by a reserved but
    unmapped guard page, created once per process and never freed.

    One per process because on ROCm (6.x/7.x, gfx1030) only the first hipMem* VMM mapping
    of a process is usable: a second one faults on its first write, even with the first
    still mapped -- which, before this, took down every later test in a full run."""
    if "va" in _GUARDED:
        assert nbytes <= _GUARDED["size"]
        return _GUARDED["va"]
    if getattr(torch.version, "hip", None) is None:
        pytest.skip("the guard page uses the HIP VMM API")
    try:
        lib = ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("libamdhip64.so not available")

    class MemAllocationProp(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_int),
            ("requestedHandleType", ctypes.c_int),
            ("location_type", ctypes.c_int),
            ("location_id", ctypes.c_int),
            ("win32HandleMetaData", ctypes.c_void_p),
            ("allocFlags", ctypes.c_uint64),
        ]

    class MemAccessDesc(ctypes.Structure):
        _fields_ = [
            ("location_type", ctypes.c_int),
            ("location_id", ctypes.c_int),
            ("flags", ctypes.c_int),
        ]

    mapped_size = (nbytes + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE
    ptr = ctypes.c_void_p()
    err = lib.hipMemAddressReserve(ctypes.byref(ptr), ctypes.c_size_t(mapped_size + PAGE_SIZE),
                                   ctypes.c_size_t(0), ctypes.c_void_p(0), ctypes.c_uint64(0))
    assert err == 0
    va = ptr.value
    prop = MemAllocationProp()
    prop.type = 1
    prop.location_type = 1
    prop.location_id = torch.cuda.current_device()
    handle = ctypes.c_void_p()
    err = lib.hipMemCreate(ctypes.byref(handle), ctypes.c_size_t(mapped_size), ctypes.byref(prop), ctypes.c_uint64(0))
    assert err == 0
    err = lib.hipMemMap(ctypes.c_void_p(va), ctypes.c_size_t(mapped_size), ctypes.c_size_t(0), handle, ctypes.c_uint64(0))
    assert err == 0
    desc = MemAccessDesc()
    desc.location_type = 1
    desc.location_id = torch.cuda.current_device()
    desc.flags = 3
    err = lib.hipMemSetAccess(ctypes.c_void_p(va), ctypes.c_size_t(mapped_size), ctypes.byref(desc), ctypes.c_size_t(1))
    assert err == 0
    _GUARDED.update(va=va, size=mapped_size)
    return va


def _run_guarded_mmq(qtype: int, row_bytes: int):
    """MMQ over a weight whose last byte is the last mapped byte before the guard page, so
    any read past the end of the tensor faults."""
    num_bytes = OUT_FEATURES * row_bytes
    # Sized for the largest parametrised case (Q8_0, 34 bytes per 32 weights).
    region = _guarded_region(OUT_FEATURES * IN_FEATURES // 32 * 34)
    start = region + _GUARDED["size"] - num_bytes
    assert start % 16 == 0

    class DeviceBuffer:
        def __init__(self, p):
            self.__cuda_array_interface__ = {
                "shape": (OUT_FEATURES, row_bytes),
                "typestr": "|u1",
                "data": (p, False),
                "version": 3,
            }

    w = torch.as_tensor(DeviceBuffer(start), device="cuda")
    w.zero_()
    x = torch.randn(13, IN_FEATURES, dtype=torch.float16, device="cuda")
    out = ggml_mul_mat_a8(w, x, qtype, OUT_FEATURES)
    torch.cuda.synchronize()
    return bool(torch.isfinite(out).all().item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/HIP")
@pytest.mark.parametrize(
    ("qtype", "row_bytes"),
    [
        (2, 640 // 32 * 18),  # Q4_0
        (3, 640 // 32 * 20),  # Q4_1
        (6, 640 // 32 * 22),  # Q5_0
        (7, 640 // 32 * 24),  # Q5_1
        (8, 640 // 32 * 34),  # Q8_0
    ],
)
def test_mmq_guarded_memory_partial_tile(qtype: int, row_bytes: int):
    """Confirm MMQ does not fault when weight ends at an unmapped guard page."""
    assert _run_guarded_mmq(qtype, row_bytes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA/HIP")
def test_q5_0_mmq_parity():
    """Verify Q5_0 MMQ parity against dequantized fp16 reference."""
    torch.manual_seed(42)
    in_features = 640
    out_features = 2560
    w_bytes = torch.randint(0, 256, (out_features, 20, 22), dtype=torch.uint8, device="cuda")
    scale_half = torch.tensor([0.05], dtype=torch.float16).view(torch.uint8)
    w_bytes[:, :, 0] = scale_half[0]
    w_bytes[:, :, 1] = scale_half[1]
    w_bytes = w_bytes.view(out_features, 440)

    w_fp16 = ggml_dequantize(w_bytes, 6, out_features, in_features, torch.float16)

    for batch in [7, 13, 32]:
        x = torch.randn(batch, in_features, dtype=torch.float16, device="cuda")
        out_mmq = ggml_mul_mat_a8(w_bytes, x, 6, out_features)
        out_ref = x @ w_fp16.T
        diff = (out_mmq - out_ref).abs()
        rel_err = diff / (out_ref.abs() + 1e-5)
        assert torch.isfinite(out_mmq).all()
        assert rel_err.mean().item() < 0.05
