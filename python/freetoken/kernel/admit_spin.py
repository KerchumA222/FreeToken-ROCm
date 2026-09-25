"""Graph-captured disk-tier admission without host nodes (``csrc/admit/admit_spin.cu``).

:class:`SpinAdmission` owns two coherent pinned buffers (request and reply), the device
sequence counter and the host poller thread. ``launch`` records one admission kernel per
MoE layer; the kernel returns at once when the layer has no GPU-cache misses, and
otherwise asks the poller, which calls ``handler(layer, ids) -> slots``.
"""

from __future__ import annotations

import ctypes
import functools
import pathlib
from typing import Callable, Sequence

import numpy as np
import torch

_CSRC = pathlib.Path(__file__).parent / "csrc" / "admit"
_HEADER = 4
# Longest a kernel waits for the poller before it gives up (flagging an error).
TIMEOUT_S = 5.0


@functools.cache
def _module():
    from freetoken.kernel.gguf import jit_extension

    return jit_extension("freetoken_admit_spin", [str(_CSRC / "admit_spin.cu")], [str(_CSRC)])


def _coherent_host_buffer(n_int32: int) -> tuple[np.ndarray, int, int]:
    """(host view, device pointer, host pointer) of fine-grained pinned memory: the GPU
    spins on it, so its reads must see the host's writes without a kernel boundary."""
    if torch.version.hip is not None:
        rt = ctypes.CDLL("libamdhip64.so")
        malloc, get_dev = rt.hipHostMalloc, rt.hipHostGetDevicePointer
        flags = 0x1 | 0x2 | 0x40000000  # Portable | Mapped | Coherent
    else:
        rt = ctypes.CDLL("libcudart.so")
        malloc, get_dev = rt.cudaHostAlloc, rt.cudaHostGetDevicePointer
        flags = 0x1 | 0x2  # Portable | Mapped (uncached by the GPU)
    host = ctypes.c_void_p()
    rc = malloc(ctypes.byref(host), ctypes.c_size_t(n_int32 * 4), ctypes.c_uint(flags))
    if rc != 0:
        raise RuntimeError(f"coherent pinned allocation failed ({rc})")
    dev = ctypes.c_void_p()
    rc = get_dev(ctypes.byref(dev), host, ctypes.c_uint(0))
    if rc != 0:
        raise RuntimeError(f"device pointer of the pinned buffer failed ({rc})")
    view = np.ctypeslib.as_array(ctypes.cast(host, ctypes.POINTER(ctypes.c_int32)), shape=(n_int32,))
    view[:] = 0
    return view, int(dev.value), int(host.value)


class SpinAdmission:
    def __init__(self, width: int, device, handler: Callable[[int, list], Sequence[int]]):
        self.width = int(width)
        self.handler = handler
        self.error: BaseException | None = None
        self._req, self._req_dev, req_host = _coherent_host_buffer(_HEADER + self.width)
        self._resp, self._resp_dev, resp_host = _coherent_host_buffer(_HEADER + self.width)
        self._counter = torch.zeros(1, dtype=torch.int32, device=device)
        mod = _module()
        mod.start_poller(req_host, resp_host, self._serve)
        self._running = True

    def _serve(self, layer: int, n: int) -> int:
        try:
            with torch.inference_mode():
                ids = self._req[_HEADER : _HEADER + n].tolist()
                self._resp[_HEADER : _HEADER + n] = self.handler(layer, ids)
            return 0
        except BaseException as exc:  # never unwind into the poller thread
            if self.error is None:
                self.error = exc
            return 1

    def launch(self, num_indices: torch.Tensor, src_indices: torch.Tensor, layer: int) -> None:
        assert src_indices.numel() <= self.width
        _module().launch_admit(num_indices, src_indices, self._req_dev, self._resp_dev,
                               self._counter, int(layer), TIMEOUT_S)

    def take_error(self) -> BaseException | None:
        """The handler's exception, or a timeout the kernel flagged; clears both."""
        if self._req[3]:
            self._req[3] = 0
            if self.error is None:
                self.error = TimeoutError(
                    f"disk-tier admission kernel waited over {TIMEOUT_S} s for the host poller")
        exc, self.error = self.error, None
        return exc

    def close(self) -> None:
        if self._running:
            _module().stop_poller()
            self._running = False


__all__ = ["SpinAdmission", "TIMEOUT_S"]
