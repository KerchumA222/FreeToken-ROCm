"""Disk-tier expert admission inside a captured CUDA graph.

Eager decode reads each layer's GPU-cache misses back to the host, admits them to the host tier (reading the host misses from disk) and rewrites
``src_indices`` from expert ids to pool slots (``OffloadMoeCache._admit_from_disk``). A graph cannot read back, so the same round trip is recorded
as stream operations around a host call (``kernel/host_call``):

    D2H   miss count and ids -> pinned
    DOORBELL, then WAIT       a host thread runs ``HostExpertTier.ensure`` and writes the pool slots
    H2D   pinned slots -> src_indices

The thread answers a layer without misses without taking the GIL. Every path out of the callback releases the stream; a failed admission zeroes the
slots (so the copy that follows reads a valid pool row), parks the exception here, and the engine raises it after the step.
"""

from __future__ import annotations

import atexit
import sys

import torch

from freetoken.kernel import host_call
from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.utils import init_logger

logger = init_logger(__name__)

# the callback thread takes the GIL once per layer with misses; the default 5 ms switch interval would make it wait for the engine thread
_SWITCH_INTERVAL_S = 2e-4


class GraphAdmission:
    def __init__(self, tier, width: int, device: torch.device):
        self.tier = tier
        self.device = device
        self.error: BaseException | None = None
        self.n = alloc_pinned_tensor(1, dtype=torch.int64)
        self.n.zero_()
        self._n = self.n.numpy()
        self.resize(width)
        sys.setswitchinterval(min(sys.getswitchinterval(), _SWITCH_INTERVAL_S))
        self._call = host_call.load().HostCall(self.n.data_ptr(), self._serve)
        self._call.start()
        self._closed = False
        atexit.register(self.close)

    def resize(self, width: int) -> None:
        """Buffers for up to ``width`` misses per layer; never while a graph is being captured (pinned allocation is illegal there)."""
        self.width = int(width)
        self.ids = alloc_pinned_tensor(self.width, dtype=torch.int32)
        self.slots = alloc_pinned_tensor(self.width, dtype=torch.int32)
        self.ids.zero_()
        self.slots.zero_()
        self._ids, self._slots = self.ids.numpy(), self.slots.numpy()

    def _serve(self, task: int) -> None:
        """Host thread: admit the layer ``task - 1``'s misses; never raises."""
        n = 0
        try:
            n = int(self._n[0])
            with torch.inference_mode():
                self._slots[:n] = self.tier.ensure(task - 1, self._ids[:n].tolist())
        except BaseException as exc:
            self._slots[:n] = 0
            if self.error is None:
                self.error = exc

    def record(self, layer_id: int, num_indices: torch.Tensor, src_indices: torch.Tensor, width: int) -> None:
        """Append layer ``layer_id``'s admission to the stream being captured; ``src_indices[:width]`` holds expert ids on entry, pool slots on exit."""
        if width > self.width:
            raise RuntimeError(f"disk tier: a graph admits {width} experts per layer but the admission buffers hold {self.width}")
        stream = torch.cuda.current_stream(self.device).cuda_stream
        self.n.copy_(num_indices, non_blocking=True)
        self.ids[:width].copy_(src_indices[:width], non_blocking=True)
        self._call.submit(stream, layer_id + 1)
        self._call.wait(stream)
        src_indices[:width].copy_(self.slots[:width], non_blocking=True)

    def take_error(self) -> BaseException | None:
        exc, self.error = self.error, None
        return exc

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._call.stop()


__all__ = ["GraphAdmission"]
