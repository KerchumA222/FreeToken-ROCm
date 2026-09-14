"""Host-function nodes: the one host callback shape that survives graph capture.

Stream memops (``cuStreamWriteValue64`` and friends) are not capturable on HIP --
they run eagerly at capture time and leave no node behind -- so a flag handshake
cannot gate a captured graph there. ``hipLaunchHostFunc`` / ``cudaLaunchHostFunc``
can: the node fires once per replay and the stream does not advance until the
callback returns, which is exactly the barrier a host-side fetch needs.

The callback runs on a driver thread and must not call back into the GPU runtime.
It may touch pinned host memory, which is all the expert admission path needs.
"""

from __future__ import annotations

import ctypes
from typing import Callable

HostFuncType = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

_launcher = None
_resolved = False


def _resolve():
    global _launcher, _resolved
    if _resolved:
        return _launcher
    _resolved = True
    for lib_name, sym in (
        ("libamdhip64.so", "hipLaunchHostFunc"),
        ("libcudart.so", "cudaLaunchHostFunc"),
        ("libcuda.so.1", "cuLaunchHostFunc"),
    ):
        try:
            lib = ctypes.CDLL(lib_name)
        except OSError:
            continue
        fn = getattr(lib, sym, None)
        if fn is None:
            continue
        fn.argtypes = [ctypes.c_void_p, HostFuncType, ctypes.c_void_p]
        fn.restype = ctypes.c_int
        _launcher = fn
        return _launcher
    return None


def available() -> bool:
    return _resolve() is not None


def make_host_func(fn: Callable[[], None]) -> HostFuncType:
    """Wrap a Python callable as a driver-callable host function.

    The returned object must be kept alive for as long as any graph holding the
    node can replay; ctypes frees the trampoline with it, and the driver would
    then call into freed memory. Exceptions must not escape into the driver, so
    ``fn`` is expected to have swallowed its own.
    """

    def _trampoline(_user_data):
        fn()

    return HostFuncType(_trampoline)


def launch_host_func(stream: int, host_func: HostFuncType) -> None:
    launcher = _resolve()
    if launcher is None:
        raise RuntimeError("no host-function launcher in this GPU runtime")
    rc = launcher(ctypes.c_void_p(stream), host_func, None)
    if rc != 0:
        raise RuntimeError(f"host function launch failed with status {rc}")


__all__ = ["available", "launch_host_func", "make_host_func", "HostFuncType"]
