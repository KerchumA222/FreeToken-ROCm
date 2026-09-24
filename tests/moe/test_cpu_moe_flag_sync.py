"""Platform policy for the graph-replayed CPU-MoE stream handshake."""

from freetoken.moe.cpu_executor import _flag_sync_platform_enabled


def test_flag_sync_requires_cuda_and_non_rocm_runtime():
    assert _flag_sync_platform_enabled(True, "cuda", is_rocm=False)
    assert not _flag_sync_platform_enabled(True, "cuda", is_rocm=True)
    assert not _flag_sync_platform_enabled(True, "cpu", is_rocm=False)
    assert not _flag_sync_platform_enabled(False, "cuda", is_rocm=False)
