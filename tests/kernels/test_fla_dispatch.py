from freetoken.kernel.fla import utils


def test_rdna2_gdn_tiles_are_explicitly_gated(monkeypatch):
    monkeypatch.setattr(utils, "is_amd", True)
    monkeypatch.setattr(utils, "_current_triton_arch", lambda: "gfx1030")
    monkeypatch.delenv("FT_RDNA2_GDN_TILES", raising=False)
    assert not utils._use_rdna2_gdn_tiles()

    monkeypatch.setenv("FT_RDNA2_GDN_TILES", "0")
    assert not utils._use_rdna2_gdn_tiles()

    monkeypatch.setenv("FT_RDNA2_GDN_TILES", "1")
    assert utils._use_rdna2_gdn_tiles()

    monkeypatch.setattr(utils, "_current_triton_arch", lambda: "gfx1100")
    assert not utils._use_rdna2_gdn_tiles()

    monkeypatch.setattr(utils, "is_amd", False)
    monkeypatch.setattr(utils, "_current_triton_arch", lambda: "gfx1030")
    assert not utils._use_rdna2_gdn_tiles()


def test_rdna2_gate_selects_chunk_h_tiles_at_import(monkeypatch):
    import importlib

    from freetoken.kernel.fla import chunk_delta_h

    try:
        for enabled, expected in ((True, (16, 8)), (False, (32, 4))):
            with monkeypatch.context() as patch:
                patch.setattr(utils, "is_rdna2_gdn", enabled)
                patch.delenv("SGLANG_GDN_CHUNK_H_BV", raising=False)
                patch.delenv("SGLANG_GDN_CHUNK_H_NUM_WARPS", raising=False)
                importlib.reload(chunk_delta_h)
                actual = (
                    chunk_delta_h.GDN_CHUNK_H_BV,
                    chunk_delta_h.GDN_CHUNK_H_NUM_WARPS,
                )
                assert actual == expected
    finally:
        # Restore module-level tiles along with the process flags.
        importlib.reload(chunk_delta_h)


def test_rdna2_gate_selects_chunk_o_and_wy_launch_tiles(monkeypatch):
    import torch

    from freetoken.kernel.fla import chunk_o, wy_fast

    class FakeKernel:
        def __init__(self):
            self.launch = None

        def __getitem__(self, _grid):
            def run(*_args, **kwargs):
                self.launch = kwargs

            return run

    q = torch.empty((1, 2, 1, 4))
    k = torch.empty_like(q)
    v = torch.empty((1, 2, 1, 4))
    state = torch.empty((1, 1, 4, 4))
    beta = torch.empty((1, 2, 1))
    decay = torch.empty_like(beta)
    matrix = torch.empty((1, 1, 16, 16))

    for enabled, expected_o, expected_wy in (
        (True, (32, 8), (2, 1)),
        (False, (128, 4), (4, 3)),
    ):
        monkeypatch.setattr(chunk_o, "is_rdna2_gdn", enabled)
        o_kernel = FakeKernel()
        monkeypatch.setattr(chunk_o, "chunk_fwd_kernel_o", o_kernel)
        chunk_o.chunk_fwd_o(q, k, v, state)
        assert (o_kernel.launch["BK"], o_kernel.launch["num_warps"]) == expected_o

        monkeypatch.setattr(wy_fast, "is_rdna2_gdn", enabled)
        wy_kernel = FakeKernel()
        monkeypatch.setattr(wy_fast, "recompute_w_u_fwd_kernel", wy_kernel)
        wy_fast.recompute_w_u_fwd(k, v, beta, decay, matrix, None)
        assert (
            wy_kernel.launch["num_warps"],
            wy_kernel.launch["num_stages"],
        ) == expected_wy


def test_rdna2_gdn_tile_variants_preserve_factual_outputs(monkeypatch):
    import importlib

    import pytest
    import torch

    from freetoken.kernel.fla import chunk_delta_h, chunk_o, utils, wy_fast

    if not torch.cuda.is_available():
        pytest.skip("GDN tile parity requires a Triton GPU")
    if not utils.is_amd or not utils._current_triton_arch().startswith("gfx103"):
        pytest.skip("GDN tile parity is specific to gfx103x")

    torch.manual_seed(2025)
    device = torch.device("cuda")
    batch, tokens, heads, kv_heads, dim = 1, 64, 2, 1, 64
    k = torch.randn((batch, tokens, kv_heads, dim), device=device, dtype=torch.float16) * 0.1
    q = torch.randn_like(k)
    v = torch.randn((batch, tokens, heads, dim), device=device, dtype=torch.float16) * 0.1
    beta = torch.rand((batch, tokens, heads), device=device, dtype=torch.float16)
    decay = torch.randn_like(beta) * 0.01
    w = torch.randn((batch, tokens, heads, dim), device=device, dtype=torch.float16) * 0.1
    u = torch.randn_like(v) * 0.1
    initial_state = torch.randn((batch, heads, dim, dim), device=device, dtype=torch.float16) * 0.1
    initial_state_indices = torch.arange(batch, device=device, dtype=torch.int32)

    def run_chunk_h(rdna2):
        with monkeypatch.context() as patch:
            patch.setattr(utils, "is_rdna2_gdn", rdna2)
            patch.delenv("SGLANG_GDN_CHUNK_H_BV", raising=False)
            patch.delenv("SGLANG_GDN_CHUNK_H_NUM_WARPS", raising=False)
            module = importlib.reload(chunk_delta_h)
            state = initial_state.clone()
            outputs = module.chunk_gated_delta_rule_fwd_h(
                k,
                w,
                u,
                g=decay,
                initial_state=state,
                initial_state_indices=initial_state_indices,
            )
            torch.cuda.synchronize()
            return (*outputs, state)

    original_rdna2 = utils.is_rdna2_gdn
    try:
        h_ref, v_new_ref, state_ref = run_chunk_h(False)
        h_alt, v_new_alt, state_alt = run_chunk_h(True)
        torch.testing.assert_close(h_alt, h_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(v_new_alt, v_new_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(state_alt, state_ref, atol=2e-2, rtol=2e-2)

        for enabled in (False, True):
            monkeypatch.setattr(chunk_o, "is_rdna2_gdn", enabled)
            output = chunk_o.chunk_fwd_o(q, k, v, h_ref, g=decay)
            if not enabled:
                output_ref = output
            else:
                torch.cuda.synchronize()
                torch.testing.assert_close(output, output_ref, atol=2e-2, rtol=2e-2)

        matrix = torch.randn(
            (batch, tokens, heads, tokens), device=device, dtype=torch.float16
        ) * 0.01
        for enabled in (False, True):
            monkeypatch.setattr(wy_fast, "is_rdna2_gdn", enabled)
            w_out, u_out = wy_fast.recompute_w_u_fwd(k, v, beta, decay, matrix, None)
            if not enabled:
                w_ref, u_ref = w_out, u_out
            else:
                torch.cuda.synchronize()
                torch.testing.assert_close(w_out, w_ref, atol=2e-2, rtol=2e-2)
                torch.testing.assert_close(u_out, u_ref, atol=2e-2, rtol=2e-2)
    finally:
        with monkeypatch.context() as patch:
            patch.setattr(utils, "is_rdna2_gdn", original_rdna2)
            importlib.reload(chunk_delta_h)
