"""The grouped MoE prefill's two down-projection layouts (``moe/fused_q4_0.py``) agree with
each other and with a plain per-(token, expert) reference."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

GGML_F16 = 1


def _reference(x, gu, dn, w, ids):
    out = torch.zeros(x.shape[0], dn.shape[1], dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        for j in range(ids.shape[1]):
            e = int(ids[t, j])
            g = gu[e].float() @ x[t].float()
            half = g.shape[0] // 2
            a = torch.nn.functional.silu(g[:half]) * g[half:]
            out[t] += w[t, j] * (dn[e].float() @ a)
    return out


@pytest.mark.parametrize("skewed", [False, True])
def test_down_flip_matches_reference(monkeypatch, skewed):
    import freetoken.moe.fused_q4_0 as m
    from freetoken.layers.activation import silu_and_mul

    torch.manual_seed(0)
    e, h, inter, t, k = 16, 256, 128, 160, 4
    gu = torch.randn(e, 2 * inter, h, device="cuda", dtype=torch.float16) * 0.05
    dn = torch.randn(e, h, inter, device="cuda", dtype=torch.float16) * 0.05
    x = torch.randn(t, h, device="cuda", dtype=torch.float16)
    ids = torch.stack([torch.randperm(e, device="cuda")[:k] for _ in range(t)]).int()
    if skewed:  # experts 0 and 1 take every token: those groups pad past the flip threshold
        ids[:, 0], ids[:, 1] = 0, 1
        ids[:, 2:] = torch.randint(2, e, (t, k - 2), device="cuda", dtype=torch.int32)
        for r in range(t):  # distinct experts per token
            ids[r, 2:] = torch.randperm(e - 2, device="cuda")[: k - 2] + 2
    w = torch.softmax(torch.randn(t, k, device="cuda"), -1)
    ref = _reference(x, gu, dn, w, ids)

    def run(flip):
        monkeypatch.setattr(m, "_DOWN_FLIP_ROWS", flip)
        return m._fused_experts_grouped(
            x, gu.view(torch.uint8), dn.view(torch.uint8), w, ids, silu_and_mul, GGML_F16, GGML_F16
        ).float()

    for flip in (0, 100, 1 << 20):  # never, mixed (skewed case), always
        got = run(flip)
        rel = ((got - ref).norm() / ref.norm()).item()
        assert rel < 2e-3, f"flip={flip} rel={rel}"
