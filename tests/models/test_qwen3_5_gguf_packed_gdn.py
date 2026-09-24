"""Packed GDN out_proj: permuting the input must equal permuting the weight's columns.

The GGUF loader reorders value heads into the kernel's order. A dense out_proj takes
that as a column permutation of its weight; a packed one cannot (columns live inside
quant blocks), so the module gathers its input back into checkpoint order instead."""

from types import SimpleNamespace

import torch

from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet
from freetoken.models.qwen3_5_moe.gguf import _v_head_permutation


def test_input_unpermute_matches_weight_column_permute():
    hk, hv, dv, hidden = 4, 16, 8, 32
    torch.manual_seed(0)
    w = torch.randn(hidden, hv * dv, dtype=torch.float64)
    x_kernel = torch.randn(3, hv * dv, dtype=torch.float64)

    perm = _v_head_permutation(hv, hk)
    cols = (perm[:, None] * dv + torch.arange(dv)).reshape(-1)
    dense = x_kernel @ w[:, cols].T

    mod = SimpleNamespace(num_k_heads=hk, num_v_heads=hv, head_v_dim=dv, _out_in_index=None)
    idx = Qwen3_5GatedDeltaNet._out_input_index(mod, torch.device("cpu"))
    packed = x_kernel.index_select(1, idx) @ w.T
    torch.testing.assert_close(packed, dense)
