"""FT_EXPERT_REMAP: pruned experts' routings land on the expert that absorbed them."""

import torch

from freetoken.moe import expert_remap


def test_remap_rewrites_ids_per_layer_and_skips_extra_layers(tmp_path, monkeypatch):
    table = torch.arange(8).repeat(2, 1)
    table[0, 1], table[0, 5] = 0, 4          # layer 0: experts 1 and 5 pruned
    table[1, 7] = 2                          # layer 1: expert 7 pruned
    path = tmp_path / "remap.pt"
    torch.save(table, path)
    monkeypatch.setattr(expert_remap, "_PATH", str(path))
    monkeypatch.setattr(expert_remap, "_table", None)

    ids = torch.tensor([[1, 5], [7, 3]], dtype=torch.int32)
    w = torch.tensor([[0.6, 0.4], [0.7, 0.3]])
    assert expert_remap.apply(0, w, ids)[1].tolist() == [[0, 4], [7, 3]]
    assert expert_remap.apply(1, w, ids)[1].tolist() == [[1, 5], [2, 3]]
    assert expert_remap.apply(2, w, ids)[1].tolist() == ids.tolist()  # e.g. an MTP draft layer
    assert expert_remap.apply(0, w, ids)[1].dtype == torch.int32
    assert torch.equal(expert_remap.apply(0, w, ids)[0], w)


def test_soft_remap_splits_a_pruned_pick_and_keeps_weight_sums(tmp_path, monkeypatch):
    E, m = 4, 2
    ids = torch.arange(E).view(1, E, 1).repeat(1, 1, m)          # kept: [e, e]
    frac = torch.tensor([1.0, 0.0]).repeat(1, E, 1)                # kept: [1, 0]
    ids[0, 3] = torch.tensor([0, 2]); frac[0, 3] = torch.tensor([0.75, 0.25])  # expert 3 pruned
    path = tmp_path / "soft.pt"
    torch.save({"ids": ids, "w": frac}, path)
    monkeypatch.setattr(expert_remap, "_PATH", str(path))
    monkeypatch.setattr(expert_remap, "_table", None)

    topk_ids = torch.tensor([[3, 1]], dtype=torch.int32)
    topk_w = torch.tensor([[0.8, 0.2]])
    w, out = expert_remap.apply(0, topk_w, topk_ids)
    assert out.tolist() == [[0, 2, 1, 1]]
    torch.testing.assert_close(w, torch.tensor([[0.6, 0.2, 0.2, 0.0]]))
    torch.testing.assert_close(w.sum(-1), topk_w.sum(-1))
