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
    assert expert_remap.apply(0, ids).tolist() == [[0, 4], [7, 3]]
    assert expert_remap.apply(1, ids).tolist() == [[1, 5], [2, 3]]
    assert expert_remap.apply(2, ids).tolist() == ids.tolist()  # e.g. an MTP draft layer
    assert expert_remap.apply(0, ids).dtype == torch.int32
