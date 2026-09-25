import torch

from freetoken.layers.q8_act import attach, attached


def test_blocks_ride_on_the_tensor_and_not_its_views():
    with torch.inference_mode():
        x = torch.zeros(2, 4)
        q = torch.ones(3, dtype=torch.int32)
        assert attach(x, q) is x
        assert attached(x) is q
        assert attached(x.view(4, 2)) is None
        assert attached(torch.zeros(2, 4)) is None


def test_an_in_place_change_drops_the_blocks_outside_inference_mode():
    x = torch.zeros(4)
    attach(x, torch.ones(1))
    assert attached(x) is not None
    x.add_(1)
    assert attached(x) is None


def test_no_blocks_attaches_nothing():
    x = torch.zeros(3)
    attach(x, None)
    assert attached(x) is None
