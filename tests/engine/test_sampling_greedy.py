"""Temperature 0 is greedy even when a model-default top_p rides along.

With sampling_defaults='model' the checkpoint's recommended top_p fills any field the
request left out, so a temperature-0 request arrives with top_p < 1. Treating that as
non-greedy routed it through the stochastic sampling kernel at T=1e-6 and silently
disabled speculative decoding, which only drafts for greedy requests."""

import torch

from freetoken.core import SamplingParams
from freetoken.engine.sample import Sampler


def test_temperature_zero_with_default_top_p_is_greedy():
    assert SamplingParams(temperature=0.0, top_p=0.95).is_greedy
    assert SamplingParams(temperature=0.7, top_k=1, top_p=0.8).is_greedy
    assert not SamplingParams(temperature=0.7, top_p=0.95).is_greedy


def test_an_all_greedy_batch_takes_the_argmax_path():
    class _Req:
        sampling_params = SamplingParams(temperature=0.0, top_p=0.95, top_k=20)

    class _Batch:
        reqs = [_Req(), _Req()]

    args = Sampler(torch.device("cpu"), vocab_size=16).prepare(_Batch())
    assert args.temperatures is None
