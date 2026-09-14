"""Building the draft head beside a loaded target model.

The head is not part of the target's module tree. It is a separate model that borrows the
target's token embedding and LM head, so keeping it outside leaves the target's strict
``load_state_dict`` alone and lets a checkpoint that ships a head serve normally when
speculation is off.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

logger = init_logger(__name__)

# The head's own tensors arrive under this prefix; the module is loaded without it.
_PREFIX = "mtp."


def build_mtp_head(model_path: str, config: "ModelConfig", *, device, dtype):
    """Construct the draft head and load its dense weights, or return ``None``.

    ``config`` must be the engine's config, which has already moved the draft block into
    the full-attention group -- that is what gives the head's block a KV layer.
    """
    from freetoken.models.register import _load_attr, get_model_spec
    from freetoken.utils.torch_utils import torch_dtype

    if not config.num_speculative_tokens or not config.num_nextn_layers:
        return None
    spec = get_model_spec(config.architectures[0])
    if spec.mtp_head is None or spec.iter_mtp_weights is None:
        raise NotImplementedError(
            f"{config.architectures[0]} has no MTP draft head adapter; speculative "
            "decoding needs one"
        )
    layer_id = config.num_layers
    head_cls = _load_attr(spec.module, spec.mtp_head)
    # Meta, like the target: the routed bank is served by the offload cache and must not
    # be materialized here (256 experts of it), and the dense tensors are assigned by the
    # load below rather than copied into pre-allocated storage.
    with torch_dtype(dtype), torch.device("meta"):
        head = head_cls(config, layer_id)

    iter_mtp_weights = _load_attr(spec.module, spec.iter_mtp_weights)
    reference = head.state_dict()
    # The reader hands back bf16; the module is built in the engine's compute dtype, which
    # on a GPU without bf16 hardware is fp16. Cast to what the module declares, the same
    # way the target's own weight load does.
    state = {}
    for name, tensor in iter_mtp_weights(model_path):
        key = name.removeprefix(_PREFIX)
        expected = reference.get(key)
        state[key] = tensor.to(
            device=device, dtype=expected.dtype if expected is not None else tensor.dtype
        )
    # The routed experts are not in that stream: they live in the packed banks the offload
    # cache serves, indexed by layer, and the head's block is one layer past the target's.
    missing = [k for k in reference if k not in state]
    routed = [k for k in missing if ".mlp.experts." in k]
    unexpected = [k for k in state if k not in reference]
    assert not unexpected, f"draft head weights with no home: {unexpected}"
    assert set(missing) == set(routed), (
        f"draft head is missing non-expert weights: {sorted(set(missing) - set(routed))}"
    )
    for key in routed:
        state[key] = reference[key]  # stays meta; the offload cache serves these
    dense = [k for k in state if k not in routed]
    # A head that loaded nothing would run on meta tensors and produce noise rather than
    # fail, so the count is checked rather than just logged.
    assert dense, f"draft head loaded no dense weights (reference had {len(reference)} keys)"
    head.load_state_dict(state)
    logger.info_rank0(
        f"MTP draft head: block {layer_id}, {len(dense)} dense tensors, "
        f"{len(routed)} routed banks from the offload cache"
    )
    return head


__all__ = ["build_mtp_head"]
