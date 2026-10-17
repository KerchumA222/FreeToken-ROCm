import pytest

from freetoken.server.args import parse_args


def test_kv_cache_dtype_defaults_to_upstream_storage():
    args, _ = parse_args(["--model", "/models/anon", "--dtype", "bfloat16"])
    assert args.kv_cache_dtype == "model"


@pytest.mark.parametrize("cache_dtype", ["q8_0", "turboquant_4bit_nc"])
def test_kv_cache_dtype_can_be_opted_into(cache_dtype):
    args, _ = parse_args([
        "--model", "/models/anon", "--dtype", "bfloat16",
        "--kv-cache-dtype", cache_dtype,
    ])
    assert args.kv_cache_dtype == cache_dtype
