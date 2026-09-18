import zlib

import pytest

from freetoken.utils import mp
from freetoken.utils.mp import NUM_ZMQ_CHANNELS, zmq_addr


@pytest.fixture
def tcp(monkeypatch):
    """Force the loopback-TCP fallback taken where libzmq has no ipc:// transport."""
    monkeypatch.setattr(mp, "_HAS_IPC", False)


@pytest.fixture
def ipc(monkeypatch):
    monkeypatch.setattr(mp, "_HAS_IPC", True)


def test_ipc_transport_used_when_available(ipc):
    assert zmq_addr(2, "-1234") == "ipc:///tmp/freetoken_2-1234"


def test_tcp_ports_stay_below_the_dynamic_range(tcp):
    # Windows hands out ephemeral ports from 49152 up; a bind there can lose to an
    # unrelated outbound connection. Sweep the whole crc32 image, not a sample.
    for block in (0, mp._ZMQ_PORT_BLOCKS - 1):
        base = mp._ZMQ_PORT_BASE + block * NUM_ZMQ_CHANNELS
        for channel in range(NUM_ZMQ_CHANNELS):
            assert 1024 < base + channel < 49152


def test_tcp_channels_are_distinct_and_deterministic(tcp):
    ports = {zmq_addr(channel, "-777") for channel in range(NUM_ZMQ_CHANNELS)}
    assert len(ports) == NUM_ZMQ_CHANNELS
    assert zmq_addr(0, "-777") == zmq_addr(0, "-777")


def test_tcp_address_does_not_depend_on_process_hash_salt(tcp):
    block = zlib.crc32(b"-4242") % mp._ZMQ_PORT_BLOCKS
    port = mp._ZMQ_PORT_BASE + block * NUM_ZMQ_CHANNELS + 3
    assert zmq_addr(3, "-4242") == f"tcp://127.0.0.1:{port}"


def test_distinct_instances_get_distinct_blocks(tcp):
    assert zmq_addr(0, "-1") != zmq_addr(0, "-2")


@pytest.mark.parametrize("channel", [-1, NUM_ZMQ_CHANNELS])
def test_channel_id_is_bounds_checked(channel):
    with pytest.raises(AssertionError):
        zmq_addr(channel, "-1")
