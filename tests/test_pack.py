import pytest
import torch

from quantkit.quant.pack import pack_int4, unpack_int4


@pytest.mark.parametrize("K,N", [(8, 1), (16, 3), (128, 7), (2048, 11), (64, 4096)])
def test_roundtrip_exact(K, N):
    g = torch.Generator().manual_seed(K * 1000 + N)
    q = torch.randint(0, 16, (K, N), generator=g, dtype=torch.uint8)
    packed = pack_int4(q)
    assert packed.shape == (K // 8, N) and packed.dtype == torch.int32
    assert torch.equal(unpack_int4(packed), q)


def test_all_values_including_sign_bit():
    # Value 15 in the top nibble sets bit 31; a correct unpack must not sign-leak it.
    q = torch.full((8, 2), 15, dtype=torch.uint8)
    q[:, 1] = torch.arange(16)[8:].to(torch.uint8)  # 8..15 in every position
    assert torch.equal(unpack_int4(pack_int4(q)), q)
    assert pack_int4(torch.full((8, 1), 15, dtype=torch.uint8)).item() == -1  # 0xFFFFFFFF


def test_nibble_order_is_little_first():
    q = torch.arange(8, dtype=torch.uint8).view(8, 1)  # k=i holds value i
    assert pack_int4(q).item() == 0x76543210


@pytest.mark.parametrize("bad", [torch.zeros(7, 4), torch.zeros(8)])
def test_rejects_bad_shapes(bad):
    with pytest.raises(ValueError):
        pack_int4(bad.to(torch.uint8))


def test_rejects_out_of_range():
    with pytest.raises(ValueError):
        pack_int4(torch.full((8, 1), 16, dtype=torch.int32))
