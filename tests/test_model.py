"""
Module 4 checkpoint: network architecture and checkpoint loading.

The load_model tests are regression tests for a real bug: different call
sites hardcoded different architectures (2 blocks/64 channels vs 4/128), so
checkpoints failed to load — or loaded into the wrong network. load_model
infers the architecture from the checkpoint, making that impossible.
"""

import pytest

torch = pytest.importorskip("torch")

from net.model import PolicyValueNetwork, auto_device, create_model, load_model


def test_forward_pass_shapes():
    model = create_model(num_res_blocks=1, num_channels=16)
    x = torch.randn(2, 20, 8, 8)
    policy_logits, value = model(x)
    assert policy_logits.shape == (2, 4672)
    assert value.shape == (2, 1)


def test_value_in_tanh_range():
    model = create_model(num_res_blocks=1, num_channels=16)
    _, value = model(torch.randn(8, 20, 8, 8))
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)


def test_legal_move_masking_zeroes_illegal_probability():
    model = create_model(num_res_blocks=1, num_channels=16)
    x = torch.randn(2, 20, 8, 8)
    mask = torch.zeros(2, 4672)
    mask[:, :20] = 1.0

    probs, _ = model.get_policy_value(x, mask)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)
    assert torch.all(probs[:, 20:] < 1e-6)


@pytest.mark.parametrize("blocks,channels", [(1, 16), (2, 64), (4, 128)])
def test_load_model_infers_architecture_from_checkpoint(tmp_path, blocks, channels):
    """Whatever architecture was trained, load_model reconstructs it."""
    original = create_model(num_res_blocks=blocks, num_channels=channels)
    path = str(tmp_path / "ckpt.pt")
    torch.save({
        'model_state_dict': original.state_dict(),
        'num_res_blocks': blocks,
        'num_channels': channels,
    }, path)

    loaded, checkpoint = load_model(path, device="cpu")
    assert loaded.num_res_blocks == blocks
    assert loaded.num_channels == channels
    assert 'model_state_dict' in checkpoint

    # Same weights → same outputs
    x = torch.randn(1, 20, 8, 8)
    original.eval()
    with torch.no_grad():
        p1, v1 = original(x)
        p2, v2 = loaded(x)
    assert torch.allclose(p1, p2)
    assert torch.allclose(v1, v2)


def test_load_model_handles_legacy_bare_state_dict(tmp_path):
    """Old best_model.pt files were bare state_dicts — must still load."""
    original = create_model(num_res_blocks=2, num_channels=32)
    path = str(tmp_path / "legacy.pt")
    torch.save(original.state_dict(), path)

    loaded, _ = load_model(path, device="cpu")
    assert loaded.num_res_blocks == 2
    assert loaded.num_channels == 32


def test_auto_device_returns_valid_device():
    device = auto_device()
    assert device.type in ("cpu", "cuda", "mps")
