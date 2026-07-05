"""
Module 6 checkpoint: self-play data generation, replay buffer, arena.

Uses the uniform fake network so no torch/trained model is needed for the
data-plumbing tests; the training-step test imports torch and is skipped
when it isn't installed.
"""

import random

import chess
import numpy as np
import pytest

from net.encoding import NUM_MOVES
from search.puct import uniform_policy_value
from selfplay.arena import play_arena_game, play_match
from selfplay.replay_buffer import ReplayBuffer
from selfplay.self_play import play_selfplay_game


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

def _dummy_example(seed=0):
    rng = np.random.default_rng(seed)
    state = rng.random((20, 8, 8), dtype=np.float32)
    policy = np.zeros(NUM_MOVES, dtype=np.float32)
    policy[seed % NUM_MOVES] = 1.0
    return state, policy, float(seed % 3 - 1)


def test_buffer_add_and_sample_shapes():
    buffer = ReplayBuffer(capacity=100)
    for i in range(10):
        buffer.add(*_dummy_example(i))

    states, policies, values = buffer.sample(4, rng=random.Random(0))
    assert states.shape == (4, 20, 8, 8)
    assert policies.shape == (4, NUM_MOVES)
    assert values.shape == (4,)


def test_buffer_evicts_oldest_when_full():
    buffer = ReplayBuffer(capacity=5)
    for i in range(8):
        buffer.add(*_dummy_example(i))
    assert len(buffer) == 5  # FIFO window, not unbounded growth


def test_buffer_save_load_round_trip(tmp_path):
    buffer = ReplayBuffer(capacity=50)
    for i in range(6):
        buffer.add(*_dummy_example(i))

    path = str(tmp_path / "buffer.npz")
    buffer.save(path)
    restored = ReplayBuffer.load(path)

    assert len(restored) == len(buffer)
    assert restored.buffer.maxlen == 50
    orig_state = buffer.buffer[0][0]
    rest_state = restored.buffer[0][0]
    np.testing.assert_array_equal(orig_state, rest_state)


def test_buffer_refuses_to_save_empty(tmp_path):
    with pytest.raises(ValueError):
        ReplayBuffer().save(str(tmp_path / "empty.npz"))


# ---------------------------------------------------------------------------
# Self-play game generation
# ---------------------------------------------------------------------------

def test_selfplay_game_produces_valid_training_examples():
    examples, result = play_selfplay_game(
        uniform_policy_value, simulations=8, max_moves=12,
        rng=random.Random(0),
    )
    assert result in ("1-0", "0-1", "1/2-1/2")
    assert len(examples) > 0

    for state, policy, value in examples:
        assert state.shape == (20, 8, 8)
        assert policy.shape == (NUM_MOVES,)
        assert abs(policy.sum() - 1.0) < 1e-5, "policy target must be a distribution"
        assert value in (-1.0, 0.0, 1.0)


def test_selfplay_values_alternate_perspective():
    """Consecutive positions have opposite side to move, so in a decisive
    game their outcome labels must have opposite signs."""
    examples, result = play_selfplay_game(
        uniform_policy_value, simulations=8, max_moves=12,
        rng=random.Random(3),
    )
    values = [value for _, _, value in examples]
    if result == "1/2-1/2":
        assert all(v == 0.0 for v in values)
    else:
        for prev, curr in zip(values, values[1:]):
            assert prev == -curr


# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------

def test_arena_game_returns_valid_result():
    result = play_arena_game(uniform_policy_value, uniform_policy_value,
                             simulations=4, max_moves=10)
    assert result in ("1-0", "0-1", "1/2-1/2")


def test_match_score_in_unit_interval():
    score = play_match(uniform_policy_value, uniform_policy_value,
                       num_games=2, simulations=4, max_moves=10, verbose=False)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Training step (needs torch)
# ---------------------------------------------------------------------------

def test_soft_target_loss_matches_hard_cross_entropy():
    """With a one-hot target, soft-target CE must equal nn.CrossEntropyLoss."""
    torch = pytest.importorskip("torch")
    from selfplay.train_loop import soft_target_loss

    logits = torch.randn(4, 10)
    target_idx = torch.tensor([1, 3, 5, 7])
    one_hot = torch.zeros(4, 10)
    one_hot[torch.arange(4), target_idx] = 1.0
    values = torch.zeros(4, 1)

    _, policy_loss, _ = soft_target_loss(logits, one_hot, values, values)
    expected = torch.nn.functional.cross_entropy(logits, target_idx)
    assert torch.allclose(policy_loss, expected, atol=1e-6)


def test_train_candidate_reduces_loss():
    torch = pytest.importorskip("torch")
    from net.model import create_model
    from selfplay.train_loop import soft_target_loss, train_candidate

    torch.manual_seed(0)
    model = create_model(num_res_blocks=1, num_channels=16)
    device = torch.device("cpu")
    model.to(device)

    buffer = ReplayBuffer(capacity=100)
    for i in range(32):
        buffer.add(*_dummy_example(i))

    def batch_loss():
        states, policies, values = buffer.sample(32, rng=random.Random(1))
        with torch.no_grad():
            logits, preds = model(torch.from_numpy(states))
            loss, _, _ = soft_target_loss(
                logits, torch.from_numpy(policies),
                preds, torch.from_numpy(values).unsqueeze(1),
            )
        return loss.item()

    before = batch_loss()
    train_candidate(model, buffer, device, train_steps=15, batch_size=16,
                    rng=random.Random(2), verbose=False)
    after = batch_loss()
    assert after < before, f"training did not reduce loss ({before:.3f} → {after:.3f})"
