"""Module 6, exercise 5 — resumable training loops."""

import numpy as np
import pytest


@pytest.mark.skip(reason="Course Module 6, exercise 5: implement "
                         "load_resume_state(checkpoint_dir) in "
                         "selfplay/train_loop.py and a --resume flag that "
                         "uses it, then delete this skip line")
def test_resume_state_round_trips(tmp_path):
    """Contract: selfplay/train_loop.py gains

        def load_resume_state(checkpoint_dir) -> (champion, buffer, iteration)

    reading the artifacts the loop already writes: champion.pt (a
    self-describing checkpoint whose 'selfplay_iteration' key gives the
    iteration) and replay_buffer.npz. run_training_loop(..., resume=True)
    should use it to continue numbering from iteration+1 instead of
    starting over. Long RL runs die; the ones that matter are the ones you
    can restart.
    """
    torch = pytest.importorskip("torch")

    from net.model import create_model
    from selfplay.replay_buffer import ReplayBuffer
    from selfplay.train_loop import load_resume_state, save_checkpoint

    # Simulate a killed run's artifacts
    model = create_model(num_res_blocks=1, num_channels=16)
    save_checkpoint(model, str(tmp_path / "champion.pt"), iteration=3)

    buffer = ReplayBuffer(capacity=50)
    rng = np.random.default_rng(0)
    for i in range(4):
        state = rng.random((20, 8, 8), dtype=np.float32)
        policy = np.zeros(4672, dtype=np.float32)
        policy[i] = 1.0
        buffer.add(state, policy, 0.0)
    buffer.save(str(tmp_path / "replay_buffer.npz"))

    champion, restored_buffer, iteration = load_resume_state(str(tmp_path))

    assert iteration == 3
    assert len(restored_buffer) == 4
    assert champion.num_res_blocks == 1 and champion.num_channels == 16
