"""Module 6 exercise 9: continue learning from a saved champion and experience.

Remove pytestmark below, then add run_training_loop(..., resume=True) and a
--resume CLI flag. File loading/validation is supplied by selfplay.run_files.load_run.
It returns saved.champion, saved.buffer, saved.iteration, saved.settings and
saved.files (the writer already carrying the old log). Reuse that writer.

Restore the saved scientific settings; iterations means ADDITIONAL iterations.
Device/verbosity can change. Use saved.require_architecture(res_blocks, channels)
for explicit size requests, and reject resume combined with init_checkpoint.

You are implementing continuation of the learning loop, not a file format.
The main regression suite tests damaged-save rejection separately. Explain why
Adam need not persist across iterations here and why the same seed is not exact
random-stream restoration. The local reference patch is a fallback after trying.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skip(reason="Module 6 exercise 9: wire saved learning state into the loop, then remove this marker")


@pytest.fixture
def completed_run(tmp_path):
    torch = pytest.importorskip('torch')
    from selfplay.train_loop import run_training_loop
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    folder = tmp_path / 'run'
    run_training_loop(checkpoint_dir=str(folder), iterations=1, games_per_iter=1,
                      simulations=2, train_steps=1, batch_size=2, arena_games=2,
                      max_moves=4, buffer_capacity=100, res_blocks=1, channels=8,
                      device='cpu', seed=7, verbose=False, promote_threshold=0.3)
    yield folder
    torch.set_num_threads(previous)


def test_resume_keeps_experience_numbering_log_and_settings(completed_run, monkeypatch):
    from selfplay.train_loop import run_training_loop
    from selfplay.run_files import load_run
    old_log = (completed_run / 'run_log.csv').read_text()
    old_config = (completed_run / 'config.json').read_bytes()
    before = load_run(completed_run)
    from selfplay import train_loop
    original_train = train_loop.train_candidate
    def check_starting_champion(model, *args, **kwargs):
        import torch
        assert all(torch.equal(value, before.champion.state_dict()[name])
                   for name, value in model.state_dict().items())
        return original_train(model, *args, **kwargs)
    monkeypatch.setattr(train_loop, 'train_candidate', check_starting_champion)
    run_training_loop(checkpoint_dir=str(completed_run), iterations=1,
                      resume=True, device='cpu', verbose=False)
    after = load_run(completed_run)
    assert after.iteration == 2
    assert len(after.buffer) == 8  # four old positions and four new ones
    for expected, actual in zip(before.buffer.buffer, after.buffer.buffer):
        import numpy as np
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)
    assert after.files.log_text.startswith(old_log)
    assert len(after.files.log_text.splitlines()) == 3
    assert after.files.log_text.splitlines()[-1].startswith('2,8,')
    assert after.champion.num_channels == 8  # omitted size inherits saved model
    assert (completed_run / 'config.json').read_bytes() == old_config
    with pytest.raises(FileExistsError):
        run_training_loop(checkpoint_dir=str(completed_run), iterations=1)


def test_resume_checks_explicit_conflicts_before_learning(completed_run):
    from selfplay.train_loop import run_training_loop
    before = (completed_run / 'completed.json').read_bytes()
    with pytest.raises(ValueError, match='architecture'):
        run_training_loop(checkpoint_dir=str(completed_run), resume=True,
                          iterations=1, channels=16, device='cpu')
    with pytest.raises(ValueError, match='init_checkpoint'):
        run_training_loop(checkpoint_dir=str(completed_run), resume=True,
                          init_checkpoint=str(completed_run / 'champion.pt'))
    assert (completed_run / 'completed.json').read_bytes() == before


def test_resume_cli_is_wired(completed_run):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([
        sys.executable, str(root / 'selfplay/train_loop.py'), '--resume',
        '--checkpoint-dir', str(completed_run), '--iterations', '1', '--device', 'cpu',
    ], cwd=root, env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'),
       capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((completed_run / 'completed.json').read_text())['iteration'] == 2
