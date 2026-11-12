"""Supplied run-file handling protects results without burdening RL exercises."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from net.model import create_model
from selfplay.replay_buffer import ReplayBuffer
from selfplay import train_loop as loop
from selfplay import run_files


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_run(path, **kwargs):
    options = dict(iterations=1, games_per_iter=1, simulations=2, train_steps=1,
                   batch_size=2, arena_games=2, max_moves=4, res_blocks=1,
                   channels=8, device='cpu', seed=7, verbose=False,
                   checkpoint_dir=str(path))
    options.update(kwargs)
    return loop.run_training_loop(**options)


def contents(folder):
    return {p.relative_to(folder): p.read_bytes() for p in folder.rglob('*') if p.is_file()}


def test_rerun_refuses_before_touching_saved_results(tmp_path, monkeypatch):
    folder = tmp_path / 'run'
    tiny_run(folder)
    original = contents(folder)
    def must_not_construct_model(*args, **kwargs):
        pytest.fail('occupied directory must be refused before creating a model')
    monkeypatch.setattr(loop, 'create_model', must_not_construct_model)
    with pytest.raises(FileExistsError, match='--checkpoint-dir runs/m6_smoke_02'):
        tiny_run(folder)
    assert original == contents(folder)


def test_unrelated_files_are_preserved(tmp_path):
    (tmp_path / 'notes.txt').write_text('keep this')
    with pytest.raises(FileExistsError, match='Nothing was overwritten'):
        run_files.RunFiles.create(tmp_path)
    assert list(tmp_path.iterdir()) == [tmp_path / 'notes.txt']


def test_config_records_loaded_architecture_and_effective_settings(tmp_path):
    warm_start = tmp_path / 'warm.pt'
    run_files.save_checkpoint(create_model(num_res_blocks=1, num_channels=8), warm_start, 0)
    folder = tmp_path / 'run'
    tiny_run(folder, init_checkpoint=warm_start, res_blocks=4, channels=128)
    config = json.loads((folder / 'config.json').read_text())
    settings = config['settings']
    assert (settings['res_blocks'], settings['channels']) == (1, 8)
    assert settings['arena_simulations'] == 2
    assert settings['seed'] == 7 and settings['device'] == 'cpu'
    assert settings['init_checkpoint'] == str(warm_start)
    assert not settings['keep_history']
    assert config['torch_version'] and config['created_utc']


def test_supplied_loader_returns_a_complete_learning_state(tmp_path):
    tiny_run(tmp_path / 'run')
    saved = run_files.load_run(tmp_path / 'run')
    assert saved.iteration == 1 and len(saved.buffer) == 4
    assert saved.buffer.buffer.maxlen == 50_000
    assert (saved.champion.num_res_blocks, saved.champion.num_channels) == (1, 8)
    assert saved.files.log_text.splitlines()[-1].startswith('1,4,')
    saved.require_architecture()
    saved.require_architecture(1, 8)
    with pytest.raises(ValueError, match='architecture'):
        saved.require_architecture(channels=16)


def test_default_retains_only_last_save_and_no_candidates(tmp_path):
    folder = tmp_path / 'run'
    tiny_run(folder, iterations=3)
    saved = run_files.load_run(folder)
    assert saved.iteration == 3 and len(saved.files.log_text.splitlines()) == 4
    snapshots = list((folder / 'iterations').iterdir())
    assert len(snapshots) == 1
    assert not list(folder.rglob('candidate*.pt'))
    assert (snapshots[0] / 'champion.pt').read_bytes() == (folder / 'champion.pt').read_bytes()
    # When supported, exported names share storage rather than duplicate models.
    try:
        os.link(folder / 'config.json', folder / 'link_probe')
    except OSError:
        return
    assert os.path.samefile(snapshots[0] / 'champion.pt', folder / 'champion.pt')


def test_keep_history_retains_champions_candidates_and_buffers(tmp_path):
    folder = tmp_path / 'run'
    tiny_run(folder, iterations=2, keep_history=True)
    saved = run_files.load_run(folder)
    assert saved.files.keep_history
    snapshots = list((folder / 'iterations').iterdir())
    assert len(snapshots) == 2
    for snapshot in snapshots:
        assert {p.name for p in snapshot.iterdir()} == {
            'champion.pt', 'candidate.pt', 'replay_buffer.npz', 'run_log.csv'}
    assert len(saved.files.log_text.splitlines()) == 3


def test_interrupted_write_preserves_last_save_then_next_save_cleans_up(tmp_path, monkeypatch):
    folder = tmp_path / 'run'
    tiny_run(folder)
    saved = run_files.load_run(folder)
    old_marker = (folder / 'completed.json').read_bytes()
    real_save = saved.buffer.save
    def interrupted_save(path):
        Path(path).write_bytes(b'incomplete buffer')
        raise OSError('simulated interruption')
    monkeypatch.setattr(saved.buffer, 'save', interrupted_save)
    with pytest.raises(OSError, match='interruption'):
        saved.files.save(saved.champion, saved.buffer, 2, '2,4,0,0,0.5,0\n')
    assert (folder / 'completed.json').read_bytes() == old_marker
    assert run_files.load_run(folder).iteration == 1
    monkeypatch.setattr(saved.buffer, 'save', real_save)
    saved.files.save(saved.champion, saved.buffer, 2, '2,4,0,0,0.5,0\n')
    assert run_files.load_run(folder).iteration == 2
    assert len(list((folder / 'iterations').iterdir())) == 1


def test_export_interruption_still_leaves_new_complete_save_loadable(tmp_path, monkeypatch):
    folder = tmp_path / 'run'
    tiny_run(folder)
    saved = run_files.load_run(folder)
    def interrupted_export(*args):
        raise OSError('export interrupted')
    monkeypatch.setattr(run_files, '_export', interrupted_export)
    with pytest.raises(OSError, match='export interrupted'):
        saved.files.save(saved.champion, saved.buffer, 2, '2,4,0,0,0.5,0\n')
    assert run_files.load_run(folder).iteration == 2
    # Top-level exports are stale, but are never used for recovery.
    assert (folder / 'run_log.csv').read_text().splitlines()[-1].startswith('1,')


@pytest.mark.parametrize('damage', ['missing_marker', 'buffer', 'config', 'path'])
def test_supplied_loader_rejects_damaged_saves(tmp_path, damage):
    folder = tmp_path / 'run'
    tiny_run(folder)
    marker_path = folder / 'completed.json'
    marker = json.loads(marker_path.read_text())
    if damage == 'missing_marker':
        marker_path.unlink()
    elif damage == 'buffer':
        (folder / marker['snapshot'] / 'replay_buffer.npz').write_bytes(b'partial')
    elif damage == 'config':
        config = json.loads((folder / 'config.json').read_text())
        config['settings']['channels'] = 16
        (folder / 'config.json').write_text(json.dumps(config))
    else:
        marker['snapshot'] = '../elsewhere'
        marker_path.write_text(json.dumps(marker))
    with pytest.raises((ValueError, FileNotFoundError)):
        run_files.load_run(folder)


def test_export_falls_back_to_copy_without_hard_links(tmp_path, monkeypatch):
    def no_links(*args):
        raise OSError('links unavailable')
    monkeypatch.setattr(run_files.os, 'link', no_links)
    tiny_run(tmp_path / 'run')
    saved = run_files.load_run(tmp_path / 'run')
    assert saved.iteration == 1
    assert (tmp_path / 'run/run_log.csv').read_text() == saved.files.log_text
