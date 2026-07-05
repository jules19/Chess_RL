"""
UCI protocol tests — every engine stage must work through a chess GUI.

These tests drive uci/engine.py as a real subprocess over stdin/stdout,
exactly like Arena or Cute Chess would. That catches a class of bug that
in-process tests can't: a stray print() anywhere in an engine's code path
corrupts the UCI protocol and hangs the GUI.
"""

import subprocess
import sys

import chess
import pytest

# Launch via the repo-root launcher — the same entry point users configure
# in their GUI — not uci/engine.py directly. (Running uci/engine.py puts
# uci/ first on sys.path, where the engine.py FILE shadows the engine/
# PACKAGE; that plus a hardcoded path once made these tests pass locally
# and fail in CI. Test the path your users actually take.)
UCI_CMD = [sys.executable, "chess_rl_uci.py"]


def run_uci_session(commands, timeout=120):
    """Send commands to the UCI engine, return its stdout lines."""
    stdin = "\n".join(commands + ["quit"]) + "\n"
    result = subprocess.run(
        UCI_CMD, input=stdin, capture_output=True, text=True, timeout=timeout,
    )
    assert result.returncode == 0, f"engine crashed:\n{result.stderr}"
    return result.stdout.splitlines()


def extract_bestmove(lines):
    for line in lines:
        if line.startswith("bestmove"):
            return line.split()[1]
    raise AssertionError(f"no bestmove in output:\n" + "\n".join(lines))


def test_handshake_declares_all_engine_stages():
    lines = run_uci_session(["uci"])
    assert "uciok" in lines
    combo = next(l for l in lines if l.startswith("option name Engine Type"))
    for stage in ("random", "material", "minimax", "mcts", "puct"):
        assert f"var {stage}" in combo, f"stage '{stage}' missing from UCI combo"


@pytest.mark.parametrize("engine_type,extra_options", [
    ("random", []),
    ("material", []),
    ("minimax", []),
    ("mcts", ["setoption name MCTS Simulations value 50"]),
])
def test_every_classical_stage_produces_a_legal_move(engine_type, extra_options):
    lines = run_uci_session([
        "uci",
        f"setoption name Engine Type value {engine_type}",
        *extra_options,
        "isready",
        "position startpos moves e2e4",
        "go depth 2",
    ])
    move = extract_bestmove(lines)

    board = chess.Board()
    board.push_uci("e2e4")
    assert chess.Move.from_uci(move) in board.legal_moves


def test_puct_without_model_falls_back_gracefully():
    """PUCT with a missing model file must still answer the GUI (with an
    'info string' explanation), never hang or crash."""
    lines = run_uci_session([
        "uci",
        "setoption name Engine Type value puct",
        "setoption name Model File value /nonexistent/model.pt",
        "isready",
        "position startpos",
        "go depth 2",
    ])
    move = extract_bestmove(lines)
    assert chess.Move.from_uci(move) in chess.Board().legal_moves
    assert any("info string" in l and "falling back" in l for l in lines)


def test_puct_with_real_checkpoint(tmp_path):
    """End-to-end Phase 3/4 stage: save a real (untrained) checkpoint, point
    the UCI engine at it, and get a legal move out of PUCT search."""
    torch = pytest.importorskip("torch")
    from net.model import create_model

    model = create_model(num_res_blocks=1, num_channels=16)
    ckpt = tmp_path / "tiny_model.pt"
    torch.save({'model_state_dict': model.state_dict()}, str(ckpt))

    lines = run_uci_session([
        "uci",
        "setoption name Engine Type value puct",
        f"setoption name Model File value {ckpt}",
        "setoption name PUCT Simulations value 16",
        "isready",
        "position startpos",
        "go",
    ])
    move = extract_bestmove(lines)
    assert chess.Move.from_uci(move) in chess.Board().legal_moves
    assert any("Loaded model" in l for l in lines)


def test_movetime_is_honored_by_iterative_deepening():
    """'go movetime 500' must return promptly, not search to fixed depth."""
    import time
    start = time.time()
    lines = run_uci_session([
        "uci",
        "setoption name Engine Type value minimax",
        "isready",
        "position startpos",
        "go movetime 500",
    ])
    elapsed = time.time() - start
    extract_bestmove(lines)
    # generous bound: subprocess startup + one overshooting iteration
    assert elapsed < 30, f"movetime ignored? took {elapsed:.1f}s"


def test_clock_time_management_from_wtime_btime():
    """A GUI sending only wtime/btime (no movetime) still gets a move."""
    lines = run_uci_session([
        "uci",
        "isready",
        "position startpos moves e2e4 e7e5",
        "go wtime 60000 btime 60000 winc 1000 binc 1000",
    ])
    move = extract_bestmove(lines)
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    assert chess.Move.from_uci(move) in board.legal_moves
