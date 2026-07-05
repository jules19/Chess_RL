"""
Smoke tests for the training/ package.

This directory previously had ZERO test coverage — and it showed:
training/evaluate.py shipped importing baseline player classes that did not
exist, wrapped in a try/except that printed a warning and skipped the
matches. The course's "evaluate your model against minimax" instruction
silently measured nothing. These tests exist so an assigned tool can never
again be broken without CI noticing.
"""

import textwrap

import chess
import pytest

torch = pytest.importorskip("torch")


SCHOLARS_MATE_PGN = textwrap.dedent("""\
    [Event "Test"]
    [Result "1-0"]

    1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0

    [Event "Test"]
    [Result "0-1"]

    1. f3 e5 2. g4 Qh4# 0-1

""")


def test_baseline_players_construct_and_move():
    """Regression: MinimaxPlayer/MCTSPlayer must exist and produce legal
    moves through the player interface evaluate.py's tournament uses."""
    from training.evaluate import MCTSPlayer, MinimaxPlayer

    board = chess.Board()

    move, value = MinimaxPlayer(depth=1).select_move(board)
    assert move in board.legal_moves
    assert -1.0 <= value <= 1.0

    move, _ = MCTSPlayer(num_simulations=8).select_move(board)
    assert move in board.legal_moves


def test_play_game_between_baselines_terminates():
    from training.evaluate import MinimaxPlayer, play_game

    result, moves = play_game(MinimaxPlayer(depth=1), MinimaxPlayer(depth=1),
                              max_moves=6)
    assert result in ("1-0", "0-1", "1/2-1/2")
    assert moves <= 6


def test_dataset_loads_pgn_and_yields_valid_examples(tmp_path):
    """The dataset must produce (board_tensor, move_index, value) triples
    with in-range policy indices — including from games that end in mate."""
    from training.dataset import ChessDataset

    pgn_path = tmp_path / "games.pgn"
    pgn_path.write_text(SCHOLARS_MATE_PGN)

    dataset = ChessDataset(str(pgn_path))
    assert len(dataset) > 0, "two decisive games must yield training examples"

    board_tensor, move_index, value = dataset[0]
    assert tuple(board_tensor.shape) == (20, 8, 8)
    assert 0 <= int(move_index) < 4672
    assert -1.0 <= float(value) <= 1.0
