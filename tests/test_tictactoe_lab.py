"""Tests for the tic-tac-toe self-play lab (selfplay/tictactoe_lab.py)."""

from search.puct import puct_search, select_move
from selfplay.tictactoe_lab import (TicTacToeBoard, TablePolicyValue, optimal_moves,
                                    reachable_positions, run, solve)


def test_solver_knows_the_game_is_a_draw():
    assert solve(TicTacToeBoard().key()) == 0
    assert len(reachable_positions()) == 4520  # non-terminal positions; 5,478 including finished ones


def test_board_is_a_drop_in_for_chess_board_in_puct_search():
    # X: 0, 1 ; O: 3, 4 ; X to move wins with square 2
    board = TicTacToeBoard([1, 1, 0, -1, -1, 0, 0, 0, 0], turn=1)
    _, visits = puct_search(board, TablePolicyValue(), simulations=50)
    assert select_move(visits) == 2
    assert optimal_moves(board) == [2]


def test_is_checkmate_means_side_to_move_has_lost():
    board = TicTacToeBoard([1, 1, 1, -1, -1, 0, 0, 0, 0], turn=-1)
    assert board.is_game_over() and board.is_checkmate()


def test_self_play_learns_and_exploration_is_load_bearing():
    rows = run(iterations=6, games_per_iter=60, seed=0, verbose=False)
    first, last = rows[0], rows[-1]
    assert first['vs_perfect'] == 0          # untrained table loses to perfect play
    assert last['vs_perfect'] >= 50          # ...and learns to hold draws
    assert last['value_mse'] < first['value_mse']

    frozen = run(iterations=6, games_per_iter=60, seed=0, root_noise=False,
                 temperature_moves=0, verbose=False)
    assert frozen[-1]['coverage'] < 2         # no exploration: the same game forever
    assert frozen[-1]['coverage'] < last['coverage'] / 5
