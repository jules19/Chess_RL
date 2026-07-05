"""Module 2, exercise 2 — mid-search abort (hard time budget)."""

import time

import chess
import pytest

# A middlegame position where the depth-4 iteration costs ~10x depth-3.
# The predictive stop assumes ~3x growth, so it happily STARTS depth 4 with
# a 6-second budget — and then blows it by roughly 10 seconds. Only a true
# mid-search abort can keep the promise.
EXPLOSIVE_FEN = "r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1B3/PPP2PPP/R2Q1RK1 w - - 0 1"


@pytest.mark.skip(reason="Course Module 2, exercise 2: implement mid-search "
                         "abort in search/minimax.py (raise a SearchTimeout "
                         "from inside minimax() when a deadline passes; catch "
                         "it in best_move_iterative and return the previous "
                         "iteration's move), then delete this skip line")
def test_time_budget_is_a_hard_promise():
    """Contract: best_move_iterative(board, max_depth, time_limit) returns
    within time_limit plus a small grace period, EVEN when an iteration's
    real cost dwarfs the predictive estimate. Suggested approach: check the
    clock inside minimax() every ~1024 nodes; on expiry raise SearchTimeout;
    best_move_iterative catches it and falls back to the last completed
    iteration's move.

    Without the abort this test takes ~18s and fails; with it, ~6s and
    passes. (Run it before implementing to watch the failure mode — that's
    the point of the exercise.)
    """
    from search.minimax import best_move_iterative

    board = chess.Board(EXPLOSIVE_FEN)
    start = time.time()
    move = best_move_iterative(board, max_depth=6, time_limit=6.0)
    elapsed = time.time() - start

    assert move in board.legal_moves
    assert elapsed < 8.0, (
        f"time_limit=6.0 but search ran {elapsed:.1f}s — the predictive "
        f"stop started an iteration it couldn't afford and nothing aborted it"
    )
