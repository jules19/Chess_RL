"""
Module 2 checkpoint: minimax, alpha-beta, quiescence, transposition table,
iterative deepening.
"""

import time

import chess
import pytest

from search.minimax import (
    best_move_iterative,
    best_move_minimax,
    minimax,
    order_moves,
)


def test_finds_mate_in_one():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    assert str(best_move_minimax(board, depth=2)) == "e1e8"


def test_captures_hanging_queen():
    # White queen on d4, attacked by the e5 pawn, defended by nothing
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/3QP3/8/PPPP1PPP/RNB1KBNR b KQkq - 0 1")
    move = best_move_minimax(board, depth=3)
    assert move.to_square == chess.D4, f"Expected exd4 queen capture, got {move}"


def test_single_legal_move_returned_instantly():
    # Back-rank check: Black's only legal move is Kh7
    board = chess.Board("4R1k1/5pp1/8/8/8/8/8/6K1 b - - 0 1")
    legal = list(board.legal_moves)
    assert len(legal) == 1
    assert best_move_minimax(board, depth=3) == legal[0]


def test_transposition_table_preserves_score_and_cuts_nodes():
    """The TT is a pure optimization: same answer, less work.

    The position matters: king-and-pawn endgames transpose constantly (the
    kings reach the same squares by different routes), so the TT shines.
    Shallow tactical searches barely transpose at all — measuring on the
    wrong position would show no benefit.
    """
    fen = "8/2k5/8/1p6/1P6/2K5/8/8 w - - 0 1"
    nodes_plain, nodes_tt = [0], [0]

    score_plain = minimax(chess.Board(fen), 6, float("-inf"), float("inf"), True, nodes_plain)
    score_tt = minimax(chess.Board(fen), 6, float("-inf"), float("inf"), True, nodes_tt, tt={})

    assert score_plain == score_tt, "TT changed the search result — that's a bug, not a speedup"
    assert nodes_tt[0] < nodes_plain[0] * 0.7, (
        f"TT should cut nodes substantially here ({nodes_plain[0]} → {nodes_tt[0]})"
    )


def test_iterative_deepening_finds_mate():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    assert str(best_move_iterative(board, max_depth=3, time_limit=10)) == "e1e8"


def test_iterative_deepening_respects_time_budget():
    """Predictive stop: never START an iteration likely to blow the budget."""
    board = chess.Board("r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1B3/PPP2PPP/R2Q1RK1 w - - 0 1")
    start = time.time()
    move = best_move_iterative(board, max_depth=8, time_limit=1.0)
    elapsed = time.time() - start
    assert move is not None
    # Soft budget: one full shallow iteration may overshoot, but not by 10x
    assert elapsed < 8.0, f"Took {elapsed:.1f}s with a 1s budget"


def test_move_ordering_prefers_captures_and_promotions():
    board = chess.Board("rnb1kbnr/pP2pppp/8/8/8/8/P1PPPPPP/RNBQKBNR w KQkq - 0 1")
    ordered = order_moves(board, list(board.legal_moves))
    # The first move should be forcing: a promotion or a capture
    first = ordered[0]
    assert first.promotion or board.is_capture(first)
