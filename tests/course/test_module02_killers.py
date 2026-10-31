"""Module 2, exercise 3 — killer moves."""

import chess
import pytest


def test_killer_moves_keep_the_score_and_cut_nodes_in_quiet_positions():
    """Contract: search/minimax.py's minimax() gains a keyword argument

        minimax(board, depth, alpha, beta, maximizing, nodes_searched=None,
                tt=None, killers=None)

    killers=None searches exactly as before. killers={} turns killer moves on:
    the dict maps depth -> up to two quiet (non-capture) moves that caused a
    cutoff at that depth, tried right after the hash move; pass the same dict
    down the recursion. Killers change the order moves are tried in, never the
    result, so the score must be identical. On the quiet starting position
    they should save a lot of work (a reference solution went from 13,592 to
    about 8,100 nodes at depth 4).
    """
    from search.minimax import minimax

    board = chess.Board()
    plain, with_killers = [0], [0]
    score_plain = minimax(board.copy(), 4, float("-inf"), float("inf"), True, plain)
    score_killers = minimax(board.copy(), 4, float("-inf"), float("inf"), True,
                            with_killers, killers={})
    assert score_killers == score_plain, "killers changed the result: that's a bug"
    assert with_killers[0] < 0.75 * plain[0], (
        f"expected far fewer nodes with killers ({plain[0]} -> {with_killers[0]})")
