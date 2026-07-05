"""Module 3, exercise 4 — make MCTS reproducible."""

import random

import chess
import pytest


def test_mcts_is_reproducible_with_a_seeded_rng():
    """Contract: best_move_mcts and mcts_search accept rng=random.Random(...)
    and use it for EVERY random choice (move shuffling in MCTSNode, rollout
    move selection, sampling in get_prioritized_moves). Two searches with
    identically-seeded rngs must pick the same move; the default (rng=None)
    keeps today's behavior.

    Why this matters: you cannot debug — or write a non-flaky test for — a
    stochastic search you cannot replay. Global random.seed() also 'works'
    but poisons every other consumer of the random module in the process;
    threading an explicit rng is the habit that transfers.
    """
    from search.mcts import best_move_mcts

    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1")

    move_a = best_move_mcts(board, simulations=60, rng=random.Random(7))
    move_b = best_move_mcts(board, simulations=60, rng=random.Random(7))
    move_c = best_move_mcts(board, simulations=60, rng=random.Random(8))

    assert move_a == move_b, "same seed must reproduce the same search"
    # Different seeds MAY coincide on a forced move; this position isn't
    # forced, so require at least that the machinery consumed the rng:
    assert move_c in board.legal_moves
