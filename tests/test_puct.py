"""
Module 5 checkpoint: PUCT search (true AlphaZero-style, no rollouts).

All tests here use uniform_policy_value — a fake network with zero chess
knowledge. That's deliberate: it proves the SEARCH machinery is correct
independently of any trained model (and lets these tests run without torch).
"""

import random

import chess
import pytest

from search.puct import (
    add_dirichlet_noise,
    best_move_puct,
    puct_search,
    select_move,
    uniform_policy_value,
    visits_to_policy,
)


def test_finds_mate_in_one_with_no_chess_knowledge():
    """Terminal values are exact, so search alone finds forced mates."""
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    move = best_move_puct(board, simulations=300)
    assert str(move) == "e1e8"


def test_visit_counts_sum_to_simulations():
    board = chess.Board()
    _, visits = puct_search(board, uniform_policy_value, simulations=50)
    # Every simulation passes through exactly one root child
    assert sum(visits.values()) == 50
    assert set(visits.keys()) == set(board.legal_moves)


def test_search_explores_and_visits_all_root_moves():
    """With uniform priors and 0-value leaves, PUCT behaves like breadth-
    first exploration: every root move gets visits, none is starved."""
    board = chess.Board()
    _, visits = puct_search(board, uniform_policy_value, simulations=100)
    assert min(visits.values()) >= 1, "some root move was never explored"


def test_search_on_finished_game_raises():
    board = chess.Board("8/8/8/8/8/1q6/2q5/K7 w - - 0 1")
    board.push_san("Qb1#") if not board.is_game_over() else None
    if not board.is_game_over():
        pytest.skip("test position not terminal")
    with pytest.raises(ValueError):
        puct_search(board, uniform_policy_value, simulations=1)


def test_visits_to_policy_temperature_extremes():
    a, b = chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")
    visits = {a: 75, b: 25}

    # τ=1: proportional to visits
    proportional = visits_to_policy(visits, temperature=1.0)
    assert abs(proportional[a] - 0.75) < 1e-9

    # τ→0: all mass on the most-visited move
    greedy = visits_to_policy(visits, temperature=0.0)
    assert greedy[a] == 1.0 and greedy[b] == 0.0


def test_select_move_greedy_and_stochastic():
    a, b = chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")
    visits = {a: 90, b: 10}

    assert select_move(visits, temperature=0.0) == a

    rng = random.Random(0)
    sampled = {select_move(visits, temperature=1.0, rng=rng) for _ in range(200)}
    assert sampled == {a, b}, "τ=1 sampling should occasionally pick the minority move"


def test_dirichlet_noise_keeps_distribution_normalized():
    board = chess.Board()
    priors, _ = uniform_policy_value(board)
    noisy = add_dirichlet_noise(priors, rng=random.Random(0))
    assert set(noisy) == set(priors)
    assert abs(sum(noisy.values()) - 1.0) < 1e-9
    assert noisy != priors  # noise actually did something
