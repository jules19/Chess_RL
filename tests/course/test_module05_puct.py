"""Module 5, exercise 1 — give PUCT the Module 1 knowledge."""

import chess
import pytest

# White queen on d4, en prise to the e5 pawn, defended by nothing.
# A ZERO-knowledge value function cannot see this (every non-terminal leaf
# looks like 0.0) — the uniform-priors baseline picks essentially anything.
# A MATERIAL value function sees +9 for Black one ply deep, and PUCT piles
# its visits onto the capture. Same search, different knowledge.
HANGING_QUEEN_FEN = "rnbqkbnr/pppp1ppp/8/4p3/3QP3/8/PPPP1PPP/RNB1KBNR b KQkq - 0 1"


def test_material_value_function_finds_the_hanging_queen():
    """Contract: search/puct.py gains

        def material_policy_value(board) -> (Dict[chess.Move, float], float)

    with UNIFORM priors over legal moves (like uniform_policy_value) but a
    material-count value: sum PIECE_VALUES over the board (White-positive),
    convert to the side-to-move perspective, squash to [-1, 1] (e.g. /1000
    and clamp). Mind the perspective convention at the top of puct.py —
    getting the sign wrong makes the search AVOID winning material, which
    is its own memorable lesson.
    """
    from search.puct import material_policy_value, puct_search, uniform_policy_value

    board = chess.Board(HANGING_QUEEN_FEN)

    _, visits = puct_search(board, material_policy_value, simulations=200)
    best = max(visits, key=visits.get)
    assert best == chess.Move.from_uci("e5d4"), (
        f"With material knowledge PUCT must take the free queen (exd4), "
        f"got {best} — if it actively avoids the capture, check your sign "
        f"convention (value must be from the side to move's perspective)"
    )

    # Sanity of the value function itself: after Black wins the queen,
    # the position is great for Black — but it's WHITE to move, so the
    # side-to-move value must be strongly negative.
    after_capture = board.copy()
    after_capture.push(chess.Move.from_uci("e5d4"))
    _, value = material_policy_value(after_capture)
    assert value < -0.5
