"""Module 1, exercise 2 — add an evaluation term."""

import chess
import pytest

# Same material on both sides in both positions; the ONLY difference is
# whether the d-file (where the white rook sits) is open or blocked by pawns.
OPEN_FILE_FEN = "4k3/ppp2ppp/8/8/8/8/PPP2PPP/3RK3 w - - 0 1"
CLOSED_FILE_FEN = "4k3/ppp1pppp/8/8/8/8/PPP1PPPP/3RK3 w - - 0 1"


@pytest.mark.skip(reason="Course Module 1, exercise 2: implement "
                         "rook_open_file_bonus(board) in engine/evaluator.py, "
                         "wire it into evaluate(), then delete this skip line")
def test_rook_on_open_file_earns_a_bonus():
    """Contract: engine/evaluator.py gains

        def rook_open_file_bonus(board) -> int   # centipawns, White-positive

    returning a positive bonus (~+25cp per rook) for each rook on a file
    with no pawns of either color, negative for Black's rooks on open files,
    and 0 when no rook sits on an open file. It must be added into
    evaluate()'s total.
    """
    from engine.evaluator import evaluate, rook_open_file_bonus

    open_board = chess.Board(OPEN_FILE_FEN)
    closed_board = chess.Board(CLOSED_FILE_FEN)

    assert rook_open_file_bonus(open_board) > 0, \
        "White rook on the open d-file should earn a bonus"
    assert rook_open_file_bonus(closed_board) == 0, \
        "A rook behind its own pawn is not on an open file"

    # Mirror check: a BLACK rook on an open file must score negative
    # (evaluation is always from White's perspective in this codebase)
    black_rook_board = chess.Board("3rk3/ppp2ppp/8/8/8/8/PPP2PPP/4K3 b - - 0 1")
    assert rook_open_file_bonus(black_rook_board) < 0

    # Integration: the term must actually reach evaluate()
    assert evaluate(open_board) > evaluate(closed_board), \
        "The bonus isn't wired into evaluate()"
