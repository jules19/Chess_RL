"""Module 4, Part A exercise 3 — horizontal mirroring (data augmentation)."""

import chess
import numpy as np
import pytest

# Castling-less middlegame position (mirroring castling rights is the trap:
# a kingside castle mirrors into a QUEENSIDE castle — using a '-' castling
# position keeps this exercise focused on geometry; handling castling is
# the stretch goal in the module).
FEN = "r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1B3/PPP2PPP/R2Q1RK1 w - - 0 1"


def test_horizontal_mirror_round_trips():
    """Contract: net/encoding.py gains

        def mirror_board_tensor(tensor) -> np.ndarray
            # (20, 8, 8) -> (20, 8, 8), a-file <-> h-file
        def mirror_move(move) -> chess.Move
            # e2e4 -> d2d4 (file mirrored, rank unchanged, promotion kept)

    Mirroring the TENSOR must equal encoding the MIRRORED BOARD — that
    equivalence is what lets you double your training data for free.
    """
    from net.encoding import board_to_tensor, mirror_board_tensor, mirror_move

    board = chess.Board(FEN)
    mirrored_board = board.transform(chess.flip_horizontal)

    np.testing.assert_array_equal(
        mirror_board_tensor(board_to_tensor(board)),
        board_to_tensor(mirrored_board),
        err_msg="mirror(encode(board)) must equal encode(mirror(board))",
    )

    # Mirroring twice is the identity
    tensor = board_to_tensor(board)
    np.testing.assert_array_equal(mirror_board_tensor(mirror_board_tensor(tensor)), tensor)

    # Moves mirror with rank preserved and promotion piece kept
    assert mirror_move(chess.Move.from_uci("e2e4")) == chess.Move.from_uci("d2d4")
    assert mirror_move(chess.Move.from_uci("a7a8q")) == chess.Move.from_uci("h7h8q")
