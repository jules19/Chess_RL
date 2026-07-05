"""
Module 4 checkpoint: board & move encoding.

The promotion tests here are REGRESSION tests for a real bug: the original
move encoding gave queen promotions indices > 4672 (silently dropped from the
legal-move mask) and made knight promotions collide with ordinary moves. A
network trained on that encoding could never learn to promote a pawn.
See the "LESSON LEARNED" note in net/encoding.py.
"""

import chess
import numpy as np
import pytest

from net.encoding import (
    NUM_MOVES,
    board_to_tensor,
    index_to_move,
    legal_moves_mask,
    move_to_index,
)

# Positions chosen to exercise every special move type
POSITIONS = {
    "start": chess.STARTING_FEN,
    "white_promotions": "rnb1kbnr/pP2pppp/8/8/8/8/P1PPPPPP/RNBQKBNR w KQkq - 0 1",
    "black_promotions": "8/8/8/8/8/k7/4p3/K4N2 b - - 0 1",
    "castling_both_sides": "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1",
    "en_passant": "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "endgame": "8/2k5/8/8/3K4/8/8/8 w - - 0 1",
}


@pytest.mark.parametrize("name,fen", POSITIONS.items())
def test_every_legal_move_round_trips(name, fen):
    """move → index → move must be the identity for ALL legal moves."""
    board = chess.Board(fen)
    for move in board.legal_moves:
        index = move_to_index(move)
        assert 0 <= index < NUM_MOVES, f"{name}: {move} encodes out of range ({index})"
        recovered = index_to_move(index, board)
        assert recovered == move, f"{name}: {move} → {index} → {recovered}"


@pytest.mark.parametrize("name,fen", POSITIONS.items())
def test_mask_matches_legal_move_count(name, fen):
    """The legal-move mask must contain exactly one 1 per legal move."""
    board = chess.Board(fen)
    mask = legal_moves_mask(board)
    assert mask.shape == (NUM_MOVES,)
    assert int(mask.sum()) == board.legal_moves.count(), (
        f"{name}: mask has {int(mask.sum())} moves, "
        f"board has {board.legal_moves.count()} — moves are being dropped"
    )


def test_promotions_get_distinct_indices():
    """All four promotion pieces of the same pawn push must encode differently."""
    board = chess.Board("rnb1kbnr/pP2pppp/8/8/8/8/P1PPPPPP/RNBQKBNR w KQkq - 0 1")
    indices = {
        promo: move_to_index(chess.Move(chess.B7, chess.B8, promotion=promo))
        for promo in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
    }
    assert len(set(indices.values())) == 4, f"Promotion indices collide: {indices}"


def test_knight_promotion_does_not_collide_with_plain_moves():
    """The original bug: knight promos shared indices with ordinary moves."""
    knight_promo = chess.Move(chess.B7, chess.B8, promotion=chess.KNIGHT)
    plain_move = chess.Move(chess.B7, chess.B8)
    assert move_to_index(knight_promo) != move_to_index(plain_move)


def test_index_to_move_rejects_illegal_indices():
    board = chess.Board()
    # a1a1 "move" (index 0) is never legal
    assert index_to_move(0, board) is None
    # Padding region decodes to nothing
    assert index_to_move(NUM_MOVES - 1, board) is None


def test_board_tensor_shape_and_contents():
    board = chess.Board()
    tensor = board_to_tensor(board)
    assert tensor.shape == (20, 8, 8)
    assert tensor.dtype == np.float32
    assert tensor[0:6].sum() == 16  # 16 white pieces
    assert tensor[6:12].sum() == 16  # 16 black pieces
    assert tensor[16, 0, 0] == 1.0  # white to move


def test_board_tensor_en_passant_plane():
    board = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    tensor = board_to_tensor(board)
    assert tensor[19].sum() == 1.0
    assert tensor[19, 5, 5] == 1.0  # f6 = rank index 5, file index 5
