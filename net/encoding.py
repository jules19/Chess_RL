"""
Board State Encoding - Convert chess positions to neural network tensors

This module handles the conversion between chess.Board objects and the
tensor representation used by the neural network.

Encoding scheme (20 planes × 8×8):
    Planes 0-11: Piece positions (6 piece types × 2 colors)
        0: White Pawns
        1: White Knights
        2: White Bishops
        3: White Rooks
        4: White Queens
        5: White Kings
        6: Black Pawns
        7: Black Knights
        8: Black Bishops
        9: Black Rooks
        10: Black Queens
        11: Black Kings

    Planes 12-15: Castling rights (binary)
        12: White kingside castling
        13: White queenside castling
        14: Black kingside castling
        15: Black queenside castling

    Plane 16: Side to move (all 1s = White to move, all 0s = Black to move)
    Plane 17: Fifty-move counter (normalized to [0, 1])
    Plane 18: Repetition counter (0, 1, or 2+ repetitions)
    Plane 19: En passant square (1 at en passant square, 0 elsewhere)

Move encoding:
    Maps chess.Move to integer in [0, 4671]
    Format: from_square * 73 + to_square_offset
    (Simplified - real AlphaZero uses complex move encoding)

For simplicity, we use a direct mapping:
    move_index = from_square * 64 + to_square
    This gives 64 * 64 = 4096 possible moves
    We pad to 4672 to match standard move encoding
"""

import chess
import numpy as np
import torch


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Convert a chess board to a tensor representation.

    Args:
        board: chess.Board object

    Returns:
        numpy array of shape (20, 8, 8) with binary/normalized features
    """
    # Initialize tensor
    tensor = np.zeros((20, 8, 8), dtype=np.float32)

    # Piece positions (planes 0-11)
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        rank = chess.square_rank(square)
        file = chess.square_file(square)

        # Calculate plane index
        # Piece type: PAWN=1, KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5, KING=6
        # Offset by color: White=0-5, Black=6-11
        if piece.color == chess.WHITE:
            plane = piece.piece_type - 1  # 0-5
        else:
            plane = piece.piece_type - 1 + 6  # 6-11

        tensor[plane, rank, file] = 1.0

    # Castling rights (planes 12-15)
    tensor[12, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    tensor[13, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    tensor[14, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    tensor[15, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))

    # Side to move (plane 16)
    tensor[16, :, :] = float(board.turn == chess.WHITE)

    # Fifty-move counter (plane 17) - normalized to [0, 1]
    tensor[17, :, :] = min(board.halfmove_clock, 100) / 100.0

    # Repetition counter (plane 18)
    # This would require game history, so we'll set it to 0 for now
    # In full implementation, track position repetitions
    tensor[18, :, :] = 0.0

    # En passant square (plane 19)
    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        tensor[19, ep_rank, ep_file] = 1.0

    return tensor


def tensor_to_board_debug(tensor: np.ndarray) -> str:
    """
    Convert tensor back to a debug string (for validation).

    Args:
        tensor: (20, 8, 8) numpy array

    Returns:
        Debug string showing piece positions
    """
    # This is just for debugging - not a full reconstruction
    piece_symbols = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']

    board_str = ""
    for rank in range(7, -1, -1):  # 8 down to 1
        for file in range(8):
            found = False
            for plane in range(12):
                if tensor[plane, rank, file] > 0.5:
                    board_str += piece_symbols[plane]
                    found = True
                    break
            if not found:
                board_str += '.'
        board_str += '\n'

    return board_str


def move_to_index(move: chess.Move) -> int:
    """
    Convert a chess move to an integer index.

    Simplified encoding: from_square * 64 + to_square
    This covers all possible from-to combinations.

    Args:
        move: chess.Move object

    Returns:
        Integer index in [0, 4095] (we'll pad to 4672)
    """
    from_square = move.from_square
    to_square = move.to_square

    # Base index
    index = from_square * 64 + to_square

    # For promotions, add offset based on promotion piece
    # This is simplified - real implementation needs more complex encoding
    if move.promotion:
        # Add offset for promotion type (4 pieces: N, B, R, Q)
        # Knight=2, Bishop=3, Rook=4, Queen=5
        promotion_offset = (move.promotion - 2) * 4096
        index += promotion_offset

    return index


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Convert an integer index back to a chess move.

    Args:
        index: Integer index
        board: Current chess board (needed for move validation)

    Returns:
        chess.Move object, or None if invalid
    """
    # Handle promotions
    if index >= 4096:
        promotion_type = (index // 4096) + 1  # 1-4 → 2-5 (N, B, R, Q)
        index = index % 4096
        promotion_piece = promotion_type + 1  # Convert to piece type
    else:
        promotion_piece = None

    # Decode from_square and to_square
    from_square = index // 64
    to_square = index % 64

    # Create move
    try:
        move = chess.Move(from_square, to_square, promotion=promotion_piece)

        # Validate move is legal
        if move in board.legal_moves:
            return move
        else:
            return None
    except:
        return None


def legal_moves_mask(board: chess.Board) -> np.ndarray:
    """
    Create a binary mask of legal moves for the current position.

    Args:
        board: chess.Board object

    Returns:
        numpy array of shape (4672,) with 1s for legal moves, 0s for illegal
    """
    mask = np.zeros(4672, dtype=np.float32)

    for move in board.legal_moves:
        index = move_to_index(move)
        if index < 4672:  # Sanity check
            mask[index] = 1.0

    return mask


def batch_board_to_tensor(boards: list) -> torch.Tensor:
    """
    Convert a batch of boards to a batched tensor.

    Args:
        boards: List of chess.Board objects

    Returns:
        torch.Tensor of shape (batch_size, 20, 8, 8)
    """
    tensors = [board_to_tensor(board) for board in boards]
    batch = np.stack(tensors, axis=0)
    return torch.from_numpy(batch)


def batch_legal_moves_mask(boards: list) -> torch.Tensor:
    """
    Create batched legal move masks.

    Args:
        boards: List of chess.Board objects

    Returns:
        torch.Tensor of shape (batch_size, 4672)
    """
    masks = [legal_moves_mask(board) for board in boards]
    batch = np.stack(masks, axis=0)
    return torch.from_numpy(batch)


if __name__ == "__main__":
    """Test encoding/decoding functions."""
    print("Testing Board Encoding Module...")
    print("=" * 70)

    # Test 1: Encode starting position
    print("\nTEST 1: Encode starting position")
    print("-" * 70)
    board = chess.Board()
    tensor = board_to_tensor(board)

    print(f"✓ Board shape: {tensor.shape}")
    assert tensor.shape == (20, 8, 8), f"Wrong shape: {tensor.shape}"
    print(f"✓ Data type: {tensor.dtype}")
    assert tensor.dtype == np.float32, f"Wrong dtype: {tensor.dtype}"

    # Check piece counts
    white_pieces = tensor[0:6].sum()
    black_pieces = tensor[6:12].sum()
    print(f"✓ White pieces encoded: {white_pieces} (expected: 16)")
    print(f"✓ Black pieces encoded: {black_pieces} (expected: 16)")
    assert white_pieces == 16, f"Wrong white piece count: {white_pieces}"
    assert black_pieces == 16, f"Wrong black piece count: {black_pieces}"

    # Check side to move
    side_to_move = tensor[16, 0, 0]
    print(f"✓ Side to move: {'White' if side_to_move > 0.5 else 'Black'}")
    assert side_to_move == 1.0, "Side to move encoding wrong"

    print("✅ Starting position encoding PASSED!\n")

    # Test 2: Move encoding/decoding
    print("TEST 2: Move encoding and decoding")
    print("-" * 70)

    # Test some standard opening moves
    test_moves = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("d2d4"),
        chess.Move.from_uci("g1f3"),
        chess.Move.from_uci("e7e5"),
    ]

    for move in test_moves:
        index = move_to_index(move)
        recovered = index_to_move(index, board)
        print(f"  {move} → {index} → {recovered}")
        assert recovered == move, f"Round-trip failed for {move}"

    print("✅ Move encoding round-trip PASSED!\n")

    # Test 3: Legal move mask
    print("TEST 3: Legal move mask")
    print("-" * 70)
    mask = legal_moves_mask(board)
    print(f"✓ Mask shape: {mask.shape}")
    print(f"✓ Legal moves in position: {board.legal_moves.count()}")
    print(f"✓ Mask sum (legal moves): {mask.sum()}")
    assert mask.sum() == board.legal_moves.count(), "Legal move count mismatch"
    print("✅ Legal move masking PASSED!\n")

    # Test 4: Batch encoding
    print("TEST 4: Batch encoding")
    print("-" * 70)
    boards = [chess.Board(), chess.Board(), chess.Board()]
    batch_tensor = batch_board_to_tensor(boards)
    batch_mask = batch_legal_moves_mask(boards)

    print(f"✓ Batch tensor shape: {batch_tensor.shape}")
    print(f"✓ Batch mask shape: {batch_mask.shape}")
    assert batch_tensor.shape == (3, 20, 8, 8), f"Wrong batch shape: {batch_tensor.shape}"
    assert batch_mask.shape == (3, 4672), f"Wrong mask shape: {batch_mask.shape}"
    print("✅ Batch encoding PASSED!\n")

    # Test 5: Different positions
    print("TEST 5: Encode different positions")
    print("-" * 70)

    # After 1.e4
    board_e4 = chess.Board()
    board_e4.push(chess.Move.from_uci("e2e4"))
    tensor_e4 = board_to_tensor(board_e4)

    # Check pawn moved
    assert tensor_e4[0, 1, 4] == 0.0, "Pawn should have moved from e2"
    assert tensor_e4[0, 3, 4] == 1.0, "Pawn should be on e4"
    print("✓ Position after 1.e4 encoded correctly")

    # Check side to move changed
    assert tensor_e4[16, 0, 0] == 0.0, "Should be Black's turn"
    print("✓ Side to move updated correctly")

    print("✅ Different position encoding PASSED!\n")

    # Test 6: Debug visualization
    print("TEST 6: Debug visualization")
    print("-" * 70)
    debug_str = tensor_to_board_debug(tensor)
    print("Starting position (visual check):")
    print(debug_str)
    print("✅ Debug visualization working\n")

    print("=" * 70)
    print("✅ ALL ENCODING TESTS PASSED!")
    print("=" * 70)
    print()
    print("Key statistics:")
    print(f"  - Tensor size per position: {tensor.nbytes:,} bytes")
    print(f"  - Batch of 64 positions: {tensor.nbytes * 64:,} bytes ({tensor.nbytes * 64 / 1024:.1f} KB)")
    print(f"  - Legal moves average: ~{mask.sum():.0f} per position")
