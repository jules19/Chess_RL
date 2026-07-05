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

Move encoding (4672-dim policy vector):
    Indices 0-4095: "from-to" moves
        move_index = from_square * 64 + to_square
        Queen promotions ALSO live in this range: a pawn moving to the last
        rank is decoded as a queen promotion (the from-to pair is enough to
        identify it, given the board).

    Indices 4096-4239: underpromotions (knight, bishop, rook)
        A pawn promotion is fully described by:
        - side (white promotes rank 7→8, black promotes rank 2→1): 2 options
        - promotion piece (N, B, R): 3 options
        - direction (capture-left, push, capture-right): 3 options
        - from-file (a-h): 8 options
        index = 4096 + side*72 + promo*24 + direction*8 + from_file
        Total: 2 * 3 * 3 * 8 = 144 indices

    Indices 4240-4671: unused padding (kept so the policy head matches the
        conventional 4672-dim AlphaZero output size).

⚠️ LESSON LEARNED (a real bug that lived in this file):
    The original encoding added `(promotion_piece - 2) * 4096` for promotions.
    Knight promotions (offset 0) collided with ordinary moves, and queen
    promotions (offset 12288) overflowed the 4672-dim policy and were
    silently dropped from the legal-move mask. The network could literally
    never learn to promote a pawn — and no test caught it, because the tests
    only round-tripped ordinary opening moves. The regression tests in
    tests/test_encoding.py now round-trip EVERY legal move in positions that
    include promotions. Moral: test the edge cases of your data encoding
    before you spend hours training on it.
"""

import chess
import numpy as np

# Size of the policy output vector
NUM_MOVES = 4672

# Underpromotion encoding constants (see module docstring)
UNDERPROMOTION_OFFSET = 4096
UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
NUM_UNDERPROMOTION_INDICES = 2 * 3 * 3 * 8  # side * piece * direction * file


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
    Convert a chess move to an integer index in [0, NUM_MOVES).

    Ordinary moves AND queen promotions: from_square * 64 + to_square.
    (A queen promotion is unambiguous from the from-to pair alone: the only
    way a pawn reaches the last rank is by promoting, and we treat queen as
    the default promotion piece.)

    Underpromotions (N, B, R) get dedicated indices starting at
    UNDERPROMOTION_OFFSET — see the module docstring for the layout.

    Args:
        move: chess.Move object

    Returns:
        Integer index in [0, 4239] (always < NUM_MOVES)
    """
    if move.promotion and move.promotion != chess.QUEEN:
        # Underpromotion: encode (side, piece, direction, from-file)
        from_rank = chess.square_rank(move.from_square)
        from_file = chess.square_file(move.from_square)
        to_file = chess.square_file(move.to_square)

        side_idx = 0 if from_rank == 6 else 1  # 0 = White (rank 7→8), 1 = Black (rank 2→1)
        promo_idx = UNDERPROMOTION_PIECES.index(move.promotion)
        dir_idx = (to_file - from_file) + 1  # -1/0/+1 → 0/1/2

        return (UNDERPROMOTION_OFFSET
                + side_idx * 72
                + promo_idx * 24
                + dir_idx * 8
                + from_file)

    # Ordinary move, or queen promotion (queen is the default on decode)
    return move.from_square * 64 + move.to_square


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Convert an integer index back to a chess move.

    The board is needed for two things:
    1. Detecting queen promotions (a pawn moving to the last rank in the
       from-to range must be promoting, and queen is the default piece)
    2. Validating that the decoded move is actually legal

    Args:
        index: Integer index in [0, NUM_MOVES)
        board: Current chess board

    Returns:
        chess.Move object, or None if the index doesn't decode to a legal move
    """
    if index >= UNDERPROMOTION_OFFSET:
        # Underpromotion: decode (side, piece, direction, from-file)
        offset = index - UNDERPROMOTION_OFFSET
        if offset >= NUM_UNDERPROMOTION_INDICES:
            return None  # Padding region — never a valid move

        side_idx, rem = divmod(offset, 72)
        promo_idx, rem = divmod(rem, 24)
        dir_idx, from_file = divmod(rem, 8)

        from_rank, to_rank = (6, 7) if side_idx == 0 else (1, 0)
        to_file = from_file + (dir_idx - 1)
        if not 0 <= to_file <= 7:
            return None  # Capture off the edge of the board

        move = chess.Move(
            chess.square(from_file, from_rank),
            chess.square(to_file, to_rank),
            promotion=UNDERPROMOTION_PIECES[promo_idx],
        )
    else:
        from_square, to_square = divmod(index, 64)

        # A pawn arriving on the last rank must promote; queen is the default
        promotion = None
        piece = board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN and chess.square_rank(to_square) in (0, 7):
            promotion = chess.QUEEN

        move = chess.Move(from_square, to_square, promotion=promotion)

    return move if move in board.legal_moves else None


def legal_moves_mask(board: chess.Board) -> np.ndarray:
    """
    Create a binary mask of legal moves for the current position.

    Args:
        board: chess.Board object

    Returns:
        numpy array of shape (4672,) with 1s for legal moves, 0s for illegal
    """
    mask = np.zeros(NUM_MOVES, dtype=np.float32)

    for move in board.legal_moves:
        # No "if index < NUM_MOVES" guard here — the old version had one, and
        # it silently swallowed the promotion-encoding bug for weeks. If the
        # encoding ever produces an out-of-range index, we WANT a loud crash.
        mask[move_to_index(move)] = 1.0

    return mask


def mirror_board_tensor(tensor: np.ndarray) -> np.ndarray:
    """
    Reference solution — course Module 4, Part A exercise 3.

    Mirror a board tensor horizontally (a-file <-> h-file). The tensor is
    (plane, rank, file), so this is a flip along axis 2. Every plane is
    either per-square (piece positions, en passant — flipping is exactly
    right) or board-constant (side to move, counters — flipping is a no-op).

    ⚠️ Castling planes are board-constant too, which is only correct for
    positions WITHOUT castling rights: mirroring a real kingside right into
    a queenside right needs plane 12<->13 and 14<->15 swaps AND the rooks/
    king to match — that's the stretch goal, and why the course test uses a
    '-' castling position.

    Args:
        tensor: (20, 8, 8) board encoding

    Returns:
        New (20, 8, 8) array, mirrored
    """
    return np.flip(tensor, axis=2).copy()


def mirror_move(move: chess.Move) -> chess.Move:
    """
    Mirror a move horizontally: e2e4 -> d2d4 (files flip, ranks stay,
    promotion piece is preserved).

    Square trick: a square index is rank*8 + file, and file XOR 7 flips
    a<->h, b<->g, ... — so square ^ 7 mirrors the file bits in place.
    """
    return chess.Move(
        move.from_square ^ 7,
        move.to_square ^ 7,
        promotion=move.promotion,
    )


def batch_board_to_tensor(boards: list):
    """
    Convert a batch of boards to a batched tensor.

    Args:
        boards: List of chess.Board objects

    Returns:
        torch.Tensor of shape (batch_size, 20, 8, 8)
    """
    import torch  # Local import: keeps the rest of this module torch-free

    tensors = [board_to_tensor(board) for board in boards]
    batch = np.stack(tensors, axis=0)
    return torch.from_numpy(batch)


def batch_legal_moves_mask(boards: list):
    """
    Create batched legal move masks.

    Args:
        boards: List of chess.Board objects

    Returns:
        torch.Tensor of shape (batch_size, 4672)
    """
    import torch  # Local import: keeps the rest of this module torch-free

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

    # Round-trip EVERY legal move in the starting position.
    # (The old test used a hardcoded list that included e7e5 — an ILLEGAL
    # move with White to play — so this self-test could never have passed.
    # Round-tripping all legal moves is both stronger and can't go stale.)
    for move in board.legal_moves:
        index = move_to_index(move)
        recovered = index_to_move(index, board)
        assert recovered == move, f"Round-trip failed for {move}: {index} → {recovered}"
    print(f"  ✓ All {board.legal_moves.count()} legal opening moves round-trip")

    # Round-trip promotions and underpromotions (the historical bug)
    promo_board = chess.Board("rnb1kbnr/pP2pppp/8/8/8/8/P1PPPPPP/RNBQKBNR w KQkq - 0 1")
    promo_moves = [m for m in promo_board.legal_moves if m.promotion]
    assert len(promo_moves) >= 8, "Expected promotions in test position"
    for move in promo_board.legal_moves:
        index = move_to_index(move)
        assert index < NUM_MOVES, f"{move} encodes out of range: {index}"
        recovered = index_to_move(index, promo_board)
        assert recovered == move, f"Round-trip failed for {move}: {index} → {recovered}"
    print(f"  ✓ All {len(promo_moves)} promotion moves round-trip (incl. underpromotions)")

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
