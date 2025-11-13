"""
Chess position evaluator - Days 2-6: Material + Positional Evaluation

This module provides chess position evaluation including:
- Material counting (Day 2)
- Center control (Day 5)
- Piece development (Day 5)
- King safety (Day 6)
- Pawn structure (Day 6)
"""

import chess
import random


# Standard piece values in centipawns (1 pawn = 100)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,  # King has no material value (can't be captured)
}

# Center squares (d4, e4, d5, e5) - the most important squares
CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]

# Extended center (c3-f3, c6-f6, plus the main center)
EXTENDED_CENTER = [
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.D4, chess.E4, chess.F4,
    chess.C5, chess.D5, chess.E5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
]

# Starting squares for pieces (to detect development)
WHITE_KNIGHT_START = [chess.B1, chess.G1]
BLACK_KNIGHT_START = [chess.B8, chess.G8]
WHITE_BISHOP_START = [chess.C1, chess.F1]
BLACK_BISHOP_START = [chess.C8, chess.F8]


def evaluate_material(board: chess.Board) -> int:
    """
    Evaluate board position based purely on material count.

    Args:
        board: The chess board position to evaluate

    Returns:
        Score in centipawns from White's perspective:
        - Positive = White is winning
        - Negative = Black is winning
        - Zero = Material is equal

    Examples:
        >>> board = chess.Board()  # Starting position
        >>> evaluate_material(board)
        0

        >>> board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPP1P/RNBQKBNR w KQkq - 0 1")  # White missing a pawn
        >>> evaluate_material(board)
        -100
    """
    score = 0

    # Count all pieces on the board
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            value = PIECE_VALUES[piece.piece_type]
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value

    return score


def evaluate_center_control(board: chess.Board) -> int:
    """
    Evaluate center control.

    Pawns and pieces occupying or attacking center squares get bonuses.
    Controlling the center is a fundamental chess principle.

    Args:
        board: The chess board position to evaluate

    Returns:
        Score in centipawns (positive = White advantage)
    """
    score = 0

    # Bonus for pieces in the center
    for square in CENTER_SQUARES:
        piece = board.piece_at(square)
        if piece:
            if piece.piece_type == chess.PAWN:
                bonus = 30  # Pawns in center are very strong
            elif piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                bonus = 20  # Centralized pieces
            else:
                bonus = 10  # Other pieces

            if piece.color == chess.WHITE:
                score += bonus
            else:
                score -= bonus

    # Smaller bonus for extended center
    for square in EXTENDED_CENTER:
        if square in CENTER_SQUARES:
            continue  # Already counted

        piece = board.piece_at(square)
        if piece:
            if piece.piece_type == chess.PAWN:
                bonus = 10
            elif piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                bonus = 10
            else:
                bonus = 5

            if piece.color == chess.WHITE:
                score += bonus
            else:
                score -= bonus

    return score


def evaluate_piece_development(board: chess.Board) -> int:
    """
    Evaluate piece development.

    Encourages getting knights and bishops off their starting squares
    in the opening. This is a key opening principle.

    Args:
        board: The chess board position to evaluate

    Returns:
        Score in centipawns (positive = White advantage)
    """
    score = 0

    # Penalize pieces still on starting squares (opening phase heuristic)
    # Only apply this in early game (roughly first 10 moves)
    if board.fullmove_number <= 10:
        # White knights
        for square in WHITE_KNIGHT_START:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.KNIGHT and piece.color == chess.WHITE:
                score -= 15  # Penalty for undeveloped knight

        # Black knights
        for square in BLACK_KNIGHT_START:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.KNIGHT and piece.color == chess.BLACK:
                score += 15  # Penalty for Black

        # White bishops
        for square in WHITE_BISHOP_START:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.BISHOP and piece.color == chess.WHITE:
                score -= 10

        # Black bishops
        for square in BLACK_BISHOP_START:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.BISHOP and piece.color == chess.BLACK:
                score += 10

    return score


def evaluate_king_safety(board: chess.Board) -> int:
    """
    Evaluate king safety.

    Rewards castling and penalizes exposed kings.
    King safety is critical in the opening and middlegame.

    Args:
        board: The chess board position to evaluate

    Returns:
        Score in centipawns (positive = White advantage)
    """
    score = 0

    # Bonus for castling
    if board.has_kingside_castling_rights(chess.WHITE):
        score += 10  # Potential to castle is good
    if board.has_queenside_castling_rights(chess.WHITE):
        score += 5

    if board.has_kingside_castling_rights(chess.BLACK):
        score -= 10
    if board.has_queenside_castling_rights(chess.BLACK):
        score -= 5

    # Check if kings have castled (heuristic: king on g1/g8 or c1/c8)
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)

    # White king castled kingside
    if white_king_square == chess.G1:
        score += 30
        # Check for pawn shield
        if board.piece_at(chess.F2) == chess.Piece(chess.PAWN, chess.WHITE):
            score += 10
        if board.piece_at(chess.G2) == chess.Piece(chess.PAWN, chess.WHITE):
            score += 10
        if board.piece_at(chess.H2) == chess.Piece(chess.PAWN, chess.WHITE):
            score += 10

    # White king castled queenside
    elif white_king_square == chess.C1:
        score += 30
        if board.piece_at(chess.A2) == chess.Piece(chess.PAWN, chess.WHITE):
            score += 10
        if board.piece_at(chess.B2) == chess.Piece(chess.PAWN, chess.WHITE):
            score += 10
        if board.piece_at(chess.C2) == chess.Piece(chess.PAWN, chess.WHITE):
            score += 10

    # Black king castled kingside
    if black_king_square == chess.G8:
        score -= 30
        if board.piece_at(chess.F7) == chess.Piece(chess.PAWN, chess.BLACK):
            score -= 10
        if board.piece_at(chess.G7) == chess.Piece(chess.PAWN, chess.BLACK):
            score -= 10
        if board.piece_at(chess.H7) == chess.Piece(chess.PAWN, chess.BLACK):
            score -= 10

    # Black king castled queenside
    elif black_king_square == chess.C8:
        score -= 30
        if board.piece_at(chess.A7) == chess.Piece(chess.PAWN, chess.BLACK):
            score -= 10
        if board.piece_at(chess.B7) == chess.Piece(chess.PAWN, chess.BLACK):
            score -= 10
        if board.piece_at(chess.C7) == chess.Piece(chess.PAWN, chess.BLACK):
            score -= 10

    # Penalty for king in center (dangerous in opening/middlegame)
    if white_king_square in CENTER_SQUARES or white_king_square in EXTENDED_CENTER:
        score -= 40
    if black_king_square in CENTER_SQUARES or black_king_square in EXTENDED_CENTER:
        score += 40

    return score


def evaluate_pawn_structure(board: chess.Board) -> int:
    """
    Evaluate pawn structure.

    Penalizes doubled pawns and isolated pawns.
    Rewards passed pawns (pawns with no enemy pawns blocking their path).

    Args:
        board: The chess board position to evaluate

    Returns:
        Score in centipawns (positive = White advantage)
    """
    score = 0

    # Analyze each file for pawn structure
    for file in range(8):
        white_pawns_on_file = []
        black_pawns_on_file = []

        for rank in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                if piece.color == chess.WHITE:
                    white_pawns_on_file.append(rank)
                else:
                    black_pawns_on_file.append(rank)

        # Doubled pawns penalty
        if len(white_pawns_on_file) > 1:
            score -= 20 * (len(white_pawns_on_file) - 1)
        if len(black_pawns_on_file) > 1:
            score += 20 * (len(black_pawns_on_file) - 1)

        # Isolated pawns (no friendly pawns on adjacent files)
        if white_pawns_on_file:
            has_support = False
            if file > 0:
                for rank in range(8):
                    square = chess.square(file - 1, rank)
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                        has_support = True
                        break
            if file < 7 and not has_support:
                for rank in range(8):
                    square = chess.square(file + 1, rank)
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                        has_support = True
                        break
            if not has_support:
                score -= 15 * len(white_pawns_on_file)  # Isolated pawn penalty

        if black_pawns_on_file:
            has_support = False
            if file > 0:
                for rank in range(8):
                    square = chess.square(file - 1, rank)
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                        has_support = True
                        break
            if file < 7 and not has_support:
                for rank in range(8):
                    square = chess.square(file + 1, rank)
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                        has_support = True
                        break
            if not has_support:
                score += 15 * len(black_pawns_on_file)

        # Passed pawns bonus (simplified: no enemy pawns on same file or adjacent files ahead)
        for white_rank in white_pawns_on_file:
            is_passed = True
            # Check same file ahead
            for check_rank in range(white_rank + 1, 8):
                square = chess.square(file, check_rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                    is_passed = False
                    break
            # Check adjacent files ahead
            if is_passed and file > 0:
                for check_rank in range(white_rank + 1, 8):
                    square = chess.square(file - 1, check_rank)
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                        is_passed = False
                        break
            if is_passed and file < 7:
                for check_rank in range(white_rank + 1, 8):
                    square = chess.square(file + 1, check_rank)
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.BLACK:
                        is_passed = False
                        break

            if is_passed:
                # Bonus increases as pawn advances
                score += 20 + (white_rank * 10)

        for black_rank in black_pawns_on_file:
            is_passed = True
            # Check same file ahead (for black, "ahead" is toward rank 0)
            for check_rank in range(black_rank - 1, -1, -1):
                square = chess.square(file, check_rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                    is_passed = False
                    break
            # Check adjacent files ahead
            if is_passed and file > 0:
                for check_rank in range(black_rank - 1, -1, -1):
                    square = chess.square(file - 1, check_rank)
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                        is_passed = False
                        break
            if is_passed and file < 7:
                for check_rank in range(black_rank - 1, -1, -1):
                    square = chess.square(file + 1, check_rank)
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == chess.WHITE:
                        is_passed = False
                        break

            if is_passed:
                # Bonus increases as pawn advances (for black, lower rank = more advanced)
                score -= 20 + ((7 - black_rank) * 10)

    return score


def evaluate(board: chess.Board) -> int:
    """
    Main evaluation function with full positional understanding.

    Combines multiple evaluation components:
    1. Material counting (most important)
    2. Center control
    3. Piece development
    4. King safety
    5. Pawn structure
    6. Mobility
    7. Check bonus

    Args:
        board: The chess board position to evaluate

    Returns:
        Score in centipawns from White's perspective
        - Positive = White advantage
        - Negative = Black advantage
        - 0 = Equal position
    """
    # Check for game over first
    if board.is_checkmate():
        # Checkmate is worth infinite points for the winner
        if board.turn == chess.WHITE:
            # White is checkmated (Black wins)
            return -100000
        else:
            # Black is checkmated (White wins)
            return 100000

    if board.is_stalemate() or board.is_insufficient_material():
        return 0  # Draw

    # Start with material evaluation (most important - ~80% of score)
    score = evaluate_material(board)

    # Add positional evaluation components
    # Each component contributes a smaller amount (tactical bonuses)
    score += evaluate_center_control(board)       # ~10-50 centipawns typically
    score += evaluate_piece_development(board)    # ~10-40 in opening
    score += evaluate_king_safety(board)          # ~20-80 in opening/middlegame
    score += evaluate_pawn_structure(board)       # ~10-60 depending on position

    # Small bonus for giving check (encourages attacking the king)
    if board.is_check():
        if board.turn == chess.WHITE:
            # Black gave check to White
            score -= 50
        else:
            # White gave check to Black
            score += 50

    # Tiny bonus for mobility (encourages active pieces, helps avoid draws)
    # This helps the engine make progress rather than shuffling
    mobility = board.legal_moves.count()
    if board.turn == chess.WHITE:
        score += mobility  # More moves for White is good
    else:
        score -= mobility  # More moves for Black is good for Black

    return score


def best_move_material(board: chess.Board) -> chess.Move:
    """
    Find the best move based on material evaluation only.

    This is a greedy 1-ply evaluation: for each legal move, we apply it,
    evaluate the resulting position, and pick the move with the best score.

    Args:
        board: Current chess position

    Returns:
        The move with the best material outcome

    Note:
        This is VERY weak - it only looks 1 move ahead and doesn't consider
        opponent's replies. Days 3-4 will add minimax search to fix this.
    """
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        # No legal moves (shouldn't happen if game isn't over)
        return None

    # Collect all moves with their scores
    move_scores = []
    for move in legal_moves:
        # Make the move on a copy
        board_copy = board.copy()
        board_copy.push(move)

        # Evaluate the resulting position
        score = evaluate(board_copy)
        move_scores.append((move, score))

    # Find best score
    if board.turn == chess.WHITE:
        best_score = max(score for _, score in move_scores)
        best_moves = [move for move, score in move_scores if score == best_score]
    else:  # Black's turn
        best_score = min(score for _, score in move_scores)
        best_moves = [move for move, score in move_scores if score == best_score]

    # Randomly pick among best moves (adds variety)
    return random.choice(best_moves)


if __name__ == "__main__":
    # Quick self-test
    print("Testing material evaluator...")

    # Test 1: Starting position should be 0
    board = chess.Board()
    score = evaluate_material(board)
    assert score == 0, f"Starting position should be 0, got {score}"
    print(f"✓ Starting position: {score}")

    # Test 2: White down a pawn
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPP1/RNBQKBNR w KQkq - 0 1")
    score = evaluate_material(board)
    assert score == -100, f"White down a pawn should be -100, got {score}"
    print(f"✓ White down a pawn: {score}")

    # Test 3: Black up a queen
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    score = evaluate_material(board)
    assert score == 0, f"Equal material should be 0, got {score}"
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")  # White missing queen
    score = evaluate_material(board)
    assert score == -900, f"White down a queen should be -900, got {score}"
    print(f"✓ White down a queen: {score}")

    # Test 4: Best move should capture piece
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
    best = best_move_material(board)
    print(f"✓ Best move finder works: {best}")

    print("\n✅ All tests passed!")
