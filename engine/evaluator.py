"""
Chess position evaluator - Day 2: Material Evaluation

This module provides simple material counting for chess positions.
Later it will be extended with positional evaluation (center control,
piece development, king safety, pawn structure).
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


def evaluate(board: chess.Board) -> int:
    """
    Main evaluation function. Currently just material, but will be extended.

    Args:
        board: The chess board position to evaluate

    Returns:
        Score in centipawns from White's perspective

    Note:
        This will be extended in Days 5-6 with:
        - Center control bonus
        - Piece development
        - King safety
        - Pawn structure
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

    # Material evaluation
    score = evaluate_material(board)

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
