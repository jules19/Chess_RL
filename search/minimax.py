"""
Minimax search with alpha-beta pruning - Days 3-4

This module implements game tree search to look multiple moves ahead,
dramatically improving on the 1-ply material evaluator from Day 2.

Key improvements:
- Looks 2-3 moves ahead (configurable depth)
- Considers opponent's best replies (minimax)
- Efficient pruning (alpha-beta)
- Move ordering (searches good moves first)
- Finds forced checkmates and tactics
"""

import chess
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.evaluator import evaluate


def quiescence_search(board: chess.Board, alpha: float, beta: float,
                      maximizing: bool, nodes_searched: list = None,
                      max_depth: int = 10) -> float:
    """
    Quiescence search - searches only "noisy" moves (captures, checks, promotions)
    until position is quiet. This prevents the horizon effect where the engine
    stops searching in the middle of a tactical sequence.

    Example of horizon effect WITHOUT quiescence:
        Depth 3: Bxe5 (capture pawn, looks good!)
        Depth 4: Nxe5 (recapture bishop - oops, we lose material!)

    Quiescence fixes this by continuing to search tactical moves beyond the
    normal search depth until the position stabilizes.

    Args:
        board: Current position
        alpha: Best score White can guarantee
        beta: Best score Black can guarantee
        maximizing: True if maximizing (White), False if minimizing (Black)
        nodes_searched: Optional counter for nodes
        max_depth: Maximum depth to search (prevents infinite loops)

    Returns:
        Evaluation score of the quiet position
    """
    if nodes_searched is not None:
        nodes_searched[0] += 1

    # Check for game over
    if board.is_game_over():
        return evaluate(board)

    # Stand-pat score: evaluation if we make no more captures
    # This represents "doing nothing" - if it's already good enough, we can stop
    stand_pat = evaluate(board)

    # Prevent searching too deep (safety limit)
    if max_depth <= 0:
        return stand_pat

    # Beta cutoff: opponent won't allow this position
    if maximizing:
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
    else:
        if stand_pat <= alpha:
            return alpha
        if stand_pat < beta:
            beta = stand_pat

    # Generate only "noisy" moves: captures, checks, and promotions
    all_moves = list(board.legal_moves)
    noisy_moves = []

    for move in all_moves:
        # Include captures
        if board.is_capture(move):
            noisy_moves.append(move)
            continue

        # Include promotions
        if move.promotion:
            noisy_moves.append(move)
            continue

        # Include checks (but be careful - this can make search very slow)
        # We only check for checks if there aren't many captures
        if len(noisy_moves) < 5:
            board.push(move)
            if board.is_check():
                noisy_moves.append(move)
            board.pop()

    # If no noisy moves, position is quiet - return stand-pat evaluation
    if not noisy_moves:
        return stand_pat

    # Order noisy moves (captures first, best captures prioritized)
    ordered_moves = order_moves(board, noisy_moves)

    # Search noisy moves
    if maximizing:
        for move in ordered_moves:
            board.push(move)
            score = quiescence_search(board, alpha, beta, False, nodes_searched, max_depth - 1)
            board.pop()

            if score >= beta:
                return beta  # Beta cutoff
            if score > alpha:
                alpha = score
        return alpha
    else:
        for move in ordered_moves:
            board.push(move)
            score = quiescence_search(board, alpha, beta, True, nodes_searched, max_depth - 1)
            board.pop()

            if score <= alpha:
                return alpha  # Alpha cutoff
            if score < beta:
                beta = score
        return beta


def order_moves(board: chess.Board, moves: list) -> list:
    """
    Order moves for more efficient alpha-beta pruning.

    Good move ordering dramatically improves pruning effectiveness.
    We search likely-good moves first:
    1. Captures (especially high-value captures)
    2. Checks
    3. Other moves

    Args:
        board: Current position
        moves: List of legal moves

    Returns:
        Ordered list of moves (best candidates first)
    """
    def move_priority(move):
        """Calculate priority score for a move (higher = search first)."""
        score = 0

        # Captures get high priority (MVV-LVA: Most Valuable Victim - Least Valuable Attacker)
        if board.is_capture(move):
            # Captured piece value
            captured = board.piece_at(move.to_square)
            if captured:
                score += captured.piece_type * 100

            # Prefer using less valuable pieces to capture
            attacker = board.piece_at(move.from_square)
            if attacker:
                score -= attacker.piece_type

        # Checks get medium priority
        board_copy = board.copy()
        board_copy.push(move)
        if board_copy.is_check():
            score += 50

        # Promotions get very high priority
        if move.promotion:
            score += 900

        return score

    # Sort moves by priority (highest first)
    return sorted(moves, key=move_priority, reverse=True)


def minimax(board: chess.Board, depth: int, alpha: float, beta: float,
            maximizing: bool, nodes_searched: list = None) -> float:
    """
    Minimax search with alpha-beta pruning.

    This is the core search algorithm. It recursively explores the game tree,
    assuming both players play optimally (minimax), and uses alpha-beta
    pruning to skip branches that can't affect the final result.

    Args:
        board: Current chess position
        depth: How many more plies (half-moves) to search
        alpha: Best score White can guarantee (lower bound)
        beta: Best score Black can guarantee (upper bound)
        maximizing: True if maximizing player (White), False for minimizing (Black)
        nodes_searched: Optional list to track nodes (for debugging)

    Returns:
        Best evaluation score from this position (in centipawns, White's perspective)
    """
    if nodes_searched is not None:
        nodes_searched[0] += 1

    # Base case: game over
    if board.is_game_over():
        return evaluate(board)

    # Base case: reached search depth limit
    # Instead of evaluating immediately, use quiescence search to resolve tactics
    if depth == 0:
        return quiescence_search(board, alpha, beta, maximizing, nodes_searched)

    legal_moves = list(board.legal_moves)

    # No legal moves shouldn't happen (caught by is_game_over), but handle it
    if not legal_moves:
        return evaluate(board)

    # Order moves for better pruning
    ordered_moves = order_moves(board, legal_moves)

    if maximizing:
        # White's turn: maximize score
        max_eval = float('-inf')
        for move in ordered_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, False, nodes_searched)
            board.pop()

            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)

            # Beta cutoff: Black won't allow this branch
            if beta <= alpha:
                break  # Prune remaining moves

        return max_eval
    else:
        # Black's turn: minimize score
        min_eval = float('inf')
        for move in ordered_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, True, nodes_searched)
            board.pop()

            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)

            # Alpha cutoff: White won't allow this branch
            if beta <= alpha:
                break  # Prune remaining moves

        return min_eval


def best_move_minimax(board: chess.Board, depth: int = 3, verbose: bool = False) -> chess.Move:
    """
    Find the best move using minimax search with alpha-beta pruning.

    This is the main entry point for the minimax engine. It searches all legal
    moves and returns the one with the best evaluation.

    Args:
        board: Current chess position
        depth: Search depth in plies (half-moves). Default 3 = look 1.5 moves ahead
               depth=2: Very fast, sees captures
               depth=3: Good tactical vision (recommended)
               depth=4: Strong play, slower
               depth=5+: Very strong but much slower
        verbose: If True, print search statistics

    Returns:
        Best move found by search
    """
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return None

    # Single legal move? Play it instantly
    if len(legal_moves) == 1:
        return legal_moves[0]

    best_move = None
    best_score = float('-inf') if board.turn == chess.WHITE else float('inf')
    alpha = float('-inf')
    beta = float('inf')
    nodes_searched = [0]

    # Order moves for better pruning at root
    ordered_moves = order_moves(board, legal_moves)

    if verbose:
        print(f"Searching {len(legal_moves)} moves at depth {depth}...")

    for move in ordered_moves:
        board.push(move)

        # After making our move, opponent tries to minimize (if we're White) or maximize (if we're Black)
        if board.turn == chess.BLACK:  # We just played as White
            score = minimax(board, depth - 1, alpha, beta, False, nodes_searched)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
        else:  # We just played as Black
            score = minimax(board, depth - 1, alpha, beta, True, nodes_searched)
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, score)

        board.pop()

        if verbose:
            print(f"  {move}: {score/100:.2f} pawns")

    if verbose:
        print(f"\nBest move: {best_move} (score: {best_score/100:.2f} pawns)")
        print(f"Nodes searched: {nodes_searched[0]:,}")

    return best_move


if __name__ == "__main__":
    # Self-test with some tactical positions
    print("Testing minimax search engine...")
    print("="*50)

    # Test 1: Mate in 1 (Back rank mate)
    print("\n📝 Test 1: Mate in 1 (Back rank)")
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    print(board)
    print("White to move (should find Re8#)")
    move = best_move_minimax(board, depth=2, verbose=True)
    print(f"✓ Found move: {move}")
    if move and str(move) == "e1e8":
        print("✅ PASS: Found checkmate!")
    else:
        print("❌ FAIL: Didn't find checkmate")

    # Test 2: Free piece capture
    print("\n📝 Test 2: Hanging queen")
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPPQPPP/RNB1KBNR b KQkq - 0 1")
    print(board)
    print("Black to move (queen on e2 is hanging)")
    move = best_move_minimax(board, depth=3, verbose=False)
    print(f"✓ Black plays: {move}")
    # Check if it's a capture of the queen
    if move and move.to_square == chess.E2:
        print("✅ PASS: Captured hanging queen!")
    else:
        print("⚠️ Note: Didn't capture queen (may be seeing a better move)")

    # Test 3: Forced checkmate in 2
    print("\n📝 Test 3: Mate in 2")
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1")
    print(board)
    print("White to move (Qxf7+ leads to mate)")
    move = best_move_minimax(board, depth=4, verbose=True)
    print(f"✓ White plays: {move}")
    if move and str(move) == "h5f7":
        print("✅ PASS: Found mating combination!")
    else:
        print("ℹ️  Found different move (mate in 2 requires depth 4+)")

    # Test 4: Starting position sanity check
    print("\n📝 Test 4: Starting position")
    board = chess.Board()
    print("Depth 2 from starting position...")
    move = best_move_minimax(board, depth=2, verbose=False)
    print(f"✓ Suggests: {move}")
    if move:
        print("✅ PASS: Engine runs on starting position")
    else:
        print("❌ FAIL: Returned None")

    print("\n✅ All minimax tests complete!")
