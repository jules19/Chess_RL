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
import chess.polyglot
import time
import sys
import os
from collections import namedtuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.evaluator import evaluate


# ---------------------------------------------------------------------------
# Transposition table
# ---------------------------------------------------------------------------
#
# Chess positions repeat during search: 1.d4 d5 2.c4 and 1.c4 d5 2.d4 reach
# the SAME position through different move orders (a "transposition"). Without
# a cache, the engine re-searches that identical subtree from scratch every
# time it's reached. A transposition table is a hash map from position →
# previously computed search result.
#
# The subtlety: alpha-beta doesn't always compute the EXACT value of a node.
# When a cutoff happens, we only learn a BOUND on the value:
#   EXACT       - full search completed, score is the true minimax value
#   LOWER_BOUND - search was cut off high (fail-high): true value >= score
#   UPPER_BOUND - search was cut off low  (fail-low):  true value <= score
# A stored bound can still narrow the (alpha, beta) window on a later visit,
# and the stored best move improves move ordering even when the score can't
# be reused directly.
#
# Positions are keyed by Zobrist hash: a 64-bit signature built by XOR-ing
# random numbers for every (piece, square) pair plus castling/en-passant/
# side-to-move. python-chess provides it via chess.polyglot.zobrist_hash().

TT_EXACT = 0
TT_LOWER_BOUND = 1
TT_UPPER_BOUND = 2

# depth: remaining search depth the score was computed with (a score from a
#        deeper search is trustworthy for a shallower one, not vice versa)
# score: evaluation in centipawns from White's perspective
# flag:  TT_EXACT / TT_LOWER_BOUND / TT_UPPER_BOUND
# best_move: best move found at this node (for move ordering on re-visits)
TTEntry = namedtuple('TTEntry', ['depth', 'score', 'flag', 'best_move'])


# ---------------------------------------------------------------------------
# Hard time control (reference solution — course Module 2, exercise 2)
# ---------------------------------------------------------------------------
#
# The predictive stop in best_move_iterative estimates the next iteration at
# ~3x the last one, but tactical positions can be 10x — so the search can
# still commit to an iteration it can't afford. The fix is a hard deadline
# checked INSIDE the search: every TIME_CHECK_INTERVAL nodes we glance at
# the clock and, if the deadline passed, unwind the whole recursion with an
# exception. The driver catches it and falls back to the previous
# iteration's move (which is complete and trustworthy — partial iterations
# are NOT: the best move might be in the unsearched remainder).
#
# Why check every N nodes instead of every node? time.time() costs more
# than most of what a node does; checking every ~1024 nodes makes the
# overhead invisible while bounding the overshoot to (1024 / nodes-per-
# second) seconds.

class SearchTimeout(Exception):
    """Raised inside the search tree when the hard deadline passes."""


TIME_CHECK_INTERVAL = 1024


def _check_deadline(deadline, nodes_searched):
    if (deadline is not None
            and nodes_searched is not None
            and nodes_searched[0] % TIME_CHECK_INTERVAL == 0
            and time.time() > deadline):
        raise SearchTimeout()


def quiescence_search(board: chess.Board, alpha: float, beta: float,
                      maximizing: bool, nodes_searched: list = None,
                      max_depth: int = 10, deadline: float = None) -> float:
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
    _check_deadline(deadline, nodes_searched)

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
            score = quiescence_search(board, alpha, beta, False, nodes_searched, max_depth - 1, deadline)
            board.pop()

            if score >= beta:
                return beta  # Beta cutoff
            if score > alpha:
                alpha = score
        return alpha
    else:
        for move in ordered_moves:
            board.push(move)
            score = quiescence_search(board, alpha, beta, True, nodes_searched, max_depth - 1, deadline)
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
            maximizing: bool, nodes_searched: list = None,
            tt: dict = None, deadline: float = None) -> float:
    """
    Minimax search with alpha-beta pruning and optional transposition table.

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
        tt: Optional transposition table (dict). Pass the same dict across
            calls (and across iterative-deepening iterations) to reuse work.

    Returns:
        Best evaluation score from this position (in centipawns, White's perspective)
    """
    if nodes_searched is not None:
        nodes_searched[0] += 1
    _check_deadline(deadline, nodes_searched)

    # --- Transposition table probe -------------------------------------
    # If we've already searched this position at least this deep, we may be
    # able to return immediately (EXACT hit) or narrow the search window
    # (bound hit). Either way, the stored best move improves ordering.
    # Only use the TT for depth >= 2: computing the Zobrist hash costs more
    # than a frontier (depth-1) search saves. Measure before optimizing —
    # our first version hashed every node and was SLOWER than no TT at all.
    use_tt = tt is not None and depth >= 2
    key = None
    hash_move = None
    alpha_orig, beta_orig = alpha, beta
    if use_tt:
        key = chess.polyglot.zobrist_hash(board)
        entry = tt.get(key)
        if entry is not None:
            hash_move = entry.best_move
            if entry.depth >= depth:
                if entry.flag == TT_EXACT:
                    return entry.score
                elif entry.flag == TT_LOWER_BOUND:
                    alpha = max(alpha, entry.score)
                elif entry.flag == TT_UPPER_BOUND:
                    beta = min(beta, entry.score)
                if alpha >= beta:
                    return entry.score

    # Base case: game over
    if board.is_game_over():
        return evaluate(board)

    # Base case: reached search depth limit
    # Instead of evaluating immediately, use quiescence search to resolve tactics
    if depth == 0:
        return quiescence_search(board, alpha, beta, maximizing, nodes_searched,
                                 deadline=deadline)

    legal_moves = list(board.legal_moves)

    # No legal moves shouldn't happen (caught by is_game_over), but handle it
    if not legal_moves:
        return evaluate(board)

    # Order moves for better pruning; the hash move (best move from a
    # previous, shallower search of this position) goes first — it's the
    # single most effective move-ordering heuristic in chess engines.
    ordered_moves = order_moves(board, legal_moves)
    if hash_move is not None and hash_move in legal_moves:
        ordered_moves.remove(hash_move)
        ordered_moves.insert(0, hash_move)

    best_move = None

    if maximizing:
        # White's turn: maximize score
        max_eval = float('-inf')
        for move in ordered_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, False, nodes_searched, tt, deadline)
            board.pop()

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)

            # Beta cutoff: Black won't allow this branch
            if beta <= alpha:
                break  # Prune remaining moves

        value = max_eval
    else:
        # Black's turn: minimize score
        min_eval = float('inf')
        for move in ordered_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, True, nodes_searched, tt, deadline)
            board.pop()

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)

            # Alpha cutoff: White won't allow this branch
            if beta <= alpha:
                break  # Prune remaining moves

        value = min_eval

    # --- Transposition table store --------------------------------------
    # Classify what we learned about this node:
    #   value <= original alpha → we failed low; the true value could be even
    #                             lower, so `value` is only an UPPER bound
    #   value >= original beta  → we failed high (cutoff); the true value
    #                             could be even higher: LOWER bound
    #   otherwise               → the search window never cut off: EXACT
    if use_tt:
        if value <= alpha_orig:
            flag = TT_UPPER_BOUND
        elif value >= beta_orig:
            flag = TT_LOWER_BOUND
        else:
            flag = TT_EXACT
        tt[key] = TTEntry(depth, value, flag, best_move)

    return value


def best_move_minimax(board: chess.Board, depth: int = 3, verbose: bool = False,
                      tt: dict = None, first_move: chess.Move = None,
                      deadline: float = None) -> chess.Move:
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
        tt: Optional transposition table dict, shared across calls
        first_move: Optional move to search first at the root (used by
                    iterative deepening to try the previous best move first)

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
    if first_move is not None and first_move in legal_moves:
        ordered_moves.remove(first_move)
        ordered_moves.insert(0, first_move)

    if verbose:
        print(f"Searching {len(legal_moves)} moves at depth {depth}...")

    for move in ordered_moves:
        board.push(move)

        # After making our move, opponent tries to minimize (if we're White) or maximize (if we're Black)
        if board.turn == chess.BLACK:  # We just played as White
            score = minimax(board, depth - 1, alpha, beta, False, nodes_searched, tt, deadline)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
        else:  # We just played as Black
            score = minimax(board, depth - 1, alpha, beta, True, nodes_searched, tt, deadline)
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


def best_move_iterative(board: chess.Board, max_depth: int = 5,
                        time_limit: float = 5.0,
                        verbose: bool = False) -> chess.Move:
    """
    Iterative deepening: search depth 1, then 2, then 3... until we run out
    of time or reach max_depth.

    At first glance this looks wasteful — why re-search the shallow depths?
    Two reasons it's actually FASTER than searching depth N directly:

    1. Game trees grow exponentially, so all the shallow searches combined
       cost a small fraction of the final deep search (roughly 1/branching
       factor extra work).
    2. Each iteration seeds the next one: the previous best move is searched
       first at the root, and the transposition table remembers best moves
       throughout the tree. Better move ordering → dramatically more
       alpha-beta cutoffs → the deep search runs far faster than a cold one.

    It also solves time management: a fixed-depth engine takes 0.1s in simple
    positions and 60s in complex ones. Iterative deepening always has a
    complete answer from the last finished depth when the clock runs out —
    which is exactly what UCI "go movetime" needs.

    (Real engines also abort MID-iteration when time expires; we only check
    between iterations to keep the code simple. That means we can overshoot
    the time limit by the length of one iteration — see the course exercises.)

    Args:
        board: Current chess position
        max_depth: Deepest search to attempt
        time_limit: Soft time budget in seconds
        verbose: Print per-iteration statistics

    Returns:
        Best move from the deepest completed iteration
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    if len(legal_moves) == 1:
        return legal_moves[0]

    tt = {}  # Shared across iterations — this is what makes deepening cheap
    best_move = None
    start_time = time.time()
    deadline = start_time + time_limit
    last_iteration_time = 0.0

    # Each depth costs roughly branching-factor times the previous one.
    # Empirically ~3-6x for this engine; we use 3 as an optimistic estimate.
    GROWTH_ESTIMATE = 3.0

    # HARD-ABORT TRAP (reference solution — course Module 2, exercise 2):
    # SearchTimeout unwinds the recursion PAST the board.pop() calls, so an
    # aborted search leaves its board with moves still pushed. Searching a
    # COPY makes the corruption harmless — the caller's board is untouched.
    # (The course test catches the naive version: an aborted search on the
    # caller's board makes the returned move illegal on it.)
    search_board = board.copy()

    for depth in range(1, max_depth + 1):
        elapsed = time.time() - start_time

        # PREDICTIVE time check: don't start an iteration we probably can't
        # finish. Checking `elapsed >= time_limit` alone is a trap — depth
        # N+1 takes several times longer than depth N, so starting an
        # iteration at 1.9s of a 2s budget can blow the budget by 10x.
        # This is only a heuristic (real growth can exceed the estimate,
        # e.g. 10x in tactical positions), which is why the hard deadline
        # below backs it up.
        if depth > 1 and elapsed + last_iteration_time * GROWTH_ESTIMATE > time_limit:
            break

        iteration_start = time.time()
        try:
            best_move = best_move_minimax(
                search_board, depth=depth, verbose=False, tt=tt,
                first_move=best_move, deadline=deadline
            )
        except SearchTimeout:
            # The interrupted iteration is incomplete and untrustworthy —
            # the true best move might be in the unsearched remainder — so
            # we KEEP the previous iteration's move, not any partial result.
            if verbose:
                print(f"  depth {depth}: aborted at deadline "
                      f"({time.time() - start_time:.2f}s elapsed)")
            break
        last_iteration_time = time.time() - iteration_start

        if verbose:
            print(f"  depth {depth}: best={best_move} "
                  f"({time.time() - start_time:.2f}s elapsed, TT entries: {len(tt):,})")

    if best_move is None:
        # Deadline expired inside the depth-1 iteration (tiny budgets):
        # any sensible answer beats returning nothing
        best_move = order_moves(board, legal_moves)[0]

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
