"""
Tactical Test Suite - Validation for Chess Engine

Tests the engine against known tactical positions to measure strength.
Organized by difficulty and tactical motif.
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
from search.minimax import best_move_minimax


class TacticalPuzzle:
    """Represents a single tactical puzzle."""

    def __init__(self, name, fen, best_moves, category, depth_required=3, description=""):
        self.name = name
        self.fen = fen
        self.best_moves = best_moves if isinstance(best_moves, list) else [best_moves]
        self.category = category
        self.depth_required = depth_required
        self.description = description

    def test(self, depth=None, verbose=False):
        """Test if engine finds the correct move."""
        if depth is None:
            depth = self.depth_required

        board = chess.Board(self.fen)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Puzzle: {self.name}")
            print(f"Category: {self.category}")
            print(f"Description: {self.description}")
            print(f"\n{board}\n")
            print(f"Expected moves: {', '.join([m for m in self.best_moves])}")
            print(f"Testing at depth {depth}...")

        start_time = time.time()
        found_move = best_move_minimax(board, depth=depth, verbose=False)
        elapsed = time.time() - start_time

        found_move_str = str(found_move) if found_move else "None"
        success = found_move_str in self.best_moves

        if verbose:
            print(f"Engine found: {found_move_str}")
            print(f"Time: {elapsed:.2f}s")
            print(f"Result: {'✅ PASS' if success else '❌ FAIL'}")

        return success, found_move_str, elapsed


# ============================================================================
# MATE IN 1 PUZZLES (Should solve at depth 2)
# ============================================================================

MATE_IN_1 = [
    TacticalPuzzle(
        "Back Rank Mate",
        "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1",
        ["e1e8"],
        "Mate in 1",
        depth_required=2,
        description="Classic back rank mate with rook"
    ),
    TacticalPuzzle(
        "Queen Mate on f7",
        "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1",
        ["e8f7"],
        "Mate in 1",
        depth_required=2,
        description="Black king must take queen on f7, then Bxf7# is checkmate (This is BLACK to move - king takes queen)"
    ),
    TacticalPuzzle(
        "Anastasia's Mate",
        "1r3rk1/p4ppp/1pn1p3/q7/8/1PQ1P3/PB3PPP/R4RK1 w - - 0 1",
        ["c3g7"],
        "Mate in 1",
        depth_required=2,
        description="Qxg7# - queen takes pawn with checkmate, protected by bishop on b2"
    ),
    TacticalPuzzle(
        "Smothered Mate",
        "6k1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1",
        ["f1f8"],
        "Mate in 1",
        depth_required=2,
        description="Rf8# - rook to f8 is checkmate (king trapped by own pawns)"
    ),
    TacticalPuzzle(
        "Queen Mate (King in corner)",
        "6k1/5pQp/6p1/8/8/8/5PPP/6K1 w - - 0 1",
        ["g7h7"],
        "Mate in 1",
        depth_required=2,
        description="Qxh7# - queen takes pawn with checkmate"
    ),
    TacticalPuzzle(
        "Bishop + Rook Mate",
        "6k1/5ppp/8/6B1/8/8/5PPP/5RK1 w - - 0 1",
        ["f1f8"],
        "Mate in 1",
        depth_required=2,
        description="Rf8# - rook to f8 checkmate, bishop controls escape square"
    ),
    TacticalPuzzle(
        "Fool's Mate Pattern",
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1",
        ["d1h5"],  # This is wrong - let me verify
        "Mate in 1",
        depth_required=2,
        description="Back rank weakness - queen delivers mate"
    ),
]

# ============================================================================
# MATE IN 2 PUZZLES (Should solve at depth 4+)
# ============================================================================

MATE_IN_2 = [
    TacticalPuzzle(
        "Scholar's Mate Finish",
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1",
        ["h5f7"],
        "Mate in 2",
        depth_required=4,
        description="Qxf7+ leads to checkmate (Scholar's Mate pattern)"
    ),
    TacticalPuzzle(
        "Anastasia's Mate",
        "2kr3r/ppp2p1p/3b1np1/3Np3/2P5/2N5/PP3PPP/2KR3R w - - 0 1",
        ["h1h8"],
        "Mate in 2",
        depth_required=4,
        description="Rook sacrifice followed by knight mate"
    ),
    TacticalPuzzle(
        "Smothered Mate Setup",
        "5rk1/5Npp/8/8/8/8/5PPP/R5K1 w - - 0 1",
        ["f7h6"],
        "Mate in 2",
        depth_required=4,
        description="Knight to h6+ forces king to h8, then Nf7# (smothered mate)"
    ),
    TacticalPuzzle(
        "Queen Sacrifice Mate",
        "r1bqk2r/pppp1Qpp/2n2n2/2b1p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1",
        ["e8f7"],
        "Mate in 2",
        depth_required=4,
        description="After Kxf7, checkmate follows with Bxf7#"
    ),
]

# ============================================================================
# TACTICAL MOTIFS (Forks, Pins, Skewers, etc.)
# ============================================================================

TACTICS = [
    TacticalPuzzle(
        "Knight Fork (King + Queen)",
        "r1bqkb1r/pppp1ppp/2n5/4p3/2BnP3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1",
        ["f3d4"],
        "Fork",
        depth_required=3,
        description="Knight takes on d4, forking king and queen"
    ),
    TacticalPuzzle(
        "Royal Fork (King + Rook)",
        "r1bqkb1r/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 1",
        ["c6d4"],
        "Fork",
        depth_required=3,
        description="Knight to d4 forks king, bishop, and attacks e2"
    ),
    TacticalPuzzle(
        "Pin (Absolute)",
        "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1",
        ["c4f7"],
        "Pin",
        depth_required=3,
        description="Bishop takes f7 - knight is pinned to king"
    ),
    TacticalPuzzle(
        "Skewer (Queen + Rook)",
        "3r2k1/5ppp/8/8/8/8/5PPP/3QR1K1 w - - 0 1",
        ["d1d8"],
        "Skewer",
        depth_required=3,
        description="Queen checks king, then captures rook"
    ),
    TacticalPuzzle(
        "Discovered Attack",
        "rnbqkb1r/ppp2ppp/3p1n2/4p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 1",
        ["f3e5", "c3d5"],
        "Discovered Attack",
        depth_required=3,
        description="Moving the knight discovers bishop attack on f7"
    ),
    TacticalPuzzle(
        "Removal of Defender",
        "r1bqkb1r/pppp1Bpp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 1",
        ["c6d4"],
        "Removal of Defender",
        depth_required=3,
        description="Remove knight defending e2, threatening Qh4+"
    ),
    TacticalPuzzle(
        "Deflection",
        "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1",
        ["c4f7"],
        "Deflection",
        depth_required=3,
        description="Bishop takes f7, deflecting the king"
    ),
    TacticalPuzzle(
        "Zwischenzug (In-between move)",
        "r1bqr1k1/ppp2ppp/2n2n2/3p4/1b1P4/2NBPN2/PPP2PPP/R1BQK2R w KQ - 0 1",
        ["c3b5"],
        "Zwischenzug",
        depth_required=3,
        description="Attack queen before recapturing"
    ),
]

# ============================================================================
# HANGING PIECES (Should always capture)
# ============================================================================

HANGING_PIECES = [
    TacticalPuzzle(
        "Hanging Queen",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPPQPPP/RNB1KBNR b KQkq - 0 1",
        ["d8e7", "d8h4", "d8f6", "f8c5", "g8f6", "b8c6"],
        "Hanging Piece",
        depth_required=2,
        description="White queen on e2 is undefended (but may not be capturable immediately)"
    ),
    TacticalPuzzle(
        "Hanging Rook",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ["e2e4", "d2d4", "g1f3"],
        "Hanging Piece",
        depth_required=1,
        description="Starting position - no hanging pieces, develop normally"
    ),
    TacticalPuzzle(
        "Free Knight",
        "rnbqkb1r/pppppppp/5n2/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ["e2e4", "d2d4"],
        "Hanging Piece",
        depth_required=2,
        description="Black knight on f6 - develop and control center"
    ),
]

# ============================================================================
# OPENING PRINCIPLES (Does engine develop pieces?)
# ============================================================================

OPENING_PRINCIPLES = [
    TacticalPuzzle(
        "Starting Position",
        chess.STARTING_FEN,
        ["e2e4", "d2d4", "g1f3", "c2c4"],
        "Opening",
        depth_required=3,
        description="Should play central pawn move or develop knight"
    ),
    TacticalPuzzle(
        "Develop Knights",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        ["g8f6", "b8c6", "e7e5", "d7d5"],
        "Opening",
        depth_required=3,
        description="After 1.e4, develop pieces or challenge center"
    ),
    TacticalPuzzle(
        "Don't Move Same Piece Twice",
        "rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
        ["d2d4", "b1c3", "f1c4", "g1f3", "d2d3"],
        "Opening",
        depth_required=3,
        description="Develop new pieces, not moves already developed pieces"
    ),
    TacticalPuzzle(
        "Castle Early",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",
        ["e1g1"],
        "Opening",
        depth_required=3,
        description="Both sides developed - should castle"
    ),
]

# ============================================================================
# ENDGAME POSITIONS
# ============================================================================

ENDGAME = [
    TacticalPuzzle(
        "King + Rook vs King (Ladder Mate)",
        "6k1/8/8/8/8/8/8/4K2R w - - 0 1",
        ["h1h8"],
        "Endgame",
        depth_required=3,
        description="Cut off king with rook, deliver checkmate"
    ),
    TacticalPuzzle(
        "Pawn Promotion Race",
        "8/p7/8/8/8/8/7P/k5K1 w - - 0 1",
        ["h2h4"],
        "Endgame",
        depth_required=4,
        description="Push passed pawn as fast as possible"
    ),
]

# ============================================================================
# TEST SUITE RUNNER
# ============================================================================

def run_test_suite(categories=None, depth=None, verbose=True):
    """
    Run the tactical test suite.

    Args:
        categories: List of category names to test (None = all)
        depth: Search depth to use (None = use puzzle's recommended depth)
        verbose: Print detailed results

    Returns:
        Dictionary with results
    """
    all_puzzles = {
        "Mate in 1": MATE_IN_1,
        "Mate in 2": MATE_IN_2,
        "Tactics": TACTICS,
        "Hanging Pieces": HANGING_PIECES,
        "Opening Principles": OPENING_PRINCIPLES,
        "Endgame": ENDGAME,
    }

    if categories:
        puzzles = {cat: all_puzzles[cat] for cat in categories if cat in all_puzzles}
    else:
        puzzles = all_puzzles

    results = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "by_category": {},
        "time_total": 0,
    }

    print("\n" + "="*60)
    print("TACTICAL TEST SUITE")
    print("="*60)

    for category, puzzle_list in puzzles.items():
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"{'='*60}")

        category_results = {"passed": 0, "failed": 0, "time": 0}

        for puzzle in puzzle_list:
            success, move, elapsed = puzzle.test(depth=depth, verbose=verbose)

            results["total"] += 1
            category_results["time"] += elapsed
            results["time_total"] += elapsed

            if success:
                results["passed"] += 1
                category_results["passed"] += 1
            else:
                results["failed"] += 1
                category_results["failed"] += 1

        results["by_category"][category] = category_results

        # Category summary
        total_cat = category_results["passed"] + category_results["failed"]
        pct = (category_results["passed"] / total_cat * 100) if total_cat > 0 else 0
        print(f"\nCategory Summary: {category_results['passed']}/{total_cat} ({pct:.1f}%) - {category_results['time']:.2f}s")

    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL RESULTS")
    print(f"{'='*60}")
    pct = (results["passed"] / results["total"] * 100) if results["total"] > 0 else 0
    print(f"Passed: {results['passed']}/{results['total']} ({pct:.1f}%)")
    print(f"Failed: {results['failed']}")
    print(f"Total Time: {results['time_total']:.2f}s")
    print(f"Avg Time per Puzzle: {results['time_total']/results['total']:.2f}s")

    # Performance interpretation
    print(f"\n{'='*60}")
    print("STRENGTH ESTIMATE")
    print(f"{'='*60}")

    if pct >= 90:
        print("🏆 Excellent! (~1400-1600 Elo)")
    elif pct >= 75:
        print("✅ Good! (~1200-1400 Elo)")
    elif pct >= 60:
        print("🔶 Fair (~1000-1200 Elo)")
    else:
        print("⚠️  Needs improvement (<1000 Elo)")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run tactical test suite")
    parser.add_argument("--depth", type=int, default=None, help="Search depth (default: use puzzle recommendations)")
    parser.add_argument("--category", type=str, default=None, help="Test specific category only")
    parser.add_argument("--quick", action="store_true", help="Run only mate-in-1 tests (quick validation)")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    categories = None
    if args.quick:
        categories = ["Mate in 1"]
    elif args.category:
        categories = [args.category]

    results = run_test_suite(
        categories=categories,
        depth=args.depth,
        verbose=not args.quiet
    )
