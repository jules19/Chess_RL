"""
Performance Profiling Tool - Measure search efficiency

Analyzes engine performance metrics:
- Nodes searched per move
- Time per move
- Alpha-beta pruning effectiveness
- Branching factor
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import time
from search.minimax import minimax, best_move_minimax


def profile_position(fen, depths=[1, 2, 3, 4], verbose=True):
    """
    Profile a single position at multiple depths.

    Args:
        fen: Position to analyze
        depths: List of depths to test
        verbose: Print detailed output

    Returns:
        Dictionary with profiling results
    """
    board = chess.Board(fen)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Position: {fen}")
        print(f"{'='*60}")
        print(board)
        print(f"\nLegal moves: {board.legal_moves.count()}")

    results = []

    for depth in depths:
        nodes = [0]  # Track nodes searched
        start_time = time.time()

        move = best_move_minimax(board, depth=depth, verbose=False)

        # Count nodes by doing a separate search
        minimax(board, depth, float('-inf'), float('inf'),
                board.turn == chess.WHITE, nodes)

        elapsed = time.time() - start_time

        # Calculate metrics
        nodes_per_sec = nodes[0] / elapsed if elapsed > 0 else 0
        avg_branching = nodes[0] ** (1.0 / depth) if depth > 0 else 0

        result = {
            "depth": depth,
            "move": str(move),
            "nodes": nodes[0],
            "time": elapsed,
            "nodes_per_sec": nodes_per_sec,
            "branching_factor": avg_branching
        }

        results.append(result)

        if verbose:
            print(f"\nDepth {depth}:")
            print(f"  Best move: {move}")
            print(f"  Nodes searched: {nodes[0]:,}")
            print(f"  Time: {elapsed:.3f}s")
            print(f"  Nodes/sec: {nodes_per_sec:,.0f}")
            print(f"  Effective branching factor: {avg_branching:.2f}")

    return results


def estimate_pruning_effectiveness(fen, depth=3):
    """
    Estimate alpha-beta pruning effectiveness.

    Compares nodes searched with alpha-beta vs theoretical minimax.

    Args:
        fen: Position to analyze
        depth: Search depth

    Returns:
        Pruning statistics
    """
    board = chess.Board(fen)
    legal_moves = board.legal_moves.count()

    # Count nodes with alpha-beta
    nodes_ab = [0]
    minimax(board, depth, float('-inf'), float('inf'),
            board.turn == chess.WHITE, nodes_ab)

    # Theoretical full minimax (no pruning)
    # This is approximately: legal_moves ^ depth
    theoretical_nodes = legal_moves ** depth

    # Pruning effectiveness
    if theoretical_nodes > 0:
        pruning_pct = (1 - nodes_ab[0] / theoretical_nodes) * 100
    else:
        pruning_pct = 0

    return {
        "nodes_searched": nodes_ab[0],
        "theoretical_nodes": theoretical_nodes,
        "pruning_pct": pruning_pct,
        "legal_moves": legal_moves,
        "depth": depth
    }


def profile_suite():
    """Profile multiple standard positions."""
    positions = [
        ("Starting Position", chess.STARTING_FEN),
        ("Open Position", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 1"),
        ("Middlegame", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1"),
        ("Endgame (Rook)", "8/5k2/3p4/1p1Pp3/pP2Pp2/P4P2/8/5K1R w - - 0 1"),
        ("Tactical", "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1"),
    ]

    print("="*60)
    print("ENGINE PERFORMANCE PROFILE")
    print("="*60)

    all_results = {}

    for name, fen in positions:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")

        # Profile at depths 1-4
        results = profile_position(fen, depths=[1, 2, 3, 4], verbose=True)
        all_results[name] = results

        # Estimate pruning effectiveness at depth 3
        pruning = estimate_pruning_effectiveness(fen, depth=3)
        print(f"\nPruning Effectiveness (depth 3):")
        print(f"  Nodes searched: {pruning['nodes_searched']:,}")
        print(f"  Theoretical (no pruning): {pruning['theoretical_nodes']:,}")
        print(f"  Pruning saved: {pruning['pruning_pct']:.1f}% of nodes")

    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for name, results in all_results.items():
        depth_3_result = [r for r in results if r["depth"] == 3][0]
        print(f"\n{name} (depth 3):")
        print(f"  Nodes: {depth_3_result['nodes']:,}")
        print(f"  Time: {depth_3_result['time']:.3f}s")
        print(f"  NPS: {depth_3_result['nodes_per_sec']:,.0f}")
        print(f"  Branching: {depth_3_result['branching_factor']:.2f}")

    # Overall assessment
    print(f"\n{'='*60}")
    print("PERFORMANCE ASSESSMENT")
    print(f"{'='*60}")

    avg_nps = sum(
        [r for r in results if r["depth"] == 3][0]["nodes_per_sec"]
        for results in all_results.values()
    ) / len(all_results)

    avg_branching = sum(
        [r for r in results if r["depth"] == 3][0]["branching_factor"]
        for results in all_results.values()
    ) / len(all_results)

    print(f"Average nodes/second (depth 3): {avg_nps:,.0f}")
    print(f"Average branching factor (depth 3): {avg_branching:.2f}")

    if avg_branching < 6:
        print("✅ Excellent pruning! (branching factor < 6)")
    elif avg_branching < 10:
        print("🔶 Good pruning (branching factor 6-10)")
    elif avg_branching < 15:
        print("⚠️  Fair pruning (branching factor 10-15)")
    else:
        print("❌ Poor pruning (branching factor > 15)")

    if avg_nps > 100000:
        print("🚀 Very fast search (>100k nodes/sec)")
    elif avg_nps > 50000:
        print("✅ Fast search (>50k nodes/sec)")
    elif avg_nps > 10000:
        print("🔶 Moderate speed (>10k nodes/sec)")
    else:
        print("⚠️  Slow search (<10k nodes/sec)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profile chess engine performance")
    parser.add_argument("--fen", type=str, default=None, help="Specific position to profile")
    parser.add_argument("--depth", type=int, default=None, help="Specific depth to test")

    args = parser.parse_args()

    if args.fen:
        # Profile single position
        depths = [args.depth] if args.depth else [1, 2, 3, 4]
        profile_position(args.fen, depths=depths, verbose=True)

        if args.depth:
            pruning = estimate_pruning_effectiveness(args.fen, depth=args.depth)
            print(f"\nPruning Effectiveness:")
            print(f"  Nodes searched: {pruning['nodes_searched']:,}")
            print(f"  Theoretical (no pruning): {pruning['theoretical_nodes']:,}")
            print(f"  Pruning saved: {pruning['pruning_pct']:.1f}% of nodes")
    else:
        # Profile suite of positions
        profile_suite()
