"""
Compare Baseline MCTS vs NN-MCTS

This script runs a quick head-to-head comparison between:
1. Baseline MCTS (hand-crafted evaluator)
2. NN-MCTS (neural network evaluator)

Purpose: Validate that neural network integration works and measure
if the NN provides any improvement over hand-crafted heuristics.

Even with a small model trained on 50 games, we should see:
- NN-MCTS makes reasonable moves
- Similar strength to baseline (or slightly better/worse)
- The pipeline works end-to-end
"""

import chess
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.mcts import mcts_search
from search.nn_mcts import NNMCTSPlayer


def play_game(white_player, black_player, max_moves=100, verbose=False):
    """
    Play a game between two players.

    Args:
        white_player: Function(board) -> move for White
        black_player: Function(board) -> move for Black
        max_moves: Maximum moves before declaring draw
        verbose: Print moves

    Returns:
        result: "1-0", "0-1", "1/2-1/2"
        move_count: Number of moves played
    """
    board = chess.Board()
    move_count = 0

    if verbose:
        print("\n" + "="*70)
        print("Starting game...")
        print("="*70)

    while not board.is_game_over() and move_count < max_moves:
        # Select move based on current player
        if board.turn == chess.WHITE:
            move = white_player(board)
            player_name = "White"
        else:
            move = black_player(board)
            player_name = "Black"

        if move is None:
            break

        if verbose:
            print(f"\nMove {move_count + 1}. {player_name} plays: {move}")

        board.push(move)
        move_count += 1

        if verbose and move_count % 10 == 0:
            print(f"\n{board}")

    # Determine result
    if board.is_checkmate():
        result = "1-0" if board.turn == chess.BLACK else "0-1"
    elif board.is_stalemate() or board.is_insufficient_material():
        result = "1/2-1/2"
    elif move_count >= max_moves:
        result = "1/2-1/2"
    else:
        result = "1/2-1/2"

    if verbose:
        print("\n" + "="*70)
        print(f"Game over: {result}")
        print(f"Total moves: {move_count}")
        print("="*70)

    return result, move_count


def run_comparison(model_path, num_games=3, simulations_per_move=50):
    """
    Run comparison between baseline MCTS and NN-MCTS.

    Args:
        model_path: Path to trained neural network model
        num_games: Number of games to play
        simulations_per_move: MCTS simulations per move (same for both)

    Returns:
        Dictionary with win/loss/draw counts
    """
    print("="*70)
    print("MCTS vs NN-MCTS Comparison")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Games: {num_games}")
    print(f"  Simulations per move: {simulations_per_move}")
    print(f"  Model: {model_path}")

    # Create NN-MCTS player (architecture read from checkpoint, device auto)
    nn_mcts_player = NNMCTSPlayer(
        model_path,
        simulations=simulations_per_move,
    )

    # Create baseline MCTS player function
    def baseline_mcts(board):
        return mcts_search(
            board,
            simulations=simulations_per_move,
            use_evaluator=True,
            verbose=False
        )

    # Create NN-MCTS player function
    def nn_mcts(board):
        return nn_mcts_player.select_move(board, verbose=False)

    results = {
        'nn_wins': 0,
        'baseline_wins': 0,
        'draws': 0
    }

    # Play games alternating colors
    for game_num in range(num_games):
        print(f"\n{'='*70}")
        print(f"Game {game_num + 1}/{num_games}")
        print(f"{'='*70}")

        # Alternate colors
        if game_num % 2 == 0:
            print("  White: NN-MCTS")
            print("  Black: Baseline MCTS")
            result, moves = play_game(nn_mcts, baseline_mcts, verbose=False)

            if result == "1-0":
                results['nn_wins'] += 1
                print(f"  Result: NN-MCTS wins! ({moves} moves)")
            elif result == "0-1":
                results['baseline_wins'] += 1
                print(f"  Result: Baseline MCTS wins! ({moves} moves)")
            else:
                results['draws'] += 1
                print(f"  Result: Draw ({moves} moves)")
        else:
            print("  White: Baseline MCTS")
            print("  Black: NN-MCTS")
            result, moves = play_game(baseline_mcts, nn_mcts, verbose=False)

            if result == "1-0":
                results['baseline_wins'] += 1
                print(f"  Result: Baseline MCTS wins! ({moves} moves)")
            elif result == "0-1":
                results['nn_wins'] += 1
                print(f"  Result: NN-MCTS wins! ({moves} moves)")
            else:
                results['draws'] += 1
                print(f"  Result: Draw ({moves} moves)")

    return results


def main():
    """Main comparison script."""
    if len(sys.argv) < 2:
        print("Usage: python3 test/compare_mcts_vs_nn_mcts.py <model_path> [num_games] [simulations]")
        print("\nExample:")
        print("  python3 test/compare_mcts_vs_nn_mcts.py checkpoints/best_model.pt 3 50")
        print("\nThis will play 3 games with 50 simulations per move")
        sys.exit(1)

    model_path = sys.argv[1]
    num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    simulations = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    # Run comparison
    results = run_comparison(model_path, num_games, simulations)

    # Print summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"  NN-MCTS wins: {results['nn_wins']}")
    print(f"  Baseline MCTS wins: {results['baseline_wins']}")
    print(f"  Draws: {results['draws']}")

    total = num_games
    nn_score = results['nn_wins'] + 0.5 * results['draws']
    baseline_score = results['baseline_wins'] + 0.5 * results['draws']

    print(f"\nScore:")
    print(f"  NN-MCTS: {nn_score}/{total} ({nn_score/total*100:.1f}%)")
    print(f"  Baseline MCTS: {baseline_score}/{total} ({baseline_score/total*100:.1f}%)")

    print("\n" + "="*70)
    print("✅ Comparison complete!")
    print("="*70)

    # Interpretation
    print("\nInterpretation:")
    if nn_score > baseline_score:
        print("  🎉 NN-MCTS is stronger! The neural network helps.")
    elif nn_score < baseline_score:
        print("  📊 Baseline MCTS is stronger (expected with only 50 training games)")
        print("  💡 Train with 1K-10K games for a stronger neural network")
    else:
        print("  ⚖️  Equal strength - both engines play similarly")

    print("\n✅ Pipeline validated! NN-MCTS integration works.")


if __name__ == "__main__":
    main()
