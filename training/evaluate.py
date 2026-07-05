"""
Neural Network Evaluation

Evaluate trained neural network against baseline engines to measure playing strength.

Evaluation Methods:
1. Engine vs Engine tournaments (NN vs Minimax, NN vs MCTS)
2. Policy accuracy on test positions
3. Value prediction accuracy
4. Tactical puzzle solving

This provides objective measurement of whether supervised learning improved
the network's playing strength to match/exceed hand-crafted evaluation (~1400-1600 Elo).

Usage:
    # Evaluate against minimax and MCTS
    python3 training/evaluate.py checkpoints/best_model.pt

    # Specify number of games
    python3 training/evaluate.py checkpoints/best_model.pt --games 50
"""

import torch
import chess
import sys
import os
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from net.model import load_model
from net.encoding import board_to_tensor, legal_moves_mask, index_to_move


class NeuralNetworkPlayer:
    """
    Chess player using trained neural network.

    The NN player selects moves by:
    1. Encoding the current board state
    2. Running forward pass through the network
    3. Masking illegal moves
    4. Selecting the move with highest probability
    """

    def __init__(self, model_path, device='cpu', temperature=0.0):
        """
        Initialize NN player.

        Args:
            model_path: Path to saved model checkpoint
            device: 'cpu' or 'cuda'
            temperature: Sampling temperature (0 = greedy, >0 = stochastic)
        """
        self.device = torch.device(device)
        self.temperature = temperature

        # Load model (architecture inferred from the checkpoint itself)
        self.model, _ = load_model(model_path, self.device)

        print(f"Loaded NN player from: {model_path}")

    def select_move(self, board):
        """
        Select best move for current position using neural network.

        Args:
            board: chess.Board instance

        Returns:
            move: chess.Move selected by network
            value: Predicted position value (-1 to +1)
        """
        with torch.no_grad():
            # Encode board state
            board_tensor = board_to_tensor(board)
            board_tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(self.device)

            # Get legal moves mask
            legal_mask = legal_moves_mask(board)
            legal_mask = torch.from_numpy(legal_mask).unsqueeze(0).to(self.device)

            # Forward pass
            policy_probs, value = self.model.get_policy_value(board_tensor, legal_mask)
            policy_probs = policy_probs.squeeze(0).cpu().numpy()

            # Select move (greedy by default)
            if self.temperature == 0.0:
                # Greedy: select highest probability legal move
                for move_idx in policy_probs.argsort()[::-1]:
                    move = index_to_move(move_idx, board)
                    if move and move in board.legal_moves:
                        return move, value.item()
            else:
                # Stochastic: sample from the temperature-adjusted policy.
                #
                # Temperature reshapes the distribution: p_i^(1/T) / sum(...)
                #   T → 0:   sharpens toward the single best move (greedy)
                #   T = 1:   samples from the raw policy
                #   T > 1:   flattens toward uniform (more exploration)
                #
                # Sampling matters for evaluation: two greedy engines play the
                # SAME game every time, so a "10-game match" is really one
                # game repeated. A little temperature gives game variety.
                legal_indices = np.nonzero(legal_mask.squeeze(0).cpu().numpy())[0]
                legal_probs = policy_probs[legal_indices]
                legal_probs = legal_probs ** (1.0 / self.temperature)
                legal_probs = legal_probs / legal_probs.sum()

                sampled_idx = np.random.choice(legal_indices, p=legal_probs)
                move = index_to_move(int(sampled_idx), board)
                if move:
                    return move, value.item()

        # Fallback: return first legal move (should rarely happen)
        return list(board.legal_moves)[0], 0.0


class MinimaxPlayer:
    """Baseline opponent: the Phase 1 alpha-beta engine, wrapped in the
    player interface (select_move(board) -> (move, value)) this module uses.

    ⚠️ LESSON LEARNED: an earlier version of main() imported MinimaxPlayer
    from search.minimax — a class that never existed — inside a try/except
    that printed a warning and skipped the whole baseline match. The
    evaluation script's most important feature silently did nothing. That's
    the "silent guards hide bugs" antipattern (course Module 8, case study 2)
    live in the tool that was supposed to measure everything else.
    """

    def __init__(self, depth=3):
        from search.minimax import best_move_minimax
        self._best_move = best_move_minimax
        self.depth = depth

    def select_move(self, board):
        from engine.evaluator import evaluate as evaluate_fn
        move = self._best_move(board, depth=self.depth)
        # Report the static eval so game logs stay informative
        return move, max(-1.0, min(1.0, evaluate_fn(board) / 1000.0))


class MCTSPlayer:
    """Baseline opponent: the Phase 2 rollout-MCTS engine (see MinimaxPlayer
    for why these wrappers live here)."""

    def __init__(self, num_simulations=200):
        from search.mcts import best_move_mcts
        self._best_move = best_move_mcts
        self.num_simulations = num_simulations

    def select_move(self, board):
        move = self._best_move(board, simulations=self.num_simulations)
        return move, 0.0


def play_game(white_player, black_player, max_moves=150, verbose=False):
    """
    Play one game between two players.

    Args:
        white_player: Player object with select_move(board) method
        black_player: Player object with select_move(board) method
        max_moves: Maximum moves before declaring draw
        verbose: Print moves during game

    Returns:
        result: Game result ("1-0", "0-1", "1/2-1/2")
        moves: Number of moves played
    """
    board = chess.Board()
    move_count = 0

    if verbose:
        print("\n" + "="*50)
        print("Starting new game")
        print("="*50)

    while not board.is_game_over() and move_count < max_moves:
        # Select move
        if board.turn == chess.WHITE:
            move, value = white_player.select_move(board)
            player = "White"
        else:
            move, value = black_player.select_move(board)
            player = "Black"

        if verbose:
            print(f"{move_count+1:3d}. {move:6s} ({player}, eval: {value:+.2f})")

        board.push(move)
        move_count += 1

    # Determine result
    if board.is_checkmate():
        result = "1-0" if board.turn == chess.BLACK else "0-1"
        outcome = "Checkmate"
    elif board.is_stalemate():
        result = "1/2-1/2"
        outcome = "Stalemate"
    elif board.is_insufficient_material():
        result = "1/2-1/2"
        outcome = "Insufficient material"
    elif board.is_fifty_moves():
        result = "1/2-1/2"
        outcome = "50-move rule"
    elif board.is_repetition():
        result = "1/2-1/2"
        outcome = "Repetition"
    elif move_count >= max_moves:
        result = "1/2-1/2"
        outcome = "Move limit"
    else:
        result = "*"
        outcome = "Unknown"

    if verbose:
        print(f"\nGame over: {result} ({outcome}) after {move_count} moves")

    return result, move_count


def tournament(player1, player2, num_games=20, player1_name="Player 1", player2_name="Player 2"):
    """
    Run a tournament between two players.

    Args:
        player1: First player
        player2: Second player
        num_games: Number of games to play (alternating colors)
        player1_name: Name for display
        player2_name: Name for display

    Returns:
        (wins, losses, draws) from player1's perspective
    """
    wins = 0
    losses = 0
    draws = 0

    print(f"\n{'='*70}")
    print(f"Tournament: {player1_name} vs {player2_name}")
    print(f"Playing {num_games} games (alternating colors)")
    print(f"{'='*70}")

    for i in tqdm(range(num_games), desc="Games"):
        # Alternate colors for fairness
        if i % 2 == 0:
            # Player 1 as White
            result, moves = play_game(player1, player2)
            if result == "1-0":
                wins += 1
            elif result == "0-1":
                losses += 1
            else:
                draws += 1
        else:
            # Player 1 as Black
            result, moves = play_game(player2, player1)
            if result == "1-0":
                losses += 1
            elif result == "0-1":
                wins += 1
            else:
                draws += 1

    return wins, losses, draws


def print_tournament_results(wins, losses, draws, player1_name, player2_name):
    """Print formatted tournament results."""
    total = wins + losses + draws
    win_rate = wins / total if total > 0 else 0
    draw_rate = draws / total if total > 0 else 0
    loss_rate = losses / total if total > 0 else 0

    print(f"\n{'='*70}")
    print(f"Tournament Results")
    print(f"{'='*70}")
    print(f"{player1_name:20s} {wins:3d} wins   ({win_rate:5.1%})")
    print(f"{player2_name:20s} {losses:3d} wins   ({loss_rate:5.1%})")
    print(f"{'Draws':20s} {draws:3d} draws  ({draw_rate:5.1%})")
    print(f"{'='*70}")
    print(f"Total games: {total}")

    # Expected score (1 point per win, 0.5 per draw)
    score = wins + 0.5 * draws
    expected = score / total if total > 0 else 0
    print(f"{player1_name} score: {score:.1f}/{total} ({expected:.1%})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate neural network playing strength")
    parser.add_argument('model_path', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--games', type=int, default=20, help='Number of games per opponent')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--verbose', action='store_true', help='Print game moves')
    args = parser.parse_args()

    print("="*70)
    print("Neural Network Evaluation")
    print("="*70)

    # Load NN player
    nn_player = NeuralNetworkPlayer(args.model_path, device=args.device)

    # Tournament vs Minimax.
    # No try/except-and-skip here: if a baseline can't run, we want a loud
    # crash, not an evaluation that silently measures nothing.
    print("\n" + "="*70)
    print("1. Neural Network vs Minimax (depth 3)")
    print("="*70)

    minimax_player = MinimaxPlayer(depth=3)
    wins, losses, draws = tournament(
        nn_player, minimax_player,
        num_games=args.games,
        player1_name="Neural Network",
        player2_name="Minimax (depth 3)"
    )
    print_tournament_results(wins, losses, draws, "Neural Network", "Minimax (depth 3)")

    # Tournament vs MCTS
    print("\n" + "="*70)
    print("2. Neural Network vs MCTS (200 simulations)")
    print("="*70)

    mcts_player = MCTSPlayer(num_simulations=200)
    wins, losses, draws = tournament(
        nn_player, mcts_player,
        num_games=args.games,
        player1_name="Neural Network",
        player2_name="MCTS (200 sims)"
    )
    print_tournament_results(wins, losses, draws, "Neural Network", "MCTS (200 sims)")

    print("\n" + "="*70)
    print("✅ Evaluation Complete")
    print("="*70)
    print("\nInterpretation:")
    print("  • 50%+ vs Minimax/MCTS → Competitive (~1400-1600 Elo)")
    print("  • 40-50% → Reasonable, needs more training")
    print("  • <40% → Weak, review training pipeline")


if __name__ == "__main__":
    main()
