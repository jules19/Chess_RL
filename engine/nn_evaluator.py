"""
Neural Network Evaluator for MCTS

This module provides a neural network-based position evaluator that can
replace the hand-crafted evaluator in MCTS rollouts.

The NN evaluator returns:
- Position value: scalar evaluation in [-1, +1] range
- Move policy: probability distribution over moves (optional)

This enables MCTS to use learned chess knowledge instead of hand-crafted heuristics.
"""

import torch
import chess
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net.model import create_model
from net.encoding import board_to_tensor, legal_moves_mask, index_to_move


class NeuralNetworkEvaluator:
    """
    Neural network-based position evaluator for chess.

    This class wraps a trained policy-value network to provide position
    evaluations for MCTS search. It caches the model and runs on the
    specified device (CPU or GPU).
    """

    def __init__(self, model_path, device='mps', num_res_blocks=2, num_channels=64):
        """
        Initialize neural network evaluator.

        Args:
            model_path: Path to trained model checkpoint (.pt file)
            device: Device to run on ('cpu', 'cuda', or 'mps')
            num_res_blocks: Number of residual blocks in the model
            num_channels: Number of channels in residual blocks
        """
        self.device = torch.device(device)

        # Load model
        self.model = create_model(num_res_blocks=num_res_blocks, num_channels=num_channels)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded NN evaluator from {model_path}")
        print(f"  Device: {self.device}")
        print(f"  Model: {num_res_blocks} ResBlocks, {num_channels} channels")

    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate a chess position using the neural network.

        Args:
            board: Chess board position to evaluate

        Returns:
            Position evaluation from White's perspective:
            +1.0 = White is winning
             0.0 = Equal position
            -1.0 = Black is winning
        """
        with torch.no_grad():
            # Encode board
            board_tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(self.device)

            # Get value prediction
            _, value = self.model(board_tensor)

            # Extract scalar value
            eval_score = value.item()

            # The network predicts from current player's perspective
            # We need to return from White's perspective
            if board.turn == chess.BLACK:
                eval_score = -eval_score

            # Value is already in [-1, 1] range from tanh
            return eval_score

    def evaluate_with_policy(self, board: chess.Board):
        """
        Evaluate position and get move policy probabilities.

        Args:
            board: Chess board position

        Returns:
            value: Position evaluation (float in [-1, 1])
            policy_probs: Dict mapping legal moves to probabilities
        """
        with torch.no_grad():
            # Encode board and legal moves
            board_tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(self.device)
            mask = torch.from_numpy(legal_moves_mask(board)).unsqueeze(0).to(self.device)

            # Get predictions
            policy_probs, value = self.model.get_policy_value(board_tensor, mask)

            # Extract value
            eval_score = value.item()
            if board.turn == chess.BLACK:
                eval_score = -eval_score

            # Convert policy to dict of moves -> probabilities
            policy_probs = policy_probs.squeeze(0).cpu().numpy()

            move_probs = {}
            for move in board.legal_moves:
                try:
                    from net.encoding import move_to_index
                    move_idx = move_to_index(move)
                    move_probs[move] = policy_probs[move_idx]
                except:
                    # If encoding fails, assign uniform probability
                    move_probs[move] = 1.0 / len(list(board.legal_moves))

            # Normalize probabilities
            total_prob = sum(move_probs.values())
            if total_prob > 0:
                move_probs = {move: prob / total_prob for move, prob in move_probs.items()}

            return eval_score, move_probs

    def get_best_move(self, board: chess.Board):
        """
        Get the move with highest policy probability.

        Args:
            board: Chess board position

        Returns:
            Best move according to the neural network policy
        """
        _, move_probs = self.evaluate_with_policy(board)

        if not move_probs:
            # Fallback to first legal move
            return list(board.legal_moves)[0] if board.legal_moves else None

        return max(move_probs.items(), key=lambda x: x[1])[0]


def create_nn_evaluator(model_path, device='mps', **kwargs):
    """
    Factory function to create a neural network evaluator.

    Args:
        model_path: Path to trained model checkpoint
        device: Device to run on ('cpu', 'cuda', or 'mps')
        **kwargs: Additional arguments for model architecture

    Returns:
        NeuralNetworkEvaluator instance
    """
    return NeuralNetworkEvaluator(model_path, device=device, **kwargs)


if __name__ == "__main__":
    """Test NN evaluator on a few positions."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 engine/nn_evaluator.py <model_path>")
        print("\nExample:")
        print("  python3 engine/nn_evaluator.py checkpoints/best_model.pt")
        sys.exit(1)

    model_path = sys.argv[1]

    print("="*70)
    print("Testing Neural Network Evaluator")
    print("="*70)

    # Create evaluator
    nn_eval = create_nn_evaluator(model_path, device='mps', num_res_blocks=2, num_channels=64)

    # Test on starting position
    print("\n1. Starting position:")
    board = chess.Board()
    print(board)

    eval_score, move_probs = nn_eval.evaluate_with_policy(board)
    print(f"\nPosition value: {eval_score:+.3f} (White's perspective)")

    print("\nTop 5 moves:")
    sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
    for i, (move, prob) in enumerate(sorted_moves[:5], 1):
        print(f"  {i}. {move}: {prob*100:.1f}%")

    # Test after e4
    print("\n2. After 1. e4:")
    board.push(chess.Move.from_uci("e2e4"))
    print(board)

    eval_score = nn_eval.evaluate(board)
    print(f"\nPosition value: {eval_score:+.3f} (White's perspective)")

    best_move = nn_eval.get_best_move(board)
    print(f"Best move: {best_move}")

    print("\n" + "="*70)
    print("✅ NN Evaluator test complete!")
    print("="*70)
