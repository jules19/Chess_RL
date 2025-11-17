"""
Neural Network MCTS - Phase 3b Integration

This module integrates the trained neural network into MCTS search,
replacing the hand-crafted evaluator with learned chess knowledge.

Key differences from baseline MCTS:
- Uses NN for position evaluation instead of hand-crafted heuristics
- Can optionally use NN policy to guide move selection (PUCT)
- Provides foundation for AlphaZero-style search

This is Phase 3b: validating that NN improves MCTS strength.
"""

import chess
import random
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.mcts import MCTSNode, backpropagate, get_prioritized_moves, mcts_search
from engine.nn_evaluator import NeuralNetworkEvaluator


def simulate_with_nn(board: chess.Board, nn_evaluator: NeuralNetworkEvaluator,
                     max_moves: int = 50, sample_size: int = 10,
                     use_policy: bool = False) -> float:
    """
    Simulate a game using neural network for guidance (NN-guided rollout).

    This replaces the hand-crafted evaluator with the neural network's
    learned evaluation function.

    Args:
        board: Starting position
        nn_evaluator: Neural network evaluator instance
        max_moves: Maximum moves to simulate
        sample_size: Number of moves to evaluate per position
        use_policy: If True, use NN policy to guide move selection

    Returns:
        Game result estimate from White's perspective (-1.0 to +1.0)
    """
    sim_board = board.copy()
    moves = 0

    while not sim_board.is_game_over() and moves < max_moves:
        legal_moves = list(sim_board.legal_moves)
        if not legal_moves:
            break

        if use_policy:
            # Use NN policy to select moves probabilistically
            _, move_probs = nn_evaluator.evaluate_with_policy(sim_board)

            # Sample move based on policy probabilities
            moves_list = list(move_probs.keys())
            probs_list = [move_probs[m] for m in moves_list]

            # Add small epsilon for exploration
            total_prob = sum(probs_list)
            if total_prob > 0:
                probs_list = [p / total_prob for p in probs_list]
                best_move = random.choices(moves_list, weights=probs_list, k=1)[0]
            else:
                best_move = random.choice(legal_moves)
        else:
            # Sample and evaluate moves like baseline MCTS
            moves_to_evaluate = get_prioritized_moves(sim_board, legal_moves, sample_size)

            # Pick move with best NN evaluation
            best_move = None
            best_eval = float('-inf') if sim_board.turn == chess.WHITE else float('inf')

            for move in moves_to_evaluate:
                sim_board.push(move)
                eval_score = nn_evaluator.evaluate(sim_board)
                sim_board.pop()

                if sim_board.turn == chess.WHITE:
                    if eval_score > best_eval:
                        best_eval = eval_score
                        best_move = move
                else:
                    if eval_score < best_eval:
                        best_eval = eval_score
                        best_move = move

        if best_move:
            sim_board.push(best_move)
        moves += 1

    # Return result or NN evaluation
    if sim_board.is_checkmate():
        return 1.0 if sim_board.turn == chess.BLACK else -1.0
    elif sim_board.is_game_over():
        return 0.0
    else:
        # Use NN evaluation at terminal position
        return nn_evaluator.evaluate(sim_board)


def nn_mcts_search(board: chess.Board, nn_evaluator: NeuralNetworkEvaluator,
                   simulations: int = 200,
                   exploration_constant: float = 1.41,
                   sample_size: int = 10,
                   use_policy: bool = False,
                   filter_blunders: bool = True,
                   verbose: bool = False) -> chess.Move:
    """
    Perform MCTS search using neural network for evaluation.

    This is identical to baseline MCTS except it uses the neural network
    for position evaluation instead of the hand-crafted evaluator.

    Args:
        board: Current board position
        nn_evaluator: Neural network evaluator instance
        simulations: Number of MCTS iterations
        exploration_constant: UCT exploration parameter
        sample_size: Number of moves to evaluate in rollouts
        use_policy: If True, use NN policy to guide rollouts
        filter_blunders: Filter moves that hang material
        verbose: Print search statistics

    Returns:
        Best move found
    """
    if board.is_game_over():
        return None

    # Create root node
    root = MCTSNode(board, filter_blunders=filter_blunders)

    import time
    start_time = time.time()

    # Run MCTS simulations
    for i in range(simulations):
        node = root
        search_board = board.copy()

        # 1. SELECTION - Walk down tree using UCT
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.best_child(exploration_constant)
            search_board.push(node.move)

        # 2. EXPANSION - Add new child if not terminal
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()
            search_board.push(node.move)

        # 3. SIMULATION - Play out game using NN
        value = simulate_with_nn(search_board, nn_evaluator,
                                 sample_size=sample_size,
                                 use_policy=use_policy)

        # Adjust value to be from leaf node parent's perspective
        if search_board.turn == chess.WHITE:
            value = -value

        # 4. BACKPROPAGATION - Update tree
        backpropagate(node, value)

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n=== NN-MCTS Search Statistics ===")
        print(f"Simulations: {simulations}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Sims/sec: {simulations/elapsed:.0f}")
        print(f"Rollout type: NN-guided {'(policy)' if use_policy else '(value)'}")
        print(f"\nTop moves by visit count:")

        # Sort children by visit count
        sorted_children = sorted(root.children.items(),
                                key=lambda x: x[1].visit_count,
                                reverse=True)

        for i, (move, child) in enumerate(sorted_children[:5]):
            win_rate = (child.get_average_value() + 1) / 2 * 100
            print(f"  {i+1}. {move}: {child.visit_count} visits, "
                  f"avg value: {child.get_average_value():+.3f}, "
                  f"win rate: {win_rate:.1f}%")

    # Return most visited move
    best_child = root.most_visited_child()
    return best_child.move if best_child else None


class NNMCTSPlayer:
    """
    Chess player combining MCTS with neural network evaluation.

    This player uses Monte Carlo Tree Search guided by a trained neural
    network instead of hand-crafted heuristics.
    """

    def __init__(self, model_path, simulations=200, device='mps',
                 num_res_blocks=2, num_channels=64, use_policy=False):
        """
        Initialize NN-MCTS player.

        Args:
            model_path: Path to trained model checkpoint
            simulations: Number of MCTS simulations per move
            device: Device to run on ('cpu', 'cuda', or 'mps')
            num_res_blocks: Number of residual blocks in model
            num_channels: Number of channels in model
            use_policy: If True, use NN policy to guide rollouts
        """
        self.nn_evaluator = NeuralNetworkEvaluator(
            model_path, device=device,
            num_res_blocks=num_res_blocks,
            num_channels=num_channels
        )
        self.simulations = simulations
        self.use_policy = use_policy

    def select_move(self, board: chess.Board, verbose=False):
        """
        Select best move using NN-MCTS.

        Args:
            board: Current board position
            verbose: Print search statistics

        Returns:
            Selected move
        """
        return nn_mcts_search(
            board,
            self.nn_evaluator,
            simulations=self.simulations,
            use_policy=self.use_policy,
            verbose=verbose
        )


if __name__ == "__main__":
    """Test NN-MCTS on a simple position."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 search/nn_mcts.py <model_path>")
        print("\nExample:")
        print("  python3 search/nn_mcts.py checkpoints/best_model.pt")
        sys.exit(1)

    model_path = sys.argv[1]

    print("="*70)
    print("Testing NN-MCTS")
    print("="*70)

    # Create NN-MCTS player
    player = NNMCTSPlayer(model_path, simulations=50, device='mps',
                         num_res_blocks=2, num_channels=64)

    # Test on starting position
    board = chess.Board()
    print("\nStarting position:")
    print(board)

    print("\nRunning NN-MCTS with 50 simulations...")
    move = player.select_move(board, verbose=True)

    print(f"\n✅ Selected move: {move}")
    print("="*70)
