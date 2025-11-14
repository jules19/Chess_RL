"""
Monte Carlo Tree Search (MCTS) - Phase 2 (Week 2)

This module implements MCTS with UCT (Upper Confidence bounds applied to Trees)
for chess move selection. MCTS is fundamentally different from minimax:

Minimax:
- Explores entire tree to fixed depth
- Assumes perfect play from both sides
- Deterministic

MCTS:
- Selectively explores promising branches
- Uses random simulations to estimate value
- Balances exploration vs exploitation
- Gets better with more simulations

Key Components:
1. Selection: Walk down tree using UCT formula
2. Expansion: Add new node to tree
3. Simulation: Play out random game (rollout)
4. Backpropagation: Update node statistics

Target Strength: ~1400-1600 Elo with 200 simulations/move
"""

import chess
import random
import math
import time
from typing import Optional, List
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.evaluator import evaluate


class MCTSNode:
    """
    Node in the MCTS search tree.

    Each node represents a board position and stores:
    - visit_count: Number of times this node has been visited
    - total_value: Sum of all simulation results (from this node's perspective)
    - parent: Parent node in the tree
    - children: Dictionary mapping moves to child nodes
    - untried_moves: Legal moves that haven't been explored yet
    """

    def __init__(self, board: chess.Board, parent: Optional['MCTSNode'] = None,
                 move: Optional[chess.Move] = None):
        """
        Initialize a new MCTS node.

        Args:
            board: The chess position this node represents
            parent: Parent node (None for root)
            move: The move that led to this position (None for root)
        """
        self.board = board.copy()
        self.parent = parent
        self.move = move  # Move that led to this position

        # MCTS statistics
        self.visit_count = 0
        self.total_value = 0.0  # Sum of values from simulations

        # Children and unexplored moves
        self.children = {}  # Dict[chess.Move, MCTSNode]
        self.untried_moves = list(board.legal_moves)
        random.shuffle(self.untried_moves)  # Randomize order to avoid bias

    def is_fully_expanded(self) -> bool:
        """Check if all legal moves have been tried."""
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        """Check if this is a terminal position (game over)."""
        return self.board.is_game_over()

    def get_average_value(self) -> float:
        """Get the average value of this node (Q-value)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def uct_value(self, exploration_constant: float = 1.41) -> float:
        """
        Calculate UCT (Upper Confidence bound for Trees) value.

        UCT formula: Q(v) + C * sqrt(ln(N(parent)) / N(v))

        Where:
        - Q(v) = average value (exploitation term)
        - C = exploration constant (typically sqrt(2) ≈ 1.41)
        - N(parent) = parent visit count
        - N(v) = this node's visit count

        The second term encourages exploration of less-visited nodes.

        Args:
            exploration_constant: C parameter (higher = more exploration)

        Returns:
            UCT value (higher = should be selected)
        """
        if self.visit_count == 0:
            return float('inf')  # Unvisited nodes have infinite priority

        # Exploitation term (average value)
        exploitation = self.get_average_value()

        # Exploration term (uncertainty bonus)
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )

        return exploitation + exploration

    def best_child(self, exploration_constant: float = 1.41) -> 'MCTSNode':
        """
        Select the best child using UCT.

        Args:
            exploration_constant: C parameter for UCT

        Returns:
            Child node with highest UCT value
        """
        return max(self.children.values(),
                  key=lambda child: child.uct_value(exploration_constant))

    def most_visited_child(self) -> Optional['MCTSNode']:
        """
        Get the child with the most visits (for final move selection).

        Returns:
            Most visited child node, or None if no children
        """
        if not self.children:
            return None
        return max(self.children.values(), key=lambda child: child.visit_count)

    def expand(self) -> 'MCTSNode':
        """
        Expand the tree by adding one new child node.

        Returns:
            The newly created child node
        """
        # Pick an untried move
        move = self.untried_moves.pop()

        # Create new board position
        new_board = self.board.copy()
        new_board.push(move)

        # Create child node
        child = MCTSNode(new_board, parent=self, move=move)
        self.children[move] = child

        return child


def simulate_random(board: chess.Board, max_moves: int = 200) -> float:
    """
    Simulate a random game from the given position (random rollout).

    This is the simplest simulation policy - just play random legal moves
    until the game ends or we hit the move limit.

    Args:
        board: Starting position
        max_moves: Maximum moves to simulate (prevents infinite games)

    Returns:
        Game result from White's perspective:
        +1.0 = White wins
        -1.0 = Black wins
         0.0 = Draw
    """
    sim_board = board.copy()
    moves = 0

    while not sim_board.is_game_over() and moves < max_moves:
        # Pick random legal move
        legal_moves = list(sim_board.legal_moves)
        if not legal_moves:
            break
        move = random.choice(legal_moves)
        sim_board.push(move)
        moves += 1

    # Determine result
    if sim_board.is_checkmate():
        # Checkmate - winner is the side that just moved
        return 1.0 if sim_board.turn == chess.BLACK else -1.0
    else:
        # Draw (stalemate, insufficient material, move limit, etc.)
        return 0.0


def simulate_with_evaluator(board: chess.Board, max_moves: int = 50) -> float:
    """
    Simulate a game using the evaluator for guidance (smart rollout).

    Instead of pure random moves, use the evaluation function to guide play.
    This gives much better estimates than random rollouts.

    Args:
        board: Starting position
        max_moves: Maximum moves to simulate (shorter than random since it's slower)

    Returns:
        Game result estimate from White's perspective (-1.0 to +1.0)
    """
    sim_board = board.copy()
    moves = 0

    while not sim_board.is_game_over() and moves < max_moves:
        legal_moves = list(sim_board.legal_moves)
        if not legal_moves:
            break

        # Pick move with simple 1-ply evaluation
        best_move = None
        best_eval = float('-inf') if sim_board.turn == chess.WHITE else float('inf')

        for move in legal_moves:
            sim_board.push(move)
            eval_score = evaluate(sim_board)
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

    # Return result or position evaluation
    if sim_board.is_checkmate():
        return 1.0 if sim_board.turn == chess.BLACK else -1.0
    elif sim_board.is_game_over():
        return 0.0
    else:
        # Game didn't finish - use evaluation function
        # Normalize evaluation to [-1, 1] range
        eval_score = evaluate(sim_board)
        # Clamp between -1 and 1 using tanh-like function
        return max(-1.0, min(1.0, eval_score / 1000.0))


def backpropagate(node: MCTSNode, value: float):
    """
    Backpropagate simulation result up the tree.

    Updates visit counts and values for all nodes from leaf to root.
    Values are negated at each level (what's good for White is bad for Black).

    Args:
        node: Leaf node where simulation started
        value: Simulation result (from White's perspective)
    """
    while node is not None:
        node.visit_count += 1
        # Negate value at each level (alternating perspectives)
        node.total_value += value
        value = -value  # Flip perspective for parent
        node = node.parent


def mcts_search(board: chess.Board, simulations: int = 200,
                use_evaluator: bool = True,
                exploration_constant: float = 1.41,
                verbose: bool = False) -> Optional[chess.Move]:
    """
    Perform MCTS search to find the best move.

    Algorithm:
    1. Selection: Starting at root, select best child using UCT until we reach
       a node that's not fully expanded or is terminal
    2. Expansion: Add one new child to the tree
    3. Simulation: Play out a random (or guided) game from the new position
    4. Backpropagation: Update all nodes from leaf to root with the result

    Repeat for the specified number of simulations, then return the most
    visited move (most simulations = most confidence).

    Args:
        board: Current board position
        simulations: Number of MCTS iterations to run
        use_evaluator: If True, use smart rollouts; if False, use random
        exploration_constant: UCT exploration parameter (higher = more exploration)
        verbose: Print search statistics

    Returns:
        Best move found, or None if no legal moves
    """
    if board.is_game_over():
        return None

    # Create root node
    root = MCTSNode(board)

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

        # 3. SIMULATION - Play out game
        if use_evaluator:
            value = simulate_with_evaluator(search_board)
        else:
            value = simulate_random(search_board)

        # Adjust value to be from the perspective of the side to move at root
        # (MCTS nodes store values from their perspective)
        if board.turn == chess.BLACK:
            value = -value

        # 4. BACKPROPAGATION - Update tree
        backpropagate(node, value)

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n=== MCTS Search Statistics ===")
        print(f"Simulations: {simulations}")
        print(f"Time: {elapsed:.2f}s")
        print(f"Sims/sec: {simulations/elapsed:.0f}")
        print(f"Rollout type: {'Evaluator' if use_evaluator else 'Random'}")
        print(f"\nTop moves by visit count:")

        # Sort children by visit count
        sorted_children = sorted(root.children.items(),
                                key=lambda x: x[1].visit_count,
                                reverse=True)

        for i, (move, child) in enumerate(sorted_children[:5]):
            win_rate = (child.get_average_value() + 1) / 2 * 100  # Convert [-1,1] to [0,100]
            print(f"  {i+1}. {move}: {child.visit_count} visits, "
                  f"avg value: {child.get_average_value():+.3f}, "
                  f"win rate: {win_rate:.1f}%")

    # Return most visited move
    best_child = root.most_visited_child()
    return best_child.move if best_child else None


def best_move_mcts(board: chess.Board, simulations: int = 200,
                   use_evaluator: bool = True, verbose: bool = False) -> Optional[chess.Move]:
    """
    Wrapper function for MCTS search (matches interface of other engines).

    Args:
        board: Current position
        simulations: Number of MCTS iterations
        use_evaluator: Use smart rollouts (True) or random (False)
        verbose: Print search statistics

    Returns:
        Best move found
    """
    return mcts_search(board, simulations=simulations,
                      use_evaluator=use_evaluator, verbose=verbose)


if __name__ == "__main__":
    """Quick test of MCTS implementation."""
    print("Testing MCTS implementation...")

    # Test position 1: Starting position
    board = chess.Board()
    print("\n1. Starting position - Random rollouts")
    move = best_move_mcts(board, simulations=100, use_evaluator=False, verbose=True)
    print(f"\nSelected move: {move}")

    # Test position 2: Starting position with evaluator
    print("\n\n2. Starting position - Evaluator rollouts")
    move = best_move_mcts(board, simulations=100, use_evaluator=True, verbose=True)
    print(f"\nSelected move: {move}")

    # Test position 3: Tactical position (mate in 2)
    print("\n\n3. Tactical position (Scholar's mate setup)")
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1")
    print(board)
    move = best_move_mcts(board, simulations=200, use_evaluator=True, verbose=True)
    print(f"\nSelected move: {move} (should be Qxf7# or similar forcing move)")

    print("\n✓ MCTS implementation complete!")
