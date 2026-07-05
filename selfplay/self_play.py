"""
Self-Play Game Generation - where the training data comes from

Each game the engine plays against itself produces one training example per
position:

    state  - the board encoding the network saw
    policy - the PUCT visit distribution over moves (NOT the single move
             played!). Search spent simulations judging every candidate;
             the full distribution is a far richer target than a one-hot
             "this was the move". This is the heart of AlphaZero: the
             network learns to imitate its own SEARCH, which is stronger
             than the raw network — so imitating it is self-improvement.
    value  - the eventual game result from the perspective of the side to
             move at that position. Every position in a game White wins
             gets +1 for White-to-move positions and -1 for Black-to-move
             ones. Yes, this credits even the bad moves of the winner —
             averaged over many games, the signal wins out over the noise.

Exploration knobs (both essential — remove either and self-play collapses
into repeating one deterministic game forever):

    Dirichlet root noise - makes SEARCH consider unlikely moves
    Temperature          - makes MOVE SELECTION stochastic early in the game
                           (τ=1 for the first `temperature_moves` plies,
                           then τ→0 so won positions get converted cleanly)
"""

import os
import random
import sys
from typing import Callable, List, Optional, Tuple

import chess
import numpy as np

# Add project root to path (for running this file directly as a script)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net.encoding import board_to_tensor, move_to_index, NUM_MOVES
from search.puct import puct_search, visits_to_policy, select_move


def play_selfplay_game(policy_value_fn: Callable,
                       simulations: int = 100,
                       temperature_moves: int = 20,
                       max_moves: int = 200,
                       rng: Optional[random.Random] = None,
                       verbose: bool = False) -> Tuple[List, str]:
    """
    Play one self-play game and return its training examples.

    Args:
        policy_value_fn: Network adapter (see search.puct.network_policy_value)
        simulations: PUCT simulations per move (AlphaZero used 800; use
                     16-64 for CPU experiments)
        temperature_moves: Plies of stochastic (τ=1) move selection at the
                     start of the game, for opening diversity
        max_moves: Safety cap; games hitting it are scored as draws
        rng: Optional random.Random for reproducibility
        verbose: Print moves as they're played

    Returns:
        (examples, result)
        examples: list of (state, policy_vector, value) ready for the
                  replay buffer; value is from the side-to-move perspective
        result: PGN result string ("1-0", "0-1", "1/2-1/2")
    """
    board = chess.Board()
    history = []  # (state, policy_vector, side_to_move) per position

    while not board.is_game_over(claim_draw=True) and board.ply() < max_moves:
        _, visit_counts = puct_search(
            board, policy_value_fn,
            simulations=simulations,
            add_root_noise=True,  # exploration in SEARCH
            rng=rng,
        )

        # Training target: raw visit proportions (τ=1), regardless of the
        # temperature used to pick the move actually played.
        target = visits_to_policy(visit_counts, temperature=1.0)
        policy_vector = np.zeros(NUM_MOVES, dtype=np.float32)
        for move, prob in target.items():
            policy_vector[move_to_index(move)] = prob

        history.append((board_to_tensor(board), policy_vector, board.turn))

        # Exploration in MOVE SELECTION: stochastic early, greedy later
        temperature = 1.0 if board.ply() < temperature_moves else 0.0
        move = select_move(visit_counts, temperature=temperature, rng=rng)

        if verbose:
            print(f"  {board.ply() + 1}. {board.san(move)}")
        board.push(move)

    # Score the game from White's perspective
    result = board.result(claim_draw=True)
    if result == "1-0":
        white_value = 1.0
    elif result == "0-1":
        white_value = -1.0
    else:
        white_value = 0.0
        result = "1/2-1/2"  # normalize "*" (max_moves cutoff) to a draw

    # Label every position with the final outcome, from the perspective of
    # the player to move at that position
    examples = [
        (state, policy, white_value if side_to_move == chess.WHITE else -white_value)
        for state, policy, side_to_move in history
    ]
    return examples, result


def generate_games(policy_value_fn: Callable,
                   num_games: int,
                   simulations: int = 100,
                   temperature_moves: int = 20,
                   max_moves: int = 200,
                   rng: Optional[random.Random] = None,
                   verbose: bool = True) -> List:
    """
    Generate a batch of self-play games.

    Returns:
        Flat list of (state, policy, value) examples from all games.
    """
    all_examples = []
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

    for game_idx in range(num_games):
        examples, result = play_selfplay_game(
            policy_value_fn,
            simulations=simulations,
            temperature_moves=temperature_moves,
            max_moves=max_moves,
            rng=rng,
        )
        all_examples.extend(examples)
        results[result] += 1
        if verbose:
            print(f"  Game {game_idx + 1}/{num_games}: {result} "
                  f"({len(examples)} positions)")

    if verbose:
        print(f"  Results: +{results['1-0']} -{results['0-1']} ={results['1/2-1/2']} "
              f"| {len(all_examples)} total positions")
    return all_examples


if __name__ == "__main__":
    """Smoke test: one tiny game with the knowledge-free uniform network."""
    from search.puct import uniform_policy_value

    print("Playing one self-play game (uniform network, 16 sims/move)...")
    examples, result = play_selfplay_game(
        uniform_policy_value, simulations=16, max_moves=60, verbose=True
    )
    print(f"\nResult: {result}, {len(examples)} training examples")
    state, policy, value = examples[0]
    assert state.shape == (20, 8, 8)
    assert policy.shape == (NUM_MOVES,)
    assert abs(policy.sum() - 1.0) < 1e-5
    assert value in (-1.0, 0.0, 1.0)
    print("✅ Self-play smoke test passed")
