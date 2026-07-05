"""
Arena - the promotion gate of the self-play loop

After training, we have a CANDIDATE network. Is it actually better than the
current CHAMPION, or did training just move the weights around? The arena
answers with games, not loss curves: the candidate must beat the champion in
a head-to-head match to take over self-play generation.

Why gate at all? Because a bad training iteration (wrong hyperparameters,
unlucky data) can make the network WORSE, and if the worse network starts
generating the training data, the whole loop can spiral downward. The gate
makes the system monotonic-ish: the data generator only ever changes to
something that demonstrated superiority.

Why a threshold above 50%? Match results are noisy — over a handful of
games a genuinely equal (or slightly worse) network wins half the time.
AlphaZero required 55% over 400 games. With few games, be conservative:
promoting a weaker net is costlier than keeping the champion one more
iteration.

Design notes:
- Colors alternate between games (White has a first-move edge).
- Matches use greedy move selection (temperature 0) and NO root noise —
  we're measuring strength, not generating diverse training data.
"""

import random
from typing import Callable, Optional

import chess

from search.puct import puct_search, select_move


def play_arena_game(white_fn: Callable, black_fn: Callable,
                    simulations: int = 50,
                    max_moves: int = 200) -> str:
    """
    Play one deterministic-ish game between two policy-value functions.

    Returns:
        Result string from White's perspective: "1-0", "0-1", or "1/2-1/2"
    """
    board = chess.Board()
    while not board.is_game_over(claim_draw=True) and board.ply() < max_moves:
        policy_value_fn = white_fn if board.turn == chess.WHITE else black_fn
        _, visit_counts = puct_search(
            board, policy_value_fn,
            simulations=simulations,
            add_root_noise=False,  # measuring strength, not exploring
        )
        board.push(select_move(visit_counts, temperature=0.0))

    result = board.result(claim_draw=True)
    return result if result != "*" else "1/2-1/2"


def play_match(candidate_fn: Callable, champion_fn: Callable,
               num_games: int = 10,
               simulations: int = 50,
               max_moves: int = 200,
               verbose: bool = True) -> float:
    """
    Play a candidate-vs-champion match with alternating colors.

    Args:
        candidate_fn: Policy-value function of the newly trained network
        champion_fn: Policy-value function of the current best network
        num_games: Total games (half as White, half as Black for each side)
        simulations: PUCT simulations per move
        max_moves: Per-game ply cap (cap = draw)
        verbose: Print per-game results

    Returns:
        Candidate's score fraction in [0, 1]: win=1, draw=0.5, loss=0.
        Promote if this exceeds your threshold (e.g. 0.55).
    """
    candidate_score = 0.0

    for game_idx in range(num_games):
        candidate_is_white = (game_idx % 2 == 0)
        if candidate_is_white:
            result = play_arena_game(candidate_fn, champion_fn,
                                     simulations=simulations, max_moves=max_moves)
            game_score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]
        else:
            result = play_arena_game(champion_fn, candidate_fn,
                                     simulations=simulations, max_moves=max_moves)
            game_score = {"1-0": 0.0, "0-1": 1.0, "1/2-1/2": 0.5}[result]

        candidate_score += game_score
        if verbose:
            color = "White" if candidate_is_white else "Black"
            print(f"  Arena game {game_idx + 1}/{num_games}: {result} "
                  f"(candidate as {color}, score {candidate_score}/{game_idx + 1})")

    return candidate_score / num_games
