"""
Tournament System - Play engines against each other to measure relative strength

Plays multiple games between different engine versions and reports win rates,
giving a clear picture of strength progression.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import random
import time
from search.minimax import best_move_minimax
from engine.evaluator import best_move_material


def random_move(board: chess.Board) -> chess.Move:
    """Random player baseline."""
    return random.choice(list(board.legal_moves))


def play_game(white_player, black_player, max_moves=200, verbose=False):
    """
    Play a single game between two players.

    Args:
        white_player: Function that takes board and returns move
        black_player: Function that takes board and returns move
        max_moves: Maximum number of moves before declaring draw
        verbose: Print game progress

    Returns:
        result: "1-0" (white wins), "0-1" (black wins), "1/2-1/2" (draw)
        moves: Number of moves played
        reason: Why the game ended
    """
    board = chess.Board()
    moves = 0

    if verbose:
        print("\nStarting new game...")
        print(f"White: {white_player.__name__ if hasattr(white_player, '__name__') else 'Unknown'}")
        print(f"Black: {black_player.__name__ if hasattr(black_player, '__name__') else 'Unknown'}")

    while not board.is_game_over() and moves < max_moves:
        if board.turn == chess.WHITE:
            move = white_player(board)
        else:
            move = black_player(board)

        if move is None or move not in board.legal_moves:
            # Invalid move - opponent wins
            result = "0-1" if board.turn == chess.WHITE else "1-0"
            reason = "illegal move"
            if verbose:
                print(f"Game over: {reason}")
            return result, moves, reason

        board.push(move)
        moves += 1

        if verbose and moves % 10 == 0:
            print(f"Move {moves}...")

    # Game over - check result
    if board.is_checkmate():
        result = "0-1" if board.turn == chess.WHITE else "1-0"
        reason = "checkmate"
    elif board.is_stalemate():
        result = "1/2-1/2"
        reason = "stalemate"
    elif board.is_insufficient_material():
        result = "1/2-1/2"
        reason = "insufficient material"
    elif board.can_claim_fifty_moves():
        result = "1/2-1/2"
        reason = "fifty-move rule"
    elif board.can_claim_threefold_repetition():
        result = "1/2-1/2"
        reason = "threefold repetition"
    elif moves >= max_moves:
        result = "1/2-1/2"
        reason = "max moves reached"
    else:
        result = "1/2-1/2"
        reason = "unknown"

    if verbose:
        print(f"Game over after {moves} moves: {result} ({reason})")

    return result, moves, reason


def run_tournament(player1, player1_name, player2, player2_name, num_games=20, verbose=False):
    """
    Run a tournament between two players.

    Args:
        player1: First player function
        player1_name: Name for reporting
        player2: Second player function
        player2_name: Name for reporting
        num_games: Number of games to play (each player gets num_games/2 as white)
        verbose: Print detailed game info

    Returns:
        Dictionary with tournament results
    """
    print(f"\n{'='*60}")
    print(f"TOURNAMENT: {player1_name} vs {player2_name}")
    print(f"{'='*60}")
    print(f"Games: {num_games} ({num_games//2} as White, {num_games//2} as Black for each)")

    results = {
        "player1_wins": 0,
        "player2_wins": 0,
        "draws": 0,
        "total_moves": 0,
        "reasons": {},
        "games": []
    }

    start_time = time.time()

    for game_num in range(num_games):
        # Alternate colors
        if game_num % 2 == 0:
            white, black = player1, player2
            white_name, black_name = player1_name, player2_name
        else:
            white, black = player2, player1
            white_name, black_name = player2_name, player1_name

        if verbose or (game_num + 1) % 5 == 0:
            print(f"\nGame {game_num + 1}/{num_games}: {white_name} (W) vs {black_name} (B)")

        result, moves, reason = play_game(white, black, verbose=verbose)

        results["total_moves"] += moves
        results["games"].append({
            "white": white_name,
            "black": black_name,
            "result": result,
            "moves": moves,
            "reason": reason
        })

        # Update reason statistics
        results["reasons"][reason] = results["reasons"].get(reason, 0) + 1

        # Update win/loss/draw
        if result == "1-0":
            if white_name == player1_name:
                results["player1_wins"] += 1
            else:
                results["player2_wins"] += 1
        elif result == "0-1":
            if black_name == player1_name:
                results["player1_wins"] += 1
            else:
                results["player2_wins"] += 1
        else:  # Draw
            results["draws"] += 1

        # Print progress
        if not verbose and (game_num + 1) % 5 == 0:
            p1_score = results["player1_wins"] + results["draws"] * 0.5
            p2_score = results["player2_wins"] + results["draws"] * 0.5
            print(f"  Progress: {player1_name} {p1_score:.1f} - {player2_name} {p2_score:.1f} ({results['draws']} draws)")

    elapsed = time.time() - start_time

    # Print summary
    print(f"\n{'='*60}")
    print("TOURNAMENT RESULTS")
    print(f"{'='*60}")

    p1_score = results["player1_wins"] + results["draws"] * 0.5
    p2_score = results["player2_wins"] + results["draws"] * 0.5
    total_score = num_games

    p1_pct = (p1_score / num_games) * 100
    p2_pct = (p2_score / num_games) * 100

    print(f"\n{player1_name}:")
    print(f"  Wins: {results['player1_wins']}")
    print(f"  Losses: {results['player2_wins']}")
    print(f"  Draws: {results['draws']}")
    print(f"  Score: {p1_score}/{num_games} ({p1_pct:.1f}%)")

    print(f"\n{player2_name}:")
    print(f"  Wins: {results['player2_wins']}")
    print(f"  Losses: {results['player1_wins']}")
    print(f"  Draws: {results['draws']}")
    print(f"  Score: {p2_score}/{num_games} ({p2_pct:.1f}%)")

    print(f"\nAverage game length: {results['total_moves'] / num_games:.1f} moves")
    print(f"Total time: {elapsed:.1f}s ({elapsed/num_games:.1f}s per game)")

    print(f"\nGame endings:")
    for reason, count in sorted(results["reasons"].items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    # Strength interpretation
    print(f"\n{'='*60}")
    print("STRENGTH ASSESSMENT")
    print(f"{'='*60}")

    if p1_pct >= 90:
        print(f"🏆 {player1_name} dominates {player2_name} ({p1_pct:.1f}% score)")
    elif p1_pct >= 75:
        print(f"✅ {player1_name} is significantly stronger ({p1_pct:.1f}% score)")
    elif p1_pct >= 60:
        print(f"🔶 {player1_name} is moderately stronger ({p1_pct:.1f}% score)")
    elif p1_pct >= 40:
        print(f"⚖️  Players are roughly equal ({p1_pct:.1f}% vs {p2_pct:.1f}%)")
    elif p1_pct >= 25:
        print(f"🔶 {player2_name} is moderately stronger ({p2_pct:.1f}% score)")
    elif p1_pct >= 10:
        print(f"✅ {player2_name} is significantly stronger ({p2_pct:.1f}% score)")
    else:
        print(f"🏆 {player2_name} dominates {player1_name} ({p2_pct:.1f}% score)")

    return results


def minimax_depth_2(board):
    """Minimax with depth 2."""
    return best_move_minimax(board, depth=2, verbose=False)


def minimax_depth_3(board):
    """Minimax with depth 3 (current engine)."""
    return best_move_minimax(board, depth=3, verbose=False)


def minimax_depth_4(board):
    """Minimax with depth 4 (stronger)."""
    return best_move_minimax(board, depth=4, verbose=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run chess engine tournament")
    parser.add_argument("--games", type=int, default=20, help="Number of games to play")
    parser.add_argument("--quick", action="store_true", help="Quick test (10 games)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    num_games = 10 if args.quick else args.games

    print("="*60)
    print("CHESS ENGINE TOURNAMENT")
    print("="*60)

    # Tournament 1: Current engine (depth 3) vs Random
    run_tournament(
        minimax_depth_3, "Minimax Depth 3 (Current)",
        random_move, "Random Player",
        num_games=num_games,
        verbose=args.verbose
    )

    # Tournament 2: Current engine (depth 3) vs Material-only
    run_tournament(
        minimax_depth_3, "Minimax Depth 3 (Current)",
        best_move_material, "Material-Only",
        num_games=num_games,
        verbose=args.verbose
    )

    # Tournament 3: Current engine (depth 3) vs Depth 2
    run_tournament(
        minimax_depth_3, "Minimax Depth 3 (Current)",
        minimax_depth_2, "Minimax Depth 2",
        num_games=num_games,
        verbose=args.verbose
    )

    # Tournament 4: Depth 3 vs Depth 4 (to see how much stronger depth 4 is)
    print(f"\n⚠️  Warning: Depth 4 is slow! This may take a while...")
    run_tournament(
        minimax_depth_3, "Minimax Depth 3 (Current)",
        minimax_depth_4, "Minimax Depth 4",
        num_games=num_games//2 if num_games > 10 else 6,  # Fewer games for slow depth-4
        verbose=args.verbose
    )

    print(f"\n{'='*60}")
    print("ALL TOURNAMENTS COMPLETE!")
    print(f"{'='*60}")
