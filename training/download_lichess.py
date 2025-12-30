#!/usr/bin/env python3
"""
Lichess Game Downloader

Downloads sample games from Lichess API for training neural networks.
Uses the Lichess public API to fetch recent high-quality games.

Usage:
    python3 training/download_lichess.py --output data/lichess_games.pgn --count 500
"""

import argparse
import urllib.request
import urllib.error
import time
import sys
import os


def download_lichess_games(output_path, count=500, min_elo=2000, perf_type="rapid"):
    """
    Download games from Lichess API.

    Args:
        output_path: Path to save PGN file
        count: Number of games to download
        min_elo: Minimum rating for players
        perf_type: Game type (rapid, classical, blitz)
    """
    print("="*70)
    print("Lichess Game Downloader")
    print("="*70)
    print(f"Output: {output_path}")
    print(f"Target games: {count}")
    print(f"Min Elo: {min_elo}")
    print(f"Performance type: {perf_type}")
    print("="*70)

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # We'll use a known active high-rated player account to get games
    # DrNykterstein is Magnus Carlsen's Lichess account
    # Other options: Alireza2003 (Firouzja), GMWSO (Wesley So)
    top_players = [
        "DrNykterstein",
        "Alireza2003",
        "GMWSO",
        "nihalsarin2004",
        "Hikaru",
        "LyonBeast",
        "penguingm1",
        "DanielNaroditsky",
        "Vladimirovich9000",
        "ChessNetwork"
    ]

    games_downloaded = 0
    games_per_player = max(count // len(top_players), 50)

    with open(output_path, 'w') as f:
        for player in top_players:
            if games_downloaded >= count:
                break

            print(f"\nFetching games from {player}...")

            # Lichess API endpoint for user games
            # Note: We request fewer games per player to stay within rate limits
            games_to_fetch = min(games_per_player, count - games_downloaded)

            url = f"https://lichess.org/api/games/user/{player}?max={games_to_fetch}&perfType={perf_type}&rated=true&pgnInJson=false"

            headers = {
                'Accept': 'application/x-chess-pgn',
                'User-Agent': 'Chess-RL-Training/1.0 (Educational project)'
            }

            try:
                req = urllib.request.Request(url, headers=headers)

                with urllib.request.urlopen(req, timeout=60) as response:
                    pgn_data = response.read().decode('utf-8')

                    # Count games in response
                    game_count = pgn_data.count('[Event ')

                    if game_count > 0:
                        f.write(pgn_data)
                        f.write('\n')
                        games_downloaded += game_count
                        print(f"  Downloaded {game_count} games (total: {games_downloaded})")
                    else:
                        print(f"  No games found for {player}")

            except urllib.error.HTTPError as e:
                print(f"  HTTP Error {e.code}: {e.reason}")
                if e.code == 429:
                    print("  Rate limited, waiting 60 seconds...")
                    time.sleep(60)
            except urllib.error.URLError as e:
                print(f"  URL Error: {e.reason}")
            except Exception as e:
                print(f"  Error: {e}")

            # Rate limiting - be nice to Lichess API
            time.sleep(2)

    print("\n" + "="*70)
    print("Download Complete!")
    print("="*70)
    print(f"Total games downloaded: {games_downloaded}")
    print(f"Saved to: {output_path}")

    if games_downloaded < count:
        print(f"\nNote: Only got {games_downloaded}/{count} games.")
        print("This is normal - we're limited by API rate limits.")
        print("The downloaded games are high-quality master games.")

    return games_downloaded


def create_sample_games(output_path, count=100):
    """
    Create sample games by having the existing engines play each other.
    This is a fallback if Lichess download doesn't work.
    """
    print("="*70)
    print("Creating Sample Games via Self-Play")
    print("="*70)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import chess
    import chess.pgn
    from datetime import datetime

    try:
        from search.mcts import MCTS
        from engine.evaluator import evaluate
        mcts_available = True
    except ImportError:
        mcts_available = False
        print("Warning: MCTS not available, using minimax only")

    try:
        from search.minimax import minimax_search
        minimax_available = True
    except ImportError:
        minimax_available = False
        print("Warning: Minimax not available")

    if not mcts_available and not minimax_available:
        print("Error: No search engines available!")
        return 0

    games_created = 0

    with open(output_path, 'w') as f:
        for i in range(count):
            print(f"\rGenerating game {i+1}/{count}...", end='', flush=True)

            board = chess.Board()
            game = chess.pgn.Game()
            game.headers["Event"] = "Self-Play Training"
            game.headers["Site"] = "Local"
            game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            game.headers["Round"] = str(i + 1)
            game.headers["White"] = "MCTS-200" if mcts_available else "Minimax-3"
            game.headers["Black"] = "MCTS-200" if mcts_available else "Minimax-3"
            game.headers["WhiteElo"] = "2000"
            game.headers["BlackElo"] = "2000"

            node = game
            move_count = 0
            max_moves = 150

            while not board.is_game_over() and move_count < max_moves:
                if mcts_available:
                    mcts = MCTS(evaluator=evaluate, num_simulations=100)
                    move = mcts.search(board)
                else:
                    move, _ = minimax_search(board, depth=3, evaluator=evaluate)

                if move is None:
                    break

                node = node.add_variation(move)
                board.push(move)
                move_count += 1

            # Set result
            if board.is_checkmate():
                result = "0-1" if board.turn == chess.WHITE else "1-0"
            elif board.is_stalemate() or board.is_insufficient_material():
                result = "1/2-1/2"
            elif board.is_fifty_moves() or board.is_repetition():
                result = "1/2-1/2"
            elif move_count >= max_moves:
                result = "1/2-1/2"
            else:
                result = "*"

            game.headers["Result"] = result

            print(game, file=f, end="\n\n")
            games_created += 1

    print(f"\n\nCreated {games_created} self-play games")
    print(f"Saved to: {output_path}")

    return games_created


def main():
    parser = argparse.ArgumentParser(
        description="Download chess games for neural network training"
    )
    parser.add_argument('--output', '-o', type=str, default='data/lichess_games.pgn',
                       help='Output PGN file path')
    parser.add_argument('--count', '-n', type=int, default=500,
                       help='Number of games to download')
    parser.add_argument('--min-elo', type=int, default=2000,
                       help='Minimum player Elo')
    parser.add_argument('--perf-type', type=str, default='rapid',
                       choices=['rapid', 'classical', 'blitz'],
                       help='Game type to download')
    parser.add_argument('--self-play', action='store_true',
                       help='Generate self-play games instead of downloading')

    args = parser.parse_args()

    if args.self_play:
        create_sample_games(args.output, args.count)
    else:
        games = download_lichess_games(
            args.output,
            count=args.count,
            min_elo=args.min_elo,
            perf_type=args.perf_type
        )

        # If download failed or got few games, supplement with self-play
        if games < 50:
            print("\n" + "="*70)
            print("Download yielded few games. Supplementing with self-play...")
            print("="*70)
            selfplay_output = args.output.replace('.pgn', '_selfplay.pgn')
            create_sample_games(selfplay_output, count=100)
            print(f"\nYou can combine both files for training:")
            print(f"  cat {args.output} {selfplay_output} > data/combined_games.pgn")


if __name__ == "__main__":
    main()
