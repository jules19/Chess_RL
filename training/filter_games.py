"""
PGN Game Filtering Utility

Filter chess games from large PGN databases to create high-quality training datasets.

Filters:
- Minimum player Elo rating
- Time control (avoid bullet/blitz, prefer rapid/classical)
- Game length (skip very short games)
- Result (skip unfinished games)

This creates a curated dataset of quality games for supervised learning.

Usage:
    python3 training/filter_games.py \
        --input lichess_games.pgn \
        --output data/filtered_games.pgn \
        --min-elo 2000 \
        --min-time 600 \
        --max-moves 80
"""

import chess.pgn
import argparse
from tqdm import tqdm


def parse_time_control(tc_string):
    """
    Parse Lichess time control string.

    Examples:
        "600+0" -> 600 seconds base time
        "180+2" -> 180 seconds + 2 second increment
        "-" -> Unknown/unlimited

    Returns:
        Base time in seconds, or None if invalid
    """
    if not tc_string or tc_string == "-":
        return None

    try:
        if "+" in tc_string:
            base, increment = tc_string.split("+")
            return int(base)
        else:
            return int(tc_string)
    except:
        return None


def filter_games(input_path, output_path, min_elo=2000, min_time=600, max_moves=150):
    """
    Filter PGN file to keep only high-quality games.

    Args:
        input_path: Path to input PGN file
        output_path: Path to output filtered PGN file
        min_elo: Minimum average Elo (both players)
        min_time: Minimum time control in seconds
        max_moves: Maximum number of moves (skip very long games)
    """
    print("="*70)
    print("PGN Game Filtering")
    print("="*70)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"\nFilters:")
    print(f"  Min Elo: {min_elo}")
    print(f"  Min Time: {min_time}s ({min_time//60}min)")
    print(f"  Max Moves: {max_moves}")
    print("="*70)

    games_processed = 0
    games_kept = 0
    games_rejected = {
        'elo': 0,
        'time': 0,
        'result': 0,
        'moves': 0,
        'other': 0
    }

    with open(input_path) as input_file, open(output_path, 'w') as output_file:
        while True:
            game = chess.pgn.read_game(input_file)
            if game is None:
                break

            games_processed += 1

            # Progress update
            if games_processed % 1000 == 0:
                print(f"Processed {games_processed:,} games, kept {games_kept:,} ({games_kept/games_processed:.1%})")

            # Filter 1: Check result (skip unfinished games)
            result = game.headers.get("Result", "*")
            if result == "*":
                games_rejected['result'] += 1
                continue

            # Filter 2: Check Elo ratings
            try:
                white_elo = int(game.headers.get("WhiteElo", "0"))
                black_elo = int(game.headers.get("BlackElo", "0"))

                # Require both players above minimum
                if white_elo < min_elo or black_elo < min_elo:
                    games_rejected['elo'] += 1
                    continue

            except ValueError:
                games_rejected['elo'] += 1
                continue

            # Filter 3: Check time control
            time_control = game.headers.get("TimeControl", "-")
            base_time = parse_time_control(time_control)

            if base_time is None or base_time < min_time:
                games_rejected['time'] += 1
                continue

            # Filter 4: Check game length
            move_count = sum(1 for _ in game.mainline_moves())

            if move_count > max_moves:
                games_rejected['moves'] += 1
                continue

            # Game passed all filters - write to output
            print(game, file=output_file, end="\n\n")
            games_kept += 1

    # Print summary
    print("\n" + "="*70)
    print("Filtering Complete")
    print("="*70)
    print(f"Total games processed: {games_processed:,}")
    print(f"Games kept: {games_kept:,} ({games_kept/games_processed:.1%})")
    print(f"\nRejection reasons:")
    print(f"  Elo too low: {games_rejected['elo']:,}")
    print(f"  Time control: {games_rejected['time']:,}")
    print(f"  No result: {games_rejected['result']:,}")
    print(f"  Too many moves: {games_rejected['moves']:,}")
    print(f"  Other: {games_rejected['other']:,}")
    print("="*70)
    print(f"\n✅ Filtered games saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter PGN games for quality training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input PGN file path')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output PGN file path')
    parser.add_argument('--min-elo', type=int, default=2000,
                        help='Minimum Elo for both players')
    parser.add_argument('--min-time', type=int, default=600,
                        help='Minimum time control in seconds')
    parser.add_argument('--max-moves', type=int, default=150,
                        help='Maximum number of moves')

    args = parser.parse_args()

    filter_games(
        args.input,
        args.output,
        min_elo=args.min_elo,
        min_time=args.min_time,
        max_moves=args.max_moves
    )


if __name__ == "__main__":
    main()
