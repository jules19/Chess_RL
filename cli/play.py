#!/usr/bin/env python3
"""
Chess game loop - Baby Steps #1 & #2

This script demonstrates:
1. Random vs Random play
2. Human vs Random play
3. Material-based player (Day 2)
4. Board visualization
5. Game termination detection
"""

import chess
import random
import sys
import os

# Add project root to path to import engine module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.evaluator import best_move_material


def random_move(board):
    """Select a random legal move."""
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    return random.choice(legal_moves)


def material_move(board):
    """Select best move based on material evaluation."""
    return best_move_material(board)


def display_board(board, move_num, last_move=None):
    """Display the current board state."""
    print(f"\n{'='*50}")
    print(f"Move {move_num}")
    if last_move:
        print(f"Last move: {last_move}")
    print(f"{'='*50}")
    print(board)
    print(f"\nFEN: {board.fen()}")
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    print()


def play_random_vs_random(verbose=True):
    """Play a complete game: Random vs Random."""
    board = chess.Board()
    move_count = 0

    if verbose:
        print("\n🎲 Random vs Random Chess Game")
        display_board(board, move_count)

    while not board.is_game_over():
        move = random_move(board)
        board.push(move)
        move_count += 1

        if verbose:
            display_board(board, move_count, move)

    # Game over
    result = board.result()
    outcome = board.outcome()

    print(f"\n{'='*50}")
    print(f"🏁 GAME OVER after {move_count} moves")
    print(f"{'='*50}")
    print(f"Result: {result}")
    if outcome:
        print(f"Termination: {outcome.termination.name}")
        if outcome.winner is None:
            print(f"Draw!")
        elif outcome.winner == chess.WHITE:
            print(f"Winner: White")
        else:
            print(f"Winner: Black")
    print()

    return result, move_count


def play_human_vs_random(human_color=chess.WHITE):
    """Play a game: Human vs Random."""
    board = chess.Board()
    move_count = 0

    print(f"\n🎮 Human ({'White' if human_color == chess.WHITE else 'Black'}) vs Random")
    display_board(board, move_count)

    while not board.is_game_over():
        if board.turn == human_color:
            # Human's turn
            while True:
                print(f"Legal moves: {', '.join([str(m) for m in board.legal_moves])}")
                move_input = input("Your move (e.g. 'e2e4' or 'q' to quit): ").strip()

                if move_input.lower() == 'q':
                    print("Game abandoned.")
                    return None, move_count

                try:
                    move = chess.Move.from_uci(move_input)
                    if move in board.legal_moves:
                        board.push(move)
                        move_count += 1
                        break
                    else:
                        print(f"Illegal move! Try again.")
                except ValueError:
                    print(f"Invalid format! Use format like 'e2e4'")
        else:
            # Computer's turn
            move = random_move(board)
            board.push(move)
            move_count += 1
            print(f"\nComputer plays: {move}")

        display_board(board, move_count, move)

    # Game over
    result = board.result()
    outcome = board.outcome()

    print(f"\n{'='*50}")
    print(f"🏁 GAME OVER after {move_count} moves")
    print(f"{'='*50}")
    print(f"Result: {result}")
    if outcome:
        print(f"Termination: {outcome.termination.name}")
        if outcome.winner is None:
            print(f"Draw!")
        elif outcome.winner == human_color:
            print(f"You win! 🎉")
        else:
            print(f"Computer wins!")
    print()

    return result, move_count


def play_human_vs_material(human_color=chess.WHITE):
    """Play a game: Human vs Material Engine."""
    board = chess.Board()
    move_count = 0

    print(f"\n🎮 Human ({'White' if human_color == chess.WHITE else 'Black'}) vs Material Engine 💎")
    display_board(board, move_count)

    while not board.is_game_over():
        if board.turn == human_color:
            # Human's turn
            while True:
                print(f"Legal moves: {', '.join([str(m) for m in board.legal_moves])}")
                move_input = input("Your move (e.g. 'e2e4' or 'q' to quit): ").strip()

                if move_input.lower() == 'q':
                    print("Game abandoned.")
                    return None, move_count

                try:
                    move = chess.Move.from_uci(move_input)
                    if move in board.legal_moves:
                        board.push(move)
                        move_count += 1
                        break
                    else:
                        print(f"Illegal move! Try again.")
                except ValueError:
                    print(f"Invalid format! Use format like 'e2e4'")
        else:
            # Material engine's turn
            print("\n💎 Material engine thinking...")
            move = material_move(board)
            board.push(move)
            move_count += 1
            print(f"Material engine plays: {move}")

        display_board(board, move_count, move)

    # Game over
    result = board.result()
    outcome = board.outcome()

    print(f"\n{'='*50}")
    print(f"🏁 GAME OVER after {move_count} moves")
    print(f"{'='*50}")
    print(f"Result: {result}")
    if outcome:
        print(f"Termination: {outcome.termination.name}")
        if outcome.winner is None:
            print(f"Draw!")
        elif outcome.winner == human_color:
            print(f"You win! 🎉")
        else:
            print(f"Material engine wins! 💎")
    print()

    return result, move_count


def run_test_suite():
    """Run multiple random games to test termination conditions."""
    print("\n🧪 Running test suite: 10 random vs random games")
    print("="*50)

    results = {
        "1-0": 0,  # White wins
        "0-1": 0,  # Black wins
        "1/2-1/2": 0,  # Draw
    }

    move_counts = []

    for i in range(10):
        print(f"\nGame {i+1}/10...", end=" ")
        result, move_count = play_random_vs_random(verbose=False)
        results[result] += 1
        move_counts.append(move_count)
        print(f"Result: {result}, Moves: {move_count}")

    print("\n" + "="*50)
    print("📊 Test Suite Results")
    print("="*50)
    print(f"White wins: {results['1-0']}")
    print(f"Black wins: {results['0-1']}")
    print(f"Draws: {results['1/2-1/2']}")
    print(f"Average game length: {sum(move_counts)/len(move_counts):.1f} moves")
    print(f"Min: {min(move_counts)}, Max: {max(move_counts)}")
    print()


def play_material_vs_random(verbose=True):
    """Play a complete game: Material evaluation vs Random."""
    board = chess.Board()
    move_count = 0

    if verbose:
        print("\n💎 Material Player (White) vs Random (Black)")
        display_board(board, move_count)

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = material_move(board)
        else:
            move = random_move(board)

        board.push(move)
        move_count += 1

        if verbose:
            display_board(board, move_count, move)

    # Game over
    result = board.result()
    outcome = board.outcome()

    if verbose:
        print(f"\n{'='*50}")
        print(f"🏁 GAME OVER after {move_count} moves")
        print(f"{'='*50}")
        print(f"Result: {result}")
        if outcome:
            print(f"Termination: {outcome.termination.name}")
            if outcome.winner is None:
                print(f"Draw!")
            elif outcome.winner == chess.WHITE:
                print(f"Material player wins! 💎")
            else:
                print(f"Random player wins!")
        print()

    return result, move_count


def test_material_vs_random(num_games=20):
    """Test material player against random player."""
    print(f"\n🧪 Testing Material vs Random: {num_games} games")
    print("="*50)

    results = {
        "1-0": 0,  # Material (White) wins
        "0-1": 0,  # Random (Black) wins
        "1/2-1/2": 0,  # Draw
    }

    move_counts = []

    for i in range(num_games):
        print(f"\rGame {i+1}/{num_games}...", end="", flush=True)
        result, move_count = play_material_vs_random(verbose=False)
        results[result] += 1
        move_counts.append(move_count)

    print()  # New line after progress
    print("\n" + "="*50)
    print("📊 Material vs Random Results")
    print("="*50)
    print(f"Material wins: {results['1-0']} ({results['1-0']/num_games*100:.1f}%)")
    print(f"Random wins: {results['0-1']} ({results['0-1']/num_games*100:.1f}%)")
    print(f"Draws: {results['1/2-1/2']} ({results['1/2-1/2']/num_games*100:.1f}%)")
    print(f"Average game length: {sum(move_counts)/len(move_counts):.1f} moves")
    print(f"Min: {min(move_counts)}, Max: {max(move_counts)}")

    # Validation
    win_rate = results['1-0'] / num_games
    if win_rate >= 0.8:
        print(f"\n✅ VALIDATION PASSED: Material player wins {win_rate*100:.1f}% (target: >80%)")
    else:
        print(f"\n⚠️ VALIDATION FAILED: Material player wins {win_rate*100:.1f}% (target: >80%)")

    print()


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("\nChess RL - Baby Steps Edition (Day 2)")
        print("="*50)
        print("1. Random vs Random (watch)")
        print("2. Human vs Random (play)")
        print("3. Human vs Material Engine (play) 💎")
        print("4. Test suite (10 random games)")
        print("5. Material vs Random (watch) 💎")
        print("6. Test Material player (20 games) - VALIDATION")
        print("="*50)
        choice = input("Choose mode (1-6): ").strip()

        if choice == "1":
            mode = "random"
        elif choice == "2":
            mode = "human"
        elif choice == "3":
            mode = "human-material"
        elif choice == "4":
            mode = "test"
        elif choice == "5":
            mode = "material"
        elif choice == "6":
            mode = "test-material"
        else:
            print("Invalid choice!")
            return

    if mode == "random":
        play_random_vs_random()
    elif mode == "human":
        color_choice = input("Play as White or Black? (w/b): ").strip().lower()
        human_color = chess.WHITE if color_choice == 'w' else chess.BLACK
        play_human_vs_random(human_color)
    elif mode == "human-material":
        color_choice = input("Play as White or Black? (w/b): ").strip().lower()
        human_color = chess.WHITE if color_choice == 'w' else chess.BLACK
        play_human_vs_material(human_color)
    elif mode == "test":
        run_test_suite()
    elif mode == "material":
        play_material_vs_random()
    elif mode == "test-material":
        test_material_vs_random()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python play.py [random|human|human-material|test|material|test-material]")


if __name__ == "__main__":
    main()
