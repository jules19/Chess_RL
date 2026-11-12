#!/usr/bin/env python3
"""
Chess game loop - Baby Steps #1, #2, and Days 3-4

This script demonstrates:
1. Random vs Random play
2. Human vs Random play
3. Material-based player (Day 2)
4. Minimax search with alpha-beta pruning (Days 3-4)
5. Board visualization
6. Game termination detection
"""

import chess
import random
import sys
import os
import time

# Add project root to path to import engine module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.evaluator import best_move_material
from search.minimax import best_move_minimax
from search.mcts import best_move_mcts
from cli.board_display import display_board_fancy, track_captured_pieces, cycle_color_scheme, get_color_scheme, COLOR_SCHEMES

# Board display configuration (mutable global)
# Set CHESS_BOARD_SIZE environment variable to 'large' for bigger display with borders
# or 'compact' (default) for the smaller 8-line display
# Can also be toggled dynamically during gameplay
_board_size = os.environ.get('CHESS_BOARD_SIZE', 'compact').lower()

def get_board_size():
    """Get current board display size."""
    return _board_size

def toggle_board_size():
    """Toggle between compact and large display modes."""
    global _board_size
    _board_size = 'large' if _board_size == 'compact' else 'compact'
    print(f"\n🎨 Display mode switched to: {_board_size.upper()}")
    return _board_size

# For backwards compatibility
BOARD_SIZE = get_board_size()


def parse_human_move(board, text):
    """Accept standard chess notation (e4, Nf3, O-O) and UCI (e2e4).

    Neither parser changes the board. Null moves and illegal moves are refused.
    """
    try:
        move = board.parse_uci(text.strip())
    except ValueError:
        move = board.parse_san(text.strip())
    if move not in board.legal_moves:
        raise ValueError("That move is not legal in this position")
    return move


def choose_mode():
    """A short learner-facing menu; advanced legacy modes remain available."""
    while True:
        print("\nChess RL — play, predict, discover")
        print("1. Play chess")
        print("2. Watch an engine think (the change-of-mind lab; needs torch)")
        print("3. Watch it learn (tic-tac-toe; no torch needed)")
        print("4. Continue the course")
        print("a. Advanced engines and display settings")
        print("q. Quit")
        choice = input("Choose 1–4, a or q: ").strip().lower()
        if choice == 'q':
            return None
        if choice == '1':
            print("1. Random   2. Material   3. Minimax   4. MCTS   b. Back")
            opponent = input("Opponent [1]: ").strip().lower() or '1'
            modes = {'1': 'human', '2': 'human-material',
                     '3': 'human-minimax', '4': 'human-mcts'}
            if opponent in modes:
                return modes[opponent]
            if opponent != 'b':
                print("Choose an opponent from 1–4, or b to go back.")
        elif choice in ('2', '3', '4'):
            return {'2': 'think', '3': 'learn', '4': 'course'}[choice]
        elif choice == 'a':
            print("Watch: random, material, minimax, minimax-material, mcts")
            print("Checks: test, test-material, test-minimax, test-minimax-material,")
            print("        test-mcts, test-mcts-minimax")
            print("Display: d = size, c = colours; b = back")
            mode = input("Mode: ").strip().lower()
            if mode == 'd':
                toggle_board_size()
            elif mode == 'c':
                cycle_color_scheme()
            elif mode in {'random', 'material', 'minimax', 'minimax-material', 'mcts',
                           'test', 'test-material', 'test-minimax', 'test-minimax-material',
                           'test-mcts', 'test-mcts-minimax'}:
                return mode
            elif mode != 'b':
                print("Unknown mode. Choose one of the names shown above.")
        else:
            print("Choose 1–4, a or q.")


def random_move(board):
    """Select a random legal move."""
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    return random.choice(legal_moves)


def material_move(board):
    """Select best move based on material evaluation."""
    return best_move_material(board)


def minimax_move(board, depth=3):
    """Select best move using minimax search."""
    return best_move_minimax(board, depth=depth, verbose=False)


def mcts_move(board, simulations=200, use_evaluator=True):
    """Select best move using MCTS."""
    return best_move_mcts(board, simulations=simulations, use_evaluator=use_evaluator, verbose=False)


def display_board(board, move_num, last_move=None):
    """Display the current board state with enhanced visualization."""
    print(f"\n{'='*50}")
    print(f"Move {move_num}")
    if last_move:
        print(f"Last move: {last_move}")
    print(f"{'='*50}")

    # Use fancy display with Unicode pieces and colors
    # Size can be toggled dynamically with toggle_board_size()
    captured = track_captured_pieces(board)
    print(display_board_fancy(board, last_move=last_move, captured_pieces=captured, size=get_board_size()))

    print(f"\nFEN: {board.fen()}")
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    print()


def pause_for_move(mode='step'):
    """
    Pause between moves in watch mode.

    Args:
        mode: 'step' (press Enter), 'auto' (auto-advance with delay), 'skip' (no pause)

    Returns:
        New mode if user changes it, or 'quit' to stop watching
    """
    if mode == 'skip':
        return 'skip'
    elif mode == 'auto':
        time.sleep(1.5)  # 1.5 second delay in auto mode
        return 'auto'
    else:  # step mode
        print("─" * 50)
        user_input = input("⏸️  [Enter]=next, [a]=auto, [s]=skip, [d]=display, [c]=colors, [q]=quit: ").strip().lower()
        if user_input == 'q':
            return 'quit'
        elif user_input == 'a':
            print("▶️  Auto-play mode activated (1.5s between moves)")
            return 'auto'
        elif user_input == 's':
            print("⏩ Skipping to end...")
            return 'skip'
        elif user_input == 'd':
            toggle_board_size()
            return 'step'  # Stay in step mode, just toggled display
        elif user_input == 'c':
            cycle_color_scheme()
            return 'step'  # Stay in step mode, just cycled colors
        else:
            return 'step'


def play_random_vs_random(verbose=True, interactive=False):
    """Play a complete game: Random vs Random."""
    board = chess.Board()
    move_count = 0
    watch_mode = 'step' if interactive else 'skip'

    if verbose:
        print("\n🎲 Random vs Random Chess Game")
        if interactive:
            print("Watch mode: [Enter]=next, [a]=auto, [s]=skip, [d]=display, [c]=colors, [q]=quit")
        display_board(board, move_count)

    while not board.is_game_over():
        move = random_move(board)
        board.push(move)
        move_count += 1

        if verbose:
            display_board(board, move_count, move)
            if interactive:
                watch_mode = pause_for_move(watch_mode)
                if watch_mode == 'quit':
                    print("\n⏹️  Stopped watching.")
                    return None, move_count

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
                print(f"Legal moves: {', '.join(board.san(m) for m in board.legal_moves)}")
                move_input = input("Your move (e.g. 'e4' or 'e2e4', 'd'=display, 'c'=colors, 'q'=quit): ").strip()

                if move_input.lower() == 'q':
                    print("Game abandoned.")
                    return None, move_count
                elif move_input.lower() == 'd':
                    toggle_board_size()
                    display_board(board, move_count)
                    continue
                elif move_input.lower() == 'c':
                    cycle_color_scheme()
                    display_board(board, move_count)
                    continue

                try:
                    move = parse_human_move(board, move_input)
                    if move in board.legal_moves:
                        board.push(move)
                        move_count += 1
                        break
                    else:
                        print(f"Illegal move! Try again.")
                except ValueError:
                    print("Move not recognised or not legal. Try e4, Nf3, O-O or e2e4; q quits.")
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
                print(f"Legal moves: {', '.join(board.san(m) for m in board.legal_moves)}")
                move_input = input("Your move (e.g. 'e4' or 'e2e4', 'd'=display, 'c'=colors, 'q'=quit): ").strip()

                if move_input.lower() == 'q':
                    print("Game abandoned.")
                    return None, move_count
                elif move_input.lower() == 'd':
                    toggle_board_size()
                    display_board(board, move_count)
                    continue
                elif move_input.lower() == 'c':
                    cycle_color_scheme()
                    display_board(board, move_count)
                    continue

                try:
                    move = parse_human_move(board, move_input)
                    if move in board.legal_moves:
                        board.push(move)
                        move_count += 1
                        break
                    else:
                        print(f"Illegal move! Try again.")
                except ValueError:
                    print("Move not recognised or not legal. Try e4, Nf3, O-O or e2e4; q quits.")
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


def play_material_vs_random(verbose=True, interactive=False):
    """Play a complete game: Material evaluation vs Random."""
    board = chess.Board()
    move_count = 0
    watch_mode = 'step' if interactive else 'skip'

    if verbose:
        print("\n💎 Material Player (White) vs Random (Black)")
        if interactive:
            print("Watch mode: [Enter]=next, [a]=auto, [s]=skip, [d]=display, [c]=colors, [q]=quit")
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
            if interactive:
                watch_mode = pause_for_move(watch_mode)
                if watch_mode == 'quit':
                    print("\n⏹️  Stopped watching.")
                    return None, move_count

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


def play_human_vs_minimax(human_color=chess.WHITE, depth=3):
    """Play a game: Human vs Minimax Engine."""
    board = chess.Board()
    move_count = 0

    print(f"\n🎮 Human ({'White' if human_color == chess.WHITE else 'Black'}) vs Minimax Engine (depth={depth}) 🧠")
    display_board(board, move_count)

    while not board.is_game_over():
        if board.turn == human_color:
            # Human's turn
            while True:
                print(f"Legal moves: {', '.join(board.san(m) for m in board.legal_moves)}")
                move_input = input("Your move (e.g. 'e4' or 'e2e4', 'd'=display, 'c'=colors, 'q'=quit): ").strip()

                if move_input.lower() == 'q':
                    print("Game abandoned.")
                    return None, move_count
                elif move_input.lower() == 'd':
                    toggle_board_size()
                    display_board(board, move_count)
                    continue
                elif move_input.lower() == 'c':
                    cycle_color_scheme()
                    display_board(board, move_count)
                    continue

                try:
                    move = parse_human_move(board, move_input)
                    if move in board.legal_moves:
                        board.push(move)
                        move_count += 1
                        break
                    else:
                        print(f"Illegal move! Try again.")
                except ValueError:
                    print("Move not recognised or not legal. Try e4, Nf3, O-O or e2e4; q quits.")
        else:
            # Minimax engine's turn
            print(f"\n🧠 Minimax engine (depth {depth}) thinking...")
            start_time = time.time()
            move = minimax_move(board, depth=depth)
            elapsed = time.time() - start_time
            board.push(move)
            move_count += 1
            print(f"Minimax plays: {move} (took {elapsed:.2f}s)")

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
            print(f"Minimax engine wins! 🧠")
    print()

    return result, move_count


def play_minimax_vs_random(depth=3, verbose=True, interactive=False):
    """Play a complete game: Minimax vs Random."""
    board = chess.Board()
    move_count = 0
    watch_mode = 'step' if interactive else 'skip'

    if verbose:
        print(f"\n🧠 Minimax (depth {depth}, White) vs Random (Black)")
        if interactive:
            print("Watch mode: [Enter]=next, [a]=auto, [s]=skip, [d]=display, [c]=colors, [q]=quit")
        display_board(board, move_count)

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            if verbose and interactive:
                print(f"🧠 Minimax thinking...")
            move = minimax_move(board, depth=depth)
        else:
            move = random_move(board)

        board.push(move)
        move_count += 1

        if verbose:
            display_board(board, move_count, move)
            if interactive:
                watch_mode = pause_for_move(watch_mode)
                if watch_mode == 'quit':
                    print("\n⏹️  Stopped watching.")
                    return None, move_count

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
                print(f"Minimax wins! 🧠")
            else:
                print(f"Random wins!")
        print()

    return result, move_count


def test_minimax_vs_random(depth=3, num_games=20):
    """Test minimax against random player."""
    print(f"\n🧪 Testing Minimax (depth {depth}) vs Random: {num_games} games")
    print("="*50)

    results = {
        "1-0": 0,  # Minimax (White) wins
        "0-1": 0,  # Random (Black) wins
        "1/2-1/2": 0,  # Draw
    }

    move_counts = []
    total_time = 0

    for i in range(num_games):
        print(f"\rGame {i+1}/{num_games}...", end="", flush=True)
        start = time.time()
        result, move_count = play_minimax_vs_random(depth=depth, verbose=False)
        total_time += time.time() - start
        results[result] += 1
        move_counts.append(move_count)

    print()  # New line after progress
    print("\n" + "="*50)
    print(f"📊 Minimax (depth {depth}) vs Random Results")
    print("="*50)
    print(f"Minimax wins: {results['1-0']} ({results['1-0']/num_games*100:.1f}%)")
    print(f"Random wins: {results['0-1']} ({results['0-1']/num_games*100:.1f}%)")
    print(f"Draws: {results['1/2-1/2']} ({results['1/2-1/2']/num_games*100:.1f}%)")
    print(f"Average game length: {sum(move_counts)/len(move_counts):.1f} moves")
    print(f"Min: {min(move_counts)}, Max: {max(move_counts)}")
    print(f"Total time: {total_time:.1f}s ({total_time/num_games:.1f}s per game)")

    # Validation
    win_rate = results['1-0'] / num_games
    if win_rate >= 0.9:
        print(f"\n✅ VALIDATION PASSED: Minimax wins {win_rate*100:.1f}% (target: >90%)")
    else:
        print(f"\n⚠️ VALIDATION FAILED: Minimax wins {win_rate*100:.1f}% (target: >90%)")

    print()


def play_minimax_vs_material(minimax_depth=3, verbose=True, interactive=False):
    """Play a complete game: Minimax vs Material."""
    board = chess.Board()
    move_count = 0
    watch_mode = 'step' if interactive else 'skip'

    if verbose:
        print(f"\n🧠 Minimax (depth {minimax_depth}, White) vs 💎 Material (Black)")
        if interactive:
            print("Watch mode: [Enter]=next, [a]=auto, [s]=skip, [d]=display, [c]=colors, [q]=quit")
        display_board(board, move_count)

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            if verbose and interactive:
                print(f"🧠 Minimax thinking...")
            move = minimax_move(board, depth=minimax_depth)
        else:
            move = material_move(board)

        board.push(move)
        move_count += 1

        if verbose:
            display_board(board, move_count, move)
            if interactive:
                watch_mode = pause_for_move(watch_mode)
                if watch_mode == 'quit':
                    print("\n⏹️  Stopped watching.")
                    return None, move_count

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
                print(f"Minimax wins! 🧠")
            else:
                print(f"Material wins! 💎")
        print()

    return result, move_count


def test_minimax_vs_material(minimax_depth=3, num_games=20):
    """Test minimax against material player."""
    print(f"\n🧪 Testing Minimax (depth {minimax_depth}) vs Material: {num_games} games")
    print("="*50)

    results = {
        "1-0": 0,  # Minimax (White) wins
        "0-1": 0,  # Material (Black) wins
        "1/2-1/2": 0,  # Draw
    }

    move_counts = []

    for i in range(num_games):
        print(f"\rGame {i+1}/{num_games}...", end="", flush=True)
        result, move_count = play_minimax_vs_material(minimax_depth=minimax_depth, verbose=False)
        results[result] += 1
        move_counts.append(move_count)

    print()  # New line after progress
    print("\n" + "="*50)
    print(f"📊 Minimax (depth {minimax_depth}) vs Material Results")
    print("="*50)
    print(f"Minimax wins: {results['1-0']} ({results['1-0']/num_games*100:.1f}%)")
    print(f"Material wins: {results['0-1']} ({results['0-1']/num_games*100:.1f}%)")
    print(f"Draws: {results['1/2-1/2']} ({results['1/2-1/2']/num_games*100:.1f}%)")
    print(f"Average game length: {sum(move_counts)/len(move_counts):.1f} moves")
    print(f"Min: {min(move_counts)}, Max: {max(move_counts)}")

    # Validation
    win_rate = results['1-0'] / num_games
    if win_rate >= 0.7:
        print(f"\n✅ VALIDATION PASSED: Minimax wins {win_rate*100:.1f}% (target: >70%)")
    else:
        print(f"\n⚠️ VALIDATION FAILED: Minimax wins {win_rate*100:.1f}% (target: >70%)")

    print()


def play_mcts_vs_random(simulations=200, verbose=True, interactive=False):
    """Play a complete game: MCTS vs Random."""
    board = chess.Board()
    move_count = 0
    watch_mode = 'step' if interactive else 'skip'

    if verbose:
        print(f"\n🌳 MCTS ({simulations} sims, White) vs Random (Black)")
        if interactive:
            print("Watch mode: [Enter]=next, [a]=auto, [s]=skip, [d]=display, [c]=colors, [q]=quit")
        display_board(board, move_count)

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            if verbose and interactive:
                print(f"🌳 MCTS thinking ({simulations} simulations)...")
            move = mcts_move(board, simulations=simulations)
        else:
            move = random_move(board)

        board.push(move)
        move_count += 1

        if verbose:
            display_board(board, move_count, move)
            if interactive:
                watch_mode = pause_for_move(watch_mode)
                if watch_mode == 'quit':
                    print("\n⏹️  Stopped watching.")
                    return None, move_count

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
                print(f"MCTS wins! 🌳")
            else:
                print(f"Random wins!")
        print()

    return result, move_count


def play_mcts_vs_minimax(mcts_simulations=200, minimax_depth=3, verbose=True, interactive=False):
    """Play a complete game: MCTS vs Minimax."""
    board = chess.Board()
    move_count = 0
    watch_mode = 'step' if interactive else 'skip'

    if verbose:
        print(f"\n🌳 MCTS ({mcts_simulations} sims, White) vs 🧠 Minimax (depth {minimax_depth}, Black)")
        if interactive:
            print("Watch mode: [Enter]=next, [a]=auto, [s]=skip, [d]=display, [c]=colors, [q]=quit")
        display_board(board, move_count)

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            if verbose and interactive:
                print(f"🌳 MCTS thinking ({mcts_simulations} simulations)...")
            move = mcts_move(board, simulations=mcts_simulations)
        else:
            if verbose and interactive:
                print(f"🧠 Minimax thinking (depth {minimax_depth})...")
            move = minimax_move(board, depth=minimax_depth)

        board.push(move)
        move_count += 1

        if verbose:
            display_board(board, move_count, move)
            if interactive:
                watch_mode = pause_for_move(watch_mode)
                if watch_mode == 'quit':
                    print("\n⏹️  Stopped watching.")
                    return None, move_count

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
                print(f"MCTS wins! 🌳")
            else:
                print(f"Minimax wins! 🧠")
        print()

    return result, move_count


def play_human_vs_mcts(human_color=chess.WHITE, simulations=200):
    """Play a game: Human vs MCTS Engine."""
    board = chess.Board()
    move_count = 0

    print(f"\n🎮 Human ({'White' if human_color == chess.WHITE else 'Black'}) vs MCTS ({simulations} sims) 🌳")
    display_board(board, move_count)

    while not board.is_game_over():
        if board.turn == human_color:
            # Human's turn
            while True:
                print(f"Legal moves: {', '.join(board.san(m) for m in board.legal_moves)}")
                move_input = input("Your move (e.g. 'e4' or 'e2e4', 'd'=display, 'c'=colors, 'q'=quit): ").strip()

                if move_input.lower() == 'q':
                    print("Game abandoned.")
                    return None, move_count
                elif move_input.lower() == 'd':
                    toggle_board_size()
                    display_board(board, move_count)
                    continue
                elif move_input.lower() == 'c':
                    cycle_color_scheme()
                    display_board(board, move_count)
                    continue

                try:
                    move = parse_human_move(board, move_input)
                    if move in board.legal_moves:
                        board.push(move)
                        move_count += 1
                        break
                    else:
                        print(f"Illegal move! Try again.")
                except ValueError:
                    print("Move not recognised or not legal. Try e4, Nf3, O-O or e2e4; q quits.")
        else:
            # MCTS engine's turn
            print(f"\n🌳 MCTS thinking ({simulations} simulations)...")
            move = mcts_move(board, simulations=simulations)
            board.push(move)
            move_count += 1
            print(f"MCTS plays: {move}")

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
            print(f"MCTS wins! 🌳")
    print()

    return result, move_count


def test_mcts_vs_random(simulations=200, num_games=20):
    """Test MCTS against random player."""
    print(f"\n🧪 Testing MCTS ({simulations} sims) vs Random: {num_games} games")
    print("="*50)

    results = {
        "1-0": 0,  # MCTS (White) wins
        "0-1": 0,  # Random (Black) wins
        "1/2-1/2": 0,  # Draw
    }

    move_counts = []

    for i in range(num_games):
        print(f"\rGame {i+1}/{num_games}...", end="", flush=True)
        result, move_count = play_mcts_vs_random(simulations=simulations, verbose=False)
        results[result] += 1
        move_counts.append(move_count)

    print()  # New line after progress
    print("\n" + "="*50)
    print(f"📊 MCTS ({simulations} sims) vs Random Results")
    print("="*50)
    print(f"MCTS wins: {results['1-0']} ({results['1-0']/num_games*100:.1f}%)")
    print(f"Random wins: {results['0-1']} ({results['0-1']/num_games*100:.1f}%)")
    print(f"Draws: {results['1/2-1/2']} ({results['1/2-1/2']/num_games*100:.1f}%)")
    print(f"Average game length: {sum(move_counts)/len(move_counts):.1f} moves")
    print(f"Min: {min(move_counts)}, Max: {max(move_counts)}")

    # Validation
    win_rate = results['1-0'] / num_games
    if win_rate >= 0.8:
        print(f"\n✅ VALIDATION PASSED: MCTS wins {win_rate*100:.1f}% (target: >80%)")
    else:
        print(f"\n⚠️ VALIDATION WARNING: MCTS wins {win_rate*100:.1f}% (target: >80%)")

    print()


def test_mcts_vs_minimax(mcts_simulations=200, minimax_depth=3, num_games=20):
    """Test MCTS against minimax player - KEY VALIDATION for Phase 2."""
    print(f"\n🧪 Testing MCTS ({mcts_simulations} sims) vs Minimax (depth {minimax_depth}): {num_games} games")
    print("="*50)

    results = {
        "1-0": 0,  # MCTS (White) wins
        "0-1": 0,  # Minimax (Black) wins
        "1/2-1/2": 0,  # Draw
    }

    move_counts = []

    for i in range(num_games):
        print(f"\rGame {i+1}/{num_games}...", end="", flush=True)
        result, move_count = play_mcts_vs_minimax(mcts_simulations=mcts_simulations,
                                                   minimax_depth=minimax_depth,
                                                   verbose=False)
        results[result] += 1
        move_counts.append(move_count)

    print()  # New line after progress
    print("\n" + "="*50)
    print(f"📊 MCTS ({mcts_simulations} sims) vs Minimax (depth {minimax_depth}) Results")
    print("="*50)
    print(f"MCTS wins: {results['1-0']} ({results['1-0']/num_games*100:.1f}%)")
    print(f"Minimax wins: {results['0-1']} ({results['0-1']/num_games*100:.1f}%)")
    print(f"Draws: {results['1/2-1/2']} ({results['1/2-1/2']/num_games*100:.1f}%)")
    print(f"Average game length: {sum(move_counts)/len(move_counts):.1f} moves")
    print(f"Min: {min(move_counts)}, Max: {max(move_counts)}")

    # Phase 2 Validation: MCTS should beat or match minimax
    win_rate = results['1-0'] / num_games
    if win_rate >= 0.55:
        print(f"\n✅ PHASE 2 VALIDATION PASSED: MCTS wins {win_rate*100:.1f}% (target: >55%)")
        print("MCTS is competitive with minimax - ready to proceed to Phase 3!")
    else:
        print(f"\n⚠️ PHASE 2 VALIDATION FAILED: MCTS wins {win_rate*100:.1f}% (target: >55%)")
        print("Consider: more simulations, tuning exploration constant, or evaluator improvements")

    print()


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = choose_mode()

    if mode is None or mode == 'q':
        return
    if mode == 'course':
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print(f"Open {root}/course/README.md")
        print("New here? Follow the opening session in course/module-00-orientation/README.md.")
        print("Returning? Open your JOURNAL.md and continue from its last checkpoint.")
        return
    if mode == 'think':
        try:
            from selfplay.mind_change_lab import run_lab
            import torch
        except ModuleNotFoundError as error:
            if error.name != 'torch':
                raise
            print('This lab needs PyTorch. Install with: pip install -e ".[train,dev]"')
            print('Or choose "Watch it learn" for the no-torch tic-tac-toe lab.')
            return
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            run_lab(pause=True)
        finally:
            torch.set_num_threads(previous_threads)
        return
    if mode == 'learn':
        from selfplay.tictactoe_lab import run
        print("Predict: can a learner improve by playing only against itself?")
        input("Press Enter to run the experiment. ")
        run(seed=0)
        print("Explain: did it learn every position, or just those its play reached?")
        print("Next: course/module-06-selfplay-rl/README.md, Part A.")
        return

    if mode == "random":
        play_random_vs_random(interactive=True)
    elif mode == "human":
        color_choice = input("Play as White or Black? (w/b): ").strip().lower()
        human_color = chess.WHITE if color_choice == 'w' else chess.BLACK
        play_human_vs_random(human_color)
    elif mode == "human-material":
        color_choice = input("Play as White or Black? (w/b): ").strip().lower()
        human_color = chess.WHITE if color_choice == 'w' else chess.BLACK
        play_human_vs_material(human_color)
    elif mode == "human-minimax":
        color_choice = input("Play as White or Black? (w/b): ").strip().lower()
        human_color = chess.WHITE if color_choice == 'w' else chess.BLACK
        depth_choice = input("Minimax depth (2-4, default 3): ").strip()
        depth = int(depth_choice) if depth_choice.isdigit() else 3
        play_human_vs_minimax(human_color, depth=depth)
    elif mode == "test":
        run_test_suite()
    elif mode == "material":
        play_material_vs_random(interactive=True)
    elif mode == "minimax":
        play_minimax_vs_random(interactive=True)
    elif mode == "minimax-material":
        play_minimax_vs_material(interactive=True)
    elif mode == "test-material":
        test_material_vs_random()
    elif mode == "test-minimax":
        test_minimax_vs_random()
    elif mode == "test-minimax-material":
        test_minimax_vs_material()
    elif mode == "human-mcts":
        color_choice = input("Play as White or Black? (w/b): ").strip().lower()
        human_color = chess.WHITE if color_choice == 'w' else chess.BLACK
        sims_choice = input("MCTS simulations (50-500, default 200): ").strip()
        simulations = int(sims_choice) if sims_choice.isdigit() else 200
        play_human_vs_mcts(human_color, simulations=simulations)
    elif mode == "mcts":
        play_mcts_vs_random(interactive=True)
    elif mode == "test-mcts":
        test_mcts_vs_random()
    elif mode == "test-mcts-minimax":
        test_mcts_vs_minimax()
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python play.py [think|learn|course|random|human|human-material|human-minimax|test|material|minimax|minimax-material|test-material|test-minimax|test-minimax-material|human-mcts|mcts|test-mcts|test-mcts-minimax]")


if __name__ == "__main__":
    main()
