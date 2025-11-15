#!/usr/bin/env python3
"""
UCI (Universal Chess Interface) Engine

This module implements the UCI protocol, allowing the chess engine
to communicate with chess GUIs like Arena, Cutechess, PyChess, etc.

Features:
- Multiple engine types: random, material, minimax, MCTS
- Configurable search depth and MCTS simulations
- UCI transaction logging for debugging
- PGN game export for analysis

Usage:
    # Basic usage (configure via GUI)
    python3 uci/engine.py

    # Enable UCI transaction logging
    python3 uci/engine.py --uci-log uci_transactions.log

    # Enable PGN game export
    python3 uci/engine.py --pgn-log games.pgn

The engine communicates via stdin/stdout using the UCI protocol.
Logging can also be configured via UCI options in your chess GUI.
"""

import sys
import chess
import random
import argparse
import os
from datetime import datetime
from typing import Optional, TextIO

# Import our engine modules
sys.path.insert(0, '/home/user/Chess_RL')
from engine.evaluator import evaluate, best_move_material
from search.minimax import best_move_minimax
from search.mcts import best_move_mcts


class UCIEngine:
    """UCI-compliant chess engine wrapper."""

    def __init__(self, uci_log_file: Optional[str] = None, pgn_log_file: Optional[str] = None):
        self.board = chess.Board()
        self.engine_type = "minimax"  # default
        self.search_depth = 3  # default
        self.mcts_simulations = 200  # default for MCTS
        self.mcts_use_evaluator = True  # default: use evaluator rollouts
        self.debug = False

        # UCI transaction logging
        self.uci_log_enabled = False
        self.uci_log_file_path = uci_log_file or "uci_transactions.log"
        self.uci_log_handle: Optional[TextIO] = None

        # PGN game export
        self.pgn_export_enabled = False
        self.pgn_export_file_path = pgn_log_file or "games.pgn"
        self.pgn_export_handle: Optional[TextIO] = None

        # Game state tracking for PGN
        self.game_moves = []  # List of moves in SAN notation
        self.game_start_fen = None
        self.game_result = "*"  # Ongoing game

        # Auto-enable logging if file paths provided via CLI
        if uci_log_file:
            self.enable_uci_log()
        if pgn_log_file:
            self.enable_pgn_export()

    def log_debug(self, message: str):
        """Log debug messages if debug mode is enabled."""
        if self.debug:
            print(f"info string DEBUG: {message}", flush=True)

    def enable_uci_log(self):
        """Enable UCI transaction logging."""
        if not self.uci_log_enabled:
            try:
                self.uci_log_handle = open(self.uci_log_file_path, 'a', encoding='utf-8')
                self.uci_log_enabled = True
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.uci_log_handle.write(f"\n{'='*60}\n")
                self.uci_log_handle.write(f"UCI Log started: {timestamp}\n")
                self.uci_log_handle.write(f"{'='*60}\n")
                self.uci_log_handle.flush()
                self.log_debug(f"UCI logging enabled: {self.uci_log_file_path}")
            except Exception as e:
                self.log_debug(f"Failed to enable UCI log: {e}")

    def disable_uci_log(self):
        """Disable UCI transaction logging."""
        if self.uci_log_enabled and self.uci_log_handle:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.uci_log_handle.write(f"\nUCI Log ended: {timestamp}\n")
            self.uci_log_handle.close()
            self.uci_log_handle = None
            self.uci_log_enabled = False
            self.log_debug("UCI logging disabled")

    def log_uci_transaction(self, direction: str, message: str):
        """Log a UCI transaction (IN or OUT)."""
        if self.uci_log_enabled and self.uci_log_handle:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self.uci_log_handle.write(f"[{timestamp}] {direction:3s}: {message}\n")
            self.uci_log_handle.flush()

    def uci_print(self, message: str, **kwargs):
        """Print to stdout and log the transaction."""
        print(message, **kwargs)
        # Log output (strip info string DEBUG messages to avoid duplication)
        if not message.startswith("info string DEBUG:"):
            self.log_uci_transaction("OUT", message)

    def enable_pgn_export(self):
        """Enable PGN game export."""
        if not self.pgn_export_enabled:
            try:
                # Open in append mode to allow multiple games
                self.pgn_export_handle = open(self.pgn_export_file_path, 'a', encoding='utf-8')
                self.pgn_export_enabled = True
                self.log_debug(f"PGN export enabled: {self.pgn_export_file_path}")
            except Exception as e:
                self.log_debug(f"Failed to enable PGN export: {e}")

    def disable_pgn_export(self):
        """Disable PGN game export."""
        if self.pgn_export_enabled:
            # Save current game if in progress
            self.save_pgn_game()
            if self.pgn_export_handle:
                self.pgn_export_handle.close()
                self.pgn_export_handle = None
            self.pgn_export_enabled = False
            self.log_debug("PGN export disabled")

    def save_pgn_game(self):
        """Save the current game to PGN file."""
        if not self.pgn_export_enabled or not self.pgn_export_handle:
            return

        if not self.game_moves and not self.game_start_fen:
            return  # No game to save

        try:
            # Write PGN headers
            self.pgn_export_handle.write(f'[Event "Chess_RL Game"]\n')
            self.pgn_export_handle.write(f'[Site "Local"]\n')
            self.pgn_export_handle.write(f'[Date "{datetime.now().strftime("%Y.%m.%d")}"]\n')
            self.pgn_export_handle.write(f'[Round "-"]\n')
            self.pgn_export_handle.write(f'[White "Player"]\n')
            self.pgn_export_handle.write(f'[Black "Chess_RL {self.engine_type}"]\n')
            self.pgn_export_handle.write(f'[Result "{self.game_result}"]\n')

            if self.game_start_fen and self.game_start_fen != chess.STARTING_FEN:
                self.pgn_export_handle.write(f'[FEN "{self.game_start_fen}"]\n')
                self.pgn_export_handle.write(f'[SetUp "1"]\n')

            # Write moves
            self.pgn_export_handle.write('\n')
            if self.game_moves:
                # Format moves with move numbers
                move_text = []
                for i, move in enumerate(self.game_moves):
                    if i % 2 == 0:
                        move_text.append(f"{i//2 + 1}. {move}")
                    else:
                        move_text.append(move)

                # Wrap at 80 characters
                line = ""
                for token in move_text:
                    if len(line) + len(token) + 1 > 80:
                        self.pgn_export_handle.write(line.rstrip() + '\n')
                        line = token + " "
                    else:
                        line += token + " "

                if line:
                    self.pgn_export_handle.write(line.rstrip())

                self.pgn_export_handle.write(f" {self.game_result}\n\n")
            else:
                self.pgn_export_handle.write(f"{self.game_result}\n\n")

            self.pgn_export_handle.flush()
            self.log_debug(f"Game saved to PGN ({len(self.game_moves)} moves)")
        except Exception as e:
            self.log_debug(f"Failed to save PGN: {e}")

    def handle_uci(self):
        """Handle 'uci' command - identify the engine."""
        self.uci_print("id name Chess_RL v0.1.0")
        self.uci_print("id author Your Name")
        self.uci_print("")

        # Declare available options
        self.uci_print("option name Engine Type type combo default minimax var random var material var minimax var mcts")
        self.uci_print("option name Search Depth type spin default 3 min 1 max 6")
        self.uci_print("option name MCTS Simulations type spin default 200 min 50 max 1000")
        self.uci_print("option name MCTS Use Evaluator type check default true")
        self.uci_print("option name Debug type check default false")

        # Logging options
        self.uci_print("option name UCI Log type check default false")
        self.uci_print(f"option name UCI Log File type string default {self.uci_log_file_path}")
        self.uci_print("option name PGN Export type check default false")
        self.uci_print(f"option name PGN Export File type string default {self.pgn_export_file_path}")
        self.uci_print("")

        self.uci_print("uciok", flush=True)

    def handle_debug(self, on: bool):
        """Handle 'debug' command."""
        self.debug = on
        self.log_debug(f"Debug mode set to {on}")

    def handle_isready(self):
        """Handle 'isready' command - confirm engine is ready."""
        self.uci_print("readyok", flush=True)

    def handle_setoption(self, name: str, value: str):
        """Handle 'setoption' command - configure engine options."""
        if name == "Engine Type":
            if value in ["random", "material", "minimax", "mcts"]:
                self.engine_type = value
                self.log_debug(f"Engine type set to {value}")
            else:
                self.log_debug(f"Unknown engine type: {value}")

        elif name == "Search Depth":
            try:
                depth = int(value)
                if 1 <= depth <= 6:
                    self.search_depth = depth
                    self.log_debug(f"Search depth set to {depth}")
                else:
                    self.log_debug(f"Search depth out of range: {depth}")
            except ValueError:
                self.log_debug(f"Invalid depth value: {value}")

        elif name == "MCTS Simulations":
            try:
                sims = int(value)
                if 50 <= sims <= 1000:
                    self.mcts_simulations = sims
                    self.log_debug(f"MCTS simulations set to {sims}")
                else:
                    self.log_debug(f"MCTS simulations out of range: {sims}")
            except ValueError:
                self.log_debug(f"Invalid MCTS simulations value: {value}")

        elif name == "MCTS Use Evaluator":
            self.mcts_use_evaluator = (value.lower() == "true")
            self.log_debug(f"MCTS Use Evaluator set to {self.mcts_use_evaluator}")

        elif name == "Debug":
            self.debug = (value.lower() == "true")
            self.log_debug(f"Debug set to {self.debug}")

        elif name == "UCI Log":
            if value.lower() == "true":
                self.enable_uci_log()
            else:
                self.disable_uci_log()

        elif name == "UCI Log File":
            self.uci_log_file_path = value
            self.log_debug(f"UCI log file path set to {value}")
            # If logging is already enabled, restart it with new file
            if self.uci_log_enabled:
                self.disable_uci_log()
                self.enable_uci_log()

        elif name == "PGN Export":
            if value.lower() == "true":
                self.enable_pgn_export()
            else:
                self.disable_pgn_export()

        elif name == "PGN Export File":
            self.pgn_export_file_path = value
            self.log_debug(f"PGN export file path set to {value}")
            # If export is already enabled, restart it with new file
            if self.pgn_export_enabled:
                self.disable_pgn_export()
                self.enable_pgn_export()

        else:
            self.log_debug(f"Unknown option: {name}")

    def handle_ucinewgame(self):
        """Handle 'ucinewgame' command - prepare for new game."""
        # Save previous game if PGN export is enabled
        if self.pgn_export_enabled and (self.game_moves or self.game_start_fen):
            self.save_pgn_game()

        # Reset board and game state
        self.board = chess.Board()
        self.game_moves = []
        self.game_start_fen = chess.STARTING_FEN
        self.game_result = "*"
        self.log_debug("New game started")

    def handle_position(self, parts: list):
        """
        Handle 'position' command - set up board position.

        Format: position [fen <fenstring> | startpos] moves <move1> ... <movei>
        """
        idx = 0

        # Parse position type
        if parts[idx] == "startpos":
            self.board = chess.Board()
            self.game_start_fen = chess.STARTING_FEN
            idx += 1
        elif parts[idx] == "fen":
            # FEN string follows
            fen_parts = []
            idx += 1
            while idx < len(parts) and parts[idx] != "moves":
                fen_parts.append(parts[idx])
                idx += 1
            fen = " ".join(fen_parts)
            try:
                self.board = chess.Board(fen)
                self.game_start_fen = fen
            except ValueError as e:
                self.log_debug(f"Invalid FEN: {e}")
                return

        # Reset game moves when setting position (position command gives full state)
        self.game_moves = []

        # Parse moves if present and track them for PGN
        if idx < len(parts) and parts[idx] == "moves":
            idx += 1
            # Create a temporary board to track SAN notation
            temp_board = chess.Board(self.game_start_fen) if self.game_start_fen else chess.Board()

            while idx < len(parts):
                move_str = parts[idx]
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in temp_board.legal_moves:
                        # Store in SAN notation for PGN
                        san_move = temp_board.san(move)
                        self.game_moves.append(san_move)
                        temp_board.push(move)
                        self.board.push(move)
                    else:
                        self.log_debug(f"Illegal move: {move_str}")
                        return
                except ValueError as e:
                    self.log_debug(f"Invalid move format: {move_str} - {e}")
                    return
                idx += 1

        self.log_debug(f"Position set: {self.board.fen()}")

    def get_best_move(self) -> Optional[chess.Move]:
        """Calculate best move using selected engine type."""
        if self.board.is_game_over():
            return None

        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None

        if self.engine_type == "random":
            return random.choice(legal_moves)

        elif self.engine_type == "material":
            return best_move_material(self.board)

        elif self.engine_type == "minimax":
            return best_move_minimax(self.board, self.search_depth)

        elif self.engine_type == "mcts":
            return best_move_mcts(self.board,
                                 simulations=self.mcts_simulations,
                                 use_evaluator=self.mcts_use_evaluator)

        # Fallback
        return random.choice(legal_moves)

    def handle_go(self, parts: list):
        """
        Handle 'go' command - calculate and return best move.

        Supported subcommands:
        - depth <n>: search to depth n
        - movetime <ms>: search for exactly ms milliseconds
        - infinite: search until 'stop' command

        For now, we implement basic support.
        """
        # Parse go parameters
        search_depth = self.search_depth  # default from options

        idx = 0
        while idx < len(parts):
            if parts[idx] == "depth":
                try:
                    search_depth = int(parts[idx + 1])
                    idx += 2
                except (ValueError, IndexError):
                    idx += 1
            elif parts[idx] == "movetime":
                # We don't implement time management yet, just acknowledge
                idx += 2
            elif parts[idx] == "infinite":
                # For now, treat as normal search
                idx += 1
            else:
                idx += 1

        # Override search depth for this move if specified in go command
        original_depth = self.search_depth
        if search_depth != original_depth:
            self.search_depth = search_depth

        # Calculate best move
        best_move = self.get_best_move()

        # Restore original depth
        self.search_depth = original_depth

        if best_move:
            # Track move for PGN export (convert to SAN notation)
            if self.pgn_export_enabled:
                try:
                    san_move = self.board.san(best_move)
                    self.game_moves.append(san_move)
                except Exception as e:
                    self.log_debug(f"Failed to convert move to SAN: {e}")

            # Send evaluation info (optional but nice for GUIs)
            if self.engine_type == "minimax":
                score = evaluate(self.board)
                self.uci_print(f"info depth {search_depth} score cp {score}", flush=True)

            self.uci_print(f"bestmove {best_move.uci()}", flush=True)

            # Check for game over after this move (for PGN export)
            if self.pgn_export_enabled:
                # Make the move on a copy to check game state
                temp_board = self.board.copy()
                temp_board.push(best_move)
                if temp_board.is_checkmate():
                    self.game_result = "1-0" if temp_board.turn == chess.BLACK else "0-1"
                elif temp_board.is_stalemate() or temp_board.is_insufficient_material():
                    self.game_result = "1/2-1/2"
                elif temp_board.is_fifty_moves() or temp_board.is_repetition():
                    self.game_result = "1/2-1/2"
        else:
            # No legal moves (shouldn't happen if board state is correct)
            self.uci_print("bestmove 0000", flush=True)

    def handle_stop(self):
        """Handle 'stop' command - stop calculating."""
        # For our simple engine, we don't support async calculation yet
        # Just acknowledge - the GUI won't send this in our implementation
        pass

    def handle_quit(self):
        """Handle 'quit' command - exit cleanly."""
        self.log_debug("Engine shutting down")

        # Close log files
        if self.uci_log_enabled:
            self.disable_uci_log()
        if self.pgn_export_enabled:
            self.disable_pgn_export()

        sys.exit(0)

    def run(self):
        """Main UCI loop - read commands and respond."""
        self.log_debug("UCI engine started")

        while True:
            try:
                line = input().strip()
                if not line:
                    continue

                # Log incoming command
                self.log_uci_transaction("IN", line)

                parts = line.split()
                command = parts[0]
                args = parts[1:]

                if command == "uci":
                    self.handle_uci()

                elif command == "debug":
                    if args and args[0] == "on":
                        self.handle_debug(True)
                    else:
                        self.handle_debug(False)

                elif command == "isready":
                    self.handle_isready()

                elif command == "setoption":
                    # Parse: setoption name <name> value <value>
                    if len(args) >= 4 and args[0] == "name" and "value" in args:
                        value_idx = args.index("value")
                        name = " ".join(args[1:value_idx])
                        value = " ".join(args[value_idx+1:])
                        self.handle_setoption(name, value)

                elif command == "ucinewgame":
                    self.handle_ucinewgame()

                elif command == "position":
                    self.handle_position(args)

                elif command == "go":
                    self.handle_go(args)

                elif command == "stop":
                    self.handle_stop()

                elif command == "quit":
                    self.handle_quit()

                else:
                    self.log_debug(f"Unknown command: {command}")

            except EOFError:
                # GUI closed connection
                break
            except Exception as e:
                self.log_debug(f"Error: {e}")
                import traceback
                self.log_debug(traceback.format_exc())


def main():
    """Entry point for UCI engine."""
    parser = argparse.ArgumentParser(
        description='Chess_RL UCI Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start engine (configure logging via UCI GUI)
  python3 uci/engine.py

  # Start with UCI transaction logging enabled
  python3 uci/engine.py --uci-log uci_transactions.log

  # Start with PGN export enabled
  python3 uci/engine.py --pgn-log games.pgn

  # Enable both logs
  python3 uci/engine.py --uci-log uci.log --pgn-log games.pgn
        """
    )
    parser.add_argument(
        '--uci-log',
        type=str,
        metavar='FILE',
        help='Enable UCI transaction logging to specified file'
    )
    parser.add_argument(
        '--pgn-log',
        type=str,
        metavar='FILE',
        help='Enable PGN game export to specified file'
    )

    args = parser.parse_args()

    # Create engine with optional log files
    engine = UCIEngine(uci_log_file=args.uci_log, pgn_log_file=args.pgn_log)
    engine.run()


if __name__ == "__main__":
    main()
