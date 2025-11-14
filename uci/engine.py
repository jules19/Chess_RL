#!/usr/bin/env python3
"""
UCI (Universal Chess Interface) Engine - Day 7

This module implements the UCI protocol, allowing the chess engine
to communicate with chess GUIs like Arena, Cutechess, PyChess, etc.

Usage:
    python3 uci/engine.py

The engine will then communicate via stdin/stdout using the UCI protocol.
"""

import sys
import chess
import random
from typing import Optional

# Import our engine modules
sys.path.insert(0, '/home/user/Chess_RL')
from engine.evaluator import evaluate, best_move_material
from search.minimax import best_move_minimax
from search.mcts import best_move_mcts


class UCIEngine:
    """UCI-compliant chess engine wrapper."""

    def __init__(self):
        self.board = chess.Board()
        self.engine_type = "minimax"  # default
        self.search_depth = 3  # default
        self.mcts_simulations = 200  # default for MCTS
        self.mcts_use_evaluator = True  # default: use evaluator rollouts
        self.debug = False

    def log_debug(self, message: str):
        """Log debug messages if debug mode is enabled."""
        if self.debug:
            print(f"info string DEBUG: {message}", flush=True)

    def handle_uci(self):
        """Handle 'uci' command - identify the engine."""
        print("id name Chess_RL v0.1.0")
        print("id author Your Name")
        print()

        # Declare available options
        print("option name Engine Type type combo default minimax var random var material var minimax var mcts")
        print("option name Search Depth type spin default 3 min 1 max 6")
        print("option name MCTS Simulations type spin default 200 min 50 max 1000")
        print("option name MCTS Use Evaluator type check default true")
        print("option name Debug type check default false")
        print()

        print("uciok", flush=True)

    def handle_debug(self, on: bool):
        """Handle 'debug' command."""
        self.debug = on
        self.log_debug(f"Debug mode set to {on}")

    def handle_isready(self):
        """Handle 'isready' command - confirm engine is ready."""
        print("readyok", flush=True)

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

        else:
            self.log_debug(f"Unknown option: {name}")

    def handle_ucinewgame(self):
        """Handle 'ucinewgame' command - prepare for new game."""
        self.board = chess.Board()
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
            except ValueError as e:
                self.log_debug(f"Invalid FEN: {e}")
                return

        # Parse moves if present
        if idx < len(parts) and parts[idx] == "moves":
            idx += 1
            while idx < len(parts):
                move_str = parts[idx]
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in self.board.legal_moves:
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
            # Send evaluation info (optional but nice for GUIs)
            if self.engine_type == "minimax":
                score = evaluate(self.board)
                print(f"info depth {search_depth} score cp {score}", flush=True)

            print(f"bestmove {best_move.uci()}", flush=True)
        else:
            # No legal moves (shouldn't happen if board state is correct)
            print("bestmove 0000", flush=True)

    def handle_stop(self):
        """Handle 'stop' command - stop calculating."""
        # For our simple engine, we don't support async calculation yet
        # Just acknowledge - the GUI won't send this in our implementation
        pass

    def handle_quit(self):
        """Handle 'quit' command - exit cleanly."""
        self.log_debug("Engine shutting down")
        sys.exit(0)

    def run(self):
        """Main UCI loop - read commands and respond."""
        self.log_debug("UCI engine started")

        while True:
            try:
                line = input().strip()
                if not line:
                    continue

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
    engine = UCIEngine()
    engine.run()


if __name__ == "__main__":
    main()
