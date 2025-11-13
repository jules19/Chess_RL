"""Debug a specific puzzle to understand why engine fails."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
from search.minimax import best_move_minimax
from engine.evaluator import evaluate

# Puzzle: "Rook Mate on 7th"
# Expected: f7g7 is checkmate
# Engine found: f7f5
fen = "6k1/5Rpp/8/8/8/8/5PPP/6K1 w - - 0 1"

board = chess.Board(fen)
print("Position:")
print(board)
print(f"\nFEN: {fen}")
print("\n" + "="*60)

# Check all legal moves and their evaluations
print("\nAnalyzing all legal moves (depth 1):")
print("="*60)

for move in board.legal_moves:
    test_board = board.copy()
    test_board.push(move)

    score = evaluate(test_board)
    is_checkmate = test_board.is_checkmate()
    is_check = test_board.is_check()

    status = ""
    if is_checkmate:
        status = " 🏆 CHECKMATE!"
    elif is_check:
        status = " ⚔️  CHECK"

    print(f"{str(move):8s} -> {score:8d} centipawns{status}")

print("\n" + "="*60)
print("Running minimax search...")
print("="*60)

best = best_move_minimax(board, depth=2, verbose=True)

print(f"\n{'='*60}")
print(f"Expected: f7g7 (checkmate)")
print(f"Engine chose: {best}")

# Verify f7g7 is actually checkmate
board.push(chess.Move.from_uci("f7g7"))
print(f"\nAfter f7g7:")
print(board)
print(f"Is checkmate: {board.is_checkmate()}")
print(f"Is check: {board.is_check()}")
print(f"Legal moves: {list(board.legal_moves)}")
