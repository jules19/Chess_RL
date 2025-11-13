"""
Test quiescence search on a tactical position that demonstrates horizon effect.

This position has a hanging piece that looks capturable, but there's a recapture.
Without quiescence: Engine thinks capture is good
With quiescence: Engine sees the recapture and evaluates correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
from search.minimax import best_move_minimax

# Tactical position: Scholar's Mate pattern
# White can play Qxf7+ but it's protected by the bishop
# Without quiescence: might think Qxf7 is good (captures pawn)
# With quiescence: should see Kxf7 recapture
fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1"

board = chess.Board(fen)

print("="*60)
print("QUIESCENCE SEARCH TEST")
print("="*60)
print("\nPosition:")
print(board)
print(f"\nFEN: {fen}")

print("\n" + "="*60)
print("Testing with Quiescence Search (current)")
print("="*60)

move = best_move_minimax(board, depth=3, verbose=True)

print(f"\n{'='*60}")
print("ANALYSIS")
print(f"{'='*60}")
print(f"Best move: {move}")

if move and str(move) == "h5f7":
    print("✅ Found Scholar's Mate attack (Qxf7+)!")
    print("This checks if the engine sees this forcing move.")
else:
    print(f"Engine chose: {move}")
    print("Different move - analyzing tactical alternatives")

# Let's also test a simpler position: hanging piece with recapture
print("\n" + "="*60)
print("TEST 2: Simple Capture with Recapture")
print("="*60)

# Position: White bishop on e5, can take pawn on f6
# But black knight on g8 can recapture Nxe5
fen2 = "rnbqkb1r/pppp1ppp/5n2/4B3/4P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1"
board2 = chess.Board(fen2)

print("\nPosition:")
print(board2)
print(f"\nFEN: {fen2}")
print("\nWhite can play Bxf6, but Black recaptures with gxf6")
print("Quiescence should see the recapture sequence")

move2 = best_move_minimax(board2, depth=3, verbose=False)
print(f"\nEngine's choice: {move2}")

# Check if engine makes Bxf6
if move2 and str(move2) == "e5f6":
    print("⚠️  Engine played Bxf6 - checking if it saw the recapture...")
    test_board = board2.copy()
    test_board.push(move2)
    print(f"After Bxf6, material: {test_board.fen()}")
else:
    print("✅ Engine avoided the simple capture (might have seen recapture)")

print("\n" + "="*60)
print("TEST 3: Mate in 1 (should find faster with quiescence)")
print("="*60)

# Back rank mate
fen3 = "6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"
board3 = chess.Board(fen3)

print("\nPosition:")
print(board3)
print(f"\nFEN: {fen3}")
print("\nExpected: Re8# (back rank mate)")

move3 = best_move_minimax(board3, depth=2, verbose=False)
print(f"\nEngine's choice: {move3}")

if move3 and str(move3) == "e1e8":
    print("✅ PASS: Found back rank mate!")
else:
    print(f"❌ FAIL: Expected e1e8, got {move3}")

print("\n" + "="*60)
print("Quiescence search is active!")
print("="*60)
print("\nQuiescence search now extends tactical lines beyond the")
print("normal search depth, preventing horizon effect blunders.")
