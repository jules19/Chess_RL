"""Verify problematic tactical puzzles."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess

def analyze_position(name, fen, expected_moves, description):
    """Analyze a single position."""
    print(f"\n{'='*70}")
    print(f"Puzzle: {name}")
    print(f"{'='*70}")
    print(f"Description: {description}")
    print(f"FEN: {fen}")
    print(f"Expected moves: {expected_moves}")
    print()

    board = chess.Board(fen)
    print(board)
    print()

    print(f"Side to move: {'White' if board.turn else 'Black'}")
    print(f"Is check: {board.is_check()}")
    print(f"Is checkmate: {board.is_checkmate()}")
    print()

    # Check if expected moves are legal
    legal_moves = [str(m) for m in board.legal_moves]
    print(f"Legal moves ({len(legal_moves)}):")
    for i, move in enumerate(legal_moves[:20], 1):  # Show first 20
        print(f"  {move}", end="  ")
        if i % 5 == 0:
            print()
    if len(legal_moves) > 20:
        print(f"\n  ... and {len(legal_moves) - 20} more")
    else:
        print()

    print()
    for expected in expected_moves:
        if expected in legal_moves:
            print(f"✅ Expected move '{expected}' is legal")
            # Try the move and see what happens
            test_board = board.copy()
            test_board.push(chess.Move.from_uci(expected))
            print(f"   After {expected}:")
            print(f"   - Check: {test_board.is_check()}")
            print(f"   - Checkmate: {test_board.is_checkmate()}")
            print(f"   - Stalemate: {test_board.is_stalemate()}")
        else:
            print(f"❌ Expected move '{expected}' is NOT legal!")

    return board, legal_moves


# Problem 1: Fool's Mate Pattern
print("\n" + "="*70)
print("PROBLEM 1: Fool's Mate Pattern")
print("="*70)

analyze_position(
    "Fool's Mate Pattern",
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1",
    ["d1h5"],
    "Back rank weakness - queen delivers mate"
)

# Check what the position actually looks like
print("\nANALYSIS:")
print("This position has Black's queen on h4 already attacking White.")
print("White is to move. The expected move d1h5 doesn't make sense.")
print("This looks like it should be BLACK to move, and Black has checkmate threats.")

# Problem 2: Queen Mate on f7
print("\n\n" + "="*70)
print("PROBLEM 2: Queen Mate on f7")
print("="*70)

analyze_position(
    "Queen Mate on f7",
    "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1",
    ["e8f7"],
    "Black king must take queen on f7, then Bxf7# is checkmate (This is BLACK to move - king takes queen)"
)

print("\nANALYSIS:")
print("White has a queen on f7, Black king on e8.")
print("Black is to move - so Kxf7 is Black taking the queen.")
print("Then after Kxf7, White plays Bxf7# which is NOT checkmate")
print("because the king is ON f7, not on e8!")
print("This puzzle is WRONG. The mate is already there - White queen on f7 with")
print("bishop on c4 is already checkmate if it were White's turn.")

# Problem 3: Hanging Queen
print("\n\n" + "="*70)
print("PROBLEM 3: Hanging Queen")
print("="*70)

analyze_position(
    "Hanging Queen",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPPQPPP/RNB1KBNR b KQkq - 0 1",
    ["d8e7", "d8h4", "d8f6", "f8c5", "g8f6", "b8c6"],
    "White queen on e2 is undefended (but may not be capturable immediately)"
)

print("\nANALYSIS:")
print("White's queen is on e2. Let's check if Black can capture it:")
for move_str in ["d8e7", "d8h4", "d8f6"]:
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPPQPPP/RNB1KBNR b KQkq - 0 1")
    move = chess.Move.from_uci(move_str)
    board.push(move)
    # Check if queen is hanging
    print(f"After {move_str}: Can White queen on e2 be captured? ", end="")
    # The queen is still on e2
    if chess.QUEEN in [board.piece_at(chess.E2).piece_type if board.piece_at(chess.E2) else None]:
        print("Queen still on e2 - not captured!")
    else:
        print("Queen captured or moved")

print("\nNone of the expected moves actually capture the white queen!")
print("The 'hanging queen' is not actually hanging.")

# Problem 4: Starting position labeled as "Hanging Rook"
print("\n\n" + "="*70)
print("PROBLEM 4: Hanging Rook (Starting Position)")
print("="*70)

analyze_position(
    "Hanging Rook",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ["e2e4", "d2d4", "g1f3"],
    "Starting position - no hanging pieces, develop normally"
)

print("\nANALYSIS:")
print("This is just the starting position. No hanging pieces at all.")
print("Should not be in 'Hanging Pieces' category.")

print("\n\n" + "="*70)
print("SUMMARY OF ISSUES")
print("="*70)
print("1. Fool's Mate Pattern: Expected move is illegal")
print("2. Queen Mate on f7: Puzzle logic is wrong (mate is already there)")
print("3. Hanging Queen: Queen is NOT hanging, none of the moves capture it")
print("4. Hanging Rook: Starting position has no hanging pieces")
print("\nRECOMMENDATION: Remove or fix these puzzles")
