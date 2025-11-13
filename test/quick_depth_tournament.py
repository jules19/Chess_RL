"""
Quick tournament: Depth 3 (with quiescence) vs Depth 2
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
from search.minimax import best_move_minimax

def minimax_depth_2(board):
    """Minimax with depth 2."""
    return best_move_minimax(board, depth=2, verbose=False)

def minimax_depth_3(board):
    """Minimax with depth 3 (current engine with quiescence)."""
    return best_move_minimax(board, depth=3, verbose=False)

# Import tournament system
from tournament import run_tournament

print("="*60)
print("DEPTH 3 (WITH QUIESCENCE) VS DEPTH 2")
print("="*60)

run_tournament(
    minimax_depth_3, "Depth 3 + Quiescence",
    minimax_depth_2, "Depth 2",
    num_games=10,
    verbose=False
)
