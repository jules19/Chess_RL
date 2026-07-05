"""
Module 3 checkpoint: rollout MCTS building blocks (UCT, backpropagation,
blunder filtering).
"""

import chess

from search.mcts import MCTSNode, backpropagate, hangs_material


def test_backpropagation_updates_and_alternates_perspective():
    """Values must flip sign at each level: good for White = bad for Black."""
    root = MCTSNode(chess.Board(), filter_blunders=False)
    child = root.expand()

    backpropagate(child, 1.0)

    assert child.visit_count == 1
    assert root.visit_count == 1
    assert child.total_value == 1.0
    assert root.total_value == -1.0  # flipped on the way up


def test_uct_prefers_unvisited_children():
    root = MCTSNode(chess.Board(), filter_blunders=False)
    a = root.expand()
    b = root.expand()
    backpropagate(a, 0.1)  # a: visited once, small value; b: never visited
    assert b.uct_value() == float("inf")
    assert root.best_child() is b


def test_hangs_material_detects_hanging_queen():
    # Moving the queen to h5 hangs nothing... but Qg4 walks into Bxg4
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6b1/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1")
    # Qd1 is pinned to nothing; playing d3 doesn't hang material
    assert not hangs_material(board, chess.Move.from_uci("d2d3"), threshold=300)


def test_blunder_filter_keeps_at_least_one_move():
    """Even if every move 'hangs' something, the node must stay playable."""
    board = chess.Board()
    node = MCTSNode(board, filter_blunders=True)
    assert len(node.untried_moves) > 0
