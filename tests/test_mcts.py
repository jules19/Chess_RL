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


def test_hangs_material_flags_hanging_and_clears_safe_moves():
    # Qa4?? steps onto the a-file, straight into the black rook's fire,
    # undefended: that hangs the queen. Qb2 is off the rook's lines: safe.
    board = chess.Board("r6k/8/8/8/8/8/8/Q6K w - - 0 1")
    assert hangs_material(board, chess.Move.from_uci("a1a4"), threshold=300)
    assert not hangs_material(board, chess.Move.from_uci("a1b2"), threshold=300)

    # A quiet opening move hangs nothing
    assert not hangs_material(chess.Board(), chess.Move.from_uci("d2d3"), threshold=300)


def test_blunder_filter_keeps_at_least_one_move():
    """Even if every move 'hangs' something, the node must stay playable."""
    board = chess.Board()
    node = MCTSNode(board, filter_blunders=True)
    assert len(node.untried_moves) > 0
