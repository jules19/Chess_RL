"""Module 6, exercise 7 — replace the tic-tac-toe table with a neural network."""

import numpy as np
import pytest


@pytest.mark.skip(reason="Course Module 6, exercise 7: implement MLPPolicyValue in "
                         "selfplay/tictactoe_lab.py and a --model mlp flag, "
                         "then delete this skip line")
def test_network_policy_value_learns_and_plugs_into_the_loop():
    """Contract: selfplay/tictactoe_lab.py gains a class

        class MLPPolicyValue:      # built with no arguments, like TablePolicyValue
            __call__(board) -> (priors, value)
                priors: {move: probability} over board.legal_moves, summing to 1
                value:  float in [-1, 1], for the side to move
            train(states, policies, values, lr)
                one gradient step on a batch from the replay buffer:
                states (B, 9) side-to-move cells in {-1, 0, 1}, policies (B, 9)
                visit distributions, values (B,) targets. Loss: soft-target
                cross-entropy (illegal squares masked) + MSE, as in
                selfplay/train_loop.py::soft_target_loss. You may ignore `lr`
                and use your own optimiser's.
            raw_move(board) -> the move with the highest prior
            value_of(key) -> value of the position with that key

    A small torch network is enough: one-hot the 9 cells into 27 inputs, two
    hidden layers of 64, a 9-way policy head and a tanh value head. Cache
    each position's outputs until the next train() call: tic-tac-toe has only
    4,520 positions, and without a cache the search spends its time building
    tiny tensors.

    Then run(model=MLPPolicyValue, ...) must work, and the CLI gains
    --model mlp.
    """
    pytest.importorskip("torch")
    from selfplay.tictactoe_lab import MLPPolicyValue, TicTacToeBoard, run

    net = MLPPolicyValue()
    board = TicTacToeBoard()
    priors, value = net(board)
    assert set(priors) == set(board.legal_moves)
    assert sum(priors.values()) == pytest.approx(1.0, abs=1e-5)
    assert -1.0 <= value <= 1.0

    # Training moves the outputs toward the targets
    state = np.zeros((1, 9), dtype=np.float32)          # empty board
    target_pi = np.zeros((1, 9), dtype=np.float32)
    target_pi[0, 4] = 1.0                               # "play the centre"
    for _ in range(300):
        net.train(state, target_pi, np.array([0.5], dtype=np.float32), lr=0.3)
    assert net.raw_move(board) == 4
    assert net(board)[1] == pytest.approx(0.5, abs=0.15)
    assert net.value_of(board.key()) == pytest.approx(net(board)[1])

    # And it plugs into the self-play loop
    rows = run(model=MLPPolicyValue, iterations=1, games_per_iter=10, verbose=False)
    assert len(rows) == 2 and rows[-1]['buffer'] > 0
