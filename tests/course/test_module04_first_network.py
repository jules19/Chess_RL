"""Module 4, exercise 4 — your first network: learn material balance from the board tensor."""

import random

import chess
import numpy as np
import pytest

PIECE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material(board):
    """White's material minus Black's, in pawns: the target the network must learn."""
    return float(sum(PIECE.get(p.piece_type, 0) * (1 if p.color == chess.WHITE else -1)
                     for p in board.piece_map().values()))


def random_positions(n, seed):
    """n positions reached by random play, 10-120 plies deep (plenty of material imbalance)."""
    rng = random.Random(seed)
    boards = []
    while len(boards) < n:
        board = chess.Board()
        for _ in range(rng.randrange(10, 120)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        boards.append(board)
    return boards


@pytest.mark.skip(reason="Course Module 4, exercise 4: implement train_material_net in "
                         "training/first_network.py, then delete this skip line")
def test_first_network_learns_material_and_generalises():
    """Contract: training/first_network.py gains

        def train_material_net(boards, steps=300) -> (model, losses)

    boards: a list of chess.Board. Encode each with net.encoding.board_to_tensor
    (shape (20, 8, 8)), and use White's material minus Black's, in pawns, as the
    target (compute it as material() in this file does; copying it is fine).
    Build a small torch network (flatten 20*8*8 = 1280 inputs, one hidden layer
    of 64 with ReLU, one output),
    train it with mean squared error and Adam for `steps` full-batch steps:
    loss.backward(), optimizer.step(), optimizer.zero_grad(). Return the model and
    the list of per-step training losses.

    Then try it with 300 positions as well as 2,000 and compare the error on
    positions the network never saw (Module 4, exercise 4 has the numbers).
    """
    torch = pytest.importorskip("torch")
    from net.encoding import board_to_tensor
    from training.first_network import train_material_net

    model, losses = train_material_net(random_positions(2000, seed=0), steps=300)
    assert losses[-1] < losses[0] / 5, "training loss should fall a lot"

    held_out = random_positions(300, seed=99)
    x = torch.from_numpy(np.stack([board_to_tensor(b) for b in held_out]))
    y = torch.tensor([material(b) for b in held_out]).unsqueeze(1)
    with torch.no_grad():
        error = torch.nn.functional.mse_loss(model(x), y).item()
    guess_the_average = ((y - y.mean()) ** 2).mean().item()
    assert error < guess_the_average / 3, (
        f"held-out error {error:.1f} should be far below guessing the average "
        f"({guess_the_average:.1f}): the network must generalise, not memorise")
