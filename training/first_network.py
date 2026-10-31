"""
Module 4, exercise 4 (reference solution): a first network that learns material balance.

A deliberately small network, to see one training loop end to end before the
real ResNet: board tensor in, one number out, mean squared error, Adam.
"""

import chess
import numpy as np
import torch
import torch.nn as nn

from net.encoding import board_to_tensor

PIECE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material(board):
    """White's material minus Black's, in pawns."""
    return float(sum(PIECE.get(p.piece_type, 0) * (1 if p.color == chess.WHITE else -1)
                     for p in board.piece_map().values()))


def train_material_net(boards, steps=300, lr=1e-2):
    """Train a 1280 -> 64 -> 1 network to predict material(board). Returns (model, losses)."""
    x = torch.from_numpy(np.stack([board_to_tensor(b) for b in boards]))
    y = torch.tensor([material(b) for b in boards]).unsqueeze(1)
    model = nn.Sequential(nn.Flatten(), nn.Linear(20 * 8 * 8, 64), nn.ReLU(), nn.Linear(64, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        loss = nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    return model, losses
