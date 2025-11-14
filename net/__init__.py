"""
Neural Network Module - Phase 3a (Week 3-4)

This module implements the policy-value neural network for chess,
following the AlphaZero architecture:

- Shared ResNet trunk (convolutional backbone)
- Policy head (predicts move probabilities)
- Value head (predicts position evaluation)

The network takes board positions as input and outputs:
1. Policy: Probability distribution over all possible moves
2. Value: Scalar evaluation of the position (-1 to +1)

Start small (4-6 ResBlocks) and scale up after validation.
"""

from .model import PolicyValueNetwork, ResidualBlock, create_model
from .encoding import (
    board_to_tensor,
    tensor_to_board_debug,
    move_to_index,
    index_to_move,
    legal_moves_mask,
    batch_board_to_tensor,
    batch_legal_moves_mask
)

__all__ = [
    'PolicyValueNetwork',
    'ResidualBlock',
    'create_model',
    'board_to_tensor',
    'tensor_to_board_debug',
    'move_to_index',
    'index_to_move',
    'legal_moves_mask',
    'batch_board_to_tensor',
    'batch_legal_moves_mask',
]
