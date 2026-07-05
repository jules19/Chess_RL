"""
Replay Buffer - the training-data memory of the self-play loop

Why a buffer at all? Two failure modes it prevents:

1. CATASTROPHIC FORGETTING / OVERFITTING TO THE LATEST GAMES
   If we trained only on the games from the current iteration, the network
   would chase its own most recent quirks. Keeping a window of the last N
   positions means each training batch mixes several generations of play.

2. CORRELATED BATCHES
   Consecutive positions from one game are nearly identical, and gradient
   descent assumes (roughly) independent samples. Sampling UNIFORMLY AT
   RANDOM from a large buffer decorrelates batches.

Why a WINDOW (deque with maxlen) and not "keep everything"? Old games were
played by a much weaker network; their policy targets are stale. AlphaZero
kept roughly the last 500,000 games for the same reason. Buffer size is a
real hyperparameter: too small → overfit to recent play; too large → learn
from obsolete data.

Each stored example is one position:
    state:  (20, 8, 8)  float32 - board encoding (net/encoding.py)
    policy: (4672,)     float32 - PUCT visit distribution (training target)
    value:  scalar      float32 - final game outcome from the perspective of
                                  the side to move at this position (+1/0/-1)
"""

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """Fixed-capacity FIFO store of (state, policy, value) training examples."""

    def __init__(self, capacity: int = 100_000):
        """
        Args:
            capacity: Maximum number of positions to keep. When full, the
                      oldest positions are evicted first (FIFO).
        """
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, state: np.ndarray, policy: np.ndarray, value: float):
        """Add a single training example."""
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            np.asarray(policy, dtype=np.float32),
            np.float32(value),
        ))

    def add_game(self, examples):
        """Add all examples from one self-play game."""
        for state, policy, value in examples:
            self.add(state, policy, value)

    def sample(self, batch_size: int, rng: random.Random = None):
        """
        Sample a random batch (uniform, without replacement).

        Returns:
            states:   (batch, 20, 8, 8) float32 array
            policies: (batch, 4672) float32 array
            values:   (batch,) float32 array
        """
        rng = rng or random
        batch = rng.sample(list(self.buffer), min(batch_size, len(self.buffer)))
        states, policies, values = zip(*batch)
        return (
            np.stack(states),
            np.stack(policies),
            np.array(values, dtype=np.float32),
        )

    def save(self, path: str):
        """Persist the buffer to a compressed .npz file."""
        if len(self.buffer) == 0:
            raise ValueError("Refusing to save an empty replay buffer")
        states, policies, values = zip(*self.buffer)
        np.savez_compressed(
            path,
            states=np.stack(states),
            policies=np.stack(policies),
            values=np.array(values, dtype=np.float32),
            capacity=np.int64(self.buffer.maxlen),
        )

    @classmethod
    def load(cls, path: str) -> 'ReplayBuffer':
        """Restore a buffer saved with save()."""
        data = np.load(path)
        buffer = cls(capacity=int(data['capacity']))
        for state, policy, value in zip(data['states'], data['policies'], data['values']):
            buffer.add(state, policy, float(value))
        return buffer
