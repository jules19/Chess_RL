"""
Self-Play Reinforcement Learning - Phase 4

This package closes the AlphaZero loop. Everything before it was
preparation:

    Phase 1-2 gave us search (minimax, MCTS)
    Phase 3a gave us a network and a supervised training pipeline
    Phase 3b gave us network-guided search (PUCT)

Phase 4 removes the human games. The engine now generates its own training
data by playing itself, and improves in a cycle:

    ┌──────────────┐   games    ┌───────────────┐
    │  SELF-PLAY   │ ─────────► │ REPLAY BUFFER │
    │ (PUCT + net) │            └───────┬───────┘
    └──────▲───────┘                    │ sampled batches
           │ promoted                   ▼
           │ champion            ┌──────────────┐
    ┌──────┴───────┐   candidate │   TRAINING   │
    │    ARENA     │ ◄────────── │ (policy+value│
    │ (gatekeeping)│             │    losses)   │
    └──────────────┘             └──────────────┘

Modules:
    self_play.py     - generate games with PUCT, record training examples
    replay_buffer.py - store and sample recent training examples
    arena.py         - candidate vs champion matches; promotion gate
    train_loop.py    - orchestrates the full cycle (the main entry point)

Run a tiny end-to-end loop (minutes on CPU, proves the plumbing):
    python3 selfplay/train_loop.py --iterations 2 --games-per-iter 2 \
        --simulations 16 --train-steps 20 --arena-games 2
"""

from .replay_buffer import ReplayBuffer
from .self_play import play_selfplay_game, generate_games
from .arena import play_match
