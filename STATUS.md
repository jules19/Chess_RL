# Project status

**Last updated:** 2026-10-30
**In one line:** Phases 0–3b are built. The neural network is trained but weak. Self-play RL (Phase 4)
hasn't started.

Every number below was **measured**, with the method named. Earlier versions of this file carried
ticked success criteria and Elo estimates that had never been measured. Those are gone; the
originals are in [`docs/journal/`](docs/journal/).

## What's built

| Phase | Component | Code | Teaching unit |
|---|---|---|---|
| 0 | Random mover and game loop | `cli/play.py` | [Unit 0](docs/learn/00_game_loop.md) |
| 1 | Evaluation (material + positional) | `engine/evaluator.py` | [Unit 1](docs/learn/01_evaluation.md) |
| 1 | Minimax, alpha-beta, move ordering, quiescence | `search/minimax.py` | [Units 2–3](docs/learn/02_minimax.md) |
| 1 | UCI protocol for chess GUIs | `uci/engine.py`, `chess_rl_uci.py` | [Unit 4](docs/learn/04_uci.md) |
| 2 | MCTS with evaluator-guided rollouts | `search/mcts.py` | [Unit 5](docs/learn/05_mcts.md) |
| 3a | Policy-value ResNet, encoding, supervised training | `net/`, `training/` | [Units 6–7](docs/learn/06_network.md) |
| 3a | Measurement notebook | `notebooks/01_checkpoint_reality_check.py` | [Unit 8](docs/learn/08_measuring.md) |
| 3b | Network inside MCTS rollouts | `search/nn_mcts.py` | [Unit 9](docs/learn/09_nn_mcts.md) |

## Measured results

### Search engines (from the starting position, CPU in the dev container)

| Measurement | Result |
|---|---|
| Minimax depth 3: nodes visited (incl. quiescence) | 2,470 (vs 8,902 positions without pruning) |
| Minimax depth 3: time | ~0.2 s |
| Minimax: tactical puzzle suite (`python test/tactical_suite.py`, puzzle-recommended depths) | **10/27 solved (37%)**; an earlier run in the journal recorded 12/28 (43%). The script's "<1000 Elo" label is its own rule of thumb, not a rating |
| MCTS: simulations per second (guided rollouts) | ~23 |
| MCTS 200 simulations: preferred move (seed 0) | `d2d3`, 18 of 200 visits (too few simulations to separate moves) |

### Trained network (`checkpoints/best_model.pt`)

2 residual blocks × 64 channels, 779,402 parameters (77% in the policy head's final layer), trained
5 epochs on 411 Lichess blitz games (average rating ~1519).

| Measurement | Result | Baseline / target |
|---|---|---|
| Validation top-1 accuracy at epoch 5 | 3.2% | Target 35% |
| Validation loss at epoch 5 (still falling) | 6.84 | Knowing nothing: ~9.3 (ln 4672 ≈ 8.45 policy + ~0.9 value) |
| Human move ranked first (3,277 positions, 50 games, in-sample) | 13.1% | Random legal move: 6.1% |
| Human move in top 3 | 26.3% | |
| Value sign matches game result | 54% | Coin flip: 50% |
| vs random mover, 10 games from 5 random openings | 0 W / 8 D / 2 L | |
| vs material-only greedy | 0 W / 2 D / 8 L | |
| vs minimax depth 1 | 0 W / 0 D / 10 L | |
| vs minimax depth 2 | 0 W / 0 D / 10 L | Target ≥ 40% vs depth 3 |
| NN-MCTS: network calls per simulation | ~465 | AlphaZero-style: 1 |

**Method:** `notebooks/01_checkpoint_reality_check.py`; the matches use the network's top policy
move with no search.

## Known issues

- **Training data too small and too weak:** 411 games against the planned 10,000 at 2000+ rating.
- **Validation leaks:** `training/dataset.py` splits by position, not by game, so validation numbers are optimistic.
- **Promotions:** `net/encoding.py` maps queen, rook and bishop promotions beyond the 4,672 policy slots
  (they're dropped from training), and knight promotions collide with ordinary moves. See unit 6, exercise 2.
- **UCI `info score`** prints a static evaluation from White's view, not the search score from the
  side to move. See unit 4, exercise 2.
- **NN-MCTS is ~460× more expensive per move than an AlphaZero-style search** would be (unit 9).

## Next steps

1. Retrain with more and stronger data, split by game; re-measure with notebook 01.
2. Fix the promotion encoding (requires retraining).
3. AlphaZero-style PUCT search: policy prior plus value at the leaf, no rollouts.
4. Phase 4: the self-play loop.
