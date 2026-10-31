# Project Status

**Last updated:** 2026-10-31

## Where things stand

| Phase | Scope | Code | Validated |
|-------|-------|------|-----------|
| 0 | Random engine | ✅ | ✅ games complete correctly |
| 1 | Minimax: eval, alpha-beta, quiescence, UCI, **transposition table, iterative deepening** | ✅ | ✅ unit tests + tactical puzzles |
| 2 | Rollout MCTS (UCT, guided rollouts, blunder filter) | ✅ | ✅ unit tests (strength vs minimax not yet measured with an interval; course Module 3 shows 6-game matches can't tell) |
| 3a | Policy-value ResNet + supervised pipeline | ✅ | ⚠️ measured on one month of Lichess games (4,794 games, 3 epochs): policy 27.7% vs a 9.3% baseline; the value head only just beats "always even" after epoch 1, then overfits. **Needs several months of data** |
| 3b | NN-guided search: `nn_mcts.py` + **true PUCT (`search/puct.py`)** | ✅ | ⚠️ unit-tested with a fake network; with the one-month network it scored 4/6 against rollout MCTS (interval 30-90%: not yet conclusive) |
| 4 | Self-play RL: replay buffer, arena gate, training loop | ✅ | ⚠️ runs end-to-end at toy scale (verified); **no long training run yet** |

**Test suite:** `pytest --collect-only -q` counts them (85 on 2026-10-31,
8 of them the deliberately-skipped course exercises in `tests/course/`), run
by GitHub Actions CI on every push (`.github/workflows/ci.yml`). All green.

## Changes

The change history is in [docs/history/CHANGELOG.md](docs/history/CHANGELOG.md).
The course itself starts at [course/README.md](course/README.md).

## What's next (in priority order)

These are experiments to *run*, not code to write — the course walks through
each one:

1. **Train on more real data** (course Module 4): the quick run and a
   one-month run are done (numbers in Module 4, exercise 7). Next is several
   months of Lichess games, until the value head beats "always even" by a
   clear margin (`training/eval_supervised.py`).
2. **Validate PUCT with the trained network** (Module 5): PUCT+NN vs rollout
   MCTS at equal simulations.
3. **Run a real self-play loop** (Module 6): warm-start from the supervised
   checkpoint, overnight run, track promotions.
4. **Measure honestly** (Module 7): cutechess-cli vs strength-limited
   Stockfish, ≥100 games, report Elo with an error bar. Replaces all
   historical guesses.

## Environment

- Development machine: an M1 Mac. Experiments run CPU-only inside a Docker
  container (no MPS there); `auto_device()` picks CUDA or MPS when it can.
  The times quoted in the course are from that 8-core CPU.
- Install: `pip install -e ".[train,dev]"`.
- No cloud spend to date; nothing requires it until self-play scaling.
