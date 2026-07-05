# Project Status

**Last updated:** 2026-07-05

## Where things stand

| Phase | Scope | Code | Validated |
|-------|-------|------|-----------|
| 0 | Random engine | ✅ | ✅ games complete correctly |
| 1 | Minimax: eval, alpha-beta, quiescence, UCI, **transposition table, iterative deepening** | ✅ | ✅ unit tests + tactical puzzles |
| 2 | Rollout MCTS (UCT, guided rollouts, blunder filter) | ✅ | ✅ unit tests, beats minimax d3-4 |
| 3a | Policy-value ResNet + supervised pipeline | ✅ | ⚠️ pipeline unit-tested; **not yet trained on real data** |
| 3b | NN-guided search: `nn_mcts.py` + **true PUCT (`search/puct.py`)** | ✅ | ⚠️ PUCT search unit-tested with fake network; needs a trained model for strength claims |
| 4 | Self-play RL: replay buffer, arena gate, training loop | ✅ | ⚠️ runs end-to-end at toy scale (verified); **no long training run yet** |

**Test suite:** 53 pytest tests in `tests/`, run by GitHub Actions CI on
every push (`.github/workflows/ci.yml`). All green.

## Recent changes (2026-07)

- **Fixed: promotion move encoding.** Queen promotions overflowed the
  4672-dim policy vector and were silently dropped; knight promotions
  collided with ordinary moves. The network could never have learned to
  promote. New scheme + regression tests (`tests/test_encoding.py`).
- **Fixed: checkpoint architecture drift.** Different files hardcoded
  different model sizes (2×64 vs 4×128). `net/model.py::load_model()` now
  infers architecture from the checkpoint weights.
- **Added:** transposition table + iterative deepening with predictive time
  management (`search/minimax.py`).
- **Added:** true AlphaZero PUCT search — no rollouts, Dirichlet noise,
  temperature selection (`search/puct.py`).
- **Added:** complete Phase 4 self-play skeleton (`selfplay/`), verified
  end-to-end at toy scale on CPU.
- **Added:** UCI now covers every engine stage — new `puct` engine type
  loads a trained checkpoint ("Model File" option) with graceful fallback,
  and `go movetime`/`wtime`/`btime` are honored via iterative deepening.
  Protocol-level subprocess tests in `tests/test_uci.py`.
- **Added:** packaging (`pyproject.toml`), pytest suite, CI.
- **Restructured:** documentation into a course (`course/`, 9 modules);
  historical docs preserved in `docs/history/` and `docs/plans/`.

## What's next (in priority order)

These are experiments to *run*, not code to write — the course walks through
each one:

1. **Train on real data** (course Module 4): download + filter Lichess games,
   train 10 epochs, hit the success criteria (val policy accuracy > 35%).
2. **Validate PUCT with the trained network** (Module 5): PUCT+NN vs rollout
   MCTS at equal simulations.
3. **Run a real self-play loop** (Module 6): warm-start from the supervised
   checkpoint, overnight run, track promotions.
4. **Measure honestly** (Module 7): cutechess-cli vs strength-limited
   Stockfish, ≥100 games, report Elo with an error bar. Replaces all
   historical guesses.

## Environment

- Development machine: Mac mini M4 (`auto_device()` picks MPS automatically);
  everything also runs CPU-only.
- Install: `pip install -e ".[train,dev]"`.
- No cloud spend to date; nothing requires it until self-play scaling.
