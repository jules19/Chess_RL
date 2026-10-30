# Project Status

**Last updated:** 2026-10-30

## Where things stand

| Phase | Scope | Code | Validated |
|-------|-------|------|-----------|
| 0 | Random engine | ✅ | ✅ games complete correctly |
| 1 | Minimax: eval, alpha-beta, quiescence, UCI, **transposition table, iterative deepening** | ✅ | ✅ unit tests + tactical puzzles |
| 2 | Rollout MCTS (UCT, guided rollouts, blunder filter) | ✅ | ✅ unit tests, beats minimax d3-4 |
| 3a | Policy-value ResNet + supervised pipeline | ✅ | ⚠️ measured on one month of Lichess games (4,794 games, 3 epochs): policy 27.7% vs a 9.3% baseline; the value head only just beats "always even" after epoch 1, then overfits. **Needs several months of data** |
| 3b | NN-guided search: `nn_mcts.py` + **true PUCT (`search/puct.py`)** | ✅ | ⚠️ PUCT search unit-tested with fake network; needs a trained model for strength claims |
| 4 | Self-play RL: replay buffer, arena gate, training loop | ✅ | ⚠️ runs end-to-end at toy scale (verified); **no long training run yet** |

**Test suite:** `pytest --collect-only -q` counts them (82 on 2026-10-30,
6 of them the deliberately-skipped course exercises in `tests/course/`), run
by GitHub Actions CI on every push (`.github/workflows/ci.yml`). All green.

## Recent changes (2026-10)

- **Course pedagogy: teach RL as RL, and make every step runnable.**
  Every module now opens with where you are, a time estimate and checkable
  objectives, and has an "RL lens" section; `course/RL_CONCEPTS.md` maps
  every RL idea onto the code. Module 6 starts with
  `selfplay/tictactoe_lab.py`, the same loop (reusing `puct_search` and the
  replay buffer) on tic-tac-toe, which learns in seconds and is scored
  against perfect play, with sabotage, gate, buffer and Monte Carlo vs TD
  experiments. New tools: `cli/match.py` (engine-vs-engine matches with a
  95% interval), `training/get_lichess.py` (download + decompress + filter),
  `training/eval_supervised.py` (network vs baselines). The self-play loop
  gained `--no-root-noise` and a per-iteration `run_log.csv`. Expected
  results quoted in the course are measured. Fixed along the way: Module 4's
  data recipe (it kept 3 games), two wrong self-check answers, and dangling
  references to Module 7. Rubric criteria 11-13 (orientation, unblocked
  path, RL understanding) added in `docs/pedagogy/RUBRIC.md`.

- **UCI `info score` fixed:** it printed a static evaluation from White's view;
  it now reports the minimax search score from the side to move, at the depth
  actually searched. Regression test in `tests/test_uci.py`.
- **Train/validation split by game:** `training/dataset.py` split by position, so
  validation accuracy was inflated by near-duplicate positions. `split_by_game()`
  now holds out whole games, with a regression test in `tests/test_training.py`.
  Both are written up as Module 8 case studies 6–7.
- **Course navigation:** Module 8 now links on to Module 9, and the older HTML
  tutorials in `docs/tutorials/` carry a banner marking them as historical
  background that the course supersedes.
- **Documentation check and teaching evaluation** in `docs/pedagogy/`:
  `check_docs.py` runs every command in the learner docs in a sandbox copy and
  checks every file reference; `RUBRIC.md` and `OUTCOME_TEST.md` describe how
  the course's teaching quality is scored. Fixed what the first run found: a
  missing script and an unexplained exercise file in `docs/tutorials/minimax_tutorial.html`,
  and README's training command not saying where its data comes from.
- **An earlier local training run, for the record:** a 2×64 network trained for 5
  epochs on 411 Lichess blitz games (~1500 rating) with the *old* move encoding
  was measured before that encoding was fixed. It was weak: the human move ranked
  first 13% of the time (random legal move: 6%), value sign right 54% of the time,
  0 wins in 40 games against random, material-only and minimax (depth 1–2)
  opponents. That checkpoint is incompatible with the fixed encoding and was
  superseded by the October runs in the table above.

## Earlier changes (2026-07)

- **Pedagogy review actioned** (external instructional-design review of the
  course): added `tests/course/` — skip-marked exercise tests that fail
  until the student does the work, making the "un-skip the test" checkpoint
  mechanic real; added "Check your understanding" predict-then-peek Q&A to
  every module; rewrote Module 6's checkpoint to grade interpretation of
  arena evidence rather than a luck-dependent promotion; added Module 4
  data logistics (small Lichess dump, zstd recipe, time estimates) and an
  honest ML-prerequisites note; fixed factual drift (parameter count is
  1.8M, not the long-claimed ~830K).
- **Fixed: `training/evaluate.py` baseline matches were silently broken** —
  it imported `MinimaxPlayer`/`MCTSPlayer` classes that never existed,
  inside a try/except that skipped the matches with a warning. Real player
  wrappers added; `training/` now has smoke tests (`tests/test_training.py`).

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
- **Restructured:** documentation into a course (`course/`, modules 0-9
  including a capstone experiment report); historical docs preserved in
  `docs/history/` and `docs/plans/`.
- **Added:** `course-solutions` branch — reference implementations for all
  six `tests/course/` exercises, skips removed, all passing at the time (71
  tests; main has more since). The answer
  key, deliberately unmerged (`git diff main course-solutions`).

## What's next (in priority order)

These are experiments to *run*, not code to write — the course walks through
each one:

1. **Train on more real data** (course Module 4): the quick run and a
   one-month run are done (numbers in Module 4, exercise 6). Next is several
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
