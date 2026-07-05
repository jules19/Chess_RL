# Chess_RL

**Build a chess engine from a random mover to AlphaZero-style self-play
reinforcement learning — incrementally, with evidence at every step.**

This repository is both a working engine ladder and a hands-on course. Every
phase produced a playable engine that was validated against the previous one
before moving on:

```
random  →  minimax (+eval)  →  MCTS  →  NN + PUCT  →  self-play RL
Phase 0        Phase 1        Phase 2     Phase 3        Phase 4
~600 Elo     ~1200-1600*    ~1400-1600*   (train it!)   (run the loop!)
```
<sub>*Historical self-estimates — measuring honestly is course Module 7.</sub>

## 🎓 The Course (start here)

**[course/README.md](course/README.md)** — nine modules that use this
codebase as the textbook. Each module: concepts → read the annotated source
→ exercises on the real code → a pytest checkpoint as your autograder.
Modules 0-3 (search engines) need no ML background; Modules 4-6 are the
AlphaZero core; Modules 7-8 cover honest measurement and engineering craft.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[train,dev]"    # ".[dev]" alone skips torch (Phases 0-2 only)
pytest -q                        # 53 tests, ~10 seconds, all green

python3 cli/play.py human-minimax   # play the Phase 1 engine (depth-3 alpha-beta)
python3 cli/play.py human-mcts      # play the Phase 2 engine (MCTS)
python3 cli/play.py mcts            # watch MCTS demolish the random engine
python3 cli/play.py                 # interactive menu with all 17 modes
```

Connect to a chess GUI (Arena, Cute Chess, PyChess) via UCI:
`python3 chess_rl_uci.py` — setup guide in [docs/UCI_SETUP_MAC.md](docs/UCI_SETUP_MAC.md).
**Every stage is playable from the GUI**: pick `random`, `material`,
`minimax`, `mcts`, or `puct` via the "Engine Type" option (`puct` loads a
trained checkpoint set by "Model File"). GUI time controls are honored via
iterative deepening.

Train the network on expert games (Phase 3a):
```bash
python3 training/train.py --pgn data/filtered.pgn --max-games 10000 --epochs 10
```

Run the self-play RL loop end-to-end at toy scale (Phase 4, ~1 min on CPU):
```bash
python3 selfplay/train_loop.py --iterations 2 --games-per-iter 2 \
    --simulations 16 --train-steps 20 --arena-games 2
```

## Repository map

| Path | What it is |
|------|------------|
| `course/` | **The course** — 9 modules with exercises and checkpoints |
| `engine/` | Evaluation: hand-crafted (`evaluator.py`) and NN wrapper (`nn_evaluator.py`) |
| `search/` | `minimax.py` (alpha-beta + quiescence + transposition table + iterative deepening), `mcts.py` (rollout MCTS/UCT), `puct.py` (true AlphaZero search, no rollouts), `nn_mcts.py` (transitional Phase 3b) |
| `net/` | `encoding.py` (board→tensor, move→index), `model.py` (ResNet policy-value network) |
| `training/` | Supervised learning: PGN dataset, training loop, evaluation, game filtering |
| `selfplay/` | Phase 4: self-play generation, replay buffer, arena promotion gate, RL training loop |
| `cli/`, `uci/` | Terminal play and UCI protocol (works with chess GUIs) |
| `tests/` | pytest suite — the course checkpoints (run in CI) |
| `test/` | Legacy experiment scripts: tournaments, tactical suites, profiling |
| `docs/` | UCI guides; `history/` (day-by-day dev diary), `plans/` (original phase plans), `tutorials/` (standalone HTML tutorials) |

## Project status

See [STATUS.md](STATUS.md). Short version: all code through Phase 4 is
implemented and unit-tested; the remaining work is *running* the big
experiments (supervised training on real Lichess data, a long self-play run,
honest Elo measurement) — which is exactly what the course walks you through.

## Design principles

1. **Every phase ships a playable engine.** No six-month dark tunnels.
2. **Evidence over vibes.** Claims are backed by games between engines and
   by tests; the course keeps that discipline.
3. **Real bugs are teaching material.** The promotion-encoding bug, the
   architecture drift, the self-test that could never pass — all preserved
   as case studies in the course rather than scrubbed from history.
