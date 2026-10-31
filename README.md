# Chess_RL

**Build a chess engine from a random mover to AlphaZero-style self-play
reinforcement learning — incrementally, with evidence at every step.**

This repository is both a working engine ladder and a hands-on course. Every
phase produced a playable engine that was validated against the previous one
before moving on:

```
random  →  minimax (+eval)  →  MCTS  →  NN + PUCT  →  self-play RL
Phase 0        Phase 1        Phase 2     Phase 3        Phase 4
~600*        ~1200-1600*    ~1400-1600*   (train it!)   (run the loop!)
```
<sub>*Historical self-estimates — measuring honestly is course Module 7.</sub>

## 🎓 The Course (start here)

**[course/README.md](course/README.md)** — ten modules that use this
codebase as the textbook. Each module: where you are and what you'll be able
to do → concepts → the reinforcement-learning idea behind it → the annotated
source → exercises with expected results → a checkpoint you can verify.
Modules 0-3 (search engines) need no ML background; Modules 4-6 are the
AlphaZero core, including a tic-tac-toe version of the full self-play loop
that learns in seconds; Modules 7-8 cover honest measurement and engineering
craft. The RL ideas, mapped onto this code: [course/RL_CONCEPTS.md](course/RL_CONCEPTS.md).

### Setting up to take the course

Work in your own fresh clone, on your own branch, so your exercise code never
mixes with the course itself:

```bash
git clone https://github.com/jules19/Chess_RL.git Chess_RL_learner
cd Chess_RL_learner
git checkout -b my-run              # your exercise work lives here
python3 -m venv .venv && source .venv/bin/activate   # Python 3.9+
pip install -e ".[train,dev]"       # ".[dev]" alone skips torch until Module 4
pytest -q                           # expect all passed, plus skips: the skips are your exercises
```

Then open [Module 0](course/module-00-orientation/README.md) and work down
the page. Next time, `cd Chess_RL_learner && source .venv/bin/activate` and
carry on where you left off.

**Running inside a Docker container?** Clone into a folder that's mounted
from the host. Anything outside a mounted folder disappears when the
container is rebuilt, and that includes your code, your downloaded data and
your trained networks. Build the `.venv` inside the container, since a
virtual environment made on the host won't run there. If a rebuild changes
the container's Python version, delete `.venv` and run the two setup lines
again; your code, data and checkpoints are unaffected.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[train,dev]"    # ".[dev]" alone skips torch (Phases 0-2 only)
pytest -q                        # fast suite, all green (skips = course exercises)

python3 cli/play.py human-minimax   # play the Phase 1 engine (depth-3 alpha-beta)
python3 cli/play.py human-mcts      # play the Phase 2 engine (MCTS)
python3 cli/play.py mcts            # watch MCTS demolish the random engine
python3 cli/play.py                 # interactive menu with every mode
python3 cli/match.py minimax:depth=2 random --games 10   # engine vs engine, with a 95% interval
python3 selfplay/tictactoe_lab.py   # the AlphaZero loop on tic-tac-toe: watch it learn (~7 s)
```

Connect to a chess GUI (Arena, Cute Chess, PyChess) via UCI:
`python3 chess_rl_uci.py` — setup guide in [docs/UCI_SETUP_MAC.md](docs/UCI_SETUP_MAC.md).
**Can't install a GUI on your machine?** Run Cute Chess in Docker, rendered
in your browser, with the engine and Stockfish pre-configured — nothing
installed on the host: [docs/DOCKER_GUI.md](docs/DOCKER_GUI.md)
(`cd docker && docker compose up --build`, then http://localhost:5800).
**Every stage is playable from the GUI**: pick `random`, `material`,
`minimax`, `mcts`, or `puct` via the "Engine Type" option (`puct` loads a
trained checkpoint set by "Model File"). GUI time controls are honored via
iterative deepening.

Train the network on expert games (Phase 3a). Get the data (17 MB download), train a quick
first network (~6 minutes on a laptop CPU), then compare it with baselines:
```bash
python3 training/get_lichess.py
python3 training/train.py --pgn data/filtered.pgn --max-games 1000 --epochs 2
python3 training/eval_supervised.py checkpoints/best_model.pt --pgn data/filtered.pgn --max-games 1000
```

Run the self-play RL loop end-to-end at toy scale (Phase 4, ~1.5 min on CPU):
```bash
python3 selfplay/train_loop.py --iterations 2 --games-per-iter 2 \
    --simulations 16 --train-steps 20 --arena-games 2
```

## Repository map

| Path | What it is |
|------|------------|
| `course/` | **The course** — 10 modules (0-9) with exercises, checkpoints, and a capstone |
| `engine/` | Evaluation: hand-crafted (`evaluator.py`) and NN wrapper (`nn_evaluator.py`) |
| `search/` | `minimax.py` (alpha-beta + quiescence + transposition table + iterative deepening), `mcts.py` (rollout MCTS/UCT), `puct.py` (true AlphaZero search, no rollouts), `nn_mcts.py` (transitional Phase 3b) |
| `net/` | `encoding.py` (board→tensor, move→index), `model.py` (ResNet policy-value network) |
| `training/` | Supervised learning: Lichess download (`get_lichess.py`), PGN dataset, training loop, evaluation against baselines (`eval_supervised.py`), game filtering |
| `selfplay/` | Phase 4: self-play generation, replay buffer, arena promotion gate, RL training loop; `tictactoe_lab.py`, the same loop on tic-tac-toe |
| `cli/`, `uci/` | Terminal play, engine-vs-engine matches with error bars (`match.py`), and UCI protocol (works with chess GUIs) |
| `tests/` | pytest suite — the course checkpoints (run in CI) |
| `experiments/` | Legacy experiment scripts: tournaments, tactical suites, profiling (not tests; the tests are in `tests/`) |
| `docs/` | UCI and Docker guides; `history/` (day-by-day dev diary, changelog, the old HTML tutorials), `plans/` (original phase plans) |

## Project status

See [STATUS.md](STATUS.md). Short version: all code through Phase 4 is
implemented and unit-tested; the network has been trained on one month of
Lichess games (numbers in course Module 4), and the remaining work is
*running* the bigger experiments (more data, a long self-play run, honest Elo
measurement against Stockfish) — which is exactly what the course walks you
through.

## Design principles

1. **Every phase ships a playable engine.** No six-month dark tunnels.
2. **Evidence over vibes.** Claims are backed by games between engines and
   by tests; the course keeps that discipline.
3. **Real bugs are teaching material.** The promotion-encoding bug, the
   architecture drift, the self-test that could never pass — all preserved
   as case studies in the course rather than scrubbed from history.
