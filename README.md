# Chess_RL

Build a chess engine in stages, from a random mover to an engine in the
style of AlphaZero (DeepMind's chess program that learned by playing against
itself). Each stage is checked with evidence before the next one starts.

This repository is both a working engine ladder (a sequence of engines, each
meant to beat the one before it) and a hands-on course. Every phase produced
a playable engine that was tested against the previous one:

```
random  →  minimax (+eval)  →  MCTS  →  NN + PUCT  →  self-play RL
Phase 0        Phase 1        Phase 2     Phase 3        Phase 4
~600*        ~1200-1600*    ~1400-1600*   (train it!)   (run the loop!)
```
<sub>*Historical estimates of Elo (the chess rating scale), not measurements. Course Module 7 shows how to measure them.</sub>

Phase 3 has two parts: 3a trains the neural network, and 3b uses it in the
search.

## The course (start here)

[course/README.md](course/README.md) has ten modules that teach from this
codebase. Each module gives the concepts, the reinforcement-learning (RL)
idea behind them, the source files to read, exercises with expected results,
and a checkpoint you can verify. RL means learning to act from rewards, here
the result of each game. [course/RL_CONCEPTS.md](course/RL_CONCEPTS.md)
explains the RL ideas in terms of this code.

Modules 0-3 (search engines) need no machine-learning background. Modules
4-6 build the AlphaZero parts, including a tic-tac-toe version of the full
self-play loop that learns in seconds. Modules 7-8 cover reliable
measurement and software engineering.

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
continue where you stopped.

**Docker.** If you work inside a Docker container, clone into a folder that
is mounted from the host. Anything outside a mounted folder, including your
code, data and trained networks, is deleted when the container is rebuilt.
Build the `.venv` inside the container, since a virtual environment made on
the host won't run there. If a rebuild changes the container's Python
version, delete `.venv` and run the two setup lines again; your code, data
and saved models are unaffected.

## Quick start

### Install and play in the terminal

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

Expected: the test run reports all tests passed, plus some skipped tests.
The skipped tests are the course exercises. A match prints a 95% interval:
the range the true score very likely lies in.

### Play in a chess GUI

To play in a chess GUI (a program with a graphical board) such as Arena,
Cute Chess or PyChess, start the engine with `python3 chess_rl_uci.py`. It
uses UCI (Universal Chess Interface), the standard protocol between GUIs and
engines. See [docs/UCI_SETUP_MAC.md](docs/UCI_SETUP_MAC.md).

If you can't install a GUI, run Cute Chess in Docker and use it in your
browser. It comes with this engine and Stockfish (a strong open-source
chess engine) already set up, and nothing is installed on the host:
[docs/DOCKER_GUI.md](docs/DOCKER_GUI.md)
(`cd docker && docker compose up --build`, then http://localhost:5800).

Every stage can be played from the GUI. Choose `random`, `material`,
`minimax`, `mcts` or `puct` with the "Engine Type" option (`puct` needs a
trained model file, a checkpoint, chosen with "Model File"). The engine
follows the GUI's time controls with iterative deepening (searching one
move deeper at a time, taught in [Module 2](course/module-02-minimax-search/README.md)).

### Train the network

Train the network on games between strong players (Phase 3a). Get the data
(17 MB download), train a quick first network (about 6 minutes on a laptop
processor), then compare it with baselines that know nothing about chess:
```bash
python3 training/get_lichess.py
python3 training/train.py --pgn data/filtered.pgn --max-games 1000 --epochs 2
python3 training/eval_supervised.py checkpoints/best_model.pt --pgn data/filtered.pgn --max-games 1000
```

### Run the self-play loop

Run the self-play RL loop from start to finish at small scale (Phase 4,
about 1.5 minutes on a laptop processor, with no GPU):
```bash
python3 selfplay/train_loop.py --iterations 2 --games-per-iter 2 \
    --simulations 16 --train-steps 20 --arena-games 2
```

## Repository map

| Path | What it is |
|------|------------|
| `course/` | The course: 10 modules (0-9) with exercises, checkpoints and a final project |
| `engine/` | Evaluation (scoring a position): hand-written (`evaluator.py`) and a neural network wrapper (`nn_evaluator.py`) |
| `search/` | Move search: four engines, listed below the table |
| `net/` | `encoding.py` (board to network input, move to a number), `model.py` (the network that suggests moves and scores positions); Module 4 |
| `training/` | Learning from human games (Module 4): Lichess download (`get_lichess.py`), reading game files, training loop, evaluation against baselines (`eval_supervised.py`), game filtering |
| `selfplay/` | Phase 4 (Module 6): self-play games, the training data store, the match that decides whether a new network replaces the old one, the RL training loop; `tictactoe_lab.py`, the same loop on tic-tac-toe |
| `cli/`, `uci/` | Terminal play, engine-against-engine matches with 95% intervals (`match.py`), and the UCI interface for chess GUIs |
| `tests/` | The pytest suite, including the course checkpoints. GitHub runs it automatically whenever new code is uploaded |
| `experiments/` | Older experiment scripts: tournaments, tactical puzzle sets, profiling (not tests; the tests are in `tests/`) |
| `docs/` | UCI and Docker guides; `history/` (day-by-day development diary, changelog, the old HTML tutorials), `plans/` (original phase plans) |

The four search engines:

- `minimax.py`: looks ahead through every line of play to a fixed depth,
  using alpha-beta to skip lines that cannot change the result (Module 2).
- `mcts.py`: Monte Carlo tree search, which plays many games out to the end
  (rollouts) and favours the moves that did well (Module 3).
- `puct.py`: AlphaZero's search, guided by the network instead of rollouts
  (Module 5).
- `nn_mcts.py`: the earlier Phase 3b design, Monte Carlo tree search with
  the network scoring positions (Module 5).

## Project status

See [STATUS.md](STATUS.md). In short, all code through Phase 4 is written
and has tests. The network has been trained on one month of Lichess games
(results in course Module 4). The remaining work is to run the larger
experiments (more data, a long self-play run, a reliable Elo measurement
against Stockfish), and the course takes you through each one.

## Design principles

1. **Every phase produces a playable engine.** No phase goes for months
   without a working result.
2. **Claims need evidence.** Every claim is backed by games between engines
   or by tests, and the course keeps to that rule.
3. **Real bugs are teaching material.** Three bugs from this project are
   kept as case studies in the course instead of being removed from the
   history. The move encoding made pawn promotion impossible for the network
   to learn. The code that loads a saved network assumed a different network
   shape from the one the training code saved. A self-test could never have
   passed, and nothing ever ran it.
