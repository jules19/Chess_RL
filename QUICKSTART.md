# Quick Start

This page gets you playing chess against the engines in this repository.
Allow a few minutes for setup; dependency downloads vary by machine and network.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # engines only (no torch)
# or: pip install -e ".[train,dev]"   # + neural network training
pytest -q                      # verify: all green
```

Expected: the test run reports all tests passed, plus some skipped tests (the
skipped tests are the course exercises).

## Play

```bash
python3 cli/play.py                  # Play / Watch it think / Watch it learn / Continue
python3 cli/play.py human            # you vs random mover (moves: e4 or e2e4, Nf3, e7e8q)
python3 cli/play.py human-minimax    # you vs the alpha-beta engine (Phase 1)
python3 cli/play.py human-mcts       # you vs the Monte Carlo engine (Phase 2)
python3 cli/play.py mcts             # watch MCTS vs random
python3 cli/play.py test             # 10 quick games + statistics (sanity check)
```

To play in a chess program with a graphical board (a chess GUI), start the
engine with `python3 chess_rl_uci.py`. It uses UCI (Universal Chess
Interface), the standard protocol that chess GUIs use to talk to engines.
Setup instructions are in [docs/UCI_SETUP_MAC.md](docs/UCI_SETUP_MAC.md).

## Learn

Try the [opening session](course/module-00-orientation/README.md#opening-session-see-the-destination-first)
first: predict, watch, and explain. It targets about 20 minutes after setup.
It starts with tic-tac-toe, which needs only the core install, and downloads
PyTorch in the background for the chess lab at the end.

This repository is also a hands-on course. It goes from these engines to an
AlphaZero-style loop, in which the engine improves by playing against itself
(self-play reinforcement learning). Start here:

[course/README.md](course/README.md)

The original development plans and the daily diary are in
[docs/plans/](docs/plans/) and [docs/history/](docs/history/).
