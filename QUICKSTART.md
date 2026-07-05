# Quick Start

Playing chess against this repo in under 5 minutes.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # engines only (no torch)
# or: pip install -e ".[train,dev]"   # + neural network training
pytest -q                      # verify: all green
```

## Play

```bash
python3 cli/play.py                  # interactive menu with all modes
python3 cli/play.py human            # you vs random mover (UCI moves: e2e4, g1f3, e7e8q)
python3 cli/play.py human-minimax    # you vs the alpha-beta engine (Phase 1)
python3 cli/play.py human-mcts       # you vs the Monte Carlo engine (Phase 2)
python3 cli/play.py mcts             # watch MCTS vs random
python3 cli/play.py test             # 10 quick games + statistics (sanity check)
```

Prefer a real chess GUI? The engine speaks UCI:
`python3 chess_rl_uci.py` — see [docs/UCI_SETUP_MAC.md](docs/UCI_SETUP_MAC.md).

## Learn

This repo doubles as a hands-on course — from these engines all the way to
an AlphaZero-style self-play loop:

**→ [course/README.md](course/README.md)**

The original development roadmap and daily diary live in
[docs/plans/](docs/plans/) and [docs/history/](docs/history/).
