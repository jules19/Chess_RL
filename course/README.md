# Build a Chess Engine, from Random Mover to AlphaZero

**A hands-on course where the codebase is the textbook.**

This repository contains a complete progression of chess engines — from a
random mover to an AlphaZero-style self-play reinforcement learning loop —
and this course walks you through it one concept at a time. Every module
follows the same rhythm:

1. **Learn** — the module README explains the concept and *why it exists*
2. **Read** — you study specific, heavily-commented source files
3. **Do** — exercises make you modify or extend the real code
4. **Checkpoint** — a pytest command tells you objectively whether it works

Two kinds of tests act as your autograder, and they check different things:

- **`tests/`** — the project's regression suite. Green means *the codebase
  works*; it passes on a fresh clone before you've done anything. Its job
  during the course is to tell you instantly when your changes break
  something.
- **`tests/course/`** — the exercise tests. Each flagship exercise has a
  **deliberately skipped** test whose docstring is the exercise contract:
  implement it, delete the `@pytest.mark.skip` line, and green means *your
  code works*. These fail until you do the work — they are the checkpoints
  that can't be faked. See [tests/course/README.md](../tests/course/README.md).

Beyond the tests, every module ends with **Check your understanding**
questions (with collapsed answers) — predict first, then peek. If your
prediction was wrong, that's the signal to reread before moving on: passing
tests with a wrong mental model is how the next module's bug gets written.

## Prerequisites

- Comfortable Python (classes, decorators are as fancy as it gets)
- Chess rules (you don't need to be good — the engine won't be either, at first)
- For Modules 4-7: basic neural-network vocabulary helps (loss, gradient,
  epoch), but the modules define what they use
- No GPU required. Everything runs on a laptop CPU at "toy scale"; the course
  points out exactly which knobs to turn when you have more compute.

## Setup

```bash
git clone <this repo> && cd Chess_RL
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[train,dev]"   # or just ".[dev]" to skip torch until Module 4
pytest -q                        # everything should pass before you start
```

## The modules

| # | Module | You build / study | Checkpoint |
|---|--------|-------------------|------------|
| 0 | [Orientation](module-00-orientation/README.md) | Run every engine, map the repo | `pytest -q` all green |
| 1 | [Board & Evaluation](module-01-board-and-evaluation/README.md) | Random player, material + positional evaluation | un-skip `tests/course/test_module01` + your arena result |
| 2 | [Minimax Search](module-02-minimax-search/README.md) | Alpha-beta, quiescence, transposition table, iterative deepening | `pytest tests/test_minimax.py` |
| 3 | [Monte Carlo Tree Search](module-03-mcts/README.md) | UCT, rollouts, exploration vs exploitation | `pytest tests/test_mcts.py` |
| 4 | [The Neural Network](module-04-neural-network/README.md) | Board encoding, ResNet policy-value net, supervised training | `pytest tests/test_encoding.py tests/test_model.py` |
| 5 | [PUCT: Search Meets Network](module-05-puct/README.md) | AlphaZero's actual search — no rollouts | `pytest tests/test_puct.py` |
| 6 | [Self-Play RL](module-06-selfplay-rl/README.md) | Replay buffer, arena gating, the full training loop | `pytest tests/test_selfplay.py` + run the loop |
| 7 | [Measure & Scale](module-07-measure-and-scale/README.md) | Real Elo measurement, scaling knobs, what to do with more compute | a measured (not guessed) rating |
| 8 | [Engineering Extras](module-08-engineering/README.md) | Packaging, CI, performance work — the software craft around the science | your own green CI run |
| 9 | [Capstone](module-09-capstone/README.md) | The experiment report: measured ladder, training story, one designed experiment, postmortem | a report you could defend |

Modules 0-3 need no ML background and no torch. Modules 4-6 are the
AlphaZero core. Modules 7-8 turn a project into an engineering habit.
Module 9 is the summative assessment — the one checkpoint with no command
to run.

## Stuck? Solutions exist — but attempt first

Reference implementations for all `tests/course/` exercises live on the
**`course-solutions`** branch (`git diff main course-solutions` shows every
solution as a readable diff). The norm: struggle for at least 30 minutes,
then peek at the *approach*, close it, and implement from memory. Solutions
you've read teach a tenth of what solutions you've fought for do — but a
permanently stuck learner learns nothing, so the rope is there.

## Two running themes

**Incremental validation.** Each phase produced a *playable* engine that was
tested against the previous one before moving on. You'll keep that
discipline: every module's exercises end with evidence, not vibes.

**Real bugs as case studies.** This codebase shipped genuine bugs that are
now teaching material — a move encoding that silently made pawn promotion
unlearnable (Module 4), architecture constants that disagreed across files
(Module 4), a self-test that could never have passed (Module 8). Each one
survived because *nothing executed the code path*. The course shows where
they hid and what testing habit would have caught them.

## Historical documents

The original day-by-day development diary and phase plans are preserved in
[`docs/history/`](../docs/history/) and [`docs/plans/`](../docs/plans/) —
useful if you want to see how the project actually unfolded, including the
dead ends. Self-contained HTML tutorials from earlier iterations live in
[`docs/tutorials/`](../docs/tutorials/).
