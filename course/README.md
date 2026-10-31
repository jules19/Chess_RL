# Build a Chess Engine, from Random Mover to AlphaZero

**A hands-on course where the codebase is the textbook.**

This repository contains a complete progression of chess engines — from a
random mover to an AlphaZero-style self-play reinforcement learning loop —
and this course walks you through it one concept at a time.

**What you'll come away with:** you'll understand reinforcement learning
well enough to explain AlphaZero as RL rather than as a pile of tricks:
what the agent, the reward and the policy are, why self-play improves
anything at all, and why it sometimes doesn't. You'll have built and
measured every piece, and you'll have watched an RL loop learn with your
own eyes.

## How to use each module page

Every module opens with the same three lines, so you always know where you
are:

> **Where you are:** which piece of the final AlphaZero loop this module builds
> **Time:** a realistic estimate, including anything that runs in the background
> **You'll be able to:** three or four things you can check you can do at the end

Then the same rhythm:

1. **Concepts** — the idea and *why it exists*, intuition first
2. **RL lens** — the reinforcement-learning idea hiding in this module, named
   and tied to the code (all of them together: [RL_CONCEPTS.md](RL_CONCEPTS.md))
3. **Read** — specific, heavily-commented source files
4. **Do** — exercises on the real code; each says what it's for and what
   result to expect
5. **Check your understanding** — predict, then open the answer
6. **Checkpoint** — a command, or a result, that tells you objectively you're done

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
- No machine learning or RL background. Module 4 gives you the four ML ideas
  it needs in one page; the RL ideas are built up module by module.
- No GPU required. Everything runs on a laptop CPU at "toy scale"; the course
  points out exactly which knobs to turn when you have more compute. The
  modules give measured times from an 8-core laptop CPU, so you can plan.

## Setup

```bash
git clone <this repo> && cd Chess_RL
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[train,dev]"   # or just ".[dev]" to skip torch until Module 4
pytest -q                        # everything should pass before you start
```

## The modules

| # | Module | You build / study | RL idea you meet | Time |
|---|--------|-------------------|------------------|------|
| 0 | [Orientation](module-00-orientation/README.md) | Run every engine, map the repo | agent, environment, state, action, reward | 1-2 h |
| 1 | [Board & Evaluation](module-01-board-and-evaluation/README.md) | Random player, material + positional evaluation | a value function, written by hand | 3-4 h |
| 2 | [Minimax Search](module-02-minimax-search/README.md) | Alpha-beta, quiescence, transposition table, iterative deepening | the Bellman backup; planning with a known model | 4-6 h |
| 3 | [Monte Carlo Tree Search](module-03-mcts/README.md) | UCT, rollouts | exploration vs exploitation (bandits); Monte Carlo evaluation | 4-6 h + background matches |
| 4 | [The Neural Network](module-04-neural-network/README.md) | Board encoding, ResNet policy-value net, supervised training | policy and value as learned functions; imitation is not RL | two sessions: 3 h + training in the background |
| 5 | [PUCT: Search Meets Network](module-05-puct/README.md) | AlphaZero's actual search — no rollouts | search as policy improvement; Q(s, a) from the tree | 4-6 h |
| 6 | [Self-Play RL](module-06-selfplay-rl/README.md) | The full loop: tic-tac-toe first (watch it learn), then chess | policy iteration, Monte Carlo targets, credit assignment, off-policy data | 5-8 h + an overnight run |
| 7 | [Measure & Scale](module-07-measure-and-scale/README.md) | Real Elo measurement, scaling knobs | evaluating an agent honestly | 3-4 h + background matches |
| 8 | [Engineering Extras](module-08-engineering/README.md) | Packaging, CI, performance work — the software craft around the science | — | open-ended |
| 9 | [Capstone](module-09-capstone/README.md) | The experiment report: measured ladder, training story, one designed experiment, postmortem | explaining your own run in RL terms | 1-2 days |

Modules 0-3 need no ML background and no torch. Modules 4-6 are the
AlphaZero core. Modules 7-8 turn a project into an engineering habit.
Module 9 is the summative assessment — the one checkpoint with no command
to run.

**Short on time?** The RL core is Modules 0, 3, 5 and 6 Part A. Module 6's
tic-tac-toe lab runs the whole AlphaZero loop in seconds and needs no
training data, so you can see the destination early. Module 6 Part B (the
chess loop) and Module 5's network exercises need a Module 4 checkpoint, so
come back for those after Module 4.

## Your toolbox

Four tools carry most of the "Do" steps. Each has `--help`.

| Tool | Use it for | Example |
|---|---|---|
| `cli/match.py` | any "play N games of A vs B": prints the score, a 95% interval and the Elo difference | `python3 cli/match.py minimax:depth=2 random --games 10` |
| `training/get_lichess.py` | Module 4 data: download, decompress and filter one month of Lichess games | `python3 training/get_lichess.py` |
| `training/eval_supervised.py` | Module 4: is the trained network better than baselines that know nothing? | `python3 training/eval_supervised.py checkpoints/best_model.pt --pgn data/filtered.pgn --max-games 1000` |
| `selfplay/tictactoe_lab.py` | Module 6: the AlphaZero loop on a game small enough to watch learn | `python3 selfplay/tictactoe_lab.py` |

A match result without its interval means little. `cli/match.py` always
prints one, and says so when the match can't tell the engines apart.

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
dead ends. The HTML tutorials from earlier iterations, which this course
replaces, are kept in [`docs/history/tutorials/`](../docs/history/tutorials/).
