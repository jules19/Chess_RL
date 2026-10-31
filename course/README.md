# Build a Chess Engine, from Random Mover to AlphaZero

A hands-on course that teaches from the code of a working project.

This repository contains a complete sequence of chess engines, from a random
mover to an AlphaZero-style engine that learns by playing against itself
(self-play reinforcement learning). AlphaZero is DeepMind's system that
learned chess this way. This course takes you through the engines one
concept at a time.

By the end, you will understand reinforcement learning (RL) well enough to
explain AlphaZero as an RL method and how its parts fit together. You will
know what the agent, the reward and the policy are; the course defines each
term when it first appears. You will know why self-play improves anything
at all, and why it sometimes doesn't. You will have built and
measured every piece, and you will have watched an RL loop learn.

## How to use each module page

Every module opens with the same three lines, so you always know where you
are:

> **Where you are:** which piece of the final AlphaZero loop this module builds
> **Time:** a realistic estimate, including anything that runs in the background
> **You'll be able to:** three or four things you can check you can do at the end

Then every module has the same sections, in this order:

1. **Concepts.** The idea and why it exists, with the intuition first.
2. **RL connection.** The reinforcement-learning idea in this module, named and
   connected to the code (all of them together: [RL_CONCEPTS.md](RL_CONCEPTS.md)).
3. **Read.** Specific source files with detailed comments.
4. **Do.** Exercises on the real code. Each says what it is for and what
   result to expect.
5. **Check your understanding.** Predict the answer, then open it.
6. **Checkpoint.** A command, or a result, that shows you have finished.

Two kinds of tests check your work, and they check different things:

- **`tests/`** is the project's main test suite. It checks that existing
  behaviour still works after a change. It passes on a fresh clone before
  you have done anything. During the course, it tells you straight away when
  your changes break something.
- **`tests/course/`** holds the exercise tests. Each main exercise has a test
  that is skipped until you do the exercise. The test's docstring states
  exactly what your code must do. Implement it, delete the
  `@pytest.mark.skip` line, and a passing test means your code works. These
  tests fail until you do the work, so they cannot pass by accident. See
  [tests/course/README.md](../tests/course/README.md).

Every module also ends with **Check your understanding** questions, with
the answers collapsed. Predict each answer first, then open it. If your
prediction was wrong, reread the module before moving on. Code can pass the
tests while your understanding of how it works is wrong, and that wrong
understanding leads to bugs in the next module.

## Prerequisites

- Comfortable Python (nothing more advanced than classes and decorators)
- Chess rules (you don't need to play well, and the engine won't either at
  first)
- No machine learning (ML) or RL background. Module 4 explains the four ML
  ideas it needs in one page, and the RL ideas are built up module by module.
- No GPU required. Everything runs on a laptop CPU at small scale, and the
  course says which settings to change when you have more computing power.
  The modules give measured times from an 8-core laptop CPU, so you can plan.

## Setup

```bash
git clone https://github.com/jules19/Chess_RL.git && cd Chess_RL
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[train,dev]"   # or just ".[dev]" to skip torch until Module 4
pytest -q                        # everything should pass before you start
```

Expected: every test passes, apart from the course exercise tests, which
show as skipped.

## The modules

Each term in the table is taught in the module on its row, and the RL terms
are also collected in [RL_CONCEPTS.md](RL_CONCEPTS.md). You don't need to
know any of them before you start. The short notes in brackets give a first
idea of the heavier ones.

| # | Module | You build / study | RL idea you meet | Time |
|---|--------|-------------------|------------------|------|
| 0 | [Orientation](module-00-orientation/README.md) | Run every engine, map the repository | agent, environment, state, action, reward | 1-2 h |
| 1 | [Board and Evaluation](module-01-board-and-evaluation/README.md) | Random player, material and positional evaluation | a value function, written by hand | 3-4 h |
| 2 | [Minimax Search](module-02-minimax-search/README.md) | Alpha-beta, quiescence, transposition table, iterative deepening (ways to search deeper in the same time) | the Bellman backup (a position's value comes from the values of the positions after it); planning with a known model (the rules) | 4-6 h |
| 3 | [Monte Carlo Tree Search](module-03-mcts/README.md) | UCT, rollouts (games played out to the end) | exploration versus exploitation (trying less-tested moves against repeating good ones, studied as the bandit problem); Monte Carlo evaluation | 4-6 h, plus matches running in the background |
| 4 | [The Neural Network](module-04-neural-network/README.md) | Board encoding, a residual network (ResNet) that suggests moves and scores positions, supervised training (learning from human moves) | policy and value as learned functions; why learning by copying human moves is not RL | two sessions: 3 h, plus training in the background |
| 5 | [PUCT: Search Meets Network](module-05-puct/README.md) | AlphaZero's search, which uses the network instead of rollouts | search as policy improvement; Q(s, a) from the tree (the average result of move a in position s) | 4-6 h |
| 6 | [Self-Play RL](module-06-selfplay-rl/README.md) | The full loop: tic-tac-toe first (watch it learn), then chess | policy iteration, Monte Carlo targets (training on final game results), credit assignment, off-policy data (training on games from earlier networks) | 5-8 h, plus an overnight run |
| 7 | [Measure and Scale](module-07-measure-and-scale/README.md) | Reliable Elo measurement, settings for larger runs | evaluating an agent reliably | 3-4 h, plus matches running in the background |
| 8 | [Engineering Extras](module-08-engineering/README.md) | Packaging, continuous integration (CI), performance work: the software engineering around the experiments | none | open-ended |
| 9 | [Capstone](module-09-capstone/README.md) | The experiment report: measured strength of each engine, training story, one designed experiment, a review of what went wrong | explaining your own run in RL terms | 1-2 days |

Modules 0-3 need no ML background and no torch (PyTorch, the neural-network library). Modules 4-6 build the
AlphaZero parts. Modules 7-8 cover measurement and software engineering
practice. Module 9 is the final assessment, and it is the one checkpoint
with no command to run.

**If you are short on time,** the RL modules are 0, 3, 5 and 6 Part A.
Module 6's tic-tac-toe lab runs the whole AlphaZero loop in seconds and needs
no training data, so you can see the complete loop early. Module 6 Part B
(the chess loop) and Module 5's network exercises need a model trained in
Module 4 (a checkpoint), so come back for those after Module 4.

## Your toolbox

Most of the "Do" steps use these four tools. Each has `--help`.

| Tool | Use it for | Example |
|---|---|---|
| `cli/match.py` | any "play N games of A against B": prints the score, a 95% interval (the range the true score very likely lies in) and the Elo difference (Elo is the chess rating scale; Module 7 explains it) | `python3 cli/match.py minimax:depth=2 random --games 10` |
| `training/get_lichess.py` | Module 4 data: download, decompress and filter one month of Lichess games | `python3 training/get_lichess.py` |
| `training/eval_supervised.py` | Module 4: is the trained network better than baselines that know nothing about chess? | `python3 training/eval_supervised.py checkpoints/best_model.pt --pgn data/filtered.pgn --max-games 1000` |
| `selfplay/tictactoe_lab.py` | Module 6: the AlphaZero loop on a game small enough to watch learn | `python3 selfplay/tictactoe_lab.py` |

A match result tells you little without its 95% interval. `cli/match.py`
always prints one, and says so when the match can't tell the engines apart.

## If you are stuck: solutions

Reference implementations for the `tests/course/` exercises are on the
`course-solutions` branch (`git diff main course-solutions` shows every
solution as a readable diff). Try each exercise yourself for at least 30
minutes first. Then look at how the solution approaches the problem, close
it, and write your own version from memory. You learn much less from reading
a solution than from working it out yourself. But a learner who stays stuck
learns nothing, so the solutions are available.

## Two themes of the course

**Incremental validation.** Each phase produced a playable engine that was
tested against the previous one before moving on. You will keep to that
rule: every module's exercises end with evidence, such as a test result or a
match score.

**Real bugs as case studies.** This codebase contained real bugs that are now
teaching material. A move encoding made pawn promotion impossible to learn,
without any error or warning (Module 4). Architecture constants disagreed
between files (Module 4). A self-test could never have passed (Module 8).
Each bug stayed in the code because no test or program ever ran the code
that contained it. The course shows where each bug was and which testing
habit would have found it.

## Historical documents

The original day-by-day development diary and phase plans are kept in
[`docs/history/`](../docs/history/) and [`docs/plans/`](../docs/plans/).
They are useful if you want to see how the project developed, including the
approaches that failed. The HTML tutorials from earlier versions, which this
course replaces, are kept in [`docs/history/tutorials/`](../docs/history/tutorials/).
