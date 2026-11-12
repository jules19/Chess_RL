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

Start with the [opening session](module-00-orientation/README.md#opening-session-see-the-destination-first):
make a prediction and watch learning happen before building the parts.

## How to use each module page

Every module opens with the same three lines, so you always know where you
are:

> **Where you are:** which piece of the final AlphaZero loop this module builds
> **Time:** a realistic estimate, including anything that runs in the background
> **You'll be able to:** three or four things you can check you can do at the end

Most modules follow this pattern:

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
  exactly what your code must do. Remove the whole skip marker to observe
  the initial failure, then implement the exercise and rerun the test. These
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
| 6 | [Self-Play RL](module-06-selfplay-rl/README.md) | The full loop: tic-tac-toe first (watch it learn), then chess | policy iteration, Monte Carlo targets (training on final game results), credit assignment, off-policy data (training on games from earlier networks) | 5-8 h, plus a long run (~3 h on the reference CPU) |
| 7 | [Measure and Scale](module-07-measure-and-scale/README.md) | Reliable Elo measurement, settings for larger runs | evaluating an agent reliably | 3-4 h, plus matches running in the background |
| 8 | [Engineering Extras](module-08-engineering/README.md) | Packaging, continuous integration (CI), performance work: the software engineering around the experiments | none | open-ended |
| 9 | [Capstone](module-09-capstone/README.md) | The experiment report: measured strength of each engine, training story, one designed experiment, a review of what went wrong | explaining your own run in RL terms | 1-2 days |

The engine exercises in Modules 0–3 need no ML background and no torch
(PyTorch, the neural-network library). The optional opening chess preview uses
torch; its tic-tac-toe alternative does not. Modules 4-6 build the
AlphaZero parts. Modules 7-8 cover measurement and software engineering
practice. Module 9 is the final assessment, and it is the one checkpoint
with no command to run.

## Choose your pace

The **core lessons** show mechanisms and check your understanding. The
**extended experiments** ask whether those mechanisms improve strength and
how confidently you can tell. You can pause after a core lesson with a real
result to explain, even when a long match would be inconclusive.

| Module | Core lesson | Extended experiment |
|---|---|---|
| 0 | Opening session: predict, watch, explain; start your journal | Play several engines and map their code |
| 1 | Add a rook-file bonus and check its sign on controlled boards | Measure its effect in matches |
| 2 | Compare node counts and implement a hard search deadline | Killer moves and transposition-table experiments |
| 3 | Compare visit distributions; make search reproducible | Rollout/filter tournaments |
| 4 | Trace encoding and train the small material network | Human-game data, full policy/value training and augmentation |
| 5 | Material-guided PUCT and the change-of-mind lab | Compare trained networks and inference speed |
| 6 | Tic-tac-toe learning and exploration; chess smoke test | Resume exercise **before** long chess runs, then larger experiments |
| 7 | Interpret uncertainty and design a comparison | Calibrated matches and scaling experiments |
| 8 | Optional engineering topics | Choose a topic that supports your experiment |
| 9 | Plan a question from your journal | Full capstone with your own measured evidence |

The core route is a conceptual introduction, not completion of every exercise.
The full engine-building route includes the exercise tests and extended work
needed by the capstone. Module 5's larger network experiments and Module 6's
long chess run need your Module 4 trained model. The opening lab's deliberately
biased checkpoint is **not** a substitute for that model.

Create `JOURNAL.md` using the template in Module 0. Add short entries only when
an experiment asks for a prediction or interpretation, and keep a note of your
next step. Bring those entries to the module checkpoints and final report.

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

Read the exercise test's contract first. If it is meant to fail before you
implement anything, remove the **whole** `@pytest.mark.skip(...)` decorator
first (it often spans several lines), then run that test. A skipped test never
checks your implementation. For the resume exercise, remove its module-level
`pytestmark` instead. Keep all unrelated exercise skips in place.

If stuck, use a hint from the [exercise recovery guide](../tests/course/README.md)
before inspecting a reference. For example, to view only the evaluator solution
without switching branches:

```bash
git fetch origin course-solutions:refs/remotes/origin/course-solutions
git diff main origin/course-solutions -- engine/evaluator.py
```

This reads reference code; it does not modify your working files. The remote
branch can lag behind the course: use the current test as the contract, not a
promise that copying an older file will pass. The revised resume exercise has
its own local reference patch, described in the recovery guide. Try the idea,
then close the solution and implement it in your own words/code.

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
