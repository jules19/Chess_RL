# Module 0 — Orientation

> **Where you are:** the start. You'll see the finished ladder of engines before building any of it.
> **Time:** 1-2 hours.
> **You'll be able to:**
> - run every engine in the repo and play a match between two of them
> - say where *search* lives in the code and where *knowledge* lives
> - describe chess as a reinforcement-learning problem: agent, environment, state, action, reward
> - use a failing test to find which part of the code you broke

The habit the whole course depends on: *a checkpoint is a command, not a feeling.*

## Concepts

The project is a ladder of engines, each meant to beat the one before it
(the matches in this course are how you check that claim, rung by rung):

```
random  →  minimax (+eval)  →  MCTS  →  NN + PUCT  →  self-play RL
Phase 0        Phase 1        Phase 2     Phase 3        Phase 4
Module 1      Modules 1-2     Module 3   Modules 4-5     Module 6
```

Two ideas make this ladder work:

1. **Every rung is playable.** You can always *play* the current engine and
   *pit it against the previous one*. Progress is measured in game results.
2. **Search and knowledge are separable.** Minimax and MCTS are pure search;
   the evaluator and the neural network are pure knowledge. Almost every
   module improves exactly one of the two — that separation is also why the
   code is testable.

## RL lens: chess as a reinforcement-learning problem

The last rung, self-play RL, is where the course is heading. It helps to
name its parts now, because every module adds one of them:

- the **agent** is the engine, the thing that chooses
- the **environment** is the rules of chess plus the opponent
- the **state** is the board position; an **action** is a legal move
- the **reward** is 0 after every move, and +1 / 0 / -1 when the game ends

That last line is what makes chess hard to learn from: the only feedback is
one number at the end of a game of 40-odd moves. Every engine on the ladder
is a different answer to "how do you choose a move when the reward is that
far away?" Minimax looks ahead and trusts a hand-written score. MCTS plays
the game out to the end, many times. AlphaZero *learns* to score positions,
by playing itself. [RL_CONCEPTS.md](../RL_CONCEPTS.md) is the reference for
these words; you don't need to read it yet.

## The map

```
engine/     evaluation functions (hand-crafted; NN wrapper)
search/     minimax.py  mcts.py  puct.py  nn_mcts.py
net/        encoding.py (board→tensor, move→index)  model.py (ResNet)
training/   supervised learning: get_lichess, dataset, train, eval_supervised
selfplay/   Phase 4: self_play, replay_buffer, arena, train_loop, tictactoe_lab
cli/        play.py (play engines in the terminal), match.py (engine vs engine)
uci/        UCI protocol (connect to chess GUIs)
tests/      pytest suite — the course checkpoints
test/       legacy experiment scripts (tournaments, profiling, puzzles)
course/     you are here
```

## Do

1. **Install and verify** (see the [course README](../README.md#setup)), then:
   ```bash
   pytest -q            # your baseline: all green (a few course tests show as skipped)
   ```
2. **Play the ladder.** Play a quick game against each engine and *feel* the
   difference:
   ```bash
   python3 cli/play.py human            # vs the random mover
   python3 cli/play.py human-minimax    # vs alpha-beta
   python3 cli/play.py human-mcts       # vs MCTS
   ```
3. **Watch engines fight.** Every module uses engine-vs-engine games as
   evidence, and `cli/match.py` is the tool for it. Run the first two rungs
   of the ladder:
   ```bash
   python3 cli/match.py material random --games 10          # ~1 second
   python3 cli/match.py minimax:depth=2 material --games 10  # ~20 seconds
   ```
   **Expected:** the stronger engine scores around 85-95%. Look at the
   last lines: the score, a *95% interval*, and the Elo difference. With 10
   games the interval is wide (we measured 54% to 96% for 8.5/10). That
   width is the most important number in the course; Module 7 explains it.
4. **Break something on purpose.** Open `engine/evaluator.py`, make the queen
   worth 1 point, and run `pytest -q`. Watch which tests catch it, then
   revert. This is the feedback loop you'll live in.

## Check your understanding

Predict each answer before opening the details.

<details>
<summary><b>Q1.</b> You lower the queen's value to 1 point in <code>engine/evaluator.py</code> and run <code>pytest -q</code>. Which test fails first, and why that one?</summary>

`tests/test_minimax.py::test_captures_hanging_queen` — the engine no longer
thinks a free queen is worth capturing, so depth-3 search picks a different
move. Notice what *didn't* fail: the encoding and PUCT tests, because they
don't depend on the evaluator. Test failures localize faults — that's why
the modules keep search and knowledge separated.
</details>

<details>
<summary><b>Q2.</b> The mate-in-1 test passes for minimax, MCTS, <em>and</em> PUCT (AlphaZero's search, Module 5) with a fake network that knows nothing. What do those three engines share that makes a forced mate findable without any chess knowledge?</summary>

Terminal positions don't need evaluation — the game rules give the exact
value (mate = win). All three searches propagate exact terminal values back
to the root, so any search that can *reach* the mate proves it. Knowledge
only matters where the game hasn't ended yet. In RL terms: the reward is
known exactly at the end of the episode; everything else is an estimate.
</details>

<details>
<summary><b>Q3.</b> In a 40-move game the engine wins, how many non-zero rewards did it receive, and why does that make learning hard?</summary>

One, at the very end. The engine made about 40 choices and got a single
number back. Working out which of those choices earned the win is the
*credit assignment problem*, and it comes back in Module 6.
</details>

## Checkpoint

- [ ] `pytest -q` passes
- [ ] You beat the random engine (if you can't, the course will be humbling)
- [ ] You ran a `cli/match.py` match and can read its interval
- [ ] You can say, without looking: where does search live? where does
      knowledge live? what is the reward in chess?

Next: [Module 1 — Board & Evaluation](../module-01-board-and-evaluation/README.md)
