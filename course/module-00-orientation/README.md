# Module 0: Orientation

> **Where you are:** the start. You'll see the finished sequence of engines before building any of it.
> **Time:** 1-2 hours.
> **You'll be able to:**
> - run every engine in the repository and play a match between two of them
> - say which parts of the code do *search* (looking ahead through possible moves) and which hold *knowledge* (judging how good a position is)
> - describe chess as a reinforcement-learning problem: agent, environment, state, action, reward
> - use a failing test to find which part of the code you broke

Throughout the course, every checkpoint is a command you can run and a
result you can check.

## Concepts

The project is a ladder of engines: a sequence of engines, each meant to
beat the one before it. The matches in this course are how you check that
claim at each step:

```
random  →  minimax (+eval)  →  MCTS  →  NN + PUCT  →  self-play RL
Phase 0        Phase 1        Phase 2     Phase 3        Phase 4
Module 1      Modules 1-2     Module 3   Modules 4-5     Module 6
```

In the diagram, *minimax* searches every line of play a few moves deep and
scores the positions at the end (eval is the evaluation function that gives
those scores). *MCTS* (Monte Carlo tree search) plays many games out to the
end and favours the moves that won most often. *NN + PUCT* is AlphaZero's
search (PUCT), guided by a neural network (NN). Later modules explain each
one in full.

Two ideas make this sequence work:

1. **Every engine is playable.** You can always play the current engine and
   play it against the previous one. Progress is measured in game results.
2. **Search and knowledge are separate.** Minimax and MCTS do only search.
   The evaluator and the neural network provide only knowledge. Almost every
   module improves one of the two and leaves the other unchanged. That
   separation is also why the code is easy to test.

## RL connection: chess as a reinforcement-learning problem

The last stage, self-play reinforcement learning (RL), is the goal of the
course. In RL, a program learns to choose actions from the rewards it
receives. It helps to name the parts now, because every module adds one of
them:

- the **agent** is the engine, the thing that chooses
- the **environment** is the rules of chess plus the opponent
- the **state** is the board position; an **action** is a legal move
- the **reward** is 0 after every move, and +1 / 0 / -1 when the game ends

That last line is what makes chess hard to learn from. The only feedback is
one number at the end of a game of about 40 moves. Every engine in the
sequence is a different answer to the question "how do you choose a move
when the reward comes so much later?" Minimax looks ahead and uses a
hand-written score. MCTS plays the game out to the end, many times.
AlphaZero (DeepMind's self-learning chess program) learns to score
positions by playing against itself. [RL_CONCEPTS.md](../RL_CONCEPTS.md) is
the reference for these words; you don't need to read it yet.

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
experiments/  legacy experiment scripts (tournaments, profiling, puzzles); not tests
course/     you are here
```

## Do

1. **Install and verify** (see the [course README](../README.md#setup)), then:
   ```bash
   pytest -q            # your baseline: all green (a few course tests show as skipped)
   ```
   **Expected:** all tests pass, and a few course tests show as skipped.
2. **Play each engine.** Play a quick game against each engine and notice
   how differently they play:
   ```bash
   python3 cli/play.py human            # vs the random mover
   python3 cli/play.py human-minimax    # vs alpha-beta
   python3 cli/play.py human-mcts       # vs MCTS
   ```
3. **Play engines against each other.** Every module uses games between
   engines as evidence, and `cli/match.py` is the tool for it. Run the first
   two steps of the sequence:
   ```bash
   python3 cli/match.py material random --games 10          # ~1 second
   python3 cli/match.py minimax:depth=2 material --games 10  # ~20 seconds
   ```
   **Expected:** the stronger engine scores around 85-95%. Look at the
   last lines: the score, a *95% interval* (the range the true score very
   likely lies in), and the Elo difference (Elo is the chess rating scale).
   With 10 games the interval is wide (a measured result of 8.5/10 gave 54% to 96%).
   You will use that width to judge every match in the course, and Module 7
   explains how it is calculated.
4. **Break something on purpose.** Open `engine/evaluator.py`, make the queen
   worth 1 point, and run `pytest -q`. Note which tests fail, then undo the
   change. You will use this cycle of change, test and fix throughout the
   course.

## Check your understanding

Predict each answer before opening the details.

<details>
<summary><b>Q1.</b> You lower the queen's value to 1 point in <code>engine/evaluator.py</code> and run <code>pytest -q</code>. Which test fails first, and why that one?</summary>

`tests/test_minimax.py::test_captures_hanging_queen`. The evaluator now
scores a free queen as worth almost nothing, so a depth-3 search (three
plies ahead; a ply is one move by one side) picks a different move. Notice
which tests did not fail: the tests for the move encoding (converting moves
to numbers for the network) and for PUCT. They don't depend on the
evaluator. A failing test points to the part of the code that is broken,
and that is why the modules keep search and knowledge separate.
</details>

<details>
<summary><b>Q2.</b> The mate-in-1 test passes for minimax, MCTS, <em>and</em> PUCT (AlphaZero's search, Module 5) with a stand-in network that has no chess knowledge. What do those three engines share that makes a forced mate findable without any chess knowledge?</summary>

A terminal position (one where the game has ended) needs no evaluation,
because the game rules give its exact value (mate = win). All three
searches pass exact terminal values back up the search tree to the root
(the current position). So any search that reaches the mate proves it.
Knowledge only matters where the game hasn't ended yet. In RL terms: the
reward is known exactly at the end of the episode (one complete game), and
everything else is an estimate.
</details>

<details>
<summary><b>Q3.</b> In a 40-move game the engine wins, how many non-zero rewards did it receive, and why does that make learning hard?</summary>

One, at the very end. The engine made about 40 choices and got a single
number back. Working out which of those choices earned the win is the
*credit assignment problem*, and it comes back in Module 6.
</details>

## Checkpoint

- [ ] `pytest -q` passes
- [ ] You beat the random engine
- [ ] You ran a `cli/match.py` match and can read its interval
- [ ] You can say, without looking: which parts of the code do search?
      which hold knowledge? what is the reward in chess?

Next: [Module 1: Board and Evaluation](../module-01-board-and-evaluation/README.md)
