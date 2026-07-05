# Module 0 — Orientation

**Goal:** run every engine in the repo, understand the map, and establish the
habit the whole course depends on: *a checkpoint is a command, not a feeling.*

## Concepts

The project is a ladder of engines, each strictly stronger than the last:

```
random  →  minimax (+eval)  →  MCTS  →  NN + PUCT  →  self-play RL
Phase 0        Phase 1        Phase 2     Phase 3        Phase 4
```

Two ideas make this ladder work:

1. **Every rung is playable.** You can always *play* the current engine and
   *pit it against the previous one*. Progress is measured in game results.
2. **Search and knowledge are separable.** Minimax and MCTS are pure search;
   the evaluator and the neural network are pure knowledge. Almost every
   module improves exactly one of the two — that separation is also why the
   code is testable.

## The map

```
engine/     evaluation functions (hand-crafted; NN wrapper)
search/     minimax.py  mcts.py  puct.py  nn_mcts.py
net/        encoding.py (board→tensor, move→index)  model.py (ResNet)
training/   supervised learning: dataset, train, evaluate, filter_games
selfplay/   Phase 4: self_play, replay_buffer, arena, train_loop
cli/        play against engines in the terminal
uci/        UCI protocol (connect to chess GUIs)
tests/      pytest suite — the course checkpoints
test/       legacy experiment scripts (tournaments, profiling, puzzles)
course/     you are here
```

## Do

1. **Install and verify** (see the [course README](../README.md#setup)), then:
   ```bash
   pytest -q            # your baseline: all green
   ```
2. **Play the ladder.** Play a quick game against each engine and *feel* the
   difference:
   ```bash
   python3 cli/play.py human            # vs the random mover
   python3 cli/play.py human-minimax    # vs alpha-beta
   python3 cli/play.py human-mcts      # vs MCTS
   ```
3. **Watch engines fight.** Every module uses engine-vs-engine games as
   evidence. Try the tournament script:
   ```bash
   python3 test/tournament.py    # (long-running; Ctrl-C when convinced)
   ```
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
<summary><b>Q2.</b> The mate-in-1 test passes for minimax, MCTS, <em>and</em> PUCT-with-a-fake-network. What do those three engines share that makes a forced mate findable without any chess knowledge?</summary>

Terminal positions don't need evaluation — the game rules give the exact
value (mate = win). All three searches propagate exact terminal values back
to the root, so any search that can *reach* the mate proves it. Knowledge
only matters where the game hasn't ended yet.
</details>

## Checkpoint

- [ ] `pytest -q` passes
- [ ] You beat the random engine (if you can't, the course will be humbling)
- [ ] You can say, without looking: where does search live? where does
      knowledge live?

Next: [Module 1 — Board & Evaluation](../module-01-board-and-evaluation/README.md)
