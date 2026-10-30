# Unit 8: Measuring honestly

**You'll be able to:**
- tell a **target** from a **result**, and an estimate from a measurement
- measure a model against baselines that make the numbers meaningful
- avoid two classic traps: deterministic engines replaying the same game, and too few games

**Before this:** [Unit 7](07_training.md). **Next:** [Unit 9, NN + search](09_nn_mcts.md). **Time:** ~60 min.

## The idea

It's easy to write "✅ policy accuracy > 35%" in a plan and later read it as a fact. This
project did exactly that: its status documents carried success criteria with ticks for months,
and Elo ratings like "~1400–1600" that were never measured. Learning to measure is as much a
part of building an engine as writing the search.

Three habits make a number trustworthy:

1. **Compare with a baseline.** "13% top-1 accuracy" means nothing on its own. Picking a random
   legal move gets about 6%, so 13% is "about twice random", which is a very different message.
2. **Make the games different.** The engines here are deterministic: the same position always
   gets the same move. Twenty games between them from the starting position are *one game
   played twenty times*. Start each pair of games from a different random opening and swap
   colours.
3. **Respect sample size.** With 10 games, a 40% score has a 95% confidence interval of
   roughly 17%–69%. Small matches can show that something is *broken*. They can't show it's
   *slightly better*.

## In the code

| Where | What to look for |
|---|---|
| `notebooks/01_checkpoint_reality_check.py` | The whole measurement, cell by cell: training curves, parameters, data, move agreement against a random baseline, matches from random openings |
| `checkpoints/model_epoch_N.pt` | Each holds `train_loss`, `val_loss` and `val_accuracy`. The training history is inside the files |
| `training/evaluate.py`, `FunctionPlayer` | Wraps any `board → move` function so it can play in a tournament |
| `test/tournament.py`, `run_tournament` | The older tournament harness used for the minimax and MCTS phases |

## Try it

Open the notebook. It's a [marimo](https://marimo.io) notebook, stored as a plain `.py` file:

```bash
marimo edit notebooks/01_checkpoint_reality_check.py --host 0.0.0.0 --port 8080
```

Or run every cell without the browser, to check it works:

```bash
python notebooks/01_checkpoint_reality_check.py
```

These are the measured results for the project's trained checkpoint (2 residual blocks,
64 channels, 779,402 parameters, trained 5 epochs on 411 games):

| Measurement | Network | Baseline |
|---|---|---|
| Validation top-1 accuracy (during training) | 3.2% | Target was 35% |
| Human's move ranked first (3,277 positions) | 13.1% | Random legal move: 6.1% |
| Human's move in top 3 | 26.3% | |
| Value sign matches game result | 54% | Coin flip: 50% |
| vs random mover (10 games) | 0 W, 8 D, 2 L | |
| vs material-only greedy | 0 W, 2 D, 8 L | |
| vs minimax depth 1 / depth 2 | 0 W, 0 D, 10 L each | |

## Exercises

**1. Predict before you look at the table:** given unit 7 (a loss that barely improved on
"knowing nothing" and 411 blitz games from ~1500-rated players), what score would you
expect the network to get against a *random* mover? Commit to a number.

<details><summary>Answer</summary>

It scored 40% (no wins, eight draws, two losses). Many people expect "surely better than
random". But a policy-only player with no search, trained on so little data, can't
convert advantages into checkmate. So most games drift into draws, and it occasionally
blunders into a loss.
</details>

**2.** The move-agreement numbers are measured on games the network trained on. Is 13.1% an
upper or a lower bound on what it would score on new games? Why?

<details><summary>Answer</summary>

An **upper bound**. A model does at least as well on data it has seen as on new data. To
measure properly, hold out whole games (unit 7, exercise 3) and never train on them.
</details>

**3.** In the notebook, the match code plays every opening twice with colours swapped. What
bias does the colour swap remove?

<details><summary>Answer</summary>

The advantage of moving first, and any opening that happens to favour one side. Each opening
is played once from each side, so those effects cancel out of the net's score.
</details>

**4. Stretch:** the old status file said the minimax engine was "~1200–1600 Elo". Design an
experiment that would actually measure it. What would you play it against?

<details><summary>Answer</summary>

Play many games (hundreds) from varied openings against opponents with *known* ratings, such as
Stockfish at fixed, calibrated skill levels, or bots on a rating server. Then fit an Elo from the
results. Elo is always relative to a pool of opponents, so a number with no reference pool is
an estimate at best.
</details>

## Check yourself

1. Why is a baseline needed to interpret an accuracy figure?
2. Why do deterministic engines need varied openings in a match?
3. Roughly how wide is the uncertainty on a 10-game match score?

<details><summary>Answers</summary>

1. Without one you can't tell good from trivial. Random legal moves already score ~6% top-1.
2. Otherwise every game between the same two engines is identical.
3. Very wide, about ±25 percentage points at 95% confidence.
</details>
