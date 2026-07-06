# Module 4 — The Neural Network

**Goal:** build the knowledge half of AlphaZero — a network that looks at a
position and outputs both *which moves look right* (policy) and *who's
winning* (value) — and train it on expert games.

Requires torch: `pip install -e ".[train]"`.

**Pace yourself — this is two sessions, not one.**
**Part A** (representation: encoding + architecture, exercises 1, 4, 5) needs
no data and no training. **Part B** (data + training, exercises 2, 3) involves
a download and hours of compute. Do Part A, take a break, then start Part B's
download while you sleep.

> **ML prerequisites, honestly stated.** This module *uses* but does not
> *teach*: what a loss function and gradient descent are, cross-entropy vs
> MSE, train/validation splits, and overfitting. The docstrings explain
> everything chess-specific (the encoding, the two heads, why the losses are
> combined). If those four ML terms are fuzzy, spend an hour with any
> introductory resource first (3Blue1Brown's neural network series + the
> PyTorch "60-minute blitz" cover all of it) — the module will land far
> better.

## Concepts

**Encoding: the board as an image.** `net/encoding.py` turns a position into
a (20, 8, 8) tensor — 12 piece planes, castling rights, side to move,
counters, en passant. A chessboard is spatially structured, which is why the
network is convolutional.

**Encoding: moves as indices.** The policy head outputs 4672 numbers, one
per possible move. `move_to_index()` / `index_to_move()` define that mapping.
Read the **"LESSON LEARNED"** section in `net/encoding.py` carefully — this
mapping shipped with a bug that made *pawn promotion literally unlearnable*
(queen promotions overflowed the vector and were silently masked out), and
no one noticed because the tests only round-tripped ordinary opening moves.
The fix and the regression tests (`tests/test_encoding.py`) are a case study
in **testing the edges of your data representation before training on it**.

**The architecture** (`net/model.py`): a ResNet trunk (4 blocks, 128
channels; 1,824,970 parameters — don't trust that number either, run
`create_model().num_parameters()` yourself. An earlier version of this
course claimed "~830K", copied from a planning doc nobody had checked
against the code: config that isn't derived drifts, including in
documentation) with two heads:
- policy head → 4672 logits (mask illegal moves, softmax)
- value head → tanh scalar in [-1, 1]
One shared trunk, two tasks — the value gradient and policy gradient
regularize each other. That's the AlphaZero design, just small.

**Supervised training first** (`training/train.py`): before any self-play,
imitate strong humans — cross-entropy on the expert's move, MSE on the game
outcome. This validates the whole pipeline (data → tensors → loss ↓ →
checkpoints) with a *stationary* dataset, so when self-play RL gets weird in
Module 6, you'll know the plumbing isn't the problem.

**Checkpoints must be self-describing.** Second case study: loading code
once hardcoded a *different* architecture than the training code saved.
`net/model.py::load_model()` now infers the architecture from the weight
shapes themselves. Config that can drift *will* drift; derive it.

## Read

- `net/encoding.py` (all of it, especially the promotion encoding)
- `net/model.py` (ResidualBlock → heads → load_model)
- `training/dataset.py` and `training/train.py`

## Do — Part A: the representation (no data needed)

1. **Trace one example by hand.** Take 1.e4: write down which tensor plane
   changes and which policy index is the target. Verify with three lines of
   Python.
2. **Exercise: find the bug's blast radius.** Check out the git history from
   before the encoding fix and count what fraction of moves in your PGN set
   would have been silently dropped (`git log -- net/encoding.py`). Write
   the number down — that's the cost of an untested encoder.
3. **Exercise: augmentation.** Chess positions can be mirrored horizontally
   (a-file ↔ h-file), doubling your training data for free. Implement
   `mirror_board_tensor` and `mirror_move` in `net/encoding.py`; the
   contract and the equivalence you must satisfy are in
   `tests/course/test_module04_augmentation.py` — un-skip it and make it
   pass. (Castling rights are the trap — the test sidesteps them and the
   stretch goal is handling them; measure the effect on val accuracy after
   Part B.)

## Do — Part B: data and training

**Data logistics (read before downloading anything).** The
[Lichess database](https://database.lichess.org/) publishes monthly dumps as
zstandard-compressed PGN. Recent months are **~30 GB compressed** — do NOT
grab one of those. Two sane options:

- **Small and instant:** the earliest months are tiny —
  `lichess_db_standard_rated_2013-01.pgn.zst` is ~17 MB (~120K games).
  Perfect for a first end-to-end run.
- **Curated strength:** the community "Lichess Elite" databases (games by
  2400+ players) if you want maximum expert signal per megabyte.

Decompress before filtering (`training/filter_games.py` reads plain PGN):
```bash
zstd -d lichess_db_standard_rated_2013-01.pgn.zst   # or: python3 -m zstandard -d ...
python3 training/filter_games.py \
    --input lichess_db_standard_rated_2013-01.pgn \
    --output data/filtered.pgn --min-elo 2000
```

**Time expectations** (estimates — measure your own and write them down,
that's Module 7 discipline arriving early): loading + encoding 10K games is
minutes; 10 epochs on ~700K positions is roughly **2–6 hours on a laptop
CPU**, meaningfully faster on Apple Silicon (MPS) or any GPU. Start with
`--max-games 1000 --epochs 2` to verify the pipeline end-to-end in ~15
minutes before committing to the real run.

4. **Train for real:**
   ```bash
   python3 training/train.py --pgn data/filtered.pgn --max-games 10000 --epochs 10
   ```
   Watch tensorboard. Success criteria from the original plan: val policy
   accuracy > 35%, val loss < 2.5. **Watch the train/val gap**: if train
   loss keeps falling while val loss rises, you're overfitting — more data
   beats more epochs.
5. **Smell-test the trained model.** `training/evaluate.py <ckpt>` runs the
   baseline matches (vs minimax depth 3 and MCTS) and prints the network's
   top move in test positions; check the opening choice is a normal
   developing move, and expect the raw policy to lose to minimax — search
   is exactly what Module 5 adds.

## Check your understanding

<details>
<summary><b>Q1.</b> Predict: what does <code>move_to_index(chess.Move.from_uci("e2e4"))</code> return? Derive it, then run it.</summary>

`e2` is square 12 (rank 1 × 8 + file 4), `e4` is square 28, and ordinary
moves encode as `from * 64 + to` → 12 × 64 + 28 = **796**. If you got the
squares' numbering backwards (file × 8 + rank), you've just met the exact
confusion the encoding tests exist for.
</details>

<details>
<summary><b>Q2.</b> The policy head outputs 4672 logits but a position averages ~35 legal moves. Why doesn't the network waste all its capacity on the ~4637 illegal ones?</summary>

Because illegal moves are masked *before* the softmax (large negative
logit → probability ~0) both at inference and implicitly in training (the
target never puts mass there). The network never receives gradient pressure
to rank illegal moves correctly — capacity follows gradient, and gradient
follows the mask.
</details>

<details>
<summary><b>Q3.</b> Both heads share one trunk. Name a concrete reason that's better than two separate networks — and one risk.</summary>

Better: the features useful for "who's winning" (king safety, material,
mobility) are the same features useful for "what's the best move," so each
head's gradient improves the other's inputs — shared trunks act as mutual
regularization (and halve inference cost, which Module 5 cares about).
Risk: the two losses can compete for capacity; if one loss dominates in
scale, the trunk optimizes for it at the other's expense — which is why the
combined loss weights them and why you watch *both* curves in tensorboard.
</details>

## Checkpoint

```bash
pytest tests/test_encoding.py tests/test_model.py -q         # still green
pytest tests/course/test_module04_augmentation.py -q         # un-skipped and passing
```

- [ ] Both green
- [ ] A trained checkpoint exists and `load_model()` loads it *without you
      telling it the architecture*
- [ ] A short training log: dataset size, wall-clock time, final train/val
      loss and val accuracy — numbers you measured, not the module's
      estimates

Next: [Module 5 — PUCT](../module-05-puct/README.md)
