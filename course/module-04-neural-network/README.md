# Module 4 — The Neural Network

> **Where you are:** the knowledge half of AlphaZero. Modules 2-3 left a hole where "how good is this position?" and "which moves are worth looking at?" should be. This network fills it.
> **Time:** two sessions. Part A (representation) is about 3 hours. Part B (data and training) is about 1 hour of your time; the training runs in the background (about 6 minutes for the quick run and 45 for the full month, on an 8-core laptop CPU).
> **You'll be able to:**
> - explain how a position becomes a tensor and a move becomes one of 4,672 indices
> - train a policy-value network on human games and judge the result against baselines, not just a loss curve
> - spot overfitting in your own run, and say which head it hit and why
> - explain why training on human games is *imitation*, not reinforcement learning, and what that limits

Requires torch: `pip install -e ".[train]"`.

**Pace yourself — this is two sessions, not one.**
**Part A** (representation: encoding + architecture, exercises 1-3) needs
no training; only exercise 2 needs the data from Part B's step 4, a 2-3
minute download. **Part B** (data + training, exercises 4-6) needs a
17 MB download and background compute. Do Part A, take a break, then start
Part B's quick run and read while it trains.

## The four ML ideas this module uses

You don't need a machine-learning course first. These four ideas are enough:

- **Loss function.** One number that says how wrong the network's outputs
  are on some examples. Training is "make this number smaller".
- **Gradient descent.** For every weight, work out which direction would
  make the loss a little smaller (the *gradient*; PyTorch computes it for
  you with `loss.backward()`), then move every weight a small step that way
  (`optimizer.step()`). Repeat millions of times. The step size is the
  *learning rate* (`--lr`). One pass over the whole training set is an *epoch*.
- **Two kinds of loss, one per head.** The policy head picks one move out
  of many, so it uses **cross-entropy**: minus the log of the probability it
  gave the move that was actually played. A network that gave that move
  probability 1 scores 0; uniform guessing over 30 moves scores about 3.4.
  The value head predicts a number in [-1, 1], so it uses **mean squared
  error (MSE)**: the average of (prediction - actual result)². Always
  guessing 0 on decisive games scores 1.0. `train.py` adds the two losses
  together.
- **Train/validation split and overfitting.** The network trains on some
  games and is scored on others it never sees (the *validation* set; this
  repo holds out whole games, see Module 8 case study 7). If the training
  loss keeps falling while the validation loss rises, the network is
  memorising its training games instead of learning chess. That's
  *overfitting*, and you will see it in Part B.

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

## Inside the network, one position at a time

If you've never looked inside a neural network, follow one position through
this one. The shapes below are measured; run the snippet at the end to see
them yourself.

1. **Input: the board as 20 stacked 8×8 grids**, shape `(1, 20, 8, 8)`:
   one grid per piece type and colour (12), castling rights (4), side to
   move, two move counters, en passant. The leading 1 is the *batch*: how
   many positions go through at once.
2. **Convolution: look at 3×3 patches.** A convolution slides a small 3×3
   window over every square and computes a weighted sum of what it sees
   there (all 20 grids at once). Each set of weights is one *filter*, a
   pattern detector, for example "my knight attacks this square". The first
   layer has 128 filters, so the output is 128 grids: `(1, 128, 8, 8)`. The
   same filter is used on every square, which is why convolutions suit
   boards and images: a pattern means the same thing anywhere.
3. **Residual blocks: refine, don't replace.** Each of the 4 blocks applies
   two more convolutions and then *adds its input back* (`out + identity` in
   `ResidualBlock.forward`). So each block only has to learn a correction to
   what the previous one saw, which makes deep stacks much easier to train.
   The shape stays `(1, 128, 8, 8)`. After a few blocks, each square's 128
   numbers summarise a growing neighbourhood of the board.
4. **Two heads read the same features.** The *policy head* squeezes the 128
   grids into 2, flattens them to 128 numbers and maps those to **4,672
   logits**, one raw score per possible move: `(1, 4672)`. The *value head*
   squeezes them into 1 grid, then to one number, and `tanh` bends it into
   [-1, 1]: `(1, 1)`.
5. **Logits → probabilities.** *Softmax* turns raw scores into
   probabilities: exponentiate each, divide by the total. A logit 1 higher
   means about 2.7 times as likely.
6. **Masking.** Only about 20-40 of the 4,672 moves are legal. Before the
   softmax, set every illegal move's logit to a huge negative number, so its
   probability becomes 0. That's `legal_moves_mask` in `net/encoding.py`.

**Measured on the starting position** with the one-month network from
exercise 6: without masking, the network already put 96.7% of its
probability on the 20 legal moves (it has learned legality); after masking
its top three were Nf3 27%, e3 20%, e4 13%, and its value was +0.02, about
even. Your own network will give different numbers.

```python
import chess, torch
from net.model import load_model
from net.encoding import board_to_tensor, legal_moves_mask
model, _ = load_model("checkpoints/best_model.pt"); model.eval()
x = torch.from_numpy(board_to_tensor(chess.Board())).unsqueeze(0)
with torch.no_grad():
    logits, value = model(x)
mask = torch.from_numpy(legal_moves_mask(chess.Board()))
probs = torch.softmax(logits[0].masked_fill(mask == 0, -1e9), dim=0)
print(x.shape, logits.shape, value.item(), probs.max().item())
```

## RL lens: imitation is not reinforcement learning

Training on human games is **supervised learning**: for each position the
data says which move a strong player chose (the policy target) and how the
game ended (the value target), and the network learns to reproduce them.
Copying an expert's actions is called *imitation learning* or *behaviour
cloning*. There is no trial and error and no reward the network earned
itself, so it is not RL.

That sets its **ceiling**: a perfect score means playing exactly like the
humans in the data, mistakes and all. It can never discover a move nobody in
the data played. Getting past the teacher needs a signal that isn't the
teacher, and Module 6's self-play provides one.

Why do it, then? Two reasons. It tests the whole pipeline (data, tensors,
loss, checkpoints) on a fixed dataset, so when self-play misbehaves you know
the plumbing isn't the cause. And it hands RL a network that already knows
something. AlphaGo started this way; AlphaZero started from random weights.

Two more ideas you'll see in the results:

- The network is a **function approximator** for the policy π(a | s) and the
  value V(s). There are far too many chess positions to store an answer for
  each, so it must *generalise* to positions it has never seen. Validation
  accuracy measures exactly that.
- The two heads get very different amounts of signal. Every position gives
  the policy head its own label (the move played). The value head gets the
  same label for every position in a game: the result. A thousand games are
  70,000 policy labels but only a thousand independent value labels. Predict
  which head overfits first before you run Part B.

## Read

- `net/encoding.py` (all of it, especially the promotion encoding)
- `net/model.py` (ResidualBlock → heads → load_model)
- `training/dataset.py` and `training/train.py`

## Do — Part A: the representation (no data needed)

1. **Trace one example by hand.** Take 1.e4: predict which tensor planes
   change and which policy index is the target (the plane layout is
   commented in `board_to_tensor`). Then check:
   ```python
   import chess, numpy as np
   from net.encoding import board_to_tensor, move_to_index
   before = board_to_tensor(chess.Board())
   board = chess.Board(); board.push_san("e4")
   print(sorted({int(p) for p in np.argwhere(before != board_to_tensor(board))[:, 0]}))
   print(move_to_index(chess.Move.from_uci("e2e4")))
   ```
   **Expected:** planes `[0, 16, 19]` and index `796`. Plane 0 is White's
   pawns (e2 empties, e4 fills); plane 16 is side to move (all ones become
   all zeros: Black to move); plane 19 marks the en-passant square, e3.
   If you predicted only plane 0, you've just found the two "bookkeeping"
   planes the network needs to know whose turn it is and what's legal.
2. **Exercise: find the bug's blast radius.** The old encoder gave every
   queen promotion an index ≥ 4672, and the old guard silently skipped
   those moves. How much training data did that throw away? Run Part B's
   step 4 first (2-3 minutes) so you have `data/filtered.pgn`, then count:
   - **With git history** (a full `git clone`): the fix is commit `9b6cb6c`.
     Save the encoder from before it with
     `git show 9b6cb6c^:net/encoding.py > old_encoding.py`, load its
     `move_to_index`, and count the moves whose index is ≥ 4672.
   - **Without history** (a downloaded zip): count the promotions instead,
     with `move.promotion` from python-chess. That's the set the bug hit.

   **Expected (measured on `data/filtered.pgn`):** with history, 678 of
   349,814 moves were dropped, **0.19%**; without it you'll count 683
   promotions (the old encoder lost all but 5 of them). That sounds harmless
   until you notice it's every one of the 660 queen promotions plus most
   under-promotions: the network could never have learned to promote at
   all. A small fraction of the data can be all
   of a skill. That's the cost of an untested encoder.
3. **Exercise: augmentation.** Chess positions can be mirrored horizontally
   (a-file ↔ h-file), doubling your training data for free. Implement
   `mirror_board_tensor` and `mirror_move` in `net/encoding.py`; the
   contract and the equivalence you must satisfy are in
   `tests/course/test_module04_augmentation.py` — un-skip it and make it
   pass. (Castling rights are the trap — the test sidesteps them and the
   stretch goal is handling them; measure the effect on val accuracy after
   Part B.)

## Do — Part B: data and training

4. **Get the data** (2-3 minutes, 17 MB download):
   ```bash
   python3 training/get_lichess.py
   ```
   This downloads January 2013 from the [Lichess database](https://database.lichess.org/)
   (free, CC0), decompresses it, keeps games where both players are rated
   1800+ with at least 3 minutes on the clock, and writes `data/filtered.pgn`.
   **Expected:** `Games kept: 4,794 (4.0%)` out of 121,332. Filters matter:
   the stricter "2000+ and 10 minutes" keeps 3 games from this month,
   because few players were rated that high in 2013. Recent months are many
   GB; the script refuses them unless you pass `--allow-large`.

5. **Quick run first, then judge it properly** (about 6 minutes of training
   on an 8-core laptop CPU, faster with a GPU):
   ```bash
   python3 training/train.py --pgn data/filtered.pgn --max-games 1000 --epochs 2
   ```
   The checkpoint lands in `checkpoints/best_model.pt` (Modules 5 and 6 use
   it); `tensorboard --logdir runs/` shows the curves if you want them.
   **Expected (measured):** 1,000 games give 67,613 positions, split by game
   into 61,097 for training and 6,516 for validation. Each epoch takes about
   3 minutes. Validation accuracy goes from about 7% to about 9.4%.

   Is 9.4% good? A bare number can't say. Before you look at the answer,
   write down your prediction from the RL lens: which head overfits? Then:
   ```bash
   python3 training/eval_supervised.py checkpoints/best_model.pt \
       --pgn data/filtered.pgn --max-games 1000
   ```
   It rebuilds the same validation games and compares the network with two
   baselines that know nothing about the position. **Expected (measured):**

   | | policy: picks the played move | |
   |---|---|---|
   | random legal move | 5.6% | |
   | most common legal move in training | 9.4% | no board understanding at all |
   | network, best legal move | **13.8%** | beats both: it learned something |
   | network, move in its top 5 | 38.5% | |

   The policy head works. The **value head doesn't**: its error (MSE 1.23) is
   *worse* than always guessing "even" (0.95), it picks the winner of
   decisive games 52% of the time (a coin flip), and it rates "White a queen
   down" as good for White. Its training loss fell while its validation loss
   rose: it memorised its 889 training games, one result each. That's the
   overfitting the RL lens predicted. The smell tests (fixed positions) show
   the raw network missing a free queen and a mate in one; expect that,
   because search is what Module 5 adds.

   Two more things the report shows: `train.py` reports 9.3%, not 13.8%,
   because its accuracy lets the network "choose" illegal moves, and 36% of
   its top picks are illegal. The engine only ever picks legal moves.

6. **The real run: more games, not more epochs.** The fix for the value
   head is more independent results, so train on the whole month:
   ```bash
   python3 training/train.py --pgn data/filtered.pgn --epochs 3
   python3 training/eval_supervised.py checkpoints/best_model.pt --pgn data/filtered.pgn
   ```
   About 320K positions (4,295 training games, 477 held out). Training time
   grows with the number of positions: expect about 15 minutes per epoch on
   an 8-core laptop CPU (we measured 19 with other jobs sharing the CPU).
   `train.py` also saves `model_epoch_1.pt`, `_2` and `_3`; evaluate epoch 1
   as well as the final model.
   **Expected (measured):**

   | | after epoch 1 | after epoch 3 ("best") |
   |---|---|---|
   | policy: best legal move (baselines 5.9% / 9.3%) | 22.1% | **27.7%** |
   | policy: in top 5 | 51.6% | 60.2% |
   | value error (always "even": 0.923) | **0.912** | 1.034 |
   | value: right winner, decisive games | 56% | 57% |
   | value: correlation with material | **+0.44** | +0.35 |
   | takes the free queen (smell test) | no | yes |

   More games helped both heads: after one epoch the value head finally beats
   "always even" (just) and has clearly learned that material matters. Then
   more epochs trade one head for the other. The policy keeps improving,
   while the value head goes back to memorising: its validation loss rose
   every epoch (0.91, 0.96, 1.03) as its training loss fell. `train.py` still
   labels epoch 3 "best", because it picks by the *sum* of the two losses
   and the policy's gains hide the value's losses. Read each head's numbers
   separately. The value head needs far more games than one month gives
   (or a smaller weight on its loss); it is the weak part of the network you
   carry into Module 5.

   **What counts as success.** Compare against the baselines, not a target
   number. As a rule of thumb: the policy's best-legal-move figure at least
   twice the "most common move" baseline, and the value head's error below
   "always even" with a correlation with material above +0.3. The one-month
   run above meets both at epoch 1 and the policy one at epoch 3. The original project plan's targets (val accuracy > 35%, val
   loss < 2.5) were never measured on this code; treat them as a stretch for
   much more data, not as the bar for this module. Watch the train/val gap
   at every size: more data beats more epochs.

   **More data, when you want it.** Each extra month is one command, and PGN
   files can simply be concatenated:
   ```bash
   python3 training/get_lichess.py --month 2013-02 --output data/filtered_2013-02.pgn
   cat data/filtered.pgn data/filtered_2013-02.pgn > data/filtered_2months.pgn
   ```
   Months grow quickly (2013-12 is several times January's size), so check
   `--help` and your disk before going far.

   **About the measured numbers.** Your figures will differ by a point or
   two: the network starts from random weights and batches are shuffled. The
   correlation with material is a plain (Pearson) correlation between the
   value head's output and the side to move's material balance, over all
   held-out positions; `training/eval_supervised.py` documents each figure
   at the top of the file.

7. **Smell-test against search.**
   `python3 training/evaluate.py checkpoints/best_model.pt --games 10` plays
   the raw policy (no search) against minimax depth 3 and rollout MCTS and
   prints the results.
   **Expected (measured with `--games 4`, about 30 minutes):** the
   one-month network lost all 4 games to minimax and scored 0.5/4 against
   MCTS. Expect the raw policy to lose: it knows something about moves, but
   it can't look ahead. That gap is exactly what Module 5 closes.

## Check your understanding

<details>
<summary><b>Q1.</b> Predict: what does <code>move_to_index(chess.Move.from_uci("e2e4"))</code> return? Derive it, then run it.</summary>

`e2` is square 12 (rank 1 × 8 + file 4), `e4` is square 28, and ordinary
moves encode as `from * 64 + to` → 12 × 64 + 28 = **796**. If you got the
squares' numbering backwards (file × 8 + rank), you've just met the exact
confusion the encoding tests exist for.
</details>

<details>
<summary><b>Q2.</b> The policy head outputs 4672 logits but a position averages ~35 legal moves. Does the network waste effort on the ~4637 illegal ones?</summary>

In training, yes, a little. At play time the engine masks illegal moves
before the softmax, so whatever the network thinks of them never matters.
But `train.py`'s loss (`nn.CrossEntropyLoss` on the raw logits) is over all
4,672 outputs, unmasked: each step pushes the played move up and every other
output down, illegal ones included. So the network has to learn legality as a
side effect, and early on it hasn't. After the quick run its favourite
*unmasked* output was illegal in 36% of positions, which is why train.py's
accuracy (9.3%) is lower than the engine's (13.8%). Masking the logits inside
the loss as well would remove that wasted effort: a good small experiment.
</details>

<details>
<summary><b>Q3.</b> Both heads share one trunk. Name a concrete reason that's better than two separate networks — and one risk.</summary>

Better: the features useful for "who's winning" (king safety, material,
mobility) are the same features useful for "what's the best move," so each
head's gradient improves the other's inputs — shared trunks act as mutual
regularization (and halve inference cost, which Module 5 cares about).
Risk: the two losses can compete for capacity; if one loss dominates in
scale, the trunk optimizes for it at the other's expense — which is why the
reason to watch *both* curves. `train.py` weights them equally (a plain
sum); a common fix when one head misbehaves is to down-weight its loss.
</details>

<details>
<summary><b>Q4 (RL lens).</b> Your supervised network reaches 50% accuracy on grandmaster games. Could more data and a bigger network take it past grandmaster strength on their own? Why or why not?</summary>

No. Its target is "the move the human played", so a perfect score means
playing exactly like the humans in the data, including their mistakes. It
has no way to find out that a different move would have been better. That
needs a signal that isn't the teacher: trying moves and seeing results,
which is RL (Module 6). Search helps too (Module 5): it can find moves
the network wouldn't.
</details>

<details>
<summary><b>Q5.</b> At 1,000 games the policy head beat its baselines but the value head didn't. Give the reason in one sentence, and the fix.</summary>

Each game gives every one of its ~70 positions its own policy label, but
only one value label (the result) shared by all of them, so 1,000 games are
just 1,000 independent value examples and the value head memorises them.
The fix is more games, not more epochs (exercise 6).
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
      loss, and the `eval_supervised.py` table — numbers you measured, not
      the module's
- [ ] One sentence on which head overfit at 1,000 games, and why

Previous: [Module 3 — Monte Carlo Tree Search](../module-03-mcts/README.md) · Next: [Module 5 — PUCT](../module-05-puct/README.md)
