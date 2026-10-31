# Module 4: The Neural Network

> **Where you are:** this module builds the part of AlphaZero that holds chess knowledge. Modules 2-3 had nothing to answer "how good is this position?" and "which moves are worth looking at?". This network answers both questions.
> **Time:** two sessions. Part A (representation) is about 3 hours. Part B (data and training) is about 1 hour of your time; the training runs in the background (about 6 minutes for the quick run and 45 for the full month, on an 8-core laptop CPU).
> **You'll be able to:**
> - explain how a position becomes a tensor (a multi-dimensional array of numbers) and a move becomes one of 4,672 indices
> - train a policy-value network (one network that suggests moves and rates positions) on human games, and judge the result against simple reference players (baselines) as well as by its error during training
> - spot overfitting (memorising the training games instead of learning chess) in your own run, and say which of the network's two outputs it affected and why
> - explain why copying human moves (imitation learning) is different from learning by trial and error (reinforcement learning), and what that limits

Requires torch (PyTorch, the deep-learning library the code uses): `pip install -e ".[train]"`.

- **Part A** (representation: encoding, architecture and a first network, exercises 1-4) needs no training. Only exercise 2 needs data, from Part B's exercise 5, which is a 2-3 minute download.
- **Part B** (data and training, exercises 5-8) needs a 17 MB download and background computation.

Do Part A, take a break, then start Part B's quick run and read while it trains.

## The four machine-learning ideas this module uses

You don't need a machine-learning (ML) course first. These four ideas are enough:

- **Loss function.** One number that says how wrong the network's outputs
  are on some examples. Training means making this number smaller.
- **Gradient descent.** For every weight, work out which direction would
  make the loss a little smaller (the *gradient*; PyTorch computes it for
  you with `loss.backward()`), then move every weight a small step that way
  (`optimizer.step()`). Repeat millions of times. The step size is the
  *learning rate* (`--lr`). One pass over the whole training set is an *epoch*.
- **Two kinds of loss, one per output.** The network has two outputs, called
  *heads*: the policy head gives a probability for every move, and the value
  head gives one number for how good the position is. The policy head's task
  is to pick one move out of many, so it uses *cross-entropy*: minus the
  logarithm of the probability it gave the move that was played. The logarithm
  turns a probability of 1 into 0 and makes small probabilities large, so the
  loss grows fast when the network gives the right move very little weight.
  A network that gave the played move probability 1 scores 0. Uniform guessing
  over 30 moves gives each move probability 1/30, and minus the logarithm of
  1/30 is about 3.4.
  The value head predicts a number in [-1, 1], so it uses *mean squared
  error (MSE)*: the average of (prediction - actual result)². Always
  guessing 0 on decisive games scores 1.0. `train.py` adds the two losses
  together.
- **Train/validation split and overfitting.** The network trains on some
  games and is scored on others it never sees (the *validation* set; this
  repository holds out whole games, see Module 8 case study 7). If the training
  loss keeps falling while the validation loss rises, the network is
  memorising its training games instead of learning chess. That is
  *overfitting*, and you will see it in Part B.

## Concepts

**Encoding: the board as an image.** `net/encoding.py` turns a position into
a (20, 8, 8) tensor: 20 stacked 8×8 grids of numbers, called *planes*. There
are 12 piece planes, plus planes for castling rights, side to move, move
counters and en passant. A chessboard has a spatial layout, like an image.
That is why the network uses convolutions (explained below), a layer type
designed for images.

**Encoding: moves as indices.** The policy head outputs 4,672 numbers, one
per possible move. `move_to_index()` / `index_to_move()` define that mapping.
Read the "LESSON LEARNED" section in `net/encoding.py` carefully. This
mapping was released with a bug that made pawn promotion impossible to learn.
Queen promotions got indices beyond the end of the 4,672-number output, and
the code dropped them without any error or warning. No one noticed, because
the tests only converted ordinary opening moves to indices and back. The fix
and the regression tests (`tests/test_encoding.py`, tests that stop a fixed bug from returning) are a case study in
testing the unusual cases of your data representation before you train on it.

**The architecture** (`net/model.py`). The network has a shared body, called
the *trunk*, and two heads. The trunk is a ResNet (residual network, explained
in step 3 below) of 4 blocks, each 128 channels wide (128 grids per layer).
The whole network has 1,824,970 parameters (learned weights).
Check that number yourself: run `create_model().num_parameters()`. An earlier version of this course said about 830,000, copied from
a planning document that nobody had checked against the code. Any setting
that is copied by hand instead of computed from the code can become wrong
over time, and documentation is no exception.

The two heads:
- policy head: 4,672 logits (raw scores, one per move). Illegal moves are
  masked out, then softmax turns the scores into probabilities (both steps are
  explained below).
- value head: a single number, passed through the tanh function so that it lies in [-1, 1].

Both heads use one shared trunk, so the trunk has to learn features (patterns
in the position, such as material or king safety) that are useful for both tasks. This acts as *regularisation*: it makes the trunk less
likely to memorise patterns that only help one task on the training data.
AlphaZero uses the same design at a much larger size.

**Supervised training first** (`training/train.py`). Supervised training means
learning from examples with known answers. Before any self-play (the engine
playing itself, Module 6), the network learns to imitate strong humans: cross-entropy on the expert's
move and MSE on the game outcome. This checks every stage of training
(data, tensors, falling loss, and checkpoints, the saved model files) on a *stationary* dataset,
one that does not change during training. If self-play training in Module 6
behaves unexpectedly, you will already know that these stages work.

**Checkpoints must be self-describing.** This is the second case study. The
loading code once had an architecture written directly into it, and that
architecture was different from the one the training code saved.
`net/model.py::load_model()` now works out the architecture from the shapes of
the saved weights. Lesson: if a setting can be computed from the data, compute
it. A copy kept by hand will eventually disagree with the original.

## Inside the network, one position at a time

Follow one position through the network. The shapes below are measured;
run the snippet at the end to see them yourself.

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
3. **Residual blocks: refine instead of replace.** Each of the 4 blocks applies
   two more convolutions and then *adds its input back* (`out + identity` in
   `ResidualBlock.forward`). So each block only has to learn a correction to
   what the previous one saw, which makes deep stacks much easier to train.
   The shape stays `(1, 128, 8, 8)`. After a few blocks, each square's 128
   numbers summarise a growing neighbourhood of the board.
4. **Two heads read the same features.** The *policy head* reduces the 128
   grids to 2, flattens them to 128 numbers and maps those to 4,672
   *logits*, one raw score per possible move: `(1, 4672)`. The *value head*
   reduces them to 1 grid, then to one number. The `tanh` function then maps
   that number into the range [-1, 1]: `(1, 1)`.
5. **Logits to probabilities.** *Softmax* turns raw scores into
   probabilities: exponentiate each, divide by the total. A logit 1 higher
   means about 2.7 times as likely, because e¹ ≈ 2.7.
6. **Masking.** Only about 20-40 of the 4,672 moves are legal. Before the
   softmax, set every illegal move's logit to a huge negative number, so its
   probability becomes 0. That's `legal_moves_mask` in `net/encoding.py`.

**Measured on the starting position** with the one-month network from
exercise 7: without masking, the network already put 96.7% of its
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

## RL connection: imitation learning and reinforcement learning

Training on human games is supervised learning: for each position the
data says which move a strong player chose (the policy target) and how the
game ended (the value target), and the network learns to reproduce them.
Copying an expert's actions is called *imitation learning* or *behaviour
cloning*. Reinforcement learning (RL) means learning from trial and error and
from rewards the learner earns itself ([RL concepts](../RL_CONCEPTS.md)).
Supervised training has neither, so it is not RL.

This sets an upper limit on its strength: a perfect score means playing
exactly like the humans in the data, mistakes and all. It can never discover
a move nobody in the data played. To become stronger than the players in the
data, the network needs a training signal that does not come from those
players. Module 6's self-play provides one.

There are two reasons to train this way first. It tests every stage of training
(data, tensors, loss, checkpoints) on a fixed dataset, as described under
Concepts. It also gives RL training a starting network that already knows
something. AlphaGo started this way; AlphaZero started from random weights.

Two more ideas you'll see in the results:

- The network is a *function approximator* for the policy π(a | s) and the
  value V(s). Read π(a | s) as "the probability of choosing move a in
  position s", and V(s) as "the expected result from position s". There are
  far too many chess positions to store an answer for each, so the network
  must *generalise* to positions it has never seen. Validation
  accuracy (the share of validation positions where the network's top move
  is the move played) measures that.
- The two heads get very different amounts of training information. Every
  position gives the policy head its own label (the move played). The value
  head gets the same label for every position in a game: the result. A
  thousand games are 70,000 policy labels but only a thousand independent
  value labels. Predict which head overfits first before you run Part B.

## Read

- `net/encoding.py` (all of it, especially the promotion encoding)
- `net/model.py` (ResidualBlock, then the heads, then load_model)
- `training/dataset.py` and `training/train.py`

## Do: Part A, the representation (no data needed)

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
   pawns (e2 empties, e4 fills). Plane 16 is side to move (all ones become
   all zeros: Black to move). Plane 19 marks the en-passant square, e3.
   If you predicted only plane 0, you have found the two planes that record
   game state rather than pieces. The network needs them to know whose turn
   it is and which moves are legal.
2. **Exercise: measure how much data the bug affected.** The old encoder gave
   every queen promotion an index ≥ 4672, and the old check skipped those
   moves without any error or warning. How much training data did that throw
   away? Run Part B's exercise 5 first (2-3 minutes) so you have
   `data/filtered.pgn`, then count:
   - **With git history** (a full `git clone`): the fix is commit `9b6cb6c`.
     Save the encoder from before it with
     `git show 9b6cb6c^:net/encoding.py > old_encoding.py`, load its
     `move_to_index`, and count the moves whose index is ≥ 4672.
   - **Without history** (a downloaded zip): count the promotions instead,
     with `move.promotion` from python-chess. That's the set of moves the bug
     affected.

   A starting point for either route:
   ```python
   import chess.pgn
   total = promotions = 0
   with open("data/filtered.pgn") as f:
       while (game := chess.pgn.read_game(f)) is not None:
           for move in game.mainline_moves():
               total += 1
               promotions += move.promotion is not None
               # with history: count old_move_to_index(move) >= 4672 instead
   print(total, promotions, f"{promotions / total:.2%}")
   ```

   **Expected (measured on `data/filtered.pgn`):** with history, 678 of
   349,814 moves were dropped, 0.19%. Without history you'll count 683
   promotions (the old encoder lost all but 5 of them). That sounds harmless,
   but it includes every one of the 660 queen promotions and most
   under-promotions. The network could never have learned to promote at all.
   Lesson: a small fraction of the data can contain all the examples of a
   skill, and an untested encoder can lose them.
3. **Exercise: augmentation.** Augmentation means creating extra training
   examples by transforming existing ones. Chess positions can be mirrored
   horizontally (a-file to h-file), which doubles your training data at no extra cost.
   Implement `mirror_board_tensor` and `mirror_move` in `net/encoding.py`. The
   requirements and the equivalence you must satisfy are in
   `tests/course/test_module04_augmentation.py`. Remove its
   `@pytest.mark.skip` line and make it pass. Castling rights are the difficult
   part: the test avoids positions with castling rights, and handling them is
   an optional extra task. Measure the effect on validation accuracy after Part B.

4. **Exercise: your first network.** Before training the real ResNet, write
   a small network and watch it learn something you can check: the material
   balance, from the board tensor.
   1. Create `training/first_network.py` and implement
      `train_material_net(boards, steps=300)` in it.
   2. Flatten the `(20, 8, 8)` tensor to 1,280 numbers. Add one hidden layer
      (a layer between input and output) of 64 numbers, and one output.
   3. Use mean squared error as the loss, and the Adam optimiser (a widely
      used form of gradient descent that adjusts the step size for each weight).
   4. Use the three lines every training loop has:
      `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`.
   5. The requirements, with a position generator and the material target,
      are in `tests/course/test_module04_first_network.py`. Remove its
      `@pytest.mark.skip` line and make it pass (about 10 seconds).
   6. Train on 300 positions instead of 2,000 and compare the error on
      positions the network never saw (the validation error).

   **Expected (measured with a reference solution):** on 2,000 positions
   the validation error is about 5, against about 49 for always guessing the
   average: it learned material. On 300 positions the training error falls
   almost to 0 but the validation error is about 53, worse than guessing.
   With 82,000 weights and 300 examples it memorised them. That is
   overfitting, and the value head in exercise 6 does the same.

## Do: Part B, data and training

5. **Get the data** (2-3 minutes, 17 MB download):
   ```bash
   python3 training/get_lichess.py
   ```
   This downloads January 2013 from the [Lichess database](https://database.lichess.org/)
   (free, public domain under the CC0 licence), decompresses it, keeps games where both players are rated
   1800 or more with at least 3 minutes on the clock, and writes `data/filtered.pgn`.
   **Expected:** `Games kept: 4,794 (4.0%)` out of 121,332. Filters matter:
   the stricter "2000+ and 10 minutes" keeps 3 games from this month,
   because few players were rated that high in 2013. Recent months are many
   GB; the script refuses them unless you pass `--allow-large`.

6. **Quick run first, then judge it against baselines** (about 6 minutes of
   training on an 8-core laptop CPU, faster with a GPU, a graphics processor):
   ```bash
   python3 training/train.py --pgn data/filtered.pgn --max-games 1000 --epochs 2
   ```
   The checkpoint is saved to `checkpoints/best_model.pt` (Modules 5 and 6 use
   it). If you want to see the loss curves, `tensorboard --logdir runs/`
   plots them in a browser (TensorBoard is PyTorch's usual plotting tool).
   **Expected (measured):** 1,000 games give 67,613 positions, split by game
   into 61,097 for training and 6,516 for validation. Each epoch takes about
   3 minutes. Validation accuracy goes from about 7% to about 9.4%.

   A single number like 9.4% does not tell you whether the network is good.
   Before you look at the answer, write down your prediction from the RL
   connection section: which head overfits? Then run:
   ```bash
   python3 training/eval_supervised.py checkpoints/best_model.pt \
       --pgn data/filtered.pgn --max-games 1000
   ```
   It rebuilds the same validation games and compares the network with two
   baselines that know nothing about the position. **Expected (measured):**

   | player | policy: picks the played move | comment |
   |---|---|---|
   | random legal move | 5.6% | |
   | most common legal move in training | 9.4% | no board understanding at all |
   | network, best legal move | 13.8% | beats both: it learned something |
   | network, move in its top 5 | 38.5% | |

   The policy head works, but the value head does not: its MSE (1.23) is
   worse than always guessing "even" (0.95). It picks the winner of
   decisive games 52% of the time, no better than a coin flip. It also rates
   "White a queen down" as good for White. Its training loss fell while its
   validation loss rose: it memorised its 889 training games, one result each.
   That is the overfitting the RL connection section predicted. The report
   ends with a few fixed test positions, and they show the raw network
   missing a free queen and a mate in one. Expect that, because search is what Module 5 adds.

   The accuracy in `train.py`'s log (about 9.4%) is lower than 13.8%. The
   match with the 9.4% baseline above is a coincidence. The log counts the
   network's top output even when that move is illegal, and 36% of its top
   picks are illegal. `eval_supervised.py` recomputes this measure as 9.3%;
   the last digit differs because it groups positions into batches
   differently. The engine only ever picks legal moves.

7. **The real run: more games instead of more epochs.** The fix for the value
   head is more independent results, so train on the whole month:
   ```bash
   python3 training/train.py --pgn data/filtered.pgn --epochs 3
   python3 training/eval_supervised.py checkpoints/best_model.pt --pgn data/filtered.pgn
   ```
   This gives about 320,000 positions (4,295 training games, 477 validation
   games). Training time grows with the number of positions. Expect about 15
   minutes per epoch on an 8-core laptop CPU. We measured 19 minutes with
   other jobs sharing the CPU. `train.py` also saves `model_epoch_1.pt`, `_2` and `_3`; evaluate
   epoch 1 as well as the final model.
   **Expected (measured):**

   | | after epoch 1 | after epoch 3 ("best") |
   |---|---|---|
   | policy: best legal move (baselines 5.9% / 9.3%) | 22.1% | 27.7% |
   | policy: in top 5 | 51.6% | 60.2% |
   | value MSE (always "even": 0.923) | 0.912 | 1.034 |
   | value: right winner, decisive games | 56% | 57% |
   | value: correlation with material | +0.44 | +0.35 |
   | takes the free queen (fixed test position) | no | yes |

   More games helped both heads. After one epoch the value head beats
   "always even" by a small margin and has clearly learned that material
   matters. After that, more epochs improve one head and make the other
   worse. The policy keeps improving, while the value head returns to
   memorising: its validation loss rose every epoch (0.91, 0.96, 1.03) as its
   training loss fell. `train.py` still labels epoch 3 "best". It chooses by
   the sum of the two losses, and the policy loss fell by more than the value
   loss rose. Read each head's numbers separately. The
   value head needs far more games than one month gives, or a smaller weight
   on its loss (multiplying the value loss by a number below 1 before adding
   it to the policy loss). It is the weakest part of the network you use in
   Module 5.

   **What counts as success.** Judge the network against the baselines
   instead of a target number. As a general guide: the policy's
   best-legal-move figure should be at least twice the "most common move"
   baseline. The value head's MSE should be below "always even", with a
   correlation with material above +0.3. The one-month run above meets both
   at epoch 1, and only the policy guide at epoch 3. The original project plan's
   targets (validation accuracy above 35%, validation loss below 2.5) were
   never measured on this code. Treat them as a goal for much more
   data, not as the standard for this module. Watch the gap between training
   and validation loss at every size: more data helps more than more epochs.

   **More data, when you want it.** Each extra month is one command, and PGN
   files can be concatenated:
   ```bash
   python3 training/get_lichess.py --month 2013-02 --output data/filtered_2013-02.pgn
   cat data/filtered.pgn data/filtered_2013-02.pgn > data/filtered_2months.pgn
   ```
   Months grow quickly (2013-12 is several times January's size), so check
   `--help` and your disk space first.

   **About the measured numbers.** Your figures will differ by a point or
   two, because of random starting weights and shuffled batches. The
   correlation with material is a plain (Pearson) correlation between the
   value head's output and the side to move's material balance, over all
   validation positions. `training/eval_supervised.py` documents each figure
   at the top of the file.

8. **Play the raw network against search.**
   `python3 training/evaluate.py checkpoints/best_model.pt --games 10` plays
   the raw policy (no search) against minimax depth 3 and rollout MCTS and
   prints the results.
   **Expected (measured with `--games 4`, about 30 minutes):** the
   one-month network lost all 4 games to minimax and scored 0.5/4 against
   MCTS. Expect the raw policy to lose: it knows something about moves, but
   it can't look ahead. Module 5 adds that look-ahead by combining the
   network with search.

## Check your understanding

<details>
<summary><b>Q1.</b> Predict: what does <code>move_to_index(chess.Move.from_uci("e2e4"))</code> return? Derive it, then run it.</summary>

`e2` is square 12 (rank 1 × 8 + file 4), `e4` is square 28, and ordinary
moves encode as `from * 64 + to`, which gives 12 × 64 + 28 = **796**. If you
got the squares' numbering backwards (file × 8 + rank), you have made the
same mistake the encoding tests are there to catch.
</details>

<details>
<summary><b>Q2.</b> The policy head outputs 4,672 logits but a position averages about 35 legal moves. Does the network waste effort on the other 4,637 or so, which are illegal?</summary>

In training, yes, a little. At play time the engine masks illegal moves
before the softmax, so whatever the network thinks of them never matters.
But `train.py`'s loss (`nn.CrossEntropyLoss` on the raw logits) is over all
4,672 outputs, unmasked: each step raises the played move's score and lowers every other
score, illegal ones included. So the network has to learn legality as a
side effect, and early on it hasn't. After the quick run its favourite
unmasked output was illegal in 36% of positions, which is why train.py's
accuracy (9.3%) is lower than the engine's (13.8%). Masking the logits inside
the loss as well would remove that wasted effort; you can try this as a small experiment.
</details>

<details>
<summary><b>Q3.</b> Both heads share one trunk. Name a concrete reason that's better than two separate networks, and one risk.</summary>

Better: the features useful for "who's winning" (king safety, material,
mobility) are the same features useful for "what's the best move". So the
training updates from each head improve the features the other head uses.
Sharing a trunk also acts as regularisation, because the trunk cannot
specialise in patterns that help only one task. It also halves the cost of
running the network on a position, which matters in Module 5, where search
runs it many times per move.
Risk: the trunk has a limited number of weights, and each loss changes them
to suit its own task. If one loss is much larger than the other, the
trunk mainly improves that loss at the other's expense. That is why you
should watch both curves. `train.py` weights the two losses equally (a plain
sum); a common fix when one head behaves badly is to give its loss a smaller weight.
</details>

<details>
<summary><b>Q4 (RL question).</b> Your supervised network reaches 50% accuracy on grandmaster games. Could more data and a bigger network take it past grandmaster strength on their own? Why or why not?</summary>

No. Its target is "the move the human played", so a perfect score means
playing exactly like the humans in the data, including their mistakes. It
has no way to find out that a different move would have been better. That
needs a training signal that does not come from the human players: trying
moves and seeing the results, which is RL (Module 6). Search helps too
(Module 5): it can find moves the network wouldn't.
</details>

<details>
<summary><b>Q5.</b> At 1,000 games the policy head beat its baselines but the value head didn't. Give the reason in one sentence, and the fix.</summary>

A game's roughly 70 positions each get their own policy label but share one
value label (the result), so 1,000 games give the value head only 1,000
independent examples, few enough to memorise.
The fix is more games instead of more epochs (exercise 7).
</details>

## Checkpoint

```bash
pytest tests/test_encoding.py tests/test_model.py -q         # still green
pytest tests/course/test_module04_augmentation.py \
       tests/course/test_module04_first_network.py -q         # un-skipped and passing
```

- [ ] Both commands pass
- [ ] A trained checkpoint exists and `load_model()` loads it without you
      telling it the architecture
- [ ] A short training log: dataset size, real elapsed time, final training and
      validation loss, and the `eval_supervised.py` table, with numbers you
      measured yourself (not the ones in this module)
- [ ] One sentence on which head overfit at 1,000 games, and why

Previous: [Module 3: Monte Carlo Tree Search](../module-03-mcts/README.md) · Next: [Module 5: PUCT](../module-05-puct/README.md)
