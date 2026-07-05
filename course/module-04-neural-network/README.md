# Module 4 — The Neural Network

**Goal:** build the knowledge half of AlphaZero — a network that looks at a
position and outputs both *which moves look right* (policy) and *who's
winning* (value) — and train it on expert games.

Requires torch: `pip install -e ".[train]"`.

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
channels, ~830K parameters) with two heads:
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

## Do

1. **Trace one example by hand.** Take 1.e4: write down which tensor plane
   changes and which policy index is the target. Verify with three lines of
   Python.
2. **Train for real.** Get expert games (e.g. the [Lichess database](https://database.lichess.org/) —
   one month of games, filter with `training/filter_games.py` to 2000+ Elo),
   then:
   ```bash
   python3 training/train.py --pgn data/filtered.pgn --max-games 10000 --epochs 10
   ```
   Watch tensorboard. Success criteria from the original plan: val policy
   accuracy > 35%, val loss < 2.5.
3. **Smell-test the trained model.** `training/evaluate.py <ckpt>` and check
   its top move in the starting position is a normal developing move, then
   play it (raw policy, no search!) against minimax depth-1.
4. **Exercise: find the bug's blast radius.** Check out the git history from
   before the encoding fix and count what fraction of moves in your PGN set
   would have been silently dropped (`git log -- net/encoding.py`). Write
   the number down — that's the cost of an untested encoder.
5. **Exercise: augmentation.** Chess positions can be mirrored horizontally
   (a-file ↔ h-file) if you also mirror castling rights and the move target.
   Implement it in the dataset and measure the effect on val accuracy.

## Checkpoint

```bash
pytest tests/test_encoding.py tests/test_model.py -q
```

- [ ] Green
- [ ] A trained checkpoint exists and `load_model()` loads it *without you
      telling it the architecture*
- [ ] Raw policy (no search) beats random play convincingly

Next: [Module 5 — PUCT](../module-05-puct/README.md)
