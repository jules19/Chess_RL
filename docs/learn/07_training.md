# Unit 7: Training on human games

**You'll be able to:**
- explain supervised training targets: "the move played" and "the final result"
- read a loss number and say whether it's better than knowing nothing
- run a small training job and interpret what it prints

**Before this:** [Unit 6](06_network.md). **Next:** [Unit 8, Measuring honestly](08_measuring.md). **Time:** ~60 min.
**Longer reading:** `neural_network_tutorial.html`, "Days 3–4" and "Days 8–10" (written before
training was implemented; the code described here is the current version).

## The idea

Before a network can learn by playing itself (AlphaZero, Phase 4), you can give it a head
start by **imitating humans**. Every position in a human game becomes one training example:

| Input | Policy target | Value target |
|---|---|---|
| the board (unit 6 tensor) | the move the human actually played | the final result: +1 if the side to move went on to win, −1 if they lost, 0 for a draw |

Training adjusts the weights to reduce two losses, added together:
- **policy loss:** cross-entropy between the predicted move probabilities and the move played
- **value loss:** mean squared error between the predicted value and the result

Both targets are noisy. Humans play different moves in the same position, and a game's result
depends on later mistakes, not just on the current position. So the network can only learn
*tendencies*, and it needs **many** games to do even that.

## In the code

| Where | What to look for |
|---|---|
| `training/dataset.py`, `ChessDataset.load_games` | One example per position: `board_to_tensor`, `move_to_index`, and `game_value` flipped to the side to move. The first 5 moves of each game are skipped |
| `training/dataset.py`, `create_dataloaders` | `random_split` into train/validation **by position** (see exercise 3) |
| `training/train.py`, `Trainer.__init__` | `CrossEntropyLoss` + `MSELoss`, Adam, learning rate halved every 5 epochs |
| `training/train.py`, `Trainer.train_epoch` | `loss = policy_loss + value_loss`, then backprop |
| `training/train.py`, `Trainer.train` | Saves `model_epoch_N.pt` (with metrics) every epoch and `best_model.pt` (weights only) on the best validation loss |
| `training/filter_games.py` | Filters a large PGN by rating and time control (for when you download more data) |

## Try it

A one-epoch training run on 50 games, saved to `/tmp` so it can't overwrite the project's
checkpoints (takes a few seconds on CPU):

```bash
python training/train.py --pgn data/sample_lichess.pgn --max-games 50 --res-blocks 2 --channels 64 --epochs 1 --device cpu --num-workers 0 --checkpoint-dir /tmp/chess_rl_try
```

Among the output you should see numbers like these (they vary a little between runs):

```
Dataset created: 2,858 training examples from 50 games
  Train - Loss: 9.0938 (policy: 8.2639, value: 0.8298)
  Val   - Loss: 9.0924 (policy: 8.2033, value: 0.8891)
  Val   - Policy Accuracy: 0.00%
```

## Exercises

**1. What does "knowing nothing" score?** A network that spreads probability evenly over all 4,672
move slots has a cross-entropy of ln(4672). Calculate it and compare it with the policy loss
above.

<details><summary>Answer</summary>

ln(4672) ≈ **8.45**. The run above gets 8.26 policy loss after one epoch on 50 games: barely
better than knowing nothing. The project's real checkpoint (5 epochs on 411 games) reached a
validation loss of 6.84 in total, still a long way from good. A useful second baseline:
spreading evenly over only the ~30 *legal* moves gives ln(30) ≈ 3.40. A network that has merely
learned the rules would reach about that.
</details>

**2.** The policy loss in training is computed on the network's raw output, **without** the legal
move mask from unit 6. What does the network therefore have to learn first?

<details><summary>Answer</summary>

Which moves are *legal*. Most of the early fall in loss comes from moving probability away from
impossible moves like `a1h8`, not from learning good chess. Masking the logits during
training too would let it spend its capacity on choosing between legal moves.
</details>

**3. Leakage:** `create_dataloaders` splits examples into training and validation sets at random **by
position**. Why does that make the validation numbers look better than they really are?

<details><summary>Answer</summary>

Positions from the *same game* end up on both sides, and so do identical opening positions
from different games. The network is then partly validated on positions it was trained on,
which measures memory, not understanding. Split **by game** (put whole games in validation)
for an honest number.
</details>

**4.** The value target for every position in a game is that game's final result. Give an
example where this labels a position wrongly.

<details><summary>Answer</summary>

A player is completely winning at move 30, then blunders their queen at move 60 and loses.
Every position before the blunder is labelled "this side loses". The network only learns the
truth by averaging over many games, which is why value heads need a lot of data.
</details>

## Check yourself

1. What are the two training targets for each position?
2. Which loss is used for each?
3. What loss value means "no better than uniform guessing over all move slots"?

<details><summary>Answers</summary>

1. The move played (policy) and the final game result from the side to move's view (value).
2. Cross-entropy for policy, mean squared error for value.
3. About 8.45 (= ln 4672).
</details>
