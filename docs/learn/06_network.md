# Unit 6: A neural network that sees the board

**You'll be able to:**
- explain how a board becomes a 20×8×8 tensor and a move becomes an index
- explain what the **policy** and **value** heads output, and what each is for
- find a real limitation in the move encoding by working out indices by hand

**Before this:** [Unit 5](05_mcts.md). Basic idea of a neural network: layers, weights, training
by gradient descent. **Next:** [Unit 7, Training](07_training.md). **Time:** ~60–90 min.
**Longer reading:** `neural_network_tutorial.html`, "Conceptual Foundation" and "Days 1–2: Neural
Network Architecture".

## The idea

Units 1–5 relied on hand-written knowledge (piece values, centre bonuses). A neural network
can **learn** this knowledge from examples instead. To do that, the board and moves need to be
numbers.

**Board → tensor.** The network sees 20 "planes", each an 8×8 grid:

| Planes | Meaning |
|---|---|
| 0–5 | White pawn, knight, bishop, rook, queen, king (1 where that piece stands) |
| 6–11 | The same for Black |
| 12–15 | Castling rights (whole plane 1 or 0) |
| 16 | Side to move (all 1 if White) |
| 17 | 50-move counter, scaled to 0–1 |
| 18 | Unused (all zeros) |
| 19 | En passant square |

**Move → index.** A move from square `f` to square `t` (squares numbered a1=0 … h8=63) gets
the index `f × 64 + t`. That covers every from/to pair in 0–4095.

**Two outputs ("heads")** share one convolutional trunk:
- **Policy:** a probability for each of 4,672 move slots ("which move would I play?").
  Illegal moves are masked out before the softmax.
- **Value:** one number in [−1, +1] ("who's winning, from the side to move's view?"),
  squashed by `tanh`.

This two-headed design comes from AlphaZero. The policy tells a search *which moves to
look at*, and the value tells it *how good a position is* without playing it out (unit 9).

## In the code

| Where | What to look for |
|---|---|
| `net/encoding.py`, `board_to_tensor` | The plane layout above |
| `net/encoding.py`, `move_to_index` / `index_to_move` | `from_square * 64 + to_square`, plus a promotion offset |
| `net/encoding.py`, `legal_moves_mask` | 1 for legal move indices, 0 otherwise |
| `net/model.py`, `ResidualBlock` | Two 3×3 convolutions plus a skip connection |
| `net/model.py`, `PolicyHead` | 1×1 conv to 2 channels, then `Linear(128, 4672)` |
| `net/model.py`, `ValueHead` | 1×1 conv to 1 channel, `Linear(64, 256)`, `Linear(256, 1)`, `tanh` |
| `net/model.py`, `get_policy_value` | Masking: illegal logits get −1e9 before the softmax |
| `net/model.py`, `load_model` | Loads a checkpoint and infers its size from the weights |

## Try it

Run the encoding self-tests and the architecture self-test:

```bash
python net/encoding.py
python net/model.py
```

You should see `✅ ALL ENCODING TESTS PASSED!` and `✅ ALL TESTS PASSED! Network architecture is ready.`
The first command also prints round-trips such as `e2e4 → 796 → e2e4`.

## Exercises

**1. By hand:** what index does `g1f3` get? (a1=0, b1=1, …, h1=7, a2=8, …)

<details><summary>Answer</summary>

g1 = 6, f3 = 2 × 8 + 5 = 21, so 6 × 64 + 21 = **405**. Check:
`python -c "import chess; from net.encoding import move_to_index; print(move_to_index(chess.Move.from_uci('g1f3')))"`
</details>

**2. Find the limitation:** promotions add `(piece − 2) × 4096` to the index, where knight = 2
and queen = 5. Work out the indices for `a7a8q` (promote to queen), `a7a8n` (promote to knight)
and the plain move `a7a8`. The policy has only 4,672 slots. What goes wrong?

<details><summary>Answer</summary>

- `a7a8` = 48 × 64 + 56 = **3128**
- `a7a8n` = 3128 + 0 × 4096 = **3128**, the *same* index as the non-promotion move
- `a7a8q` = 3128 + 3 × 4096 = **15416**, far beyond 4,672

So queen, rook and bishop promotions **can't be represented**, and knight promotions collide.
`training/dataset.py` silently skips moves with an index ≥ 4672. In the project's training data
that dropped 82 of 27,708 moves: small, but every one was a promotion, which is exactly
the kind of move an engine must get right. AlphaZero's real 4,672-slot encoding (73 move types
× 64 from-squares) is designed to avoid this. Fixing it here would mean retraining the
network.
</details>

**3.** Read the `PolicyValueNetwork` docstring in `net/model.py`, which lists what the 20 input
planes contain. Compare it with `board_to_tensor`. Do they agree?

<details><summary>Answer</summary>

No. The docstring mentions "2 repetition counters", but the code has no repetition planes.
It has en passant (plane 19) and an unused plane (18). **The code is the source of truth.**
Comments drift. This is a good habit for any codebase: when a comment and the code disagree,
trust the code and fix the comment.
</details>

**4.** The value target is from the **side to move's** view, but the board planes always show
the board from White's side. How can the network know whose view to take?

<details><summary>Answer</summary>

Plane 16 (side to move) tells it. The network has to learn to "flip" its judgement depending on
that plane, which costs capacity. Many engines instead flip the board so the side to move
always appears at the bottom. That's a design choice worth knowing about.
</details>

## Check yourself

1. What shape is the input tensor?
2. What do the policy and value heads each output?
3. Why are illegal moves masked before the softmax rather than after?

<details><summary>Answers</summary>

1. 20 × 8 × 8 (a batch adds a first dimension).
2. Policy: a probability for each of 4,672 move slots. Value: one number in [−1, +1] for the side to move.
3. So the probabilities over legal moves add up to 1. Masking after the softmax would leave
   probability mass on illegal moves.
</details>
