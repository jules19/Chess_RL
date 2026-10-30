# Unit 1: Judging a position

**You'll be able to:**
- explain what an evaluation function is and what its units mean
- read the project's evaluation and say what each term rewards
- spot a design quirk in it by predicting a number and checking it

**Before this:** [Unit 0](00_game_loop.md). **Next:** [Unit 2, Minimax](02_minimax.md). **Time:** ~45 min.
**Longer reading:** `minimax_tutorial.html`, sections "Day 2" and "Day 5–6".

## The idea

To choose a move better than random, the program needs some way to say "this position is
better than that one". An **evaluation function** turns a board into a single number:

- **positive** means good for White, **negative** means good for Black, 0 means equal
- the unit is the **centipawn**: 100 = one pawn's worth of advantage

The simplest evaluation just **counts material**: pawn 100, knight 300, bishop 300,
rook 500, queen 900. Real positions involve more than material, so the project adds smaller
bonuses for things strong players care about: controlling the centre, developing pieces,
keeping the king safe, and avoiding weak pawns.

With an evaluation, you can already build a **1-ply greedy player**: try each legal move,
evaluate the result, and play the best one. That's `best_move_material`.

## In the code

| Where | What to look for |
|---|---|
| `engine/evaluator.py`, `PIECE_VALUES` | The material values, in centipawns |
| `engine/evaluator.py`, `evaluate_material` | Adds White's pieces, subtracts Black's |
| `engine/evaluator.py`, `evaluate` | The full evaluation: checkmate first, then material plus the positional terms, a check bonus and a **mobility** term |
| `engine/evaluator.py`, `evaluate_center_control` … `evaluate_pawn_structure` | One function per idea, each typically worth tens of centipawns |
| `engine/evaluator.py`, `best_move_material` | The 1-ply greedy player |

## Try it

```bash
python -c "import chess; from engine.evaluator import evaluate; print(evaluate(chess.Board()))"
```

Expected output: `20`

## Exercises

**1. Predict, then check:** the starting position is perfectly symmetrical. So why is its
evaluation **20** and not 0? Read the end of `evaluate` before opening the answer.

<details><summary>Answer</summary>

The **mobility** term: `mobility = board.legal_moves.count()` counts the moves of the
**side to move only**, adding them if White is to move and subtracting them if Black is.
At the start White has 20 legal moves, so +20. The score therefore depends on whose turn it
is, not just the position. That's a real quirk worth knowing: a symmetrical evaluation would
count both sides' mobility.
</details>

**2.** Without running anything, what does `evaluate_material` return for the starting position
with Black's queen removed?

<details><summary>Answer</summary>

`900`: White is a queen up. Check it:

```bash
python -c "import chess; from engine.evaluator import evaluate_material; print(evaluate_material(chess.Board('rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')))"
```
</details>

**3.** After `1. e4` the evaluation is `10`, although White has just grabbed the centre.
Explain it using exercise 1.

<details><summary>Answer</summary>

Now it's **Black** to move, so the mobility term *subtracts* Black's 20 legal moves. The
centre and development bonuses White earned are mostly cancelled by that swing. Lesson:
when comparing evaluations, compare positions with the **same side to move**.

```bash
python -c "import chess; from engine.evaluator import evaluate; b = chess.Board(); b.push_uci('e2e4'); print(evaluate(b))"
```
</details>

**4. Stretch:** why does `evaluate` return ±100000 for checkmate instead of something like
±5000?

<details><summary>Answer</summary>

Checkmate must outrank *any* material advantage. Even all the pieces on the board add up to
under 4,000 centipawns, so 100000 can never be outweighed and the search will always prefer
a mate.
</details>

## Check yourself

1. What is a centipawn?
2. What does the sign of an evaluation tell you?
3. What is the weakness of a 1-ply greedy player? (Unit 3 answers this properly.)

<details><summary>Answers</summary>

1. A hundredth of a pawn: the unit evaluations are measured in.
2. Who's better: positive means White, negative means Black.
3. It never considers the opponent's reply. It will happily grab a pawn that's defended and
   lose its queen next move.
</details>
