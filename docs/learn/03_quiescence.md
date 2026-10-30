# Unit 3: The horizon effect and quiescence search

**You'll be able to:**
- explain the horizon effect with a concrete position
- explain how quiescence search fixes it, and why it needs a "stand pat" score
- show the difference by running three engines on the same position

**Before this:** [Unit 2](02_minimax.md). **Next:** [Unit 4, UCI](04_uci.md). **Time:** ~45 min.
**Longer reading:** `minimax_tutorial.html`, section "Day 7: Quiescence Search".

## The idea

A fixed-depth search stops at its **horizon** and evaluates whatever position is there,
even if it's in the middle of an exchange. Suppose the last move searched is "my queen
takes a pawn". The evaluation sees "I'm a pawn up", but the search never looked at the
next ply, where the opponent takes back the queen. This blindness is the **horizon effect**.

**Quiescence search** ("searching until things are quiet") fixes it. At the horizon,
instead of evaluating immediately, keep searching **only the noisy moves** (captures and
promotions, plus checks when there are only a few captures) until none are left, then evaluate.

One detail matters: in a quiescence position you're never *forced* to capture. So the
side to move can also "**stand pat**", meaning take the current evaluation as its score. If
standing pat is already better than any capture, stop. That bound is what stops quiescence
from exploding.

## In the code

| Where | What to look for |
|---|---|
| `search/minimax.py`, `quiescence_search` | `stand_pat = evaluate(board)`, then only noisy moves are searched (captures, promotions, and checks when captures are few), with a `max_depth` safety limit |
| `search/minimax.py`, `minimax` | `if depth == 0: return quiescence_search(...)`: every leaf goes through quiescence |
| `engine/evaluator.py`, `best_move_material` | A player **with no lookahead at all**, for comparison |

## Try it

White has a queen on e2 and can take the pawn on e5. But that pawn is defended by the pawn on
d6. Three players, same position:

```
4k3/8/3p4/4p3/8/8/4Q3/4K3 w - - 0 1
```

```bash
python -c "import chess; from engine.evaluator import best_move_material as f; print(f(chess.Board('4k3/8/3p4/4p3/8/8/4Q3/4K3 w - - 0 1')))"
python -c "import chess; from search.minimax import best_move_minimax as f; print(f(chess.Board('4k3/8/3p4/4p3/8/8/4Q3/4K3 w - - 0 1'), 1))"
```

Expected output:

```
e2e5
e2h5
```

The greedy player grabs the pawn (`Qxe5??`, and `dxe5` wins the queen). Minimax at depth **1**,
still looking just one move ahead, avoids it. Quiescence searched the recapture beyond
the horizon. (It plays `Qh5+` instead. The check bonus from unit 1 makes checks attractive.)

## Exercises

**1. Predict:** a depth-1 search **without** quiescence (play each move, call `evaluate`, take
the best) would choose what, and with roughly what score?

<details><summary>Answer</summary>

`e2e5` with a score of about **+820**. It sees the queen safely a pawn up (plus positional
bonuses) because it never looks at `dxe5`. You can verify it in a Python shell:

```python
import chess
from engine.evaluator import evaluate
board = chess.Board("4k3/8/3p4/4p3/8/8/4Q3/4K3 w - - 0 1")
scores = {}
for move in board.legal_moves:
    board.push(move); scores[move.uci()] = evaluate(board); board.pop()
print(max(scores, key=scores.get), max(scores.values()))   # e2e5 820
```
</details>

**2.** Why does quiescence search only noisy moves (mainly captures), not all moves?

<details><summary>Answer</summary>

Captures change material a lot, so they're what makes an evaluation unreliable. They're
also *finite*: every capture removes a piece, so the sequence has to end. Searching
every move would just be a deeper minimax with the same horizon problem one level further down.
</details>

**3.** What would go wrong if quiescence search had no stand-pat option?

<details><summary>Answer</summary>

The side to move would be *forced* to make a capture whenever one exists, even a terrible
one, like taking a defended pawn with a queen. Positions would then look much worse than they
are. Stand pat models the fact that you can always decline to capture.
</details>

## Check yourself

1. In one sentence, what is the horizon effect?
2. Which moves does quiescence search consider?
3. What is "stand pat"?

<details><summary>Answers</summary>

1. A fixed-depth search evaluates positions in the middle of a tactical sequence and
   misjudges them because it can't see the next move.
2. "Noisy" moves: captures and promotions, plus checks when there are only a few captures.
3. The option to take the current static evaluation instead of capturing. It's a lower
   bound for the side to move.
</details>
