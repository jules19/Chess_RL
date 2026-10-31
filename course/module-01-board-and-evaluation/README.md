# Module 1 — Board & Evaluation

> **Where you are:** the bottom rung. Everything later is built on a board you can query and a number that says who's better.
> **Time:** 3-4 hours.
> **You'll be able to:**
> - evaluate any position with `evaluate()` and explain each term that contributes
> - add a new evaluation term, with a test that pins its behaviour
> - measure whether a change made the engine stronger, and say how sure you are
> - explain what a *value function* is, using `evaluate()` as the example

## Concepts

**The board library does the hard part.** `python-chess` gives us legal move
generation, check/mate detection, FEN parsing. Writing that from scratch is a
fine project — but a different one. We build *engines*, not rules.

**Evaluation = a position → a number.** By convention throughout this repo:
centipawns, from White's perspective (+100 ≈ White is a pawn up). Every
search algorithm in this course is a way of *looking ahead before trusting
that number*.

**Material first, position second.** Counting pieces (P=1 N=3 B=3 R=5 Q=9)
gets you surprisingly far. The rest of the evaluator encodes classical
principles as small bonuses: center control, piece development, king
safety, pawn structure, mobility. Each is worth ~10-50 centipawns. (The
project diary estimated "~1000 Elo" for material with shallow search and
"+200 Elo" for the positional terms. Those are historical self-estimates, not
measurements; Module 7 is where you replace them.)

**The baseline matters.** A random mover sounds useless, but it's the floor
that makes every later claim testable: "my new engine beats random 100-0"
is the first rung of evidence.

## RL lens: a value function, written by hand

In RL, a **value function** V(s) answers "how good is this state for me?"
`evaluate()` is exactly that, written by a person who knows chess: state in,
number out. Two details differ from the RL version you'll meet later. It
scores from *White's* point of view rather than the side to move's, and in
centipawns rather than an expected game result between -1 and +1.

Keep the idea in mind. In Module 4 a network *learns* a value function from
games, and in Module 6 it learns one from its own play. This module is the
version where all the knowledge is typed in, so you can see what the learned
versions have to discover on their own.

## Read

- `engine/evaluator.py` — the whole knowledge system of Phases 0-2.
  Read `evaluate()` top to bottom; note how every term is a small,
  independent, *individually testable* function.
- `cli/play.py` — the game loop: how an "engine" is just
  `f(board) -> move`.

## Do

1. **Calibrate your intuition.** Pick 3 positions from a game of yours
   (or any online game), evaluate them, and check the sign at least agrees
   with your judgement:
   ```python
   import chess
   from engine.evaluator import evaluate
   print(evaluate(chess.Board("<paste a FEN here>")))
   ```
   **Expected:** a pawn up shows as roughly +100 (or -100 if Black is up).
2. **Add an evaluation term.** Rooks on open files get a bonus (~+25cp).
   *First*, copy the current `evaluate()` to a new function
   `evaluate_baseline()` with the same body, so you have something to
   measure against. Then write `rook_open_file_bonus(board)` and wire it
   into `evaluate()`.
3. **Prove it with games, not vibes.** Play the new evaluator against the
   baseline, 20 games at depth 2 (about a minute):
   ```bash
   python3 cli/match.py minimax:depth=2 \
       minimax:depth=2,eval=engine.evaluator:evaluate_baseline --games 20
   ```
   **Expected:** a +25cp term rarely shows clearly in 20 games. A score
   between 40% and 60%, with an interval that includes 50%, is the normal
   result, and `match.py` will tell you the match can't separate the
   engines. That is the lesson of this step: decide *before* the match how
   many games would convince you. If the score is far below 50%, your
   bonus has a sign or colour bug.
   *(This "arena test" habit becomes literal infrastructure in Module 6.)*
4. **Un-skip your first exercise test.** Open
   `tests/course/test_module01_evaluator.py` — its docstring is the exact
   contract for exercise 2. Delete the `@pytest.mark.skip` line and make it
   pass. Then add one more assertion of your own (a position the contract
   doesn't cover — e.g. two rooks doubled on the same open file).

## Check your understanding

<details>
<summary><b>Q1.</b> Predict: what does <code>evaluate()</code> return on the starting position? Then run it, and explain the number.</summary>

It returns **20**, not 0. Material, centre, development, king safety and
pawn structure are all exactly 0, because the position is symmetric. The 20
comes from the *mobility* term at the end of `evaluate()`: it adds the side
to move's number of legal moves (20 at the start) as a small bonus for the
side to move. With Black to move, the same term counts against White.

So the real symmetry test is not "the start position scores 0". It's "swap
the colours and the side to move, and the score flips sign":
`evaluate(board.mirror()) == -evaluate(board)` (20 and -20 here). A term that
fails *that* test is the most common evaluator bug: a bonus computed for
White and never negated for Black.
</details>

<details>
<summary><b>Q2.</b> Your new rook bonus is worth +25cp and your 20-game match shows 11.5–8.5. Has the bonus been proven to help?</summary>

No. With 20 games the 95% interval around a 50% score is roughly ±22
percentage points, i.e. ±4.4 points out of 20, and 11.5/20 = 57.5% is well
inside it. `cli/match.py` prints that interval for you (it shows 36% to 76%
for this score). It's *evidence*, not proof: either play more games or
accept the uncertainty explicitly. Asking "how many games would convince
me?" is the actual lesson of this exercise.
</details>

<details>
<summary><b>Q3.</b> In RL terms, what is <code>evaluate()</code>, and what's the one thing an RL agent would have to do that this module did by hand?</summary>

It's a value function: state in, "how good" out. Here a person wrote the
knowledge in (piece values, bonuses). An RL agent has to *learn* that
function from experience, i.e. from the results of games, which is what
Modules 4 and 6 do.
</details>

## Checkpoint

- [ ] `pytest -q` green, **including the un-skipped**
      `tests/course/test_module01_evaluator.py`
- [ ] A `cli/match.py` result for your term, with its interval, and one
      sentence on what it does and doesn't show (a documented "can't tell
      yet" passes; so does a negative result you reverted)

Previous: [Module 0 — Orientation](../module-00-orientation/README.md) · Next: [Module 2 — Minimax Search](../module-02-minimax-search/README.md)
