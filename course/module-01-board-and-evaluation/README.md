# Module 1 — Board & Evaluation

**Goal:** understand the two primitives everything else builds on: a board
you can query, and a number that says who's better.

## Concepts

**The board library does the hard part.** `python-chess` gives us legal move
generation, check/mate detection, FEN parsing. Writing that from scratch is a
fine project — but a different one. We build *engines*, not rules.

**Evaluation = a position → a number.** By convention throughout this repo:
centipawns, from White's perspective (+100 ≈ White is a pawn up). Every
search algorithm in this course is a way of *looking ahead before trusting
that number*.

**Material first, position second.** Counting pieces (P=1 N=3 B=3 R=5 Q=9)
gets you surprisingly far — roughly 1000 Elo when combined with shallow
search. The rest of the evaluator encodes classical principles as small
bonuses: center control, piece development, king safety, pawn structure.
Each is worth ~10-50 centipawns; together they were measured at roughly +200
Elo over material alone (see `docs/history/DAYS_5_6_SUMMARY.md`).

**The baseline matters.** A random mover sounds useless, but it's the floor
that makes every later claim testable: "my new engine beats random 100-0"
is the first rung of evidence.

## Read

- `engine/evaluator.py` — the whole knowledge system of Phases 0-2.
  Read `evaluate()` top to bottom; note how every term is a small,
  independent, *individually testable* function.
- `cli/play.py` — the game loop: how an "engine" is just
  `f(board) -> move`.

## Do

1. **Calibrate your intuition.** Pick 3 positions from a game of yours
   (or any online game), evaluate them with `evaluate()`, and check the sign
   at least agrees with your judgement.
2. **Add an evaluation term.** Rooks on open files get a bonus (~+25cp).
   Write `rook_open_file_bonus(board)`, wire it into `evaluate()`.
3. **Prove it with games, not vibes.** Play 20 games of new-evaluator vs
   old-evaluator at fixed depth (adapt `test/tournament.py`). Report the
   score. If it's ≤ 50%, your bonus is miscalibrated — tune or revert.
   *(This "arena test" habit becomes literal infrastructure in Module 6.)*
4. **Un-skip your first exercise test.** Open
   `tests/course/test_module01_evaluator.py` — its docstring is the exact
   contract for exercise 2. Delete the `@pytest.mark.skip` line and make it
   pass. Then add one more assertion of your own (a position the contract
   doesn't cover — e.g. two rooks doubled on the same open file).

## Check your understanding

<details>
<summary><b>Q1.</b> Predict: <code>evaluate()</code> on the starting position returns what value, and why must it be exactly that?</summary>

0 (or within a few centipawns of it). The position is symmetric — every
material and positional term White earns, Black earns identically with
opposite sign. If it's not ~0, one of your terms isn't color-symmetric,
which is the single most common evaluator bug (a bonus computed for White
but not negated for Black).
</details>

<details>
<summary><b>Q2.</b> Your new rook bonus is worth +25cp and your 20-game arena shows 11.5–8.5. Has the bonus been proven to help?</summary>

No. 11.5/20 is well inside noise for 20 games (compute it in Module 7:
the 95% band around 50% is roughly ±11 points at n=20). It's *evidence*,
not proof — either play more games or accept the uncertainty explicitly.
The habit of asking "how many games would convince me?" is the actual
lesson of this exercise.
</details>

## Checkpoint

- [ ] `pytest -q` green, **including the un-skipped**
      `tests/course/test_module01_evaluator.py`
- [ ] Your evaluation term wins (or you learned why it doesn't and reverted
      — a documented negative result also passes this course)

Next: [Module 2 — Minimax Search](../module-02-minimax-search/README.md)
