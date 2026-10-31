# Module 1: Board and Evaluation

> **Where you are:** the first stage. Everything later is built on a board you can query and a number that says who's better.
> **Time:** 3-4 hours.
> **You'll be able to:**
> - evaluate any position with `evaluate()` and explain each term that contributes
> - add a new evaluation term, with a test that checks its behaviour
> - measure whether a change made the engine stronger, and say how sure you are
> - explain what a *value function* is, using `evaluate()` as the example

## Concepts

**The board library handles the rules.** `python-chess` generates legal
moves and detects check and checkmate. It also reads FEN (Forsyth-Edwards
Notation), a one-line text format for a chess position. Writing that from scratch would be a separate project. This
course builds engines on top of the rules.

**Evaluation turns a position into a number.** In this repository the number
is in centipawns (cp, hundredths of a pawn), from White's perspective. For
example, +100 means White is about a pawn up. Every search algorithm
in this course looks ahead through possible moves before relying on that
number.

**Material first, position second.** Counting pieces (pawn 1, knight 3, bishop 3,
rook 5, queen 9) already gives a reasonable engine. The rest of the evaluator turns classical
chess principles into small bonuses: centre control, piece development, king
safety, pawn structure, mobility. Each is worth about 10-50 centipawns. (The
project diary estimated "~1000 Elo" for material with a search only a few moves deep, and
"+200 Elo" for the positional terms. Elo is the chess rating scale. Those
figures are historical self-estimates, not measurements; Module 7 is where
you replace them.)

**The baseline matters.** A random mover is the lowest level of play,
and it makes every later claim testable. "My new
engine beats random 100-0" is the first piece of evidence.

## RL connection: a value function, written by hand

In reinforcement learning (RL), a *value function* V(s) answers the question
"how good is this state s for me?" Here, `evaluate()` is a value function
written by a person who knows chess. It takes a state and returns a
number. It differs from
the RL version you'll meet later in two ways. It scores from White's point
of view instead of the side to move's. And it gives centipawns instead of
an expected game result between -1 and +1.

Keep the idea in mind. In Module 4 a network learns a value function from
games, and in Module 6 it learns one from its own play. In this module all
the knowledge is typed in, so you can see what the learned versions must
learn from games instead.

## Read

- `engine/evaluator.py`: all the chess knowledge used in Phases 0-2.
  Read `evaluate()` from top to bottom. Note that every term is a small,
  independent function that can be tested on its own.
- `cli/play.py`: the game loop. It shows that an "engine" is a function
  `f(board) -> move`.

## Do

1. **Check the evaluator against your own judgement.** Pick 3 positions from
   one of your games (or any online game) and evaluate them. Check that at
   least the sign agrees with your judgement:
   ```python
   import chess
   from engine.evaluator import evaluate
   print(evaluate(chess.Board("<paste a FEN here>")))
   ```
   **Expected:** a pawn up shows as roughly +100 (or -100 if Black is up).
2. **Add an evaluation term.** Rooks on open files (files with no pawns on
   them) get a bonus (about +25 cp).
   First, copy the current `evaluate()` to a new function
   `evaluate_baseline()` with the same body, so you have something to
   measure against. Then write `rook_open_file_bonus(board)` and call it
   from `evaluate()`.
3. **Test the change with games.** Play the new evaluator against the
   baseline: 20 games at depth 2, which takes about a minute. Depth 2 means
   searching two plies ahead (a ply is one move by one side; Module 2
   explains depth):
   ```bash
   python3 cli/match.py minimax:depth=2 \
       minimax:depth=2,eval=engine.evaluator:evaluate_baseline --games 20
   ```
   **Expected:** a +25 cp term rarely shows a clear effect in 20 games. A
   score between 40% and 60%, with a 95% interval (the range the true score
   very likely lies in) that includes 50%, is the normal result. `match.py`
   will tell you the match can't separate the engines. The lesson of this
   step is to decide before the match how many games would convince you. If
   the score is far below 50%, your bonus probably has a sign or colour bug: it rewards the wrong side,
   for example a bonus computed for White and never negated for Black.
   (In Module 6 this habit becomes part of the code: a match, called the
   arena, decides whether a new network replaces the old one.)
4. **Enable your first exercise test.** Open
   `tests/course/test_module01_evaluator.py`. Its docstring states exactly
   what exercise 2 must do. Delete the `@pytest.mark.skip` line and make the
   test pass. Then add one more assertion of your own, for a position the
   test doesn't cover (for example, two rooks doubled on the same open file).

## Check your understanding

<details>
<summary><b>Q1.</b> Predict: what does <code>evaluate()</code> return on the starting position? Then run it, and explain the number.</summary>

It returns 20. Material, centre, development, king safety and
pawn structure are all exactly 0, because the position is symmetric. The 20
comes from the *mobility* term at the end of `evaluate()`. It adds the
number of legal moves of the side to move (20 at the start) as a small
bonus for that side. With Black to move, the same term counts against White.

So "the start position scores 0" is the wrong symmetry test. The right one
is "swap the colours and the side to move, and the score changes sign":
`evaluate(board.mirror()) == -evaluate(board)` (20 and -20 here). A term
that fails this test is the most common evaluator bug: a bonus computed
for White and never negated for Black.
</details>

<details>
<summary><b>Q2.</b> Your new rook bonus is worth +25cp and your 20-game match shows 11.5–8.5. Has the bonus been proven to help?</summary>

No. With 20 games the 95% interval around a 50% score is roughly ±22
percentage points, that is ±4.4 points out of 20, and 11.5/20 = 57.5% is
well inside it. `cli/match.py` prints that interval for you (it shows 36% to
76% for this score). The result is evidence but not proof. Either play more
games or state the uncertainty clearly. Before each match, ask "how many
games would convince me?"
</details>

<details>
<summary><b>Q3.</b> In RL terms, what is <code>evaluate()</code>, and what's the one thing an RL agent would have to do that this module did by hand?</summary>

It's a value function: it takes a state and returns how good that state
is. Here a person wrote the
knowledge in (piece values, bonuses). An RL agent has to learn that
function from experience, that is, from the results of games. Modules 4 and
6 do this.
</details>

## Checkpoint

- [ ] `pytest -q` passes, including
      `tests/course/test_module01_evaluator.py` with its skip line removed
- [ ] A `cli/match.py` result for your term, with its interval, and one
      sentence on what it does and doesn't show (a documented "can't tell
      yet" passes; so does a negative result you reverted)

Previous: [Module 0: Orientation](../module-00-orientation/README.md) · Next: [Module 2: Minimax Search](../module-02-minimax-search/README.md)
