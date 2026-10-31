# Module 2: Minimax Search

> **Where you are:** the first search. The evaluation function from Module 1 becomes an engine that looks several moves ahead before it uses that evaluation.
> **Time:** 4-6 hours.
> **You'll be able to:**
> - explain minimax and alpha-beta (two ways to search ahead), and predict when alpha-beta skips (*prunes*) much of the search
> - measure a search optimisation by counting the positions searched, instead of guessing
> - say why a stored search result (a transposition-table entry) needs a flag marking its score as exact or a bound
> - write minimax as a Bellman backup, the equation that reinforcement learning (RL) methods are built on

## Concepts

A search builds a *game tree*: each *node* is a position and each branch is
a legal move. A *ply* (one move by one side) is one level of the tree. The
*leaves* are the positions where the search stops and evaluates.

**A small worked example first.** White to move, two candidate moves, and
Black has two replies to each. The numbers at the bottom are `evaluate()`
scores from White's point of view, after Black's reply:

```
                 White to move (picks the MAX)
                /                       \
          move a                       move b
     Black picks the MIN          Black picks the MIN
        /        \                  /          \
      +3          +5             +1           ???
```

*Minimax:* if White plays a, Black answers with the reply worth +3 (Black
wants the smallest number), so a is worth +3. If White plays b, Black can
reach +1 at least, so b is worth at most +1. White picks a: the position is
worth +3. Minimax values each move by the opponent's best reply to it.

*Alpha-beta:* after searching a, White knows it can get +3. In b, the first
Black reply already gives +1. Whatever the `???` is, Black will never let b
be worth more than +1, which is worse for White than a's +3. So `???` is
never searched: that branch is *pruned*, with the same result. Ending a
branch's search early like this is a *cutoff*. In a real search the pruned
part is a whole subtree. Searching the
best moves first (move ordering, below) makes cutoffs happen sooner.

Read these in `search/minimax.py`, where each has a teaching docstring:

1. **Minimax.** Assume the opponent replies with *their* best move.
   `minimax()` alternates between maximising plies (White picks the highest
   score) and minimising plies (Black picks the lowest).
2. **Alpha-beta pruning.** Stop searching a branch as soon as you prove the
   opponent would never allow it. It gives the same answer as plain minimax
   with much less work. How much less depends on move ordering (item 3).
3. **Move ordering.** Searching good moves first makes cutoffs happen early.
   See `order_moves()`. *MVV-LVA* ("most valuable victim, least valuable
   attacker") tries captures first, most valuable victim and cheapest
   capturer first: pawn takes queen before queen takes pawn. Winning
   captures are the moves most likely to cause a cutoff. The *hash move* is
   the best move that a previous, shallower search stored for this position
   in the transposition table (item 5). It is usually still best, so trying it first is the most
   effective single ordering rule.
4. **Quiescence search.** Never evaluate in the middle of a capture
   sequence. The search cannot see past its depth limit (its horizon), so a
   search that stops just after a capture misses the recapture: the
   *horizon effect*. `quiescence_search()` extends the
   search with captures, checks and promotions until none are left (the
   position is *quiet*).
5. **Transposition table.** Different move orders can reach the same
   position (a *transposition*). The transposition table (TT) stores
   results for reuse, under a *key* computed from the position: its
   *Zobrist hash*. This gives every (piece, square) pair a fixed random
   64-bit number (a random integer of 64 binary digits) and combines the
   numbers for the pieces on the board with XOR (bitwise exclusive or, which
   is undone by applying it again). Making a move only XORs a couple of
   numbers in and out, so updating the key costs very little.
   One detail needs care: alpha-beta often proves only a *bound* on a score
   (for example "at least +2"). Each entry therefore stores a *flag* that says
   which kind of score it holds. See the
   `TT_EXACT` / `TT_LOWER_BOUND` / `TT_UPPER_BOUND` discussion in the file.
6. **Iterative deepening.** Search to depth 1, then 2, then 3, and so on.
   Repeating shallow depths costs little, and it handles *time management*
   (staying within the time for a move): you play the deepest finished
   search's move. See `best_move_iterative()` and its *predictive* time
   check, which estimates whether the next depth
   can finish in time.

## RL connection: the Bellman backup, and planning with a model

Minimax computes a value with one rule, applied recursively: *the value of
a position is the best value you can reach in one move, seen from your
opponent's side*. With V(s) as the value of position s for the side to
move, the rule is:

```
V(s) = max over legal moves a of  -V(s after a)
```

The minus sign turns the opponent's value into yours. This is the
**Bellman optimality equation** for a two-player zero-sum game (one side's
gain is exactly the other side's loss). It defines the value of a *state*
(in chess, a position) by the values of the states one step later. Almost every RL method is a way of solving this
equation approximately when you can't afford to apply it everywhere.

Minimax can apply it exactly because it has a *model* of the environment:
the rules tell it what position each move leads to. RL with a known
model is called *model-based*, and looking ahead with it is called
*planning*. Minimax plans to a fixed depth, then has to rely on `evaluate()`
at the leaves. Better leaf values (Module 4) and better choices about which
lines to plan (Module 3) are what the rest of the course adds.

## Two measurement lessons (from this repository's history)

- **Measure before optimising.** The first TT version stored and looked up
  every node, and was slower than no TT at all: at *frontier nodes* (nodes
  next to the leaves), hashing cost more than it saved. The fix,
  `use_tt = depth >= 2`, is two lines, and it was found by timing the code.
- **Measure on the right workload.** The TT gives little benefit in shallow
  tactical positions (positions full of captures and threats, where few
  move orders lead to the same position). It halves the node count in king-and-pawn endgames.
  `tests/test_minimax.py::test_transposition_table_preserves_score_and_cuts_nodes`
  checks this.

## Do

1. **See how much alpha-beta prunes.** Count the nodes alpha-beta visits at
   depth 4, with move ordering and without it. `minimax()` counts nodes into
   a one-element list you pass as `nodes_searched`:
   ```python
   import chess
   import search.minimax as mm
   fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
   original = mm.order_moves
   for label, order in [("ordered", original), ("unordered", lambda b, moves: list(moves))]:
       mm.order_moves = order          # minimax looks order_moves up at call time
       nodes = [0]
       mm.minimax(chess.Board(fen), 4, -1e9, 1e9, True, nodes)
       print(label, nodes[0])
   mm.order_moves = original           # put it back, or later exercises measure the wrong thing
   ```
   **Expected (measured):** about 23,000 nodes ordered and 97,000 unordered
   (2.7 s and 12 s): ordering cuts the node count by a factor of four. Now
   try the *starting* position: about 13,600 and 15,200, nearly the same,
   because MVV-LVA only reorders
   captures and the start position has none. Node counts are deterministic,
   so any difference you see is a real effect.
2. **Exercise: mid-search abort.** `best_move_iterative()` only checks the
   clock between iterations. Its predictive stop assumes each depth costs
   about 3 times the previous one, but in tactical positions it can be 10
   times. So the search can still exceed a 6-second budget by 10 seconds.
   Add a hard deadline that aborts inside `minimax()`:
   - about every 1024 nodes, raise a `SearchTimeout` if the deadline has passed
   - catch it in the function that runs the iterations, and play the
     previous iteration's move

   Your checkpoint is `tests/course/test_module02_search.py`. Run it before
   implementing. Expected: it fails after about 18 seconds. Then delete the
   `@pytest.mark.skip` line above the test (it tells pytest to skip the
   test), and make the test pass.
3. **Exercise: killer moves.** A move that is not a capture and that caused
   a cutoff at one node often causes cutoffs at *sibling nodes*
   (other nodes at the same depth). Keep two such killer moves per depth and
   try them right after the hash move. Captures don't count as killers,
   because MVV-LVA already orders them first. Add it as a `killers`
   argument to `minimax()`. The test is
   `tests/course/test_module02_killers.py` (delete its skip line, as in
   exercise 2, and make it pass). Measure with exercise 1's code, on its position and on the starting
   position, at depth 4.
   **Expected (measured with a reference solution):** the score is
   unchanged, because the search is still alpha-beta. A changed score means
   a bug. Node counts: the starting position drops from 13,592 to about
   8,100 (−40%), and exercise 1's position from 23,228 to about 21,700 (−7%).
   It helps most in positions with few captures, where MVV-LVA had nothing
   to reorder.
   Your numbers will differ a little with how you store and try killers.
4. **Does the TT change the move?** Predict first, then check. Pass
   `verbose=True` and read the `Nodes searched:` line it prints last:
   ```python
   board = chess.Board("8/2k5/8/1p6/1P6/2K5/8/8 w - - 0 1")   # king-and-pawn endgame
   mm.best_move_minimax(board, depth=6, verbose=True)
   mm.best_move_minimax(board, depth=6, tt={}, verbose=True)
   ```
   Then repeat at depth 3 on exercise 1's opening position.
   **Expected:** the same move every time. In the endgame the TT more than
   halves the node count (measured: 11,066 to 5,072). In the opening at
   depth 3 it changes nothing (measured: 3,141 nodes both ways), because
   short tactical searches have few transpositions.
   Q2 below explains why the move can't change. The strength gain appears
   only when a *time budget* lets the faster engine search deeper.

## Check your understanding

<details>
<summary><b>Q1 (explain it in your own words).</b> Why does a TT entry need a <em>flag</em> as well as a score?</summary>

Because alpha-beta often does not compute the true value of a node. It stops
as soon as it can prove the node will not affect the result. The *search
window* (alpha, beta) is the range of scores that could still change the
move chosen higher up. After a cutoff you know only a one-sided bound:

- "at least X": the score reached beta or above (*fail-high*), stored as
  TT_LOWER_BOUND
- "at most X": the score stayed at alpha or below (*fail-low*), stored as
  TT_UPPER_BOUND

Reusing a bound as if it were exact gives wrong answers when the search
window has changed. The flag records which of the three facts
you proved. To see it break, change the store logic to always write
`TT_EXACT` and run `test_transposition_table_preserves_score_and_cuts_nodes`.
Expected: it fails on the "same score" assertion.
</details>

<details>
<summary><b>Q2.</b> Predict: at a <em>fixed</em> depth of 3, does adding the TT change which move the engine picks? Its strength?</summary>

The TT changes neither the move nor the strength. At fixed depth it only
makes the search faster: same tree, fewer node visits. Strength changes only
when a *time budget* lets the faster engine reach a deeper iteration than it otherwise would. This
is exercise 4's answer, and in general a speed optimisation adds strength
only by letting the engine search deeper in the same time.
</details>

<details>
<summary><b>Q3.</b> Move ordering never changes the final minimax value. So why did it cut the node count by a factor of four in exercise 1?</summary>

Alpha-beta prunes a branch only after it has seen a move good enough to
prove the branch irrelevant. If the best move is searched first, that proof
comes at once, and the search cuts off one sibling after another. If it is
searched last, the whole subtree is explored before the search learns it
wasn't needed. The answer is the same, but the node counts are very
different (exercise 1 gives the numbers).
</details>

<details>
<summary><b>Q4 (RL question).</b> Minimax applies V(s) = max over a of -V(s after a) exactly. What does it need to do that, which an RL agent in an unknown environment wouldn't have, and what does it still have to guess?</summary>

It needs a model: the rules, which say exactly which position each move
leads to, so it can look ahead without playing. An agent that doesn't know
its environment's rules can only learn by acting. What minimax still guesses
is the value at the leaves of its search, where it stops and uses
`evaluate()`. Every later engine in the course is a better answer to that
same leaf-value question.
</details>

## Checkpoint

```bash
pytest tests/test_minimax.py -q                 # regression suite still green
pytest tests/course/test_module02_search.py tests/course/test_module02_killers.py -q  # un-skipped and passing
```

- [ ] Both commands pass
- [ ] Your node counts from exercises 1 and 4, written down next to the expected ones
- [ ] You answered Q1 correctly before looking at the answer

Previous: [Module 1: Board and Evaluation](../module-01-board-and-evaluation/README.md) · Next: [Module 3: Monte Carlo Tree Search](../module-03-mcts/README.md)
