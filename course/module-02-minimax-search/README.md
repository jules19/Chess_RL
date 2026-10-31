# Module 2 — Minimax Search

> **Where you are:** the first search. Module 1's number becomes an engine that looks ahead before trusting it.
> **Time:** 4-6 hours.
> **You'll be able to:**
> - explain minimax and alpha-beta, and predict when alpha-beta prunes a lot or a little
> - measure a search optimisation with node counts instead of guessing
> - say why a transposition-table entry needs a flag, not just a score
> - write minimax as a Bellman backup, the equation RL methods are built on

## Concepts

Read these in `search/minimax.py` — each has a teaching docstring at the
exact place it's implemented:

1. **Minimax** — assume the opponent replies with *their* best move.
   `minimax()` alternates maximizing (White) and minimizing (Black) levels.
2. **Alpha-beta pruning** — stop searching a branch the moment you prove the
   opponent would never allow it. Same answer as plain minimax, a fraction
   of the work. Effectiveness depends entirely on…
3. **Move ordering** — searching good moves first makes cutoffs happen early.
   See `order_moves()`. *MVV-LVA* ("most valuable victim, least valuable
   attacker") tries captures first, biggest prize first and cheapest
   capturer first: pawn takes queen before queen takes pawn, because winning
   captures are the moves most likely to cause a cutoff. The *hash move* is
   the best move found for this same position by a previous, shallower
   search; it's usually still best, so trying it first is the strongest
   single ordering trick.
4. **Quiescence search** — never evaluate in the middle of a capture
   sequence (the "horizon effect"). `quiescence_search()` extends the search
   with captures/checks/promotions until the position is quiet.
5. **Transposition table** — different move orders reach the same position;
   cache results by Zobrist hash. A *Zobrist hash* gives every
   (piece, square) pair a fixed random 64-bit number and XORs together the
   numbers for the pieces on the board. Making a move only XORs a couple of
   numbers in and out, so the position's key updates almost for free. The subtle part is that alpha-beta often
   proves only a *bound*, not an exact value — see the `TT_EXACT` /
   `TT_LOWER_BOUND` / `TT_UPPER_BOUND` discussion in the file.
6. **Iterative deepening** — search depth 1, 2, 3… re-searching shallow
   depths is nearly free and solves time management. See
   `best_move_iterative()` and its *predictive* time check.

## RL lens: the Bellman backup, and planning with a model

Minimax computes a value with one rule, applied recursively: *the value of
a position is the best value you can reach in one move, seen from your
opponent's side*. In the side-to-move convention that's

```
V(s) = max over legal moves a of  -V(s after a)
```

That is the **Bellman optimality equation** for a two-player zero-sum game:
the value of a state, defined by the values of the states one step later.
Almost every RL method is a way of solving this equation approximately when
you can't afford to apply it everywhere.

Minimax can apply it exactly because it has a **model** of the environment:
the rules tell it precisely what position each move leads to. RL with a known
model is called *model-based*, and looking ahead with it is called
*planning*. Minimax plans to a fixed depth, then has to trust `evaluate()`
at the leaves. Better leaf values (Module 4) and smarter choices about which
lines to plan (Module 3) are what the rest of the course adds.

## Two measurement lessons (from this repo's history)

- **Measure before optimizing.** The first TT version hashed *every* node
  and was slower than no TT at all — Zobrist hashing at frontier nodes cost
  more than it saved. The fix (`use_tt = depth >= 2`) is two lines; finding
  it required timing, not cleverness.
- **Measure on the right workload.** The TT barely helps in shallow tactical
  positions (few transpositions) but *halves* the node count in king-and-pawn
  endgames. `tests/test_minimax.py::test_transposition_table_preserves_score_and_cuts_nodes`
  encodes this.

## Do

1. **Feel the pruning.** Count the nodes alpha-beta visits at depth 4, with
   move ordering and without it. `minimax()` counts nodes into a one-element
   list you pass as `nodes_searched`:
   ```python
   import chess
   import search.minimax as mm
   fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
   for label, order in [("ordered", mm.order_moves), ("unordered", lambda b, moves: list(moves))]:
       mm.order_moves = order          # minimax looks order_moves up at call time
       nodes = [0]
       mm.minimax(chess.Board(fen), 4, -1e9, 1e9, True, nodes)
       print(label, nodes[0])
   ```
   **Expected (measured):** about 23,000 nodes ordered vs 97,000 unordered
   (2.7 s vs 12 s), four times fewer. Now try the *starting* position: about
   13,600 vs 15,200, barely different. Why? MVV-LVA only reorders captures,
   and the start position has none. Node counts are deterministic, so any
   difference you see is real, not noise.
2. **Exercise: mid-search abort.** `best_move_iterative()` only checks the
   clock *between* iterations, and its predictive stop assumes each depth
   costs ~3x the last — but in tactical positions the real ratio can be 10x,
   so the search can still blow a 6-second budget by ten seconds. Add a hard
   deadline that aborts *inside* `minimax()` (raise a `SearchTimeout`
   every ~1024 nodes when past the deadline; catch it in the driver and
   fall back to the previous iteration's move). Your checkpoint is
   `tests/course/test_module02_search.py` — run it *before* implementing
   to watch the ~18-second failure, then un-skip and make it pass.
3. **Exercise: killer moves.** Quiet moves that caused a cutoff at one node
   often cause cutoffs at sibling nodes. Keep two "killer" moves per depth
   and try them right after the hash move. Only *quiet* moves (not
   captures) count as killers: captures are already ordered first by
   MVV-LVA. Measure with exercise 1's snippet, on its position *and* on the
   starting position, at depth 4.
   **Expected (measured with a reference solution):** the score must not
   change (it's still alpha-beta; if it does, you've broken something). Node
   counts: the starting position drops from 13,592 to about 8,100 (−40%),
   exercise 1's position from 23,228 to about 21,700 (−7%). Notice where it
   helps: quiet positions, exactly where MVV-LVA had nothing to reorder.
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
   halves the node count (we measured 11,066 → 5,072); in the opening at depth 3 it changes nothing (we measured
   3,141 nodes both ways), because short tactical searches barely transpose.
   Q2 below explains why the move can't change. The strength gain appears
   only when a *time budget* lets the faster engine search deeper.

## Check your understanding

<details>
<summary><b>Q1 (the rubber-duck question).</b> Why does a TT entry need a <em>flag</em>, not just a score?</summary>

Because alpha-beta often *doesn't compute the true value* of a node — it
stops as soon as it can prove the node irrelevant. If a cutoff fired, all
you learned is a one-sided bound: "at least X" (fail-high → LOWER_BOUND) or
"at most X" (fail-low → UPPER_BOUND). Reusing a bound as if it were exact
gives wrong answers when the new search window differs from the old one.
The flag records *which* of the three facts you actually proved. If you
want to see it break: change the store logic to always write `TT_EXACT` and
watch `test_transposition_table_preserves_score_and_cuts_nodes` fail on the
"same score" assertion.
</details>

<details>
<summary><b>Q2.</b> Predict: at a <em>fixed</em> depth of 3, does adding the TT change which move the engine picks? Its strength?</summary>

Neither — at fixed depth the TT is a pure speed optimization (same tree,
fewer node visits). Strength changes only when a *time budget* lets the
faster engine reach a deeper iteration than it otherwise would. This is
exercise 4's answer, and it generalizes: optimizations buy strength only
through the exchange rate of time-per-move.
</details>

<details>
<summary><b>Q3.</b> Move ordering never changes the final minimax value. So why did it cut the node count four times over in exercise 1?</summary>

Alpha-beta prunes a branch only after it has seen a move good enough to
prove the branch irrelevant. Search the best move first and the proof
arrives immediately — sibling after sibling gets cut. Search it last and
you've paid for the whole subtree before learning you didn't need it. Same
answer, wildly different node counts (exercise 1 puts numbers on it).
</details>

<details>
<summary><b>Q4 (RL lens).</b> Minimax applies V(s) = max over a of -V(s after a) exactly. What does it need to do that, which an RL agent in an unknown environment wouldn't have, and what does it still have to guess?</summary>

It needs a **model**: the rules, which say exactly which position each move
leads to, so it can look ahead without playing. An agent that doesn't know
its environment's rules can only learn by acting. What minimax still guesses
is the value at the leaves of its search, where it stops and trusts
`evaluate()`. Every later engine in the course is a better answer to that
same leaf-value question.
</details>

## Checkpoint

```bash
pytest tests/test_minimax.py -q                 # regression suite still green
pytest tests/course/test_module02_search.py -q  # un-skipped and passing
```

- [ ] Both green
- [ ] Your node counts from exercises 1 and 4, written down next to the expected ones
- [ ] You answered Q1 correctly *before* peeking

Previous: [Module 1 — Board & Evaluation](../module-01-board-and-evaluation/README.md) · Next: [Module 3 — Monte Carlo Tree Search](../module-03-mcts/README.md)
