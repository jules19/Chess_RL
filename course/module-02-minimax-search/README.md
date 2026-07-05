# Module 2 — Minimax Search

**Goal:** turn a 1-ply evaluator into an engine that *looks ahead*, then make
that search fast enough to matter.

## Concepts

Read these in `search/minimax.py` — each has a teaching docstring at the
exact place it's implemented:

1. **Minimax** — assume the opponent replies with *their* best move.
   `minimax()` alternates maximizing (White) and minimizing (Black) levels.
2. **Alpha-beta pruning** — stop searching a branch the moment you prove the
   opponent would never allow it. Same answer as plain minimax, a fraction
   of the work. Effectiveness depends entirely on…
3. **Move ordering** — searching good moves first makes cutoffs happen early.
   See `order_moves()` (MVV-LVA for captures) and the *hash move* trick.
4. **Quiescence search** — never evaluate in the middle of a capture
   sequence (the "horizon effect"). `quiescence_search()` extends the search
   with captures/checks/promotions until the position is quiet.
5. **Transposition table** — different move orders reach the same position;
   cache results by Zobrist hash. The subtle part is that alpha-beta often
   proves only a *bound*, not an exact value — see the `TT_EXACT` /
   `TT_LOWER_BOUND` / `TT_UPPER_BOUND` discussion in the file.
6. **Iterative deepening** — search depth 1, 2, 3… re-searching shallow
   depths is nearly free and solves time management. See
   `best_move_iterative()` and its *predictive* time check.

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

1. **Feel the pruning.** From the starting position, count nodes at depth 4
   with move ordering enabled vs `order_moves` replaced by an identity
   function. Write the two numbers in a comment or notebook.
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
   and try them right after the hash move. Measure the node reduction.
4. **Ladder evidence.** 10 games: depth-3 with TT+deepening vs depth-3
   without. Any difference at fixed depth? Why/why not? (Answer: at *fixed*
   depth the TT changes speed, not moves — the strength gain appears when a
   time budget lets the faster engine search deeper.)

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
<summary><b>Q3.</b> Move ordering never changes the final minimax value. So why is it worth ~half the engine's speed?</summary>

Alpha-beta prunes a branch only after it has seen a move good enough to
prove the branch irrelevant. Search the best move first and the proof
arrives immediately — sibling after sibling gets cut. Search it last and
you've paid for the whole subtree before learning you didn't need it. Same
answer, wildly different node counts (exercise 1 puts numbers on it).
</details>

## Checkpoint

```bash
pytest tests/test_minimax.py -q                 # regression suite still green
pytest tests/course/test_module02_search.py -q  # un-skipped and passing
```

- [ ] Both green
- [ ] You answered Q1 correctly *before* peeking

Next: [Module 3 — Monte Carlo Tree Search](../module-03-mcts/README.md)
