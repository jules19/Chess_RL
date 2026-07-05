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
   clock *between* iterations. Add a hard deadline that aborts *inside*
   `minimax()` (hint: raise a `SearchTimeout` exception every 1024 nodes if
   past the deadline; catch it in the driver and fall back to the previous
   iteration's move). Then tighten the assertion in
   `test_iterative_deepening_respects_time_budget` from `< 8.0` seconds to
   `< 2.0` and make it pass.
3. **Exercise: killer moves.** Quiet moves that caused a cutoff at one node
   often cause cutoffs at sibling nodes. Keep two "killer" moves per depth
   and try them right after the hash move. Measure the node reduction.
4. **Ladder evidence.** 10 games: depth-3 with TT+deepening vs depth-3
   without. Any difference at fixed depth? Why/why not? (Answer: at *fixed*
   depth the TT changes speed, not moves — the strength gain appears when a
   time budget lets the faster engine search deeper.)

## Checkpoint

```bash
pytest tests/test_minimax.py -q
```

- [ ] All green, including your tightened time-budget test (exercise 2)
- [ ] You can explain to a rubber duck why a TT entry needs a *flag*, not
      just a score

Next: [Module 3 — Monte Carlo Tree Search](../module-03-mcts/README.md)
