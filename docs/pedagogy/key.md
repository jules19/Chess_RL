# Answer key (verified against the code on 2026-10-30)

Score each answer 0 (wrong / don't know / generic), 0.5 (partly right), 1 (right, with the key point).

1. **20.** The mobility term counts legal moves of the side to move only (+20 with White to move). Key point: side-to-move mobility asymmetry.
2. **2,470** at depth 3. Depth 1 gives 41 because `nodes_searched` also counts quiescence-search calls (root + 20 + 20). Key point: quiescence nodes are counted. (Number within ±5% or the reason alone = 0.5.)
3. **The horizon effect** (evaluating mid-exchange). It searches captures and promotions, plus checks when there are fewer than 5 noisy moves, with a stand-pat bound. Key point: captures + promotions, and checks conditionally. ("Captures only" = 0.5.)
4. The **most-visited** child; **C = 1.41** (≈√2).
5. ~20 root moves share 200 simulations (~10 visits each), and the rollouts are noisy, so differences between moves are within the noise. Key point: too few simulations per move / statistical noise.
6. **20 × 8 × 8** input; **4,672** policy outputs.
7. **Queen, rook and bishop promotions** get an index ≥ 4672, outside the policy, and are dropped from training. **Knight promotions** collide with the non-promotion move's index. Key point: the offset `(piece-2)*4096` overflows or collides. (Either half = 0.5.)
8. Policy: **the move actually played**, cross-entropy loss. Value: **the final game result** from the side to move's view, MSE loss.
9. The train/validation split is **by position, not by game**, so positions from the same games (and shared openings) appear in both. ("Too little data" alone = 0.)
10. Weak. For example: validation top-1 ≈ 3%; it ranks the human move first 13% of the time vs 6% for random; it won **0 of 40** games against random, material and minimax opponents. Any measured figure showing weakness = 1. Quoting a target or an Elo estimate as the result = 0.
11. **~465** calls per simulation (rollouts call the net at every ply for ~10 candidates). AlphaZero uses **one** call per simulation: the value head at the leaf instead of a rollout, with the policy prior guiding selection (PUCT). (Order of magnitude "hundreds" = 1; "one call/PUCT" without the number = 0.5.)
12. **No** (Phase 4 not started).
13. A **static evaluation of the current position from White's perspective**, not the search result and not from the side to move (engine's) point of view. (Either point = 0.5.)
14. **No, never measured.** They were estimates or targets: no rated-opponent matches exist. Evidence: STATUS/README say so, or the absence of any measurement. Answering "yes" = 0.
