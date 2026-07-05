# Module 3 — Monte Carlo Tree Search

**Goal:** learn the *other* family of game-tree search — the one AlphaZero
is built on — and understand exactly where a neural network will later slot in.

## Concepts

Minimax explores the tree *exhaustively to a fixed depth*. MCTS explores it
*selectively to no fixed depth*, spending simulations where they matter.
Every iteration has four steps (see `search/mcts.py`):

1. **Selection** — walk down the tree picking children by **UCT**:
   `Q + C·√(ln N_parent / N_child)`. The first term exploits (good average
   results), the second explores (rarely-visited children get a bonus).
   This is the multi-armed-bandit tradeoff, applied recursively.
2. **Expansion** — add one new node.
3. **Simulation (rollout)** — estimate the leaf's value by playing the game
   out. Pure random rollouts are hilariously weak; this repo uses
   *evaluator-guided* rollouts (`simulate_with_evaluator`) — 1-ply greedy
   play with the Module 1 evaluator.
4. **Backpropagation** — push the result up the path, **flipping the sign
   each ply** (good for White = bad for Black). Sign-flipping is the #1
   source of MCTS bugs; `tests/test_mcts.py` pins it down.

**Why visits, not values, pick the final move:** average value is noisy for
rarely-visited moves; a move only accumulates visits by surviving repeated
UCT scrutiny. Visit count *is* the search's confidence.

**Where this is heading:** notice that step 3 is just "estimate who's
winning" and step 1 needs "which moves are promising". Those are precisely
the *value head* and *policy head* of the network you build in Module 4.
Module 5 swaps them in — and deletes rollouts entirely.

## Read

- `search/mcts.py` — full teaching docstrings throughout
- `tests/test_mcts.py` — how the tricky invariants get pinned

## Do

1. **Feel the rollout quality cliff.** Play `use_evaluator=False` (random
   rollouts) vs `use_evaluator=True` at 200 simulations, 10 games. The gap
   is the value of rollout knowledge — remember it for Module 5, where a
   network replaces the rollout entirely.
2. **Exercise: exploration constant sweep.** Play C=0.5 vs C=1.41 vs C=3.0
   (10 games each pairing, 200 sims). Which wins? Write one paragraph on
   what too-low and too-high C *look like* in the visit distributions
   (`verbose=True` prints them).
3. **Exercise: the blunder filter is a crutch — measure it.**
   `filter_blunders=True` papers over weak rollouts with a hand-written
   tactical veto. Measure its Elo contribution (10 games on vs off). Keep
   this number in mind when the neural network makes the crutch unnecessary.
4. **Write a failing test first.** Add a test asserting MCTS (200 sims,
   evaluator rollouts) finds mate-in-1 from
   `6k1/5ppp/8/8/8/8/5PPP/4R1K1 w`. If it flakes, make it deterministic by
   seeding `random` — learning to de-flake stochastic tests is part of the
   module.

## Checkpoint

```bash
pytest tests/test_mcts.py -q
```

- [ ] Green, including your new mate-in-1 test
- [ ] You can point at the two lines of `mcts.py` that Module 5 will replace
      with neural network calls

Next: [Module 4 — The Neural Network](../module-04-neural-network/README.md)
