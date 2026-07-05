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
4. **De-flake the search.** First write a test asserting MCTS (200 sims,
   evaluator rollouts) finds mate-in-1 from
   `6k1/5ppp/8/8/8/8/5PPP/4R1K1 w` — run it 10 times and watch it flake.
   The cure is reproducibility: thread an optional `rng: random.Random`
   parameter through `mcts_search`/`best_move_mcts` and every place
   `search/mcts.py` reaches for the global `random` module. (Why not just
   `random.seed(7)`? It "works" — and silently changes the behavior of
   every other consumer of `random` in the process. Explicit rngs are the
   habit that transfers.) Checkpoint:
   `tests/course/test_module03_mcts.py` — un-skip and make it pass.

## Check your understanding

<details>
<summary><b>Q1 (the rubber-duck question).</b> Which two pieces of <code>mcts.py</code> will Module 5 replace with neural network calls?</summary>

The **simulation step** (`simulate_with_evaluator` — the whole rollout is
replaced by one call to the value head) and the **uniform treatment of
untried moves** in expansion/selection (replaced by policy-head priors in
the PUCT formula). Everything else — selection walk, expansion,
backpropagation with sign flips — survives into `puct.py` almost unchanged.
</details>

<details>
<summary><b>Q2.</b> Predict: you delete the <code>value = -value</code> sign flip in <code>backpropagate</code>. What does the engine now do, concretely?</summary>

It plays for its *opponent*: every simulation result credited to a node is
also credited (same sign) to the parent, so moves that are good for the
side who just moved look good for the side choosing — the engine actively
walks into mates. Run `tests/test_mcts.py::test_backpropagation_updates_and_alternates_perspective`
after making the change and watch it pin the bug in milliseconds. This is
the #1 real-world MCTS bug; you'll meet it again wearing NN clothes in
Module 5.
</details>

<details>
<summary><b>Q3.</b> Why is the final move chosen by <em>visit count</em> instead of best average value?</summary>

A move visited 3 times with a lucky 0.9 average is noise; a move visited
150 times with 0.6 held that value under repeated adversarial scrutiny —
UCT only keeps visiting a child that keeps looking good. Visits are
value + confidence in one number. (Foreshadowing: in Module 6 the visit
distribution literally becomes the training target.)
</details>

## Checkpoint

```bash
pytest tests/test_mcts.py -q                  # regression suite still green
pytest tests/course/test_module03_mcts.py -q  # un-skipped and passing
```

- [ ] Both green
- [ ] Q1 answered correctly before peeking

Next: [Module 4 — The Neural Network](../module-04-neural-network/README.md)
