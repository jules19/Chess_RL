# Module 3 — Monte Carlo Tree Search

> **Where you are:** the second search, the one AlphaZero is built on. You'll find the two holes a neural network will later fill.
> **Time:** 4-6 hours, plus matches that run in the background (MCTS is slow in Python; times below are measured).
> **You'll be able to:**
> - walk through one MCTS iteration (select, expand, simulate, backpropagate) and say what each step is for
> - explain the exploration/exploitation trade-off and what the constant C controls
> - estimate a position's value by Monte Carlo rollouts, and say why that estimate is noisy
> - name the two parts of `mcts.py` that Module 5 replaces with the network

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

**What the blunder filter is.** With `filter_blunders=True` (the default),
a node never even considers a move that leaves one of its pieces worth a
knight or more attacked and undefended (`hangs_material()` at the top of
`search/mcts.py`, a one-ply attacker/defender count). It's hand-written
chess knowledge bolted onto a search that is supposed to need none, which
is why exercise 3 measures it.

**Where this is heading:** notice that step 3 is just "estimate who's
winning" and step 1 needs "which moves are promising". Those are precisely
the *value head* and *policy head* of the network you build in Module 4.
Module 5 swaps them in — and deletes rollouts entirely.

## RL lens: exploration, bandits, and Monte Carlo evaluation

Three RL ideas sit inside this one algorithm.

**Exploration vs exploitation.** At each node MCTS must choose: spend the
next simulation on the move that looks best so far (exploit), or on one it
knows little about, in case it's better (explore)? The bare form of this
dilemma is the *multi-armed bandit*: several slot machines, unknown payouts,
limited pulls. UCT's formula is the classic bandit algorithm UCB1, applied at
every node: average result plus a bonus that shrinks the more a move is
tried. C sets the price of exploring.

**Monte Carlo evaluation.** A rollout estimates "how good is this position?"
by playing it to the end and recording the result. Average many and you get
an estimate of the value V(s) that needs no knowledge at all. Its weakness
is noise: one game is a single, very random sample. Better rollout policies
(exercise 1) reduce that noise.

**Action values in the tree.** Each child's average result is an estimate
of Q(s, a): how good it is to play move a here. MCTS never stores a learned
Q; it builds these estimates fresh for the current position, every move.
AlphaZero keeps this exactly, and learns V and the move priors instead.

## Read

- `search/mcts.py` — full teaching docstrings throughout
- `tests/test_mcts.py` — how the tricky invariants get pinned

## Do

**About time.** MCTS here is pure Python. At 200 simulations a move takes
about 10 seconds with evaluator rollouts and 1.4 seconds with random
rollouts (measured on an 8-core laptop CPU), so a 10-game match at 200 sims
takes hours. The matches below use 50 simulations; start one, then read
`search/mcts.py` while it runs. Before a long match, check your setup in
about a minute with a tiny one, e.g.
`python3 cli/match.py mcts:sims=10 mcts:sims=10,rollouts=random --games 2 --max-plies 20 --adjudicate 3`:
it proves the command works, not who is stronger.

At 50 simulations MCTS rarely finishes a game: it can win material but not
convert it, so most games would hit the length cap as draws and tell you
nothing. `--max-plies 120 --adjudicate 3` ends each game after 60 moves and
gives it to whichever side is at least 3 pawns of material ahead, the way
engine tournaments stop games that would otherwise run forever.

1. **Feel the rollout quality cliff.** Evaluator rollouts against random
   rollouts:
   ```bash
   python3 cli/match.py mcts:sims=50 mcts:sims=50,rollouts=random --games 6 \
       --max-plies 120 --adjudicate 3
   ```
   **Expected (measured, about 15 minutes):** evaluator rollouts scored
   3.5/6 (58%), with a 95% interval of 24% to 86%, and `match.py` reports
   that the match can't tell the engines apart. Six games are enough to
   catch a disaster, not to measure a gap like this. If you want the answer,
   run `--games 40` overnight; the exercise's real point is learning to read
   the interval before you believe the score.
   The gap is the value of rollout knowledge. Remember it for Module 5,
   where a network replaces the rollout entirely.
2. **Exercise: exploration constant sweep.** First look, then play. On one
   position, compare the visit distributions:
   ```python
   import chess
   from search.mcts import best_move_mcts
   board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
   for c in (0.5, 1.41, 3.0):
       print("C =", c)
       best_move_mcts(board, simulations=200, exploration_constant=c, verbose=True)
   ```
   **Expected (measured, ~9 s per run):** with C = 0.5 the top three moves
   took 32, 31 and 31 of the 200 visits: the search commits early to a
   handful. With C = 3.0 the top move got only 12, close to an even spread
   over 27 legal moves: it never commits. Notice also *which* moves led at
   C = 0.5: Ke2 was among them. That's what noisy Monte Carlo values look
   like at 200 simulations. Then play `mcts:sims=50,c=0.5` against
   `mcts:sims=50,c=3.0` with `cli/match.py` (same flags as exercise 1), and
   write one paragraph on what
   too-low and too-high C look like and which won, with its interval.
3. **Exercise: the blunder filter is a crutch — measure it.**
   ```bash
   python3 cli/match.py mcts:sims=50 mcts:sims=50,filter=0 --games 6 \
       --max-plies 120 --adjudicate 3
   ```
   **Expected (measured, about 50 minutes, because both sides use evaluator
   rollouts):** 3.5/6 for the side with the filter, interval 24% to 86%:
   again, not enough games to say.
   Report the score with its interval, not an Elo number: six games can't
   pin an Elo down (Module 7). Keep the result in mind when the neural
   network makes the crutch unnecessary.
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

<details>
<summary><b>Q4 (RL lens).</b> A rollout from a position returns +1. Another from the same position returns -1. What does that tell you about Monte Carlo value estimates, and what are the two ways to make them better?</summary>

That a single rollout is one noisy sample of the position's value: the same
position can be won or lost depending on random choices later. You improve
the estimate by **averaging more rollouts** (more simulations: slower), or by
**making each rollout less random**, with a better rollout policy (exercise
1). AlphaZero takes a third way: it stops rolling out and asks a learned
value function instead.
</details>

## Checkpoint

```bash
pytest tests/test_mcts.py -q                  # regression suite still green
pytest tests/course/test_module03_mcts.py -q  # un-skipped and passing
```

- [ ] Both green
- [ ] Q1 answered correctly before peeking

Previous: [Module 2 — Minimax Search](../module-02-minimax-search/README.md) · Next: [Module 4 — The Neural Network](../module-04-neural-network/README.md)
