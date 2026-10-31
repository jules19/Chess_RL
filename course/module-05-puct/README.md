# Module 5 — PUCT: Search Meets Network

> **Where you are:** search meets knowledge. Module 3's search, with Module 4's network plugged into its two holes. This is AlphaZero's actual search, and the agent of Module 6.
> **Time:** 4-6 hours, plus background matches.
> **You'll be able to:**
> - explain the PUCT formula term by term, and what the prior and the value each replace
> - show that search correctness and network quality can be tested separately
> - measure how much better the network makes the search, at equal simulations
> - explain why search is a *policy improvement* operator, the idea Module 6 runs on

The one-sentence version: **the network call replaces the rollout.**

## Concepts

All in `search/puct.py`, with teaching docstrings:

**No rollouts.** Classic MCTS estimates a leaf by playing the game out.
PUCT asks the value head instead: one forward pass, no simulation. That's
both faster *and* more accurate once the network knows anything.

**Priors steer exploration.** UCT explores every untried move (uniform
prior). PUCT weights the exploration bonus by the policy head:
```
score = Q + c_puct · P(move) · √N_parent / (1 + N_child)
```
With good priors, 800 simulations go *deep* on 3 candidate moves instead of
*wide* on 30. That's the entire practical difference.

**Perspective discipline.** The value is always "from the side to move".
It flips sign every ply on backpropagation, and the parent selects with
`-Q(child)` because a good position for the child's mover is bad for the
parent. This convention is documented once at the top of `puct.py` and used
consistently — sign bugs here are the classic AlphaZero implementation
failure, which is why `tests/test_puct.py` exists.

**Decoupling makes it testable.** `puct_search()` takes any
`policy_value_fn(board) -> (priors, value)`. The tests pass in
`uniform_policy_value` — a fake network with zero chess knowledge — and the
search *still finds forced mates*, because terminal nodes return exact
values. Search correctness and network quality are verified separately.

**Dirichlet noise & temperature** exist for self-play (Module 6): noise
makes *search* occasionally examine despised moves; temperature makes *move
selection* stochastic early in the game. Neither is used in serious play.

How the noise works (`add_dirichlet_noise` in `search/puct.py`): at the root
only, each prior becomes `(1 − ε)·P + ε·noise`, with ε = 0.25. The `noise` is
a random set of probabilities over the legal moves, drawn from a *Dirichlet*
distribution, which is simply a way of drawing random probability vectors.
Its parameter α sets their shape: a small α (0.3 here, as in AlphaZero's
chess) puts most of the noise on a few random moves, so each search gives a
couple of unfancied moves a real chance of being looked at, instead of
sprinkling a little on everything. Keeping 75% of the network's prior means
good moves still lead.

## RL lens: search improves the policy

The policy head gives a probability for each move, π(a | s). Use it on its
own and you get a fast, shallow player that never looks ahead. Run PUCT with
it and the visit counts at the root give a *different* distribution over
moves, one that has looked ahead and checked the network's hunches against
the values it finds. That distribution is a **better policy** than the
network's own: search(net) plays better than net.

An operation that turns a policy into a better one is called **policy
improvement**, and it's the engine of Module 6. If search always beats the
raw network, you can train the network to imitate its own search, then
search with the improved network, and repeat. That loop is AlphaZero.

Inside the tree, each child's average backed-up value is an estimate of the
action value Q(s, a). Unlike Q-learning, nothing stores those Qs between
moves: the network supplies V and π, and the search computes Q fresh.

## Read

- `search/puct.py` — the whole file; it's the heart of the course
- `search/nn_mcts.py` — the transitional Phase 3b design (rollout MCTS with
  a NN evaluator). Compare it to PUCT and note what it still wastes time on.

## Do

1. **Bridge Module 1 into PUCT.** Before touching the neural network,
   prove to yourself that PUCT's strength comes from *whatever* knowledge
   you plug in: implement `material_policy_value(board)` in
   `search/puct.py` — uniform priors, but a material-count value from the
   side-to-move's perspective. The contract (and the sign-convention trap)
   is in `tests/course/test_module05_puct.py`; un-skip it and make it pass.
   With zero knowledge, PUCT can't see a hanging queen (the test shows
   uniform picking an unrelated move); with mere piece-counting it piles
   visits onto the capture. That difference *is* the value head's job
   description.
2. **Watch priors work.** Run PUCT on the starting position with no
   knowledge and with your Module 4 network, and compare where the 400
   visits go. Start `python3` in the repository root and paste the snippet
   in; it needs `checkpoints/best_model.pt` from Module 4 Part B:
   ```python
   import chess
   from net.model import load_model
   from search.puct import best_move_puct, network_policy_value, uniform_policy_value
   model, _ = load_model("checkpoints/best_model.pt"); model.eval()
   best_move_puct(chess.Board(), uniform_policy_value, simulations=400, verbose=True)
   best_move_puct(chess.Board(), network_policy_value(model), simulations=400, verbose=True)
   ```
   **Expected (measured with the Module 4 quick-run network):** uniform
   priors spread the visits evenly, 20 per move across all 20 moves. The
   network put 177 visits on Nc3, 111 on Nf3 and 38 on d4, and only 5 moves
   got 10 or more. That's priors steering the search *deep* instead of
   *wide*. The network run takes about 8 seconds; each simulation is one
   forward pass.
3. **The Module 3 rematch.** PUCT + network against rollout MCTS, at equal
   simulations:
   ```bash
   python3 cli/match.py puct:sims=64,net=checkpoints/best_model.pt mcts:sims=64 \
       --games 6 --max-plies 120 --adjudicate 3
   ```
   (Module 3 explains the adjudication flags; `python3 cli/match.py --help`
   lists every engine spec and its options.)
   Then play it again with `mcts:sims=64,filter=0` and see whether the
   blunder filter still matters (Module 3, exercise 3, explains the filter).
   **Expected (measured, about 14 minutes, with the one-month network from
   Module 4 exercise 7):** PUCT scored 4/6 (67%), with a 95% interval of 30%
   to 90%. That leans PUCT's way, but six games can't settle it, and the
   network's value head is its weak part (Module 4). Write down your score
   and interval, and rerun with more games or a better network later. The claim to test is "a good value head makes rollouts
   and the crutch redundant", and you now have a way to test it.
4. **Exercise: c_puct sweep with a real network.** Play
   `puct:sims=64,net=checkpoints/best_model.pt,c=0.5` against the same with
   `c=3.0` (and each against the default 1.5), with the same
   `--max-plies 120 --adjudicate 3` flags. With *good* priors, lower
   exploration usually wins; confirm or refute with your own data, intervals
   included.
   **Expected (measured, 6 games, about 45 minutes, one-month network):**
   c = 0.5 scored 2.5/6 against c = 3.0, five draws and one loss: 42%, with
   an interval of 14% to 76%. So no conclusion, and with this network's weak
   value head, "low exploration wins" isn't even the leaning. That's a fair
   finding to report: the textbook result assumes good priors *and* a good
   value head.
5. **Exercise: batch the network calls.** Profile `puct_search` — nearly all
   the time is single-position forward passes. Collect N leaf evaluations
   before calling the network once. The catch: if you run selection N times
   without backing anything up, all N walks pick the *same* leaf, because
   nothing in the tree changed. **Virtual loss** fixes that: while a leaf is
   waiting for its evaluation, pretend every node on its path just lost a
   game (add a visit and a -1 to its value), so the next walk prefers a
   different branch. When the real values arrive, remove the pretend loss
   and back up the real one. This is *the* optimisation that makes real
   AlphaZero implementations fast; even a simple version is a big win on
   GPU.
   **Expected:** first measure the ceiling. Time `model(x)` for a batch of
   1, 8 and 32 copies of a position:
   ```python
   import time, chess, torch
   from net.encoding import board_to_tensor
   from net.model import load_model
   model, _ = load_model("checkpoints/best_model.pt"); model.eval()
   x = torch.from_numpy(board_to_tensor(chess.Board())).unsqueeze(0)
   with torch.no_grad():
       for b in (1, 8, 32):
           batch = x.repeat(b, 1, 1, 1); model(batch)          # warm up
           start = time.time()
           for _ in range(20):
               model(batch)
           per_call = (time.time() - start) / 20
           print(b, f"{per_call * 1000 / b:.2f} ms per position")
   ```
   We measured, on an 8-core CPU, 21 ms
   per position one at a time, 4.7 ms each in a batch of 8 and 1.7 ms each
   in a batch of 32. So batching 8 leaves can make the network part about
   4-5 times faster; the Python tree walk, which batching doesn't touch,
   caps the total. Two checks that your version is right: with a batch size
   of 1 it must give *exactly* the same visit counts as `puct_search`, and
   with any batch size the root visit counts must still add up to the
   simulation budget.

## Check your understanding

<details>
<summary><b>Q1 (the rubber-duck question).</b> How can the tests prove the search is correct using a network that knows nothing about chess?</summary>

Because search correctness and knowledge quality are *different claims*.
The search's job is mechanical: honor the PUCT formula, keep perspectives
straight, propagate terminal values exactly. All of that is observable with
uniform priors and zero values — mate-in-1 must dominate visit counts
purely through terminal values, visit counts must sum to the simulation
budget, sign conventions must round-trip. If those hold with a know-nothing
network, then any strength change when you plug in a real network is
attributable to the *network* — you've isolated the variables.
</details>

<details>
<summary><b>Q2.</b> Predict: in the PUCT score, what happens to the U term of a child as its sibling gets visited instead? Why is that the load-bearing property?</summary>

It *rises* — U scales with √N(parent), which grows on every simulation
regardless of which child it went through, while the ignored child's own
N stays put. So neglected children accumulate pressure and eventually get
rechecked, no matter how confident the priors were. That's the guarantee
that a wrong prior is an inefficiency rather than a permanent blind spot.
</details>

<details>
<summary><b>Q3.</b> Your Module 4 network's value head is decent but its policy head is garbage (say, near-uniform). Is PUCT-with-it closer to Module 3's MCTS, or broken?</summary>

Closer to a *rollout-free UCT*: uniform priors make the U term explore
breadth-first (like UCT), while the value head still replaces rollouts.
Weak priors cost efficiency, not correctness — you degrade gracefully
toward wider search. (This is also why the uniform fake network is a fair
baseline rather than a rigged one.)
</details>

<details>
<summary><b>Q4 (RL lens).</b> Why does it matter for self-play that the visit distribution after search is a better policy than the network's raw output? What would happen if it weren't?</summary>

Module 6 trains the network toward the search's visit counts. If search
produces a better policy, the network is trained toward something better
than itself and improves; repeat and it keeps improving. That's policy
improvement. If search were no better than the raw network, the network
would be trained to copy itself and nothing would change, however long the
loop ran.
</details>

<details>
<summary><b>Q5.</b> The network gives a move a prior of 0.001. With ε = 0.25 and 20 legal moves, roughly what prior can noise give it, and why use a small α rather than a large one?</summary>

At least 0.75 × 0.001, and up to about 0.25 if the noise happens to land on
it. With a large α the Dirichlet draw is close to uniform, about 1/20 = 0.05
per move, so the move would get roughly 0.75 × 0.001 + 0.25 × 0.05 ≈ 0.013
every time: a small, even nudge for everything. A small α concentrates the
noise, so in a given search one or two moves get a big boost and the rest
almost none. That's what gives a despised move a real chance of being
searched deeply enough to discover it's good.
</details>

## Checkpoint

```bash
pytest tests/test_puct.py -q                  # regression suite still green
pytest tests/course/test_module05_puct.py -q  # un-skipped and passing
```

- [ ] Both green
- [ ] Your Do #3 match result with its interval, and one sentence on what
      it says about your network's value head

Previous: [Module 4 — The Neural Network](../module-04-neural-network/README.md) · Next: [Module 6 — Self-Play RL](../module-06-selfplay-rl/README.md)
