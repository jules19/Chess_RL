# Unit 9: Neural network + search, and the road to AlphaZero

**You'll be able to:**
- explain how `search/nn_mcts.py` combines the network with MCTS, and measure its cost
- explain the two changes AlphaZero makes (a policy prior in selection, and the value at the leaf
  instead of a rollout)
- describe the self-play loop that Phase 4 of this project would add

**Before this:** [Unit 5](05_mcts.md) and [Unit 8](08_measuring.md). **Next:** this is the end of
the path so far. See "Where to go from here". **Time:** ~60–90 min.
**Longer reading:** `neural_network_tutorial.html`, "Integration: From Neural Network to NN-MCTS";
`chess_rl_complete_tutorial.html`, "AlphaZero: Historical Timeline".

## The idea

The obvious way to use a trained network in MCTS is to swap it in wherever unit 5 called
`evaluate`. That's what `search/nn_mcts.py` does: the MCTS loop is unchanged, and the rollouts
use the network's value to pick moves. It works, but it's **very slow**, because every rollout
calls the network hundreds of times, one position at a time.

AlphaZero (2017) changed MCTS itself:

| | This project's NN-MCTS | AlphaZero-style (PUCT) |
|---|---|---|
| Choosing which child to explore | UCT: average value + exploration | **PUCT**: average value + exploration **weighted by the policy's probability** for that move |
| Judging a new leaf | Play a rollout, calling the net at each ply | **No rollout**: one network call, use its value directly |
| Network calls per simulation | ~465 (measured below) | **1** |

The policy head (unit 6) finally earns its keep: it tells the search which moves deserve
attention, so the search no longer wastes simulations on moves like `a2a3`.

**Reinforcement learning** closes the loop. Use that search to play games against itself, train
the network to predict the search's move choices (policy) and the game results (value), and repeat.
The better network makes the search stronger, and the stronger search produces better training
data. That loop is Phase 4 of this project, and it **hasn't been built yet**.

## In the code

| Where | What to look for |
|---|---|
| `search/nn_mcts.py`, `nn_mcts_search` | The same four steps as `mcts_search` (unit 5), but calling `simulate_with_nn` |
| `search/nn_mcts.py`, `simulate_with_nn` | The rollout: up to 50 plies, each scoring ~10 candidate moves with `nn_evaluator.evaluate` |
| `engine/nn_evaluator.py`, `NeuralNetworkEvaluator` | `evaluate` (value only) and `evaluate_with_policy` (value plus move probabilities), one position per call |
| `search/mcts.py`, `MCTSNode.uct_value` | The selection rule a PUCT version would replace |

## Try it

Count network calls during a 10-simulation search:

```bash
python -c "import chess, random; from engine.nn_evaluator import NeuralNetworkEvaluator; from search.nn_mcts import nn_mcts_search; ev = NeuralNetworkEvaluator('checkpoints/best_model.pt', verbose=False); n = [0]; f = ev.evaluate; ev.evaluate = lambda b: (n.__setitem__(0, n[0] + 1), f(b))[1]; random.seed(0); nn_mcts_search(chess.Board(), ev, simulations=10); print(n[0], 'network calls for 10 simulations')"
```

Expected output (takes ~10–15 seconds on CPU): `4652 network calls for 10 simulations`, i.e.
about **465 per simulation**. At the default 200 simulations that's ~93,000 calls for a single move.

## Exercises

**1. Back-of-envelope:** one network call on a single position costs about 1.8 ms on this
machine's CPU; scoring 64 positions in one batch costs about 0.27 ms *per position*. How long
would one move take at 200 simulations with (a) this project's NN-MCTS, (b) an AlphaZero-style
search with one call per simulation?

<details><summary>Answer</summary>

(a) 200 × 465 × 1.8 ms ≈ **170 seconds** per move.
(b) 200 × 1 × 1.8 ms ≈ **0.36 seconds**, and less again if leaves are batched.
That's a ~460× difference from the algorithm alone, before any hardware or batching tricks.
</details>

**2.** In PUCT, the exploration term is multiplied by the policy probability P(move). What
happens to a move the policy gives almost zero probability?

<details><summary>Answer</summary>

It gets almost no exploration bonus, so it's only searched if its value looks good anyway. This
focuses the search, but it also means a **bad policy can hide good moves**. AlphaZero adds random
noise to the root's policy during self-play to make sure every move gets some attention.
</details>

**3.** Why would PUCT with *this project's* current network (unit 8) probably play worse than plain
MCTS with the hand-written evaluator?

<details><summary>Answer</summary>

The network's value is barely better than a coin flip (54%), and its policy ranks the human
move first only 13% of the time. PUCT trusts both completely: no rollouts to correct a bad
value, and the policy decides where to look. A search is only as good as the judgement guiding
it, so the network has to be trained much better first (more data, held-out validation, or
self-play).
</details>

**4. Design:** sketch the Phase 4 self-play loop in five steps. What gets stored for each move?

<details><summary>Answer</summary>

1. Play a game against itself, choosing each move with PUCT search.
2. For every position store **(board, search visit distribution, final result)**. The visit
   distribution is a better policy target than one human move.
3. Train the network on a buffer of recent games.
4. Pit the new network against the current best, and keep it only if it wins clearly.
5. Repeat from 1 with the best network.
</details>

## Check yourself

1. What are AlphaZero's two changes to MCTS?
2. Roughly how many network calls per simulation does this project's NN-MCTS make, and how many
   does AlphaZero make?
3. What does self-play use as the policy target instead of human moves?

<details><summary>Answers</summary>

1. A policy-weighted exploration term in selection (PUCT), and using the value head at the leaf instead of a rollout.
2. ~465 vs 1.
3. The visit-count distribution produced by the search.
</details>

## Where to go from here

- Improve the network first: more and stronger games, a validation split by game, and longer
  training. Then rerun `notebooks/01_checkpoint_reality_check.py` and compare.
- Implement PUCT in a new search function (keep `nn_mcts_search` for comparison) and measure it
  against MCTS at **equal time**, not an equal number of simulations.
- Then build the self-play loop from exercise 4.
