# Module 5 — PUCT: Search Meets Network

**Goal:** combine Modules 3 and 4 into AlphaZero's actual search algorithm —
and understand the one-sentence version: **the network call replaces the
rollout.**

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

## Read

- `search/puct.py` — the whole file; it's the heart of the course
- `search/nn_mcts.py` — the transitional Phase 3b design (rollout MCTS with
  a NN evaluator). Compare it to PUCT and note what it still wastes time on.

## Do

1. **Watch priors work.** Run PUCT on the starting position with
   `uniform_policy_value` and with your Module 4 network
   (`network_policy_value(model)`), 400 sims each, `verbose=True`. Compare
   how concentrated the visit counts are.
2. **The Module 3 rematch.** Play PUCT+network vs the Module 3 rollout MCTS,
   equal simulations, 20 games. Then re-enable Module 3's blunder filter and
   see if it still matters. (The crutch should now be redundant — the value
   head sees hanging pieces.)
3. **Exercise: c_puct sweep with a real network.** 0.5 / 1.5 / 3.0, ten
   games each. With *good* priors, lower exploration usually wins; confirm
   or refute with your own data.
4. **Exercise: batch the network calls.** Profile `puct_search` — nearly all
   the time is single-position forward passes. Collect N leaf evaluations
   before calling the network once (hint: "virtual loss" is the standard
   trick for selecting N distinct leaves). This is *the* optimization that
   makes real AlphaZero implementations fast; even a simple version is a
   big win on GPU.

## Checkpoint

```bash
pytest tests/test_puct.py -q
```

- [ ] Green
- [ ] PUCT+your-network beats rollout MCTS at equal simulations (Do #2)
- [ ] You can explain why the tests can prove search correctness with a
      network that knows nothing about chess

Next: [Module 6 — Self-Play RL](../module-06-selfplay-rl/README.md)
