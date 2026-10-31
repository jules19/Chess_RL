# Module 5: PUCT, Search Combined with a Network

> **Where you are:** this module adds Module 4's network to Module 3's search in two places. The value head replaces rollouts, so positions are scored by the network instead of by playing games out. The policy head replaces the equal treatment of every untried move, so the network's move probabilities decide which moves the search explores first. The result is PUCT, the search AlphaZero uses. In Module 6 it becomes the agent, the player that learns.
> **Time:** 4-6 hours, plus background matches.
> **You'll be able to:**
> - explain the PUCT formula term by term, and what the prior (the policy head's probability for a move) and the value each replace
> - show that search correctness and network quality can be tested separately
> - measure how much better the network makes the search, with the same number of simulations (search iterations) on both sides
> - explain why search is a policy improvement step (it turns a policy into a better one), the idea Module 6 is built on

The name PUCT stands for Predictor + Upper Confidence bounds for Trees. The
"predictor" is the network's prior for each move. "Upper confidence bound"
is the idea behind UCT (Upper Confidence bounds applied to Trees) in Module 3: add a bonus to moves the search knows
little about, so they still get tried.

## Concepts

Everything below is in `search/puct.py`, which has docstrings written for teaching.

The search builds a tree of positions, as in
[Module 3](../module-03-mcts/README.md). Each position in the tree is a
node. A node's children are the positions one move later, and the node is
their parent. The root is the current position, and a leaf is a node the
search has not yet expanded.

**No rollouts.** Module 3's rollout MCTS (Monte Carlo tree search) estimates
a leaf by playing the game out. PUCT uses the value head's output instead: one forward pass
(one run of the network on one position). Once the network has learned
anything, this is both faster and more accurate.

**Priors guide exploration.** UCT (Upper Confidence bounds applied to Trees,
the selection rule of rollout MCTS) explores every untried move, which is
the same as giving every move an equal prior.
PUCT scales the exploration bonus by the policy head's prior. At each node,
the search picks the child with the highest score:
```
score = Q + c_puct · P(move) · √N_parent / (1 + N_child)
```
The symbols mean:
- Q: the child's average backed-up value so far (the average of the results
  passed back up through it), from the parent's point of view.
- P(move): the prior, the policy head's probability for this move.
- N_parent and N_child: how many times the search has visited the parent
  and this child.
- c_puct: a constant that sets how much weight exploration gets (default 1.5).

The second term is *U*, the exploration bonus. It is large for moves with a
high prior and few visits. With good priors, 800 simulations explore 3
candidate moves deeply instead of 30 moves shallowly. That is the main
practical difference from UCT.

**Sign convention.** The code stores each node's value from the point of
view of the side to move in that node. When the search passes a value back
up the tree, the sign flips at every ply (one move by one side). The code
calls this step backpropagation. In neural-network training the same word
means computing the gradients, which `loss.backward()` does in Module 4; the
two are unrelated.

Because a child's stored value is from the child's side to move, a good
position for the child is a bad one for the parent. So the code computes
the formula's Q, which is from the parent's point of view, as `-Q(child)`.
This convention is documented once at the top of `puct.py`. Sign mistakes
here are easy to make and hard to notice, which is why `tests/test_puct.py`
exists.

**Separating search from the network makes it testable.** `puct_search()` takes any
`policy_value_fn(board) -> (priors, value)`. The tests pass in
`uniform_policy_value`, a fake network with no chess knowledge. The search
still finds forced mates, because terminal nodes (checkmate or draw) return
exact values.

**Dirichlet noise and temperature** exist for self-play (Module 6). Noise
makes the search sometimes examine moves the network rates very low.
Temperature controls how random the final move choice is: at 0 the engine
plays the most-visited move; higher values pick moves at random, weighted by
visit count. Self-play uses it early in the game. Neither is used when the
engine plays to win.

How the noise works (`add_dirichlet_noise` in `search/puct.py`): at the root
only, each prior becomes `(1 − ε)·P + ε·noise`. Here ε (epsilon) is the
weight given to the noise, 0.25. The `noise` is a random set of probabilities
over the legal moves, drawn from a *Dirichlet* distribution, which is a way of
drawing random probability vectors. Its parameter α (alpha) sets how evenly the
noise is spread across moves. A small α (0.3 here, as in AlphaZero's chess) puts most of the noise on a few
random moves, so each search gives a couple of low-rated moves a real
chance of being looked at. Keeping 75% of the network's prior means good
moves still get the most visits.

## RL connection: search improves the policy

The policy head gives a probability for each move, π(a | s): the probability
of choosing move a in position s. Use it on its own and you get a fast
player that never looks ahead. Run PUCT with it, and the visit counts
at the root give a different distribution over moves, one that takes the
search's look-ahead into account.
That distribution is a better policy than the network's own: search with the
network plays better than the network alone.

An operation that turns a policy into a better one is called *policy
improvement*, and Module 6 is built on it. If search always beats the
raw network, you can train the network to imitate its own search, then
search with the improved network, and repeat. That repeated cycle is the
AlphaZero training method.

Inside the tree, each child's average backed-up value is an estimate of the
action value Q(s, a), the expected result of playing move a in position s.
Q-learning ([RL concepts](../RL_CONCEPTS.md)) stores and learns Q values.
PUCT stores none between moves. The network supplies the value V(s) and the
policy π, and the search computes Q again for each move.

## Read

- `search/puct.py`: the whole file. It is the central file of the course.
- `search/nn_mcts.py`: an earlier design (rollout MCTS that uses the network
  to evaluate positions). Compare it to PUCT and note what it still spends
  time on unnecessarily.

## Do

1. **Use Module 1's material count as PUCT's value.** Before you use the neural network,
   check for yourself that PUCT's strength comes from whatever knowledge
   you give it. Implement `material_policy_value(board)` in
   `search/puct.py`: uniform priors, but a material-count value from the
   side to move's point of view. The requirements, including how to get the
   sign convention right, are in `tests/course/test_module05_puct.py`. Remove
   its `@pytest.mark.skip` line and make it pass.
   **Expected:** with no knowledge, PUCT picks an unrelated move instead of
   capturing an undefended queen. With only piece-counting,
   it puts most of its visits on the capture. That difference is what the
   value head is for.
2. **Compare where the visits go, with and without priors.** Run PUCT on the starting position with no
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
   got 10 or more. The priors make the search deep on a few moves instead of
   wide on all of them. The network run takes about 8 seconds; each
   simulation is one forward pass.
3. **Exercise: PUCT against rollout MCTS.** Play PUCT with the network
   against rollout MCTS, with the same number of simulations each:
   ```bash
   python3 cli/match.py puct:sims=64,net=checkpoints/best_model.pt mcts:sims=64 \
       --games 6 --max-plies 120 --adjudicate 3
   ```
   1. Run the command above. Module 3 explains the adjudication flags, and
      `python3 cli/match.py --help` lists every engine and its options.
   2. Write down the score and its 95% interval. The 95% interval is the
      range the true score probably lies in
      ([Module 3](../module-03-mcts/README.md), exercise 1, explains how to
      read it).
   3. Play the match again with `mcts:sims=64,filter=0` and see whether the
      blunder filter still matters (Module 3, exercise 3, explains the filter).

   **Expected (measured, about 14 minutes, with the one-month network from
   Module 4 exercise 7, which was trained on one month of Lichess games):** PUCT scored 4/6 (67%), with a 95% interval of 30%
   to 90%. The result favours PUCT, but six games are too few to decide, and
   the network's value head is its weak part (Module 4). Rerun with more
   games or a better network later. The claim to test is that a good value
   head makes both rollouts and the blunder filter unnecessary.
4. **Exercise: compare several c_puct values with a real network.** With
   good priors, lower exploration is expected to win. Test this with three
   matches, each with the same `--max-plies 120 --adjudicate 3` flags:
   1. `puct:sims=64,net=checkpoints/best_model.pt,c=0.5` against the same
      engine with `c=3.0`.
   2. c = 0.5 against the default 1.5.
   3. c = 3.0 against the default 1.5.

   Report each score with its 95% interval.
   **Expected (measured for match 1 only, 6 games, about 45 minutes,
   one-month network; matches 2 and 3 have no reference result yet):**
   c = 0.5 scored 2.5/6 against c = 3.0, five draws and one loss: 42%, with
   a 95% interval of 14% to 76%. So there is no conclusion. With this
   network's weak value head, the result does not even point towards "low
   exploration wins". That is a fair finding to report: the expectation that
   low exploration wins depends on both good priors and a good value head.
5. **Exercise: batch the network calls.** The task is to collect N leaf
   evaluations and then call the network once for all of them.

   The difficulty: if you run selection N times without backing anything up,
   all N walks down the tree pick the same leaf, because nothing in the tree
   changed. *Virtual loss* solves this. While a leaf is waiting for its
   evaluation, treat every node on its path as if it had just lost a game
   (add a visit and a -1 to its value), so the next walk prefers a different
   branch. When the real values arrive, remove the temporary loss and back
   up the real value. Real AlphaZero implementations rely on this
   optimisation, and even a simple version gives a large speed-up on a GPU
   (graphics processor).

   1. Measure where `puct_search` spends its time with a profiler (a tool
      that reports time per function). Nearly all of it goes on forward
      passes for one position at a time.
   2. Measure the largest possible speed-up. Time `model(x)` for a batch of
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
   **Expected:** on an 8-core CPU we measured 21 ms per position one at a
   time, 4.7 ms each in a batch of 8 and 1.7 ms each in a batch of 32. So
   batching 8 leaves can make the network part about 4-5 times faster. The
   Python tree walk is not affected by batching, so it limits the total
   speed-up.
   3. Implement batched search with virtual loss. **Check:** with a batch
      size of 1 it must give the same visit counts as `puct_search`, and
      with any batch size the root visit counts must still add up to the
      number of simulations.

## Check your understanding

<details>
<summary><b>Q1 (explain it in your own words).</b> How can the tests prove the search is correct using a network that knows nothing about chess?</summary>

Because search correctness and knowledge quality are separate claims.
The search's job is mechanical: apply the PUCT formula correctly, keep the
sign convention consistent, and pass terminal values up exactly. All of that
can be checked with uniform priors and zero values: mate-in-1 must receive
most of the visits through terminal values alone, visit counts must sum to
the number of simulations, and signs must stay correct. If those hold with a
network that knows nothing, then any change in strength with a real network
comes from the network.
</details>

<details>
<summary><b>Q2.</b> Predict: in the PUCT score, what happens to the U term (the exploration bonus) of a child as its sibling gets visited instead? Why is that property essential?</summary>

It rises. U grows with √N_parent, which increases on every simulation
whichever child it went through, while the ignored child's own N_child does
not change. So a child that is not visited gets a steadily larger bonus and
is eventually checked again, however confident the priors were. This
guarantees that a wrong prior only makes the search slower. It cannot stop
the search from ever examining a move.
</details>

<details>
<summary><b>Q3.</b> Your Module 4 network's value head is reasonably accurate but its policy head is very poor (say, near-uniform). Is PUCT with this network closer to rollout MCTS, or broken?</summary>

It is closer to rollout MCTS without the rollouts. Uniform priors make the U
term explore all moves evenly (as UCT does), while the value head still replaces rollouts.
Weak priors make the search less efficient, but it still finds correct
results: it searches more widely. (This is also why the uniform fake network
is a fair baseline: it does not favour any move over another.)
</details>

<details>
<summary><b>Q4 (RL question).</b> Why does it matter for self-play that the visit distribution after search is a better policy than the network's raw output? What would happen if it weren't?</summary>

Module 6 trains the network toward the search's visit counts. If search
produces a better policy, the network is trained toward something better
than itself and improves; repeat and it keeps improving. That's policy
improvement. If search were no better than the raw network, the network
would be trained to copy itself and nothing would change, however long the
loop ran.
</details>

<details>
<summary><b>Q5.</b> The network gives a move a prior of 0.001. With ε = 0.25 and 20 legal moves, roughly what prior can noise give it, and why use a small α rather than a large one?</summary>

At least 0.75 × 0.001, and up to about 0.25 if the noise happens to fall on
it. With a large α the Dirichlet draw is close to uniform, about 1/20 = 0.05
per move, so the move would get roughly 0.75 × 0.001 + 0.25 × 0.05 ≈ 0.013
every time: a small, equal increase for every move. A small α concentrates
the noise, so in a given search one or two moves get a large increase and the
rest almost none. That is what gives a low-rated move a real chance of being
searched deeply enough to discover it's good.
</details>

## Checkpoint

```bash
pytest tests/test_puct.py -q                  # regression suite still green
pytest tests/course/test_module05_puct.py -q  # un-skipped and passing
```

- [ ] Both commands pass
- [ ] Your match result from Do, exercise 3, with its 95% interval, and one
      sentence on what it says about your network's value head

Previous: [Module 4: The Neural Network](../module-04-neural-network/README.md) · Next: [Module 6: Self-Play Reinforcement Learning](../module-06-selfplay-rl/README.md)
