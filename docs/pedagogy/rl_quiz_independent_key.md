# Answer Key: Reinforcement Learning Assessment (quiz2)

Scoring: each question is worth 1 point. Award **1** only when the answer contains the reasoning listed, not just the right vocabulary. Award **0.5** when the answer is correct in direction but misses a required element, or names the right concept without explaining the mechanism. Award **0** for wrong answers, pure jargon with no mechanism, or answers that only restate the code. A NOT COVERED answer scores 0, but record it separately: it is evidence about the course, not the learner.

General rule: an answer that just recites a repo fact ("the buffer is a deque with maxlen") without the RL reason scores at most 0.5.

---

## Q1. The formalism

**1 point** requires all of:
- States are board positions (plus whatever history the rules need); actions are legal moves; transitions are deterministic given both players' moves. From one player's view the opponent is part of the environment, and in self-play that opponent is the agent itself, so the environment changes as the agent learns.
- Reward is sparse and terminal: +1/0/-1 at game end, zero at every other step. Episodic with no discount (gamma = 1), so the return from any position is simply the final outcome.
- The value head estimates the expected final outcome **from the perspective of the side to move** at that position, **under the self-play policy that generated the data**: search-improved play with noise and temperature, mixed over the recent champions in the buffer. It is not the value under optimal play, and not the value under the raw network policy.

**0.5:** gets states/actions/terminal reward right but says the value is "the probability of winning" or "how good the position is" without naming the policy or perspective; or misses that the opponent is non-stationary.

---

## Q2. Why self-play improves

**1 point** requires:
- Search is a **policy improvement operator**: with lookahead and value backups, the visit distribution is (in expectation) stronger than the raw prior, because it corrects the prior using values of positions further ahead. Training the policy toward the visit counts distils that improvement into the network. Training the value head on outcomes of the improved play is **policy evaluation**. Alternating the two is (approximate) generalised policy iteration. Each improved network makes the next search stronger.
- New information enters from the **environment's rules and terminal outcomes**: exact terminal values inside search (checkmate = -1, draw = 0 in `puct_search`) and the real game results used as value targets. With random weights, the first signal comes from games that happen to reach mate or draw, and from search finding terminal states near the root. The improvement is not "the network teaching itself out of nothing": it is grounded in rewards.
- Should note the condition: improvement only holds if search is actually better than the prior, i.e. the value estimates are not badly wrong. Otherwise the loop can reinforce its own errors.

**0.5:** says "search is stronger than the network so imitating it improves it" but cannot say where new information comes from, or does not frame it as improvement plus evaluation.

---

## Q3. Value versus policy [Transfer]

**1 point** requires:
- Policy head: supplies priors P that decide **where search spends simulations** (breadth / which moves are worth exploring). Value head: replaces rollouts as the **leaf evaluation** that is backed up into Q (depth / judging outcomes).
- (a) Good policy, random value: search visits sensible moves, but Q values are noise, so backups carry no information. Visit counts stay roughly proportional to the priors (U term dominates), so search plays about as well as the raw policy, and more simulations add little except where it reaches real terminal states.
- (b) Good value, uniform policy: search wastes simulations spreading across all moves, which makes it shallow, but every backup is informative. As simulations grow it approaches a correct value-based search (like UCT with a good evaluator).
- Conclusion: (b) copes better as simulations increase, because more search converts a good value into good play, while more search cannot fix a value signal that carries no information. Credit an argued case that (a) is better at very **low** simulation counts.

**0.5:** describes the two jobs correctly but predicts the broken cases without the "more simulations helps only if backups carry information" reasoning, or gets one case wrong.

---

## Q4. Exploration

**1 point** requires:
- Without exploration, a deterministic search plus a deterministic network would play the same game over and over. The data would cover a tiny part of the state space and moves the network currently undervalues would never be tested, so their value estimates could never be corrected. That is the exploration/exploitation problem.
- **Root noise** perturbs what search *considers*: it forces some simulations onto low-prior moves at the root, so search can discover that a disliked move is actually good. That discovery then shows up in the visit-count targets. **Temperature** perturbs what is *played*: it diversifies the trajectories and so the states that reach the buffer, especially openings. They act at different levels. Noise alone still gives a near-deterministic argmax move sequence, so games stay similar. Temperature alone samples from a prior that never gets corrected, so a move with near-zero prior gets almost no visits and almost never gets played.
- Arena turns both off because it measures the strength of the **greedy (exploiting) policy**: exploration noise would add variance and measure a different, weaker policy than the one deployed.
- Strong answers note that because play is greedy after `temperature_moves`, value targets reflect a mix of exploratory and greedy behaviour (links to Q6).

**0.5:** explains that both "add randomness for diversity" without distinguishing search-level from action-level exploration, or misses why the arena disables them.

---

## Q5. Bootstrapping versus Monte Carlo [Transfer]

**1 point** requires:
- Game-outcome targets are **Monte Carlo**: unbiased samples of return under the behaviour policy, but **high variance**, because one result reflects every later move by both sides plus noise and temperature. Leaf evaluations in search are **bootstrapped**: they use the network's own estimate in place of the true return, so low variance but **biased** by the network's errors.
- Using root Q as the target is a TD-style / bootstrapped target. Gain: lower variance, faster learning, and more informative targets in long games where the outcome is far away.
- Risk early in training: the network's values are near-random, so root Q mostly reflects the network's own errors. Training on it makes the value function chase its own predictions (self-reinforcing bias, possibly drifting or collapsing to a constant such as 0) with no real outcome to correct it. Should mention that the only correction comes from terminal states search happens to reach. A good answer suggests mixing (e.g. averaging outcome and root Q, or n-step) or annealing from MC to bootstrapped.

**0.5:** classifies both correctly and states bias/variance, but cannot say what specifically goes wrong when bootstrapping from an untrained network; or gets the classification right with the bias/variance direction reversed (then score 0).

---

## Q6. Off-policy data

**1 point** requires:
- Value target: **no**, not unbiased for the current network. It is the outcome under older champions' search-plus-exploration policies, which differ from the current policy. That is off-policy (behaviour is not target), and exploration (noise, temperature) also biases it toward the exploratory policy's value.
- Policy target: different. It is a **supervised distillation target** (match this search distribution), not a return-weighted policy-gradient estimate. Old targets are "stale advice", lower quality because older search was weaker, but no estimator becomes mathematically biased in the way a likelihood-ratio gradient would. So importance sampling is not what is missing.
- Why it works without correction: the behaviour policies in the window are all close to the current one (recent champions, gate means slow change), so the mismatch is small, and the fixed point being sought (good play) is not destroyed by slightly stale targets. A FIFO window bounds the staleness.
- `buffer_capacity` trade-off: larger means more diverse, less correlated data and less forgetting, but older (more off-policy, weaker) targets. Smaller means fresher targets but overfitting to recent games and risk of oscillation.

**0.5:** says "the data is off-policy, the buffer window limits staleness" but does not separate value from policy targets, or claims importance sampling is needed or used.

---

## Q7. Stability with function approximation

**1 point** requires two distinct failure modes, each paired with a mechanism and how it acts. Acceptable pairs (any two):
- **Self-reinforcing collapse / degenerate data**: a worse network generates worse data, which trains a worse network. Dampened by the **arena gate** (only a candidate that demonstrated superiority becomes the data generator, so data quality cannot fall), and by exploration keeping coverage.
- **Overfitting to recent, correlated data / forgetting**: training on one game's nearly identical consecutive positions, or only the latest iteration, makes the network chase recent quirks and forget earlier knowledge. Dampened by the **replay buffer** with uniform random sampling (decorrelates batches, mixes generations).
- **Large destabilising updates**: a single bad batch causes a big weight change (e.g. value loss spikes on noisy outcomes). Dampened by `clip_grad_norm_`, a modest learning rate, weight decay, and a fixed, small number of `train_steps` per iteration (keeps the candidate close to the champion).
- **Moving targets**: targets depend on the network's own outputs. Dampened by decoupling: the candidate trains while the *frozen champion* generates data, a bit like a target network.

Must explain the mechanism (e.g. "the gate makes the data generator monotone-ish") rather than just naming the component.

**0.5:** one well-explained pair, or two pairs named without explaining how each dampens the problem.

---

## Q8. Credit assignment

**1 point** requires:
- Why it still works: the label is an unbiased sample of the return from that position under the behaviour policy. Positions that more often lead to wins get higher average targets **across many games**, and the blunder's +1 is offset by the many games in which a similar position-plus-blunder led to a loss. Credit assignment is solved statistically by averaging, not per game. Function approximation also generalises across similar positions.
- Cost: **high variance**. Each label is noisy, and the noise is larger for positions far from the end (more later decisions mixed in), so learning needs many more games. Draws give a 0 label everywhere, so draw-heavy play gives a weak signal.
- Move-level credit comes from the **search / policy target**: the visit distribution at each position comes from lookahead that evaluated each candidate move directly, so the policy is trained on a per-move judgement, not on the game result. Bootstrapped leaf values inside search also assign credit locally.

**0.5:** says "it averages out over many games" without explaining why (unbiased samples, offset by losing games), or omits the role of search in move-level credit.

---

## Q9. Evaluating agents [Transfer]

**1 point** requires all three parts in substance:
- (a) Score > 0.55 over 6 games means at least 3.5/6. For equal strength, the score distribution is symmetric around 3, so P(promote) = (1 - P(exactly 3)) / 2. Without draws that is (1 - 20/64)/2, about 34%. With draws it is somewhat lower, but still roughly one in four to one in three. So the gate barely filters: equal or slightly weaker networks are promoted often, and a slightly stronger one is frequently rejected. Any reasoned estimate in the 20-40% range earns credit. Extra credit (not required): arena games use temperature 0 with no noise, so they are near-deterministic. With alternating colours there may be only about 2 distinct games repeated, so the effective sample is even smaller than 6.
- (b) Loss is measured against targets the system produces itself, and those targets change as the champion and buffer change. Lower loss can mean the network matches its own (possibly poor) search better, or has overfitted the buffer, or that targets got easier (e.g. more draws, so value loss near 0). Loss measures agreement with the data, not performance in the environment, so strength must be measured by play.
- (c) Design: play a **fixed external reference panel** (e.g. the frozen iteration-0 net, `uniform_policy_value` at fixed sims, the minimax/MCTS baselines), plus several past champions, not only the immediate predecessor. Use enough games, varied openings or noise for distinct games, and confidence intervals or Elo with error bars. This exposes **non-transitivity / cycling**: A beats B, B beats C, C beats A, so "beats predecessor" can hold forever without absolute progress.

**0.5:** two of the three parts done properly, or all three gestured at without the probability reasoning in (a) or the non-transitivity point in (c).

---

## Q10. Truncation [Transfer]

**1 point** requires:
- Effect on targets: most positions get value 0, so the value head learns "everything is a draw". Its output loses information and backups inside search become uninformative (links to Q3a). Wins become rare, so the reward signal becomes extremely sparse, and play optimised for this objective learns that nothing it does matters within 40 plies (or only very fast mates do).
- Mismatch: the outcome now depends on **how many plies remain before the cap**, but the board encoding (`board_to_tensor`) does not include the ply count or time remaining. The same board can be "winning" with 30 plies left and "draw" with 2 plies left. So the target is not a function of the observed state: the problem is **no longer Markov in the network's input** (state aliasing / partial observability), which adds irreducible noise to the value targets.
- Principled fix, any one well argued: treat the cap as **truncation, not termination**, and bootstrap the value at the cutoff from the network (or the search value) instead of scoring 0. Or add remaining-plies to the state so the problem is Markov again. Or use adjudication with a strong evaluator. Raising the cap alone is acceptable only if the answer explains why (the cap becomes rare, so the aliasing affects few targets).

**0.5:** explains the draw-flooding of targets but not the non-Markov / missing-time-in-state problem, or gives a fix without explaining why it removes the bias.
