# Quiz 3 Marking Key

Each question is worth 1 mark. Award 1, 0.5 or 0. Full marks need the *reasoning*; naming the right term without the mechanism earns at most 0.5. An answer of NOT COVERED scores 0 but should be recorded separately (it signals a gap in the course rather than in the learner). Answers that are correct but rely on obvious outside knowledge should still be marked on content; flag them.

Transfer questions: Q3, Q4, Q5, Q7, Q8, Q9, Q10 (7 of 10).

---

## Q1. The formalism

**1 mark:** States = board positions (including side to move and enough history for repetition/castling rules); actions = legal moves; transitions are deterministic given the move, but from one player's viewpoint the opponent's reply makes the next own-turn state depend on the opponent's policy. Reward is 0 at every step and +1/0/-1 at the terminal position; the objective is the expected final outcome (the return). It is well defined without discounting because every episode terminates (game rules, plus the move cap here), so the undiscounted sum is finite. Zero-sum shows up as negation: values are stored from the side-to-move's perspective, and backup flips the sign each ply (a parent selects using -Q of the child), so one network serves both players.

**0.5:** Gets states/actions/terminal reward right but omits either (i) why no discounting is fine (episodic termination) or (ii) the sign-flip / side-to-move convention. Or describes the sign flip but not what is being maximised.

**0:** Talks about "rewards for capturing pieces" as the reward function, or cannot identify the return/objective.

## Q2. Why self-play improves

**1 mark:** Search (PUCT) combines the network's priors with lookahead and value evaluations of many future positions, so its visit distribution is usually a *better* policy than the raw policy head: search acts as a policy improvement operator. Training the policy head toward that improved distribution, and the value head toward outcomes of games played by that improved policy (policy evaluation), gives a cycle of evaluation and improvement (policy iteration). Next iteration, the better network makes search better still. The condition: search must actually be stronger than the raw network, which requires the value estimates (and enough simulations) to carry real information; if search adds nothing over the prior, the target equals the current policy and learning stalls. Credit also for noting that exploration must keep producing informative, diverse data.

**0.5:** Says "the search is stronger so imitating it helps" without explaining *why* search is stronger (lookahead + value) or without stating the improvement condition. Or describes policy iteration abstractly without connecting it to visit counts and z.

**0:** Says it improves because "it plays lots of games" or "it learns from its mistakes" with no mechanism.

## Q3. [T] Monte Carlo vs bootstrapped value targets

**1 mark:** z is an unbiased sample of the return under the policy that played the game, but high variance: one game's result depends on every later move, including exploratory and blundering ones. The search root value is a bootstrapped estimate: lower variance, but biased by the current network's own errors, since it is built from the network's value outputs at leaves. From random weights, the root value is essentially the network's noise reflected back; training on it would teach the network to agree with itself, with no anchor to the real game outcome. It could converge to a self-consistent but wrong value function (or collapse toward a constant such as 0). The true signal only enters through terminal positions, which are rare and shallow early on. Compromise: mix the two (e.g. average z and root Q, or a schedule that starts mostly z and shifts toward Q as the network becomes reliable), or use n-step targets that bootstrap only after some real moves.

**0.5:** Correctly states the bias/variance trade-off but does not predict the early-training failure (bootstrapping from garbage, no grounding), or predicts the failure without explaining that it comes from bias. Or offers a compromise with no justification.

**0:** Claims the root value is strictly better because it is "computed by search", or confuses which target is biased.

## Q4. [T] The move cap and a sparse reward

**1 mark:** If nearly every game is capped and scored 0, nearly every value target is 0, so the value head learns to predict about 0 everywhere and carries no information about which positions are good. Search then gets no useful leaf values (only the rare real checkmates inside the tree give signal), so the visit distributions are close to the priors and the policy targets do not improve either: without a reward signal there is nothing to propagate, a sparse-reward / credit-assignment failure. Material scoring: benefit is a denser, informative signal early, which can bootstrap learning. Risk: it changes the objective (reward shaping that is not potential-based), so the agent may optimise material rather than winning, for example by refusing sacrifices or grabbing material into lost positions, and the bias can persist after the network is strong. Credit for suggesting it be phased out, or used only as an adjudication for clearly decided positions.

**0.5:** Identifies value collapse to 0 but not the knock-on effect on policy targets, or gives the material benefit but no concrete risk (or vice versa).

**0:** Says capped draws are harmless "because draws are a legitimate result", or misses that the value head learns nothing.

## Q5. [T] Exploration

**1 mark (all three parts substantively right):**
(a) Without noise and with greedy selection, self-play becomes deterministic: the same champion plays essentially the same game every time, so the buffer fills with near-duplicate positions from one line. The network overfits to that line, never sees the consequences of moves it currently rates low (so it can never discover they are good), and the value/policy estimates elsewhere stay wrong. Improvement stalls, or the agent learns a narrow exploitable repertoire.
(b) The target should represent search's judgement of all moves (a soft distribution carrying relative information), not which move happened to be chosen; the choice of the played move is an exploration/behaviour decision and should not change what search believed. Training on one-hot greedy choices would make the policy collapse prematurely and throw away information.
(c) The arena is measuring strength of the policy you would actually deploy; exploration noise would add random weaker moves and measure a different (noisier, weaker) policy. Exploration is for generating data, not for evaluation. (Strong answers note this then creates the determinism issue in Q9.)

**0.5:** Two of three parts right with reasons, or all three asserted without mechanism (e.g. "exploration is needed to explore").

**0:** Says noise in the arena would be fine or beneficial, or cannot say what happens to the data in (a).

## Q6. Off-policy data

**1 mark:** z estimates the expected outcome under the *behaviour* policy that generated that game: the champion of that time *plus* its exploration (noise in the early moves, temperature), against itself. That differs from the value of the current greedy player because (i) older champions were weaker, and (ii) exploratory moves (mainly early in the game) made games worse or noisier than greedy play would. So the value target is off-policy and slightly stale, and no importance correction is applied; this is tolerated because exploration is limited to early plies/root and the window keeps data fairly recent. Window size: too small means training only on the latest games, which gives highly correlated data, overfitting to recent quirks, and forgetting (instability); too large means training on obsolete targets from much weaker players, slowing or biasing learning.

**0.5:** Explains the window trade-off well but is vague on which policy z belongs to, or says "it's off-policy" without saying in what way it differs from the current greedy policy.

**0:** Claims the data is on-policy, or that z is the true value of the position under optimal play.

## Q7. [T] Stability and the gate

**1 mark:** Each iteration the candidate restarts from the champion, so if a candidate is rejected, *all* its training is thrown away. With few train steps, a candidate can only move a tiny distance from the champion, so it is nearly identical and will score about 50%, below the 55% bar. Meanwhile the champion never changes, so self-play keeps generating similar data. The loop can be stuck forever even though cumulative training would help. Fixes (any two, justified): more train steps per iteration; keep training a persistent learner across iterations and only gate *which net generates data* (accumulate progress rather than discarding it); lower the threshold or use a statistically grounded test; drop the gate entirely and always use the latest net (AlphaZero-style); more arena games so real small improvements can be detected. What the gate protects against: the network generates its own training data, so a worse network produces worse data, which trains a worse network, a feedback loop that can spiral downward. The gate keeps the data generator from regressing, a role similar to a target network or slow-moving reference that stabilises learning with function approximation.

**0.5:** Explains the feedback-loop rationale for the gate but not the "discarded progress" mechanism, or diagnoses the stall but cannot say what the gate is for. Fixes given without reasons count for little.

**0:** Concludes the network "can't learn" or the learning rate must be wrong, with no reference to the restart-from-champion structure.

## Q8. [T] Credit assignment

**1 mark:** z is a noisy label; the blunder position is one sample. Across many games, positions resembling that blunder will mostly be followed by losses (when the opponent punishes it), so the regression target *averages* toward the true expected outcome: noise averages out, the bias does not. Finer credit comes from search: (i) the policy target for the blunder position comes from visit counts, and if search saw the rook loss it would have put few visits on the blunder, so the policy is not trained toward it even though it was played (it may have been played because of temperature or noise); (ii) during search, the value head is applied to the positions *after* the blunder, and lookahead evaluates individual moves, which gives per-move credit that the game result alone cannot. More damaging when: opponents systematically fail to punish (weak or same self-play agent that shares the blind spot, so the "mistake" really does lead to wins in the data), few games so averaging cannot happen, or long games where one result is smeared over many moves.

**0.5:** Gives the "averages out over many games" argument only, or only the search argument, without the other; or names "credit assignment problem" without the mechanism.

**0:** Says the network will learn the blunder is good and nothing corrects it, or that the label is wrong and should be changed by hand.

## Q9. [T] Honest evaluation

**1 mark:** (a) With no noise, temperature 0 and deterministic networks, every game with the candidate as White is identical, and likewise as Black, so the 6 games are really 2 distinct games repeated three times (sample size is effectively 2, not 6). The score can only take a few values (0, 25, 50, 75, 100%) and is dominated by luck in two specific games; it says almost nothing about strength, and with 2 effective games the 55% bar is easily passed or failed by chance. (b) Strength is not guaranteed to be transitive: a candidate can learn to exploit the champion's particular weaknesses (or beat it in its one opening) while getting worse against other play, so "beats predecessor" can cycle or drift. (c) Better: diversify openings (a book of varied starting positions, or small randomness in the first few moves, applied the same way to both sides with colours swapped), play many more games, report a confidence interval or use a sequential test rather than a fixed threshold on a small sample, and measure against fixed external references (older checkpoints, a fixed baseline such as the uniform network or a classical engine, tactical test positions) to track absolute progress.

**0.5:** Notes "6 games is too few / noisy" but misses the determinism point in (a), or gets (a) but gives no reason for (b). A protocol with no justification for its parts is half credit for (c).

**0:** Says 6 games at 55% is a reliable test, or that determinism makes the result *more* reliable.

## Q10. [T] Roles of policy and value

**1 mark:** (a) Value always 0: Q is uninformative except at real terminal nodes inside the tree, so search is guided almost entirely by the priors and visit counts follow the policy. The engine plays plausible-looking "intuitive" moves, finds short forced mates or tactics that end inside the search horizon, but cannot judge positions, so it misjudges quiet positions and long-term consequences, and search adds little over the raw policy. (b) Uniform priors with a good value head: search spreads simulations evenly across all legal moves (UCT-like), so with a fixed budget it goes wide and shallow; judgement of each candidate is good but it cannot afford depth, so it misses deeper tactics and wastes simulations on absurd moves. Policy = where to look (focusing search, pruning breadth); value = how good a position is (replacing playouts, limiting depth). A value head lets a leaf be scored with one network call: a learned estimate of the expected game outcome from that position, which is what a random rollout was trying to estimate, but with far lower variance (and less bias than random play, which does not resemble real play).

**0.5:** Both cases described but without the breadth/depth framing or without saying what the engine would be weak at; or the rollout explanation is missing or just "the network is faster."

**0:** Swaps the roles of the two heads, or says either sabotage would have little effect.
