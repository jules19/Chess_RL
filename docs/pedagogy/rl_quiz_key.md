# Answer key (0 / 0.5 / 1 per question). Full marks need the ideas, not wording.

Q1. Agent: the PUCT-search + network player (playing both sides; the opponent is
itself). Environment: chess rules (python-chess) + the opponent's replies. State:
board position. Action: a legal move. Reward: 0 every move, game result (+1/0/-1)
only at the end (sparse / terminal). 0.5 if reward timing or opponent-is-self missing.

Q2. No - supervised learning / imitation learning (behaviour cloning) on expert
moves (+ outcome regression). Ceiling: it can at best match the teacher
(humans in the data); it cannot discover better play; no feedback from its own
actions. 0.5 if named "supervised" but no ceiling.

Q3. Improvement: PUCT search using the current net (search(net) stronger than net;
visit distribution is an improved policy). Evaluation: value head trained on game
outcomes of the current champion's play. Back into net: policy head trained toward
the visit distributions (soft-target cross-entropy), then the arena gate. 0.5 if
only roles partially right.

Q4. Monte Carlo (full-episode return, no bootstrapping). Advantage: no bootstrap
bias / simple / unbiased for the policy that played. Disadvantage: high variance;
credit spread over whole game; needs many games. 0.5 if MC but no pros/cons.

Q5. Dirichlet noise on root priors (inside the search: which moves get examined);
temperature on move selection early in the game (which move is played -> data
diversity); PUCT/UCT exploration bonus U term (inside search, every node). Full
marks needs noise + temperature with correct levels; 0.5 for one.

Q6. Buffer fills with data from old, weak policies (stale / off-policy data); value
targets describe how old policies' games ended, not the current one; training
chases old play, improvement slows/stalls; also correlated old data dominates.
0.5 for "old data" without why it hurts.

Q7. +1 from the winner's perspective (side to move at that position who won).
Credit assignment problem / noisy Monte Carlo labels; tolerated because averaged over
many games the signal dominates noise (statistics, not per-position cleverness).
0.5 if target right but no name/reason.

Q8. No. It learns a state-value V(s) (value head) and a policy pi(a|s) (policy head).
Action values Q(s,a) are estimated by the search tree (mean backed-up values per
child). 0.5 if says V and policy without where Q comes from.

Q9. Change: rules/move generation (python-chess), board encoding (net/encoding.py
board_to_tensor), move indexing/policy size, network input/output shapes, terminal
detection. Unchanged: PUCT search logic, self-play loop, replay buffer, arena
gate, training losses. 0.5 if partial.

Q10. Training on finite noisy self-generated data with function approximation does
not guarantee improvement; a worse network generates worse data -> feedback loop
can spiral/collapse; gate only promotes a candidate that demonstrates >threshold
score. 0.5 if just "newest might be worse".

Q11. At toy CPU scale: promotions are rare/noisy; small arena matches (6-20 games)
have huge binomial error bars; expect plumbing proof not strength. Tell real
learning from noise: larger matches, error bars, champion vs initial checkpoint,
watch losses / value accuracy. 0.5 for partial.
