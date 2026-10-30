# Module 6 — Self-Play Reinforcement Learning

> **Where you are:** the top rung. Remove the human games: the engine generates its own experience, and every piece from Modules 1-5 becomes one step of a reinforcement-learning loop.
> **Time:** 5-8 hours, plus one overnight run. Part A (tic-tac-toe) takes about an hour and every run finishes in seconds. Part B (chess) is mostly waiting.
> **You'll be able to:**
> - explain the AlphaZero loop as RL: which part evaluates the policy, which improves it, and where the reward comes from
> - watch an RL loop learn, read its learning curve, and break it on purpose in three different ways
> - compare Monte Carlo and TD value targets from your own measurements
> - run the chess loop, and write a report that separates real learning from noise

## Concepts

The loop (see the diagram in `selfplay/__init__.py`):

```
self-play (champion + exploration) → replay buffer → train candidate
        ↑                                                   │
        └────────── arena gate: promote only if >55% ───────┘
```

### The RL view of this loop

Every RL term from [RL_CONCEPTS.md](../RL_CONCEPTS.md) has a concrete home
here:

| RL idea | In this loop |
|---|---|
| agent | PUCT search + the champion network, playing both sides |
| environment | the rules + the opponent, who is the agent itself |
| reward | 0 per move; +1 / 0 / -1 at the end of each game |
| **policy improvement** | search: the visit counts are a better policy than the raw network (Module 5) |
| **policy evaluation** | the value head learns, from game results, how good positions are *for the current player* |
| writing the improvement back | the policy head is trained toward the visit counts |
| experience | the replay buffer of (state, visit counts, result) examples |

Put together, that's **policy iteration**: evaluate the current policy,
improve it, repeat. Exact policy iteration is guaranteed to get better each
round. This sampled version, with a neural network, is not, which is what the
arena gate is about.

**The training target is the search, not the move** (`selfplay/self_play.py`).
Each position's policy target is the full PUCT *visit distribution*. The
network learns to predict what search-amplified-network concludes — and since
search(net) is stronger than net, imitating it is self-improvement. This one
idea is AlphaZero.

**The value target is a Monte Carlo return.** Every position gets the final
result of its game, from the point of view of the side to move there: +1 in
the winner's positions, −1 in the loser's, including the winner's mistakes. That's a *Monte Carlo* target: the real return, no
guessing. The alternative is a *temporal-difference (TD)* target, which
bootstraps from the network's own estimate of the next position. Part A lets
you run both. Labelling a whole game with one result is also the crudest
answer to the **credit assignment** problem (which moves earned the win?):
don't decide per position, and let averaging over thousands of games sort
the signal from the noise.

**The replay buffer is a window** (`selfplay/replay_buffer.py`): big enough
to decorrelate batches and span several generations, small enough that
ancient weak-network games age out. In RL terms, training on games played by
*earlier* champions is **off-policy** data: it describes how a different,
weaker player's games went. A little is useful; too much and the network
learns about players that no longer exist. Capacity is the dial.

**The arena is a gate, not a scoreboard** (`selfplay/arena.py`): a bad
training iteration can make the network worse, and if the worse network
generates the next data, the loop can spiral. Candidates must *demonstrate*
superiority (>55%, colors alternated, no exploration noise) to take over.
AlphaGo Zero used this gate. AlphaZero dropped it and always trained the
newest network; Part A shows one reason why.

**Exploration is load-bearing.** Without it, self-play replays one game
forever. There are three levels: the U term inside every search, Dirichlet
noise on the root priors (which moves get examined at all), and temperature
for the first moves (which move is actually played). Part A shows what
happens when you remove them.

**Soft-target loss** (`selfplay/train_loop.py::soft_target_loss`): the
supervised loss was cross-entropy against a class index; now the target is a
distribution, so it's `-Σ π(a)·log p(a)`. The test suite proves the two
coincide for one-hot targets.

## Read

All four files in `selfplay/` that make up the loop — they're short and
heavily documented — then `selfplay/tictactoe_lab.py`, which reuses
`puct_search` and the replay buffer unchanged on a different game.

## Do — Part A: watch it learn (tic-tac-toe)

Chess self-play at laptop scale proves the plumbing but rarely shows
learning. Tic-tac-toe is small enough that the same loop learns in seconds,
and it's solved, so the lab can score the agent against perfect play.
The "network" is a table with one entry per position; Q2 below asks what
that changes.

1. **Run it** (about 7 seconds):
   ```bash
   python3 selfplay/tictactoe_lab.py
   ```
   Each row is one iteration (100 self-play games, then training).
   The columns are the learning curve:

   | Column | Means |
   |---|---|
   | `value_mse` | how far the learned values are from the true game values, over all 4,520 positions (lower is better) |
   | `best_move` | % of positions where the table's favourite move is a perfect move |
   | `coverage` | % of positions the agent has ever seen in self-play |
   | `vs_perfect` | the table alone (no search) vs a perfect player: % of games not lost. You can't beat a perfect player, so 100% means it always draws |
   | `vs_random` | the table alone vs a random mover: score % |

   **Expected (measured, seed 0):**
   ```
   iter value_mse best_move  coverage vs_perfect vs_random
      0     0.767      58.7       0.0          0        64
      5     0.695      65.6      21.5         38        65
     10     0.663      68.4      27.5         85        83
     20     0.636      70.2      32.3         89        80
   ```
   From losing every game against perfect play to drawing about 9 in 10, by
   playing only against itself. `vs_perfect` bounces around by ±10 between
   iterations: it's 100 games, so it's noisy.

   Two things to notice. `coverage` stops near a third: the agent only
   learns about positions its own play reaches, and a table can't
   generalise to the rest. And `vs_random` is *lower* than `vs_perfect`: a
   random opponent steers the game into positions the agent never saw, where
   the table knows nothing. Generalising to unseen positions is exactly what
   the chess network must do and the table can't.

2. **Sabotage: remove exploration.** Predict first: what will `coverage` do?
   ```bash
   python3 selfplay/tictactoe_lab.py --no-root-noise --temperature-moves 0
   ```
   <details><summary>What we measured</summary>

   Coverage stays at 0.5% and `value_mse` never moves (0.766). With no noise
   and no temperature, every game is the same game. The buffer fills with
   one game repeated, and the agent learns that one line perfectly and
   nothing else.
   </details>

3. **Turn the gate on.** Predict first: with a 55% promotion threshold,
   what happens once both sides play well?
   ```bash
   python3 selfplay/tictactoe_lab.py --gate
   ```
   <details><summary>What we measured</summary>

   It learns for 7 iterations, then freezes: the last promotion is at
   iteration 7, and from iteration 8 every arena score is 42-50%. The
   champion never changes again. (Its `vs_perfect` still wobbles between 72
   and 82: that's the *same* table re-measured on 100 new games, which tells
   you how noisy that column is.) Good tic-tac-toe is a draw, so a better candidate can't
   *show* it's better: it draws, scores 50%, and fails a 55% gate. The gate
   protects against regressions, and it also blocks progress whenever
   progress is hard to measure. That's one reason AlphaZero dropped it.
   </details>

4. **Off-policy data: a buffer that never forgets.** Predict first, then run
   three seeds of each, because one run proves nothing:
   ```bash
   for seed in 0 1 2; do
     python3 selfplay/tictactoe_lab.py --seed $seed | tail -1
     python3 selfplay/tictactoe_lab.py --seed $seed --buffer-capacity 1000000 | tail -1
   done
   ```
   <details><summary>What we measured</summary>

   A small effect. The final `value_mse` was worse with the endless buffer
   in all three seeds (0.636 vs 0.642, 0.621 vs 0.638, 0.603 vs 0.622):
   old games describe how weaker players' games ended. `vs_perfect` went
   both ways (89 vs 77, 84 vs 82, 83 vs 85), which is noise at 100 games.
   Concluding from one seed would have got this wrong in either direction.
   </details>

5. **Monte Carlo vs TD value targets.** `--td` replaces the game result with
   minus the table's own value of the next position (the opponent's view,
   flipped). Predict: which learns values faster early on?
   ```bash
   python3 selfplay/tictactoe_lab.py --td
   ```
   <details><summary>What we measured</summary>

   TD starts slower. At iteration 5 its `value_mse` was worse in all three
   seeds (about 0.72 vs 0.69); by iteration 20 the two are level. TD learns
   from its own guesses, which start at 0 everywhere, so news of a win has
   to travel back one move per update. Monte Carlo gives every position the
   real result at once. TD's advantage, lower variance, matters more in long
   games with noisy outcomes than in tic-tac-toe.
   </details>

6. **Exercise: build n-step returns.** Monte Carlo and TD are the two ends
   of one dial. An *n-step* target looks n moves ahead: if the game ends
   within n moves, use the real result; otherwise use the table's value of
   the position n moves later. Rewards are 0 mid-game, so that's the whole
   target. One subtlety: the side to move alternates, so the value from n
   moves ahead must be negated when n is odd. n = 1 is TD; n larger than any
   game is Monte Carlo.
   Implement `n_step_targets(results, values, n)` in
   `selfplay/tictactoe_lab.py`, give `selfplay_game` and `run` an `n_step`
   parameter that uses it, and add a `--n-step` flag. The contract is
   `tests/course/test_module06_nstep.py`: un-skip it and make it pass. Then
   measure n = 1, 3 and the default Monte Carlo over three seeds, looking at
   `value_mse` at iterations 5 and 20.
   **What to expect:** if your n = 1 doesn't reproduce `--td`'s numbers,
   your sign is wrong. Predict before you measure: where will n = 3 land?
   <details><summary>What we measured (reference solution, 3 seeds)</summary>

   Not in between. At iteration 5, n = 3 was as good as Monte Carlo or a
   little better (`value_mse` 0.677-0.693, vs 0.684-0.695 for Monte Carlo
   and 0.713-0.719 for TD), and at iteration 20 it was lowest in two seeds
   of three. Three moves is long enough to reach the real result in most of
   a tic-tac-toe game, and the bootstrapped part averages away some of
   Monte Carlo's noise. A middle setting can beat both ends: that's why
   practical RL systems tune n.
   </details>

## Do — Part B: the chess loop

7. **Run the toy loop** (about 1.5 minutes on an 8-core CPU; it proves
   plumbing, not strength):
   ```bash
   python3 selfplay/train_loop.py --iterations 2 --games-per-iter 2 \
       --simulations 16 --train-steps 20 --arena-games 2
   ```
   **Expected (measured):** 0 promotions, arena scores of 50%, a policy loss
   around 7.7 and a value loss near 0. The policy loss is close to ln(4672) ≈
   8.4, which is what a network scores when it spreads its probability evenly
   over all 4,672 outputs: an untrained network knows nothing yet. With
   `--arena-games 2` the arena can only score 0%, 25%, 50%, 75% or 100%, so
   it is plumbing here, not a measurement. Each iteration also appends a row to
   `selfplay_checkpoints/run_log.csv`. Why is the value loss near 0? A
   random network never finishes a game inside the 200-ply cap, so every
   game is scored a draw and every reward is 0. That is the **sparse reward
   problem** in action: the value head has nothing to learn from until games
   start ending. It's also why warm-starting from Module 4 helps so much.
8. **Run a real (small) loop overnight.** Warm-start from your Module 4
   supervised checkpoint — this is what AlphaGo did before AlphaZero went
   tabula rasa:
   ```bash
   python3 selfplay/train_loop.py --init-checkpoint checkpoints/best_model.pt \
       --iterations 10 --games-per-iter 20 --simulations 64 --train-steps 300
   ```
   Expect several hours on a laptop CPU (a rough estimate scaled from the
   toy run; measure yours from the first iteration's timing and write it
   down). `selfplay_checkpoints/run_log.csv` records, per iteration, the
   buffer size, the policy and value losses, the arena score and whether the
   candidate was promoted. A healthy run shows the policy loss falling
   between iterations; promotions may be rare (Q1). Then test whether the
   loop gained anything:
   ```bash
   python3 cli/match.py puct:sims=64,net=selfplay_checkpoints/champion.pt \
       puct:sims=64,net=checkpoints/best_model.pt --games 20
   ```
   Twenty games is a smoke test (±22 points, Module 7 Part 0): it can catch a
   big gain or a collapse. For a claim you'd defend, run 100 overnight.
9. **Exercise: sabotage study, chess edition.** Three short runs:
   (a) `--no-root-noise`, (b) `--temperature-moves 0`,
   (c) `--promote-threshold 0.30`. Use the toy settings from step 7 with
   `--iterations 5`, e.g.
   `python3 selfplay/train_loop.py --iterations 5 --games-per-iter 2 --simulations 16 --train-steps 20 --arena-games 2 --no-root-noise --checkpoint-dir sabotage_a`.
   Predict each failure mode from what Part A showed you *before* running,
   then check `run_log.csv` (look at the promoted column and whether the
   policy loss moves). Write three
   sentences, including whether chess at this scale even shows the effect.
10. **Exercise: buffer-size sweep.** `--buffer-capacity 2000` vs `50000` at
   fixed games per iteration. Which promotes more often early? Later? How
   many seeds would you need to believe the answer (Part A, step 4)?
11. **Exercise: resumable loops.** The loop already saves
   `champion.pt`/`replay_buffer.npz`; implement
   `load_resume_state(checkpoint_dir)` and a `--resume` flag so a killed
   run continues (load buffer + champion, keep iteration numbering). Long
   RL runs die; checkpointing discipline is part of the craft. Contract
   and checkpoint: `tests/course/test_module06_resume.py` — un-skip it.

## Check your understanding

<details>
<summary><b>Q1.</b> Predict: at this module's scale (tens of games, 6 arena games per gate), roughly how often should you <em>expect</em> a genuinely-better candidate to fail promotion?</summary>

More often than not. A candidate whose true score is 55% needs at least
4 wins out of 6 to clear a 55% gate (with no draws), which happens only
about 44% of the time, so it fails promotion more often than it passes.
Binomial noise at n=6 is enormous (Module 7, Part 0, has the formula). So "no promotion this iteration"
is weak evidence of anything. This is why the checkpoint below grades your
*interpretation* of the arena numbers, not the numbers themselves.
</details>

<details>
<summary><b>Q2.</b> The policy target is the visit distribution rather than the move played. Give the one-sentence reason this makes self-play <em>self-improving</em> rather than self-imitating.</summary>

Search-amplified-network is stronger than the raw network, so training the
network to match the *search's* conclusions pulls it toward a stronger
player than itself — repeat, and the target keeps moving up. Imitating
only the move played would throw away almost all of the search's judgment
(everything it learned about the moves it *didn't* pick).
</details>

<details>
<summary><b>Q3.</b> You remove Dirichlet noise but keep temperature (sabotage study, run (a)). What failure do you predict, mechanically?</summary>

Temperature still varies which *root* move gets played, but within each
search the priors alone decide what gets explored — moves the current
network dislikes get almost no visits, so their true value is never
discovered, so the training data never contradicts the network's
prejudices. The loop converges on a self-consistent, possibly wrong,
opening repertoire: exploration in move *selection* can't compensate for
no exploration in *search*.
</details>

<details>
<summary><b>Q4 (RL lens).</b> In the tic-tac-toe lab, what does the table do that the chess network can't, and what does the chess network do that the table can't?</summary>

The table stores an exact answer for every position it has seen, with no
interference between positions. That's possible only because tic-tac-toe has
4,520 non-terminal positions. The network can't store chess's ~10^44, but it
can **generalise**: give a sensible answer for a position it has never seen,
because similar positions share features. The table's `coverage` column and
its weak `vs_random` score show what having no generalisation costs.
</details>

<details>
<summary><b>Q5 (RL lens).</b> In a self-play game the winner blundered on move 20, but the opponent didn't punish it. What value target does the move-20 position get, what's the name of the problem, and why does the algorithm put up with it?</summary>

+1 if the eventual winner is the side to move in that position (-1 if it's
the loser): the Monte Carlo target is the game result, whatever happened
along the way.
That's the **credit assignment** problem: one reward at the end, and no
direct way to tell which moves earned it. The algorithm puts up with it
because over thousands of games a blunder loses more often than it wins, so
the average label is right even though single labels are wrong. The policy
target helps too: it's the search's opinion of each position, not the
result.
</details>

<details>
<summary><b>Q6 (RL lens).</b> Why does the gate exist at all, when policy iteration is supposed to improve every round?</summary>

Exact policy iteration improves every round. This loop is approximate: a
network trained for a few hundred steps on a few dozen noisy games can come
out *worse*, and a worse champion then generates worse training data, which
can spiral. The gate keeps a candidate out until it has shown it's better.
The cost, as Part A step 3 showed, is that when improvement is hard to
measure, the gate stops the loop learning.
</details>

## Checkpoint

```bash
pytest tests/test_selfplay.py tests/test_tictactoe_lab.py -q   # regression suite still green
pytest tests/course/test_module06_nstep.py tests/course/test_module06_resume.py -q   # un-skipped and passing
```

- [ ] Both green
- [ ] **Part A:** your predictions for steps 2-5 written down *before* you
      opened the answers, and one sentence each on what you saw; your n-step
      measurements from step 6
- [ ] **A run report, not just a run**: for your overnight loop (Do #8),
      using `run_log.csv` — iterations completed, positions in the buffer,
      the arena score *per iteration*, promotions, and the champion vs
      initial match with its interval. Then one paragraph interpreting it:
      if you got promotions, why do you believe they're real given Q1?
      If you got none, is that the candidate, the gate's sample size, or
      the scale — and what single change would you try next? A well-argued
      "no promotions, here's why, here's the fix" **passes** this module;
      an uninterpreted lucky promotion doesn't.

Previous: [Module 5 — PUCT: Search Meets Network](../module-05-puct/README.md) · Next: [Module 7 — Measure & Scale](../module-07-measure-and-scale/README.md)
