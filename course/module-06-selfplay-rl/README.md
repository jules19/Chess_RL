# Module 6: Self-Play Reinforcement Learning

> **Where you are:** the last step in building the engine. You remove the human games: the engine generates its own experience, and every piece from Modules 1-5 becomes one step of a reinforcement-learning loop.
> **Time:** 5-8 hours, plus one overnight run. Part A (tic-tac-toe) takes 2-3 hours, including two coding exercises; most runs finish in seconds. Part B (chess) is mostly waiting.
> **You'll be able to:**
> - describe the AlphaZero loop in terms of reinforcement learning (RL), which is learning by trial and error from rewards. Say which part evaluates the policy, which part improves it, and where the reward comes from
> - watch an RL loop learn, read its learning curve, and break it on purpose in three different ways
> - compare three kinds of value target from your own measurements: Monte Carlo, temporal-difference (TD) and n-step. All three are explained below. Then build n-step yourself
> - replace a lookup table with a neural network, and measure what generalisation (giving sensible answers for positions never seen in training) gains and costs
> - run the chess loop, and write a report that separates real learning from noise

## Concepts

The loop (see the diagram in `selfplay/__init__.py`):

```
self-play (champion + exploration) → replay buffer → train candidate
        ↑                                                   │
        └────────── arena gate: promote only if >55% ───────┘
```

### The RL view of this loop

Each RL term from [RL_CONCEPTS.md](../RL_CONCEPTS.md) maps to a part of this loop:

| RL idea | In this loop |
|---|---|
| agent | PUCT search (Predictor + Upper Confidence bounds for Trees, Module 5's network-guided search) and the champion (the current best network), playing both sides |
| environment | the rules and the opponent, who is the agent itself |
| reward | 0 per move; +1, 0 or -1 at the end of each game (win, draw or loss) |
| policy improvement | search: the visit counts (how often search tried each move) are a better policy than the raw network (Module 5) |
| policy evaluation | the value head learns from game results how good positions are for the player to move. The value head is the network output that scores a position ([Module 4](../module-04-neural-network/README.md)) |
| writing the improvement back | the policy head (the network output that suggests moves) is trained toward the visit counts |
| experience | the replay buffer of (state, visit counts, result) examples |

Together, these steps form *policy iteration*: evaluate the current policy,
improve it, and repeat. Exact policy iteration is guaranteed to improve every
round. This sampled version, with a neural network, is not, and that is why
the arena gate exists.

**The policy target is the search result** (`selfplay/self_play.py`). Each
position's policy target is the full PUCT *visit distribution*: the share of
visits each move received. The search guided by the network is stronger than
the raw network, so training the network to predict the search's conclusions
makes it stronger. This is the central idea of AlphaZero.

**The value target is a Monte Carlo return.** Every position gets the final
result of its game, from the point of view of the side to move there: +1 in
the winner's positions and −1 in the loser's, including the winner's
mistakes. That is a *Monte Carlo* target: the real return (the reward from
that position to the end of the game), with no estimate. The alternative is
a *temporal-difference (TD)* target, which uses the network's own estimate
of the next position as the target. This is called bootstrapping. Part
A lets you run both.

Giving every position in a game the same result as its target is also the simplest answer to the
*credit assignment* problem (which moves earned the win?). It makes no
decision per position. It relies on averaging over thousands of games, so
that random effects cancel out.

**The replay buffer holds recent positions** (`selfplay/replay_buffer.py`).
It must be large enough to hold several iterations of games, and to make each
training batch (the examples for one update) mix many games, because
positions from one game are similar. It must be small enough that positions
from much older, weaker networks are dropped. In RL terms, games played by
earlier champions are *off-policy* data: they describe how a different,
weaker player's games went. A little is useful. With too much, the network
learns from games played by old versions that are no longer used. The buffer capacity controls this
balance.

**The arena gate decides which network becomes champion**
(`selfplay/arena.py`). The arena is a match between the newly trained
candidate and the champion. A bad training iteration can make the network
worse, and if that network generates the next data, the networks after it
can keep getting worse. To replace the champion, a candidate must score above 55%. The arena
alternates colours and uses no exploration noise. AlphaGo Zero, DeepMind's
Go program that learned only from self-play, used this gate. AlphaZero, the
later version that also plays chess, removed it and always trained the
newest network. Part A shows one reason why.

**Exploration is necessary.** Without it, self-play repeats the same game
every time. Exploration happens at three levels:

- the U term in every search: the exploration bonus in the PUCT score
  ([Module 5](../module-05-puct/README.md)) that favours moves with few
  visits;
- Dirichlet noise on the root priors: random noise added to the network's
  move probabilities (priors) at the root, that is, the current position at
  the top of the search tree. Module 5 explains it. It decides which
  moves get examined at all;
- temperature for the first moves: the move is drawn at random in
  proportion to visit counts instead of always being the most-visited one.
  It decides which move is played.

Part A shows what happens when you remove them.

**Soft-target loss** (`selfplay/train_loop.py::soft_target_loss`). In Module
4 the policy target was one move, given as a class index (the number of the
correct output), and the loss was cross-entropy: minus the log of the
probability the network gave that move. Now the target is a probability
distribution, and the loss is small when the network gives high probability to
the moves the search visited most. The formula is `-Σ π(a)·log p(a)`: π(a) is
the target share of visits for move a, p(a) is the network's probability for
it, and the sum runs over all moves. For a one-hot target (all weight on one
move) this equals the Module 4 loss, and the test suite checks that.

## Read

Read all four files in `selfplay/` that make up the loop. They are short and
well documented. Then read `selfplay/tictactoe_lab.py`, which reuses
`puct_search` and the replay buffer unchanged on a different game.

## Do, Part A: watch it learn (tic-tac-toe)

With the computing power of a laptop, chess self-play shows that the training pipeline works, but
it rarely shows learning. Tic-tac-toe is small enough that the same loop
learns in seconds. It is also solved (the best move is known for every
position), so the lab can score the agent against perfect play. The "network" here is a table with one entry per position; Q4
below asks what that changes.

1. **Run it.** It takes about 7 seconds. The default seed is 0, so your
   numbers should match the table below exactly:
   ```bash
   python3 selfplay/tictactoe_lab.py
   ```
   Each row is one iteration (100 self-play games, then training).
   The columns form the learning curve:

   | Column | Means |
   |---|---|
   | `value_mse` | mean squared error (MSE): the average squared difference between the learned values and the true game values, over all 4,520 positions (lower is better) |
   | `best_move` | % of positions where the table's favourite move is a perfect move |
   | `coverage` | % of positions the agent has ever seen in self-play |
   | `vs_perfect` | the table alone (no search) against a perfect player: % of games not lost. You can't beat a perfect player, so 100% means it always draws |
   | `vs_random` | the table alone against a player that moves at random: score % |
   | `buffer` | positions in the replay buffer (it stops growing at its capacity, 4,000) |
   | `arena`, `prom` | the arena score and whether the candidate was promoted (the arena column is blank without `--gate`) |

   **Expected (measured, seed 0):**
   ```
   iter value_mse best_move  coverage vs_perfect vs_random
      0     0.767      58.7       0.0          0        64
      5     0.695      65.6      21.5         38        65
     10     0.663      68.4      27.5         85        83
     20     0.636      70.2      32.3         89        80
   ```
   The agent goes from losing every game against perfect play to drawing
   about 9 in 10, by playing only against itself. `vs_perfect` varies by up
   to ±10 between iterations: it is measured on 100 games, so it is noisy.

   Notice two things. First, `coverage` stops near a third. The agent only
   learns about positions its own play reaches, and a table can't give
   answers for the rest. Second, `vs_random` is lower than `vs_perfect`. A random
   opponent's moves lead to positions the agent never saw, where the table
   has no information. The chess network must generalise to unseen
   positions, and the table can't.

2. **Break it on purpose: remove exploration.** Predict first: what will `coverage` do?
   ```bash
   python3 selfplay/tictactoe_lab.py --no-root-noise --temperature-moves 0
   ```
   <details><summary>What we measured</summary>

   Coverage stays at 0.5% and `value_mse` does not change from 0.766. With no
   noise and no temperature, every game is the same game. The buffer fills
   with that one game repeated, and the agent learns that one line perfectly
   and nothing else.
   </details>

3. **Turn the gate on.** Predict first: with a 55% promotion threshold,
   what happens once both sides play well?
   ```bash
   python3 selfplay/tictactoe_lab.py --gate
   ```
   <details><summary>What we measured</summary>

   It learns for 7 iterations and then stops improving. The last promotion
   is at iteration 7, and from iteration 8 every arena score is 42-50%. The
   champion never changes again. (Its `vs_perfect` still varies between 72
   and 82. That is the same table measured again on 100 new games, which
   shows how noisy that column is.) Good tic-tac-toe ends in a draw, so a
   better candidate can't show that it is better: it draws, scores 50%, and
   fails a 55% gate. The gate protects against regressions (candidates that
   got worse), but it also blocks progress whenever progress is hard to
   measure. That is one reason AlphaZero removed it.
   </details>

4. **Off-policy data: a buffer that keeps every game.** Predict first, then
   run three seeds of each, because one run proves nothing. `tail -1` keeps
   the last row (iteration 20), with the same columns as step 1:
   ```bash
   for seed in 0 1 2; do
     python3 selfplay/tictactoe_lab.py --seed $seed | tail -1
     python3 selfplay/tictactoe_lab.py --seed $seed --buffer-capacity 1000000 | tail -1
   done
   ```
   <details><summary>What we measured</summary>

   The effect is small. The final `value_mse` was worse with the unlimited
   buffer in all three seeds (0.636 vs 0.642, 0.621 vs 0.638, 0.603 vs
   0.622): old games describe how weaker players' games ended. `vs_perfect`
   went both ways (89 vs 77, 84 vs 82, 83 vs 85), which is noise at 100
   games. A conclusion from any single seed could have been wrong in either
   direction.
   </details>

5. **Monte Carlo and TD value targets.** `--td` replaces the game result
   with minus the table's own value of the next position (the opponent's
   view, with the sign flipped). Predict: which learns values faster early
   on?
   ```bash
   python3 selfplay/tictactoe_lab.py --td
   ```
   <details><summary>What we measured</summary>

   TD starts slower. At iteration 5 its `value_mse` was worse in all three
   seeds (about 0.72 against 0.69). By iteration 20 the two are level. TD
   learns from its own estimates, which start at 0 everywhere, so the
   information that a game was won reaches only one earlier position per
   update. Monte Carlo gives every position the real result at once. TD's
   advantage is lower variance (less random variation in its targets). That
   matters more in long games with noisy outcomes than in tic-tac-toe.
   </details>

6. **Exercise: build n-step returns.** Monte Carlo and TD are the two
   extremes of one setting. An *n-step* target looks n moves ahead. If the
   game ends within n moves, use the real result. Otherwise, use the table's
   value of the position n moves later. Rewards are 0 during the game, so
   that is the whole target. One detail matters: the side to move
   alternates, so the value from n moves ahead must be negated when n is
   odd. n = 1 is TD, and n larger than any game is Monte Carlo.

   1. Implement `n_step_targets(results, values, n)` in
      `selfplay/tictactoe_lab.py`.
   2. Give `selfplay_game` and `run` an `n_step` parameter that uses it, and
      add a `--n-step` flag.
   3. Remove the pytest skip marker from `tests/course/test_module06_nstep.py`
      (it sets out what your code must do) and make it pass.
   4. Measure n = 1, n = 3 and the default Monte Carlo over three seeds.
      Look at `value_mse` at iterations 5 and 20.

   **What to expect:** if your n = 1 doesn't reproduce the numbers from
   `--td`, your sign is wrong. Predict before you measure: how will
   n = 3 compare with the other two?
   <details><summary>What we measured (reference solution, 3 seeds)</summary>

   n = 3 did not fall between the other two. At iteration 5 it was as good
   as Monte Carlo or slightly better (`value_mse` 0.677-0.693, against
   0.684-0.695 for Monte Carlo and 0.713-0.719 for TD). At iteration 20 it
   was the lowest in two seeds of three. Three moves is enough to reach the
   real result from most positions in a tic-tac-toe game, and the estimated
   part averages away some of Monte Carlo's noise. A middle setting can do
   better than both extremes, which is why practical RL systems tune n.
   </details>

7. **Exercise: replace the table with a neural network.** The table can't
   generalise: an unseen position keeps a uniform prior (equal probability
   for every move) and a value of 0. The chess engine uses a
   network because a network can generalise. Here you can measure that.

   1. Install torch: `pip install -e ".[train]"`.
   2. Write `MLPPolicyValue` in `selfplay/tictactoe_lab.py`. MLP stands for
      multi-layer perceptron, a network of fully connected layers. It has
      the same four methods as `TablePolicyValue`, backed by a small torch
      network with:
      - 27 inputs: each of the 9 cells becomes three numbers (mover's mark,
        opponent's mark, empty), with 1 for the content present and 0 for
        the others (one-hot encoding);
      - two hidden layers (the layers between input and output) of 64 units each;
      - a policy head with 9 outputs, one per cell;
      - a value head with a tanh output (it maps any number into the range
        -1 to 1);
      - the soft-target loss from `selfplay/train_loop.py`.
   3. `run(model=...)` already accepts it. Add a `--model mlp` flag.
   4. Remove the pytest skip marker from
      `tests/course/test_module06_network.py` and make it pass. It includes
      a hint about caching outputs so that the search stays fast.
   5. Compare the network with the table over three seeds (one to two
      minutes per run).

   Predict first: on which columns will the network beat the table, and on
   which will it lose?
   <details><summary>What we measured (reference solution, seeds 0-2, iteration 20)</summary>

   | | table | network |
   |---|---|---|
   | `best_move` (all 4,520 positions) | 70.2 / 70.0 / 72.0 | **76.9 / 75.5 / 75.1** |
   | `vs_random` | 80 / 84 / 82 | 82 / 90 / 83 |
   | `vs_perfect` | **89 / 84 / 83** | 19 / 72 / 68 |

   The network wins where generalisation matters. It plays a perfect move
   in more positions, including ones self-play never reached. It does at
   least as well against a random opponent, whose moves reach those
   positions. It loses on precision and stability. Every update
   changes its answers for all positions at once, so learning one line can
   unlearn another. Its `vs_perfect` and `value_mse` rise and fall between
   iterations, and in one seed the score against perfect play fell to 19%.
   That instability is the strongest reason for an arena gate. But with the network,
   turning `--gate` on mostly stopped learning, as it did with the table in
   step 3. This conflict
   between generalisation and stability is the central difficulty of RL with
   function approximation (a learned function, such as a network, in place
   of a table), and the chess network has it at a far larger scale.
   </details>

## Do, Part B: the chess loop

8. **Run the small-scale loop.** It takes about 1.5 minutes on an 8-core
   CPU. It checks that the training pipeline works; it does not measure
   strength:
   ```bash
   python3 selfplay/train_loop.py --iterations 2 --games-per-iter 2 \
       --simulations 16 --train-steps 20 --arena-games 2
   ```
   **Expected (measured):** 0 promotions, arena scores of 50%, a policy loss
   around 7.7 and a value loss near 0. The policy loss is close to ln(4672) ≈
   8.4. That is the loss of a network that spreads its probability evenly
   over all 4,672 outputs: an untrained network knows nothing yet. With
   `--arena-games 2` the arena can only score 0%, 25%, 50%, 75% or 100%, so
   here it only checks that the code runs and is not a measurement. Each
   iteration also appends a row to `selfplay_checkpoints/run_log.csv`.

   The value loss is near 0 because a random network never finishes a game
   within the 200-ply limit (a ply is one move by one side), so every game
   is scored as a draw with reward 0. This is the *sparse reward
   problem*: the value head has nothing to learn from until games start
   ending in a win or a loss. It is also why warm-starting from Module 4
   (starting from your trained network) helps so much.
9. **Run a real (small) loop overnight.** Warm-start from your Module 4
   supervised checkpoint. AlphaGo, DeepMind's first Go program, did the
   same; AlphaZero started from random weights.
   ```bash
   python3 selfplay/train_loop.py --init-checkpoint checkpoints/best_model.pt \
       --iterations 10 --games-per-iter 20 --simulations 64 --train-steps 300
   ```
   **Expected (measured):** about 40 seconds per game at 64 simulations,
   with a 4×128 network (4 residual blocks of 128 channels, as in Module 4)
   on an 8-core CPU. One iteration (20 self-play games, training and a
   6-game arena) takes about 18 minutes, and the whole run about 3 hours.
   The run ends after its 10 iterations; there is no early stop. In a
   healthy run the policy loss falls between iterations. Promotions may be
   rare (see Q1).

   In our one-iteration trial with the Module 4 network trained on one month
   of games, both self-play games were drawn, so every value target
   was 0 and the value loss read 0.0000. This is the sparse reward problem
   again, even with a warm start. The candidate scored 25% and was rejected.

   While the run is going, watch the value loss column for the same reason.
   `selfplay_checkpoints/run_log.csv` records, for each iteration, the
   buffer size, the policy and value losses, the arena score and whether the
   candidate was promoted.

   When the run has finished, test whether the loop gained anything:
   ```bash
   python3 cli/match.py puct:sims=64,net=selfplay_checkpoints/champion.pt \
       puct:sims=64,net=checkpoints/best_model.pt --games 20
   ```
   Twenty games is a quick check: its 95% interval is about ±22
   percentage points ([Module 7](../module-07-measure-and-scale/README.md),
   Part 0), so it can detect a large gain or a collapse. For a claim you can
   defend, run 100 games overnight.
10. **Exercise: break the chess loop on purpose.** Do three short runs:
    (a) `--no-root-noise`, (b) `--temperature-moves 0`,
    (c) `--promote-threshold 0.30`. Use the small-scale settings from step 8
    with `--iterations 5`.

    1. Before running, predict each failure from what Part A showed you.
    2. Run each one. The command for (a) is:
       `python3 selfplay/train_loop.py --iterations 5 --games-per-iter 2 --simulations 16 --train-steps 20 --arena-games 2 --no-root-noise --checkpoint-dir sabotage_a`
    3. Check `run_log.csv`: look at the promoted column and at whether the
       policy loss changes.
    4. Write three sentences, including whether chess at this scale shows
       the effect at all.
    <details><summary>A hint for (c), measured</summary>

    With untrained networks every arena game is a draw, so every candidate
    scores 50%, which clears a 30% threshold: the `promoted` column is all
    1s. A gate below 50% promotes candidates that only tie, and later ones
    that lose, so it no longer protects anything. (a) and (b) are harder to
    see at this scale, which makes them worth predicting.
    </details>
11. **Exercise: buffer-size sweep.** Compare `--buffer-capacity 2000` with
    `50000` at a fixed number of games per iteration. For example, run step
    9's command with `--iterations 5`, once with
    `--buffer-capacity 2000 --checkpoint-dir buf_small` and once with
    `--buffer-capacity 50000 --checkpoint-dir buf_large`. Then compare the
    `promoted` columns of the two `run_log.csv` files. Each run takes a few
    hours on a laptop CPU.

    **Laptop version (about 10 minutes each):** use step 8's small-scale
    settings with `--iterations 4 --buffer-capacity 300`, and again with
    `--buffer-capacity 50000`. A game at these settings is a few hundred
    positions long, so a 300-position buffer drops almost everything each
    iteration. This version shows the mechanics: watch
    `positions_in_buffer` in `run_log.csv` stop growing. It does not show
    which setting learns better. That needs the full runs, several seeds, or
    the tic-tac-toe version in step 4.

    Which setting promotes more often early in the run, and which later? How
    many seeds would you need to believe the answer (Part A, step 4)?
12. **Exercise: resumable loops.** The loop already saves
    `champion.pt`/`replay_buffer.npz`. Implement
    `load_resume_state(checkpoint_dir)` in `selfplay/train_loop.py`. Add a
    `--resume` flag in its `main()` that uses it. A stopped run should then
    continue: it loads the buffer and the champion and keeps the iteration
    numbering. Long
    RL runs are often interrupted, so resuming matters.
    `tests/course/test_module06_resume.py` sets out what the code must do
    and is part of this module's checkpoint; remove its pytest skip marker.

## Check your understanding

<details>
<summary><b>Q1.</b> Predict: at this module's scale (tens of games, 6 arena games per gate), roughly how often should you expect a candidate that is stronger than the champion to fail promotion?</summary>

More often than not. A candidate whose true score is 55% needs at least
4 wins out of 6 to clear a 55% gate (assuming no draws), which happens only
about 44% of the time. The
number of wins in 6 games varies a lot by chance: it follows a binomial
distribution (successes in a fixed number of independent trials). Module 7,
Part 0, has the formula. So "no
promotion this iteration" is weak evidence of anything. This is why the
checkpoint below grades your interpretation of the arena numbers, rather
than the numbers themselves.
</details>

<details>
<summary><b>Q2.</b> The policy target is the visit distribution rather than the move played. Give the one-sentence reason this makes self-play self-improving rather than self-imitating.</summary>

The search guided by the network is stronger than the raw network, so
training the network to match the search's conclusions moves it toward a
stronger player than itself. Repeat this, and the target becomes stronger
each round. Imitating only the move played would discard almost all of the
search's judgment (everything it learned about the moves it did not pick).
</details>

<details>
<summary><b>Q3.</b> You remove Dirichlet noise but keep temperature (step 10, run (a)). What failure do you predict, mechanically?</summary>

Temperature still varies which move is played at the root (the current position). But within each
search, the priors alone decide what gets explored. Moves that the current
network rates low get almost no visits, so their true value is never
discovered. The training data then never shows the network that its low
ratings were wrong. The loop ends up repeating a set of openings that
matches the network's own ratings, and those openings may be bad. Exploration in move selection can't make up for having no
exploration in the search.
</details>

<details>
<summary><b>Q4 (RL question).</b> In the tic-tac-toe lab, what does the table do that the chess network can't, and what does the chess network do that the table can't?</summary>

The table stores an exact answer for every position it has seen, with no
interference between positions. That is possible only because tic-tac-toe
has 4,520 non-terminal positions. The network can't store chess's roughly
10^44, but it can *generalise*: give a sensible answer for a position it has
never seen, because similar positions share features. The table's
`coverage` column and its weak `vs_random` score show what having no
generalisation costs.
</details>

<details>
<summary><b>Q5 (RL question).</b> In a self-play game the winner blundered on move 20, but the opponent didn't punish it. What value target does the move-20 position get, what's the name of the problem, and why does the algorithm accept it?</summary>

+1 if the eventual winner is the side to move in that position (-1 if it is
the loser): the Monte Carlo target is the game result, whatever happened
during the game.
This is the credit assignment problem: there is one reward at the end, and
no direct way to tell which moves earned it. The algorithm accepts this
because over thousands of games a blunder loses more often than it wins, so
the average target is right even though individual targets are wrong. The
policy target also helps: it is the search's assessment of each position,
which does not depend on the result.
</details>

<details>
<summary><b>Q6 (RL question).</b> Why does the gate exist at all, when policy iteration is supposed to improve every round?</summary>

Exact policy iteration improves every round. This loop is approximate: a
network trained for a few hundred steps on a few dozen noisy games can come
out worse. A worse champion then generates worse training data, and the next
network can be worse again. The gate keeps a candidate out until it has
shown it is better. The cost, as Part A step 3 showed, is that when
improvement is hard to measure, the gate stops the loop from learning.
</details>

## Checkpoint

```bash
pytest tests/test_selfplay.py tests/test_tictactoe_lab.py -q   # regression suite still green
pytest tests/course/test_module06_nstep.py tests/course/test_module06_network.py \
       tests/course/test_module06_resume.py -q                  # un-skipped and passing
```

- [ ] Both commands pass
- [ ] **Part A:** your predictions for steps 2-5, written down before you
      opened the answers, with one sentence each on what you saw; your
      n-step measurements from step 6; and your table-against-network
      comparison from step 7
- [ ] You can answer R5-R8 in the "Going deeper" section of
      [RL_CONCEPTS.md](../RL_CONCEPTS.md) before opening the answers
- [ ] **Run report** for your overnight loop (step 9):
      1. From `run_log.csv`, report the iterations completed, positions in
         the buffer, the arena score for each iteration and the promotions.
      2. Report the match between the champion and the initial network,
         with its 95% interval.
      3. Write one paragraph interpreting the results. If you got
         promotions, why do you believe they are real, given Q1? If you got
         none, is the cause the candidate, the gate's sample size or the
         scale? What single change would you try next?

      A well-argued "no promotions, here is why, here is the fix" passes
      this module. An uninterpreted lucky promotion does not.

Previous: [Module 5: PUCT, Search Meets Network](../module-05-puct/README.md) · Next: [Module 7: Measure and Scale](../module-07-measure-and-scale/README.md)
