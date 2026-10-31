# Reinforcement learning, mapped onto this codebase

This page is the course's RL thread in one place. Each module has a short
**RL lens** section that introduces one or two of these ideas at the moment
you meet them in code. Come back here when a term feels slippery. Every entry
gives the plain idea first, then where it lives in this repo.

## The setting

**Reinforcement learning (RL)** is learning to act by trial and error. Nobody
gives the learner the right answer for each situation. It acts, sees what
happens, and gets a number now and then saying how well things went. It must
work out which of its earlier choices deserve the credit or blame.

| RL word | Plain meaning | In this project |
|---|---|---|
| **Agent** | the thing that chooses | the engine: PUCT search guided by the network (Modules 5-6). In self-play it plays both sides. |
| **Environment** | everything the agent doesn't control | the rules of chess (`python-chess`) plus the opponent's replies. In self-play the opponent is the agent itself. |
| **State** *s* | what the agent sees before choosing | a board position, encoded as a (20, 8, 8) tensor by `net/encoding.py::board_to_tensor` |
| **Action** *a* | one choice | a legal move; one of 4,672 slots via `move_to_index` |
| **Reward** *r* | the number that says how it went | 0 after every move; +1 / 0 / -1 for win / draw / loss **only when the game ends**. That makes chess a *sparse-reward* problem. |
| **Episode** | one run from start to finish | one game |
| **Return** *G* | the total reward from a state onward | with one reward at the end, the return from any position is just the game result, from the point of view of the side to move there |

Two features make board games friendlier than most RL problems, and AlphaZero
exploits both:

- **The agent knows the rules.** It can simulate "what happens if I play
  this?" exactly, which is what search does. RL with a known model of the
  environment is called *model-based*. (MuZero, AlphaZero's successor, learns
  the model too.)
- **It's two-player and zero-sum.** My +1 is your -1, so one value function
  serves both sides if it's always read *from the side to move's point of view*.
  That convention is why the code flips signs every ply (Modules 3 and 5).

## The two things an agent can learn

**Policy** π(a | s): *what to do*, a probability for each legal move.
**Value** V(s): *how good this is*, the expected return from s.

This project's network learns both, as its two heads (`net/model.py`):

- **policy head**: 4,672 logits, masked to legal moves and softmaxed into π(a | s)
- **value head**: one number in [-1, 1], an estimate of V(s) for the side to move

Neither is learned directly as an *action value* Q(s, a), "how good is this
move here". Q-values come from the **search** instead. Each child in the
PUCT tree keeps the average value backed up through it (`PUCTNode.q_value()`
in `search/puct.py`). That is a Q(s, a) estimate built fresh, by search, for
the current position. (Q-learning, the other classic family, learns Q(s, a)
directly and never searches; see "Where AlphaZero sits" below.)

## Ideas you meet along the way

**Value function by hand (Module 1).** `engine/evaluator.py::evaluate()` is a
value function written by a person: position in, "how good" out. It's in
centipawns from White's view rather than [-1, 1] from the mover's view, but
it is the same idea the value head later *learns*.

**Bellman backup (Module 2).** The value of a position is the best value
reachable in one move, seen from the opponent's side: V(s) = max over a of
-V(s after a). That one line is minimax, written as the *Bellman optimality
equation* for a two-player game. Minimax applies it exactly, to a fixed depth,
then trusts `evaluate()` at the leaves.

**Exploration vs exploitation (Module 3).** Play the move that looks best
(exploit), or try one you know less about in case it's better (explore)?
The pure form of this dilemma is the *multi-armed bandit*. UCT, the selection
rule in `search/mcts.py`, is the bandit algorithm UCB1 applied at every node
of the tree.

**Monte Carlo evaluation (Module 3).** Estimate how good a position is by
playing it out to the end, many times, and averaging the results. That's a
rollout. It needs no knowledge and is unbiased if the rollout policy is the
one you care about, but it's noisy and slow.

**Imitation learning (Module 4).** Training on human games is *supervised*
learning: predict the move a strong player made (behaviour cloning) and the
result of the game. No trial and error is involved, so it is **not RL**. Its
ceiling is the teacher: it can at best play like the humans in the data.
It is still a good start. It checks the plumbing, and it gives RL a network
that already knows something (AlphaGo started this way; AlphaZero didn't).

**Function approximation and generalisation (Module 4).** A table with one
entry per position works for tic-tac-toe (5,478 legal positions, 4,520 of
them where the game isn't over yet). Chess has about
10^44, so the agent must *generalise*: give sensible answers in positions it
has never seen. The network is a function approximator. Its errors in unseen
positions are the price, and the reason the arena gate exists. Module 6,
exercise 7 has you swap the tic-tac-toe table for a small network and
measure both sides of that trade.

**Search as policy improvement (Module 5).** Run PUCT with the current network
and the visit counts it produces are a *better policy than the network alone*:
search looks ahead, the raw policy doesn't. An operation that turns a policy
into a better one is called *policy improvement*. Here it's `puct_search`.

**The AlphaZero loop is policy iteration (Module 6).** Classic *policy
iteration* alternates two steps until nothing changes:

1. **policy evaluation**: work out V for the current policy
2. **policy improvement**: make a better policy using V

AlphaZero does an approximate, sampled version of both, every iteration:

| Step | In `selfplay/` |
|---|---|
| improve: search beats the raw network | `self_play.py`: every move comes from `puct_search` with the champion network |
| evaluate: learn V of the policy that played | value head trained on game results (`train_loop.py::soft_target_loss`, MSE part) |
| write the improvement back into the network | policy head trained toward the visit counts (soft-target cross-entropy) |
| keep only real improvements (optional) | `arena.py`: promote the candidate only if it scores > 55% |

**Monte Carlo vs temporal-difference targets (Module 6).** The value target
for every position is the final game result: a **Monte Carlo** target, a real
return with no guessing. The alternative is a **temporal-difference (TD)**
target: use the network's own estimate of the *next* position, V(s'), as the
target for V(s). That's called *bootstrapping*. TD can learn before the game
ends and has lower variance, but it learns from its own possibly wrong guesses
(bias). Monte Carlo is unbiased but noisy: one lucky blunder in a won game
labels every position in that game +1. TD-Gammon (1992) used TD; AlphaZero
uses Monte Carlo.

**Credit assignment (Module 6).** A game is won at the end. Which of the 40
moves deserves the credit? Monte Carlo targets give every position the same
label and let averaging over many games sort it out. Search helps too: the
policy target is the search's opinion of each position, not the game result.

**Exploration in self-play (Module 6).** Without it, self-play replays one
game forever and learns nothing (the tic-tac-toe lab shows this in seconds).
This project explores at three levels:

| Mechanism | Level | Where |
|---|---|---|
| U term of PUCT / UCT | inside the search, at every node | `PUCTNode.select_child` |
| Dirichlet noise | the root of each self-play search: which moves get examined at all | `add_dirichlet_noise`, `add_root_noise=True` |
| temperature | which move is actually played, for the first N moves | `select_move(temperature=1.0)` while `ply < temperature_moves` |

**On-policy vs off-policy data (Module 6).** *On-policy* learning trains only
on data from the current policy. *Off-policy* learning uses data from other
policies, such as older versions of itself. The replay buffer mixes games
from several recent champions, so training is slightly off-policy. That's
useful (more data, less correlated batches) up to a point. Games from a much
weaker champion describe how *that* player's games ended, which is the wrong
answer for today's policy. The buffer's capacity is the dial (`selfplay/replay_buffer.py`).

**Why a gate, and why AlphaZero dropped it (Module 6).** In exact policy
iteration each step is guaranteed to be at least as good as the last. With a
neural network trained on a few noisy games, it isn't: a candidate can be
worse, and if a worse network generates the next data, the loop can spiral
downward. AlphaGo Zero therefore promoted a candidate only if it won 55% of
an arena match. AlphaZero (2018) dropped the gate and always trained the
latest network: at scale the risk was small and the gate slowed progress.
The tic-tac-toe lab shows the trade-off: once both sides draw every game,
no candidate can clear 55% and a gated loop stops learning.

**Measuring an RL agent (Module 7).** Results are noisy: a game is a coin
flip with unknown bias. Every score needs an interval, and "beats its
previous self" says little about absolute strength.

## Going deeper: the questions a practitioner would ask

These sharpen the ideas above. Read them after Module 6, once you've run the
tic-tac-toe lab.

**Episodes, and no discount.** A game always ends, so this is an *episodic*
problem. Many RL methods shrink future rewards by a discount γ < 1 per step;
here γ = 1, nothing is discounted, so the return from any position is simply
the final result. (Discounting matters in problems that never end, or when
sooner rewards really are worth more.)

**Whose value, exactly?** The value head learns the expected result of the
games *as they were actually played*: the champion's search **plus** its
exploration noise and temperature, and, because the buffer mixes several
champions, an average over recent ones. That *behaviour policy* is a bit
weaker and more random than the greedy policy the arena tests. The opponent
is also the agent, so the "environment" changes every time the agent does:
self-play is *non-stationary*, unlike learning against a fixed opponent.

**Generalised policy iteration, and its condition.** Search improves the
policy; training on results evaluates it; both happen a little at a time.
That pattern is called *generalised policy iteration*. Its catch: search
only improves on the network if the values it backs up are roughly right.
If the value head is badly wrong, search confidently follows the error and
training then copies it, and the loop reinforces its own mistakes. The real
corrections come from the rules: terminal positions that search reaches,
and game results.

**What each head contributes to search.** With random values, backed-up Q
values are noise, so visits end up following the priors and more simulations
add nothing. With good values and uniform priors, more simulations keep
helping, because each one adds real information (Module 5, Q3). So the value
head is what makes extra search worth paying for.

**Why both kinds of exploration.** Dirichlet noise changes which moves the
search *looks at*, but with temperature 0 the most-visited move is still
played, so games stay nearly deterministic. Temperature changes which move is
*played*, but without noise the search never examines moves the network
dislikes. Each covers a gap the other leaves. The arena turns both off
because it measures the greedy player you would actually deploy; noise would
add variance and measure a weaker player.

**Off-policy data, more precisely.** The *value* target from an old game is
the result of an older, weaker behaviour policy: a biased label for today's.
The *policy* target is different: it's advice from an older search,
distilled into the network the way supervised learning copies a teacher,
not an estimate weighted by returns. So it's stale rather than biased in the
same way. Off-policy methods often correct with *importance sampling*
(reweighting each example by how likely today's policy was to produce it).
AlphaZero gets away without it because the window only holds recent
champions, and the gate keeps each step of change small, so the data is
nearly on-policy.

**What keeps the loop stable.** Several pieces dampen the feedback between
"the network makes its own data" and "the network trains on that data":
- the champion is *frozen* while the candidate trains, so the data source
  doesn't move under the learner (a role like the "target network" in deep
  Q-learning)
- the replay buffer mixes generations and decorrelates batches
- `train_steps`, the learning rate and gradient clipping
  (`clip_grad_norm_` in `train_candidate`) keep each update small
- the gate, at the price you saw in the lab

**Credit assignment gets worse with distance.** A label from the end of the
game is nearly right for positions close to the end, and mostly noise for
positions in the opening, where many later moves decided the result. Draws
add a further problem: a draw labels every position 0 and says almost
nothing about which moves were good.

**How much a gate can tell you.** A candidate of exactly equal strength
needs 4 wins out of 6 (with no draws) to beat 55%, which happens about 34%
of the time. So a 6-game gate promotes a no-better network about one time in
three. Arena games are also nearly deterministic, so six games may hold
fewer than six independent samples. And strength isn't always transitive:
B can beat A and C beat B while C still loses to A. A chain of "beats its
predecessor" can circle without progress, which is why Module 7 measures
against a fixed, calibrated opponent.

**Truncation is not termination.** The self-play loop scores a game that
hits `max_moves` as a draw, as if the game had ended. But it was *cut off*,
not finished. The state the network sees has no move counter that would show
the cap coming (`board_to_tensor` has a fifty-move counter, but nothing
counts plies). So the same position gets labelled 0 near the cap and its
real result elsewhere. Two positions that look identical to the network but
need different answers is called *state aliasing*: the state isn't fully
*Markov*. The principled fixes are to treat the cap as truncation and
bootstrap the target from the network's value of the last position (a TD
target, exactly where Monte Carlo has no result to give), or to add the
remaining plies to the input so the state really says what it needs to.

## Where AlphaZero sits among RL methods

| Family | Learns | Example | Compared with AlphaZero |
|---|---|---|---|
| Value-based | Q(s, a) directly, by TD | Q-learning, DQN (Atari) | no search; acts by picking the highest Q |
| Policy gradient | π directly, by pushing up the probability of actions that led to high return | REINFORCE, PPO | no search; learns from returns only |
| Model-based + search | π and V, improved by planning with a model | **AlphaZero**, MuZero | search turns a weak network into a strong policy, and the network learns to imitate the search |

## Check yourself

Each module's questions test its own RL lens. These test the whole picture.

<details>
<summary><b>R1.</b> Self-play with no exploration at all (no noise, temperature 0). Predict what the replay buffer contains after 10 iterations.</summary>

The same game, many times over. With no randomness, the same network plays
the same moves from the start position every game. It never sees a position
outside that one line, so it can't learn anything about the rest of the game.
Run `python3 selfplay/tictactoe_lab.py --no-root-noise --temperature-moves 0`:
coverage stays below 1% of positions.
</details>

<details>
<summary><b>R2.</b> Why can't supervised training on grandmaster games ever produce a superhuman engine on its own?</summary>

Its target is "the move the human played". A perfect score means playing
exactly like the humans in the data, mistakes included. Getting past the
teacher needs a signal that isn't the teacher. In AlphaZero that's search
plus game results: the search can find moves no human in the data played,
and the result says whether they worked.
</details>

<details>
<summary><b>R3.</b> Name the policy-evaluation step and the policy-improvement step in this repo's loop.</summary>

Evaluation: training the value head on the results of the champion's
self-play games. Improvement: PUCT search with the champion network, whose
visit counts are a better policy than the network's raw output. The policy
head is then trained toward those visit counts, which writes the improvement
back into the network.
</details>

<details>
<summary><b>R4.</b> Swap the Monte Carlo value target for a TD target. What would you change in <code>selfplay/self_play.py</code>, and what risk do you take on?</summary>

Instead of labelling each position with the final result, label it with the
current network's value of the *next* position, sign-flipped for the side to
move (and the final result for the last position). You'd learn from your own
estimates, so early on the network trains toward its own random guesses. That
is bias, which can slow or derail learning. In exchange you get lower
variance, because one lucky game no longer labels 40 positions.
</details>

<details>
<summary><b>R5.</b> Someone sets <code>max_moves=40</code> to speed up self-play. What happens to the value targets, and why can't the network learn the right answer even with infinite data?</summary>

Most games hit the cap and every position in them is labelled 0. Worse, the
network can't see how close the cap is (the input has no ply counter), so
the same position is labelled 0 in one game and ±1 in another: state
aliasing. Infinite data would only average those labels, not resolve them.
Fix it by bootstrapping from the network's value at the cut-off instead of
calling it a draw, or by putting the remaining moves into the input.
</details>

<details>
<summary><b>R6.</b> Candidates keep beating their predecessors in the arena, but you suspect the engine isn't improving. Give two reasons that can happen, and the evaluation that would settle it.</summary>

A 6-game gate promotes an equal network about a third of the time, so some
"wins" are luck; and strength can be non-transitive, so a chain of
head-to-head wins can cycle. Settle it against a *fixed* yardstick: the
initial checkpoint, or better a calibrated opponent such as Stockfish at a
set rating, with enough games for a narrow interval (Module 7).
</details>

<details>
<summary><b>R7.</b> Why does AlphaZero not reweight old replay-buffer examples with importance sampling, as many off-policy methods do?</summary>

The buffer only holds games from recent champions, and a champion changes
only when a candidate beats it, so the policies that produced the data are
close to today's: the data is nearly on-policy. The policy targets are also
distilled advice from search rather than return estimates, so the usual
correction doesn't apply to them in the same way. A much larger buffer
would erode that justification.
</details>

<details>
<summary><b>R8.</b> The value head is badly miscalibrated early in training. Why can search make things worse rather than better, and what stops that going on forever?</summary>

Search trusts the values it backs up. If they're wrong, it concentrates
visits on moves that only look good to the broken value head, and training
then copies those visit counts into the policy: the loop reinforces its own
error. What breaks the cycle is information from outside the network:
exact values at terminal positions the search reaches, and the real game
results used as value targets.
</details>

Back to [the course](README.md)
