# Reinforcement learning, mapped onto this codebase

This page collects the course's reinforcement learning ideas in one place.
Each module's short **RL connection** section introduces one or two of them
where you meet them in code. Each entry here gives the plain idea, then where
it appears in this repository.

## The setting

**Reinforcement learning (RL)** is learning to act by trial and error. Nobody
gives the learner the right answer for each situation. It acts, sees what
happens, and gets a number now and then saying how well things went. It must
work out which of its earlier choices deserve the credit or blame.

*AlphaZero* is the system this course rebuilds. It learned chess with no
human games, only by playing against itself. Its search algorithm is *PUCT*
(Predictor + Upper Confidence bounds for Trees): a tree search (Module 3) in
which a neural network suggests which moves to look at first (Module 5).

A few search words appear throughout. A search builds a *tree* of positions.
Each position in it is a *node*; the current position is the *root*, the
positions one move later are its *children*, and the positions where the
search stops are *leaves*. One *simulation* is one walk from the root to a
leaf and back, and a node's *visit count* is how many simulations passed
through it.

| RL word | Plain meaning | In this project |
|---|---|---|
| **Agent** | the thing that chooses | the engine: PUCT search guided by the network (Modules 5-6). In self-play it plays both sides. |
| **Environment** | everything the agent doesn't control | the rules of chess (`python-chess`) and the opponent's replies. In self-play the opponent is the agent itself. |
| **State** *s* | what the agent sees before choosing | a board position, stored as a (20, 8, 8) tensor (a multi-dimensional array of numbers: 20 layers of 8 by 8) by `net/encoding.py::board_to_tensor` |
| **Action** *a* | one choice | a legal move; one of 4,672 numbered move slots via `move_to_index` |
| **Reward** *r* | the number that says how it went | 0 after every move; +1 / 0 / -1 for win / draw / loss, given only when the game ends. A problem where rewards are this rare is called a *sparse-reward* problem. |
| **Episode** | one run from start to finish | one game |
| **Return** *G* | the total reward from a state onward | with one reward at the end, the return from any position is the game result, from the point of view of the side to move there |

Two features make board games easier than most RL problems, and AlphaZero
uses both:

- **The agent knows the rules.** It can work out exactly what a move leads
  to without playing it, which is what search does. RL with a known
  model of the environment is called *model-based*. (MuZero, AlphaZero's
  successor, also learns the model.)
- **It's two-player and zero-sum.** *Zero-sum* means one side's gain is
  exactly the other side's loss: my +1 is your -1. So one estimate of "how good is this position" (a *value*,
  defined below) works for both sides, if it is always read from the point of
  view of the side to move. This is why the code flips the sign of a value at
  every ply (one move by one side) in Modules 3 and 5.

## The two things an agent can learn

**Policy** π(a | s): *what to do*, a probability for each legal move.
**Value** V(s): *how good this is*, the expected return from s. A function
that gives this number for any position is a *value function*.

This project's network learns both. It has two output parts, called *heads*
(`net/model.py`):

- **policy head**: 4,672 raw scores (*logits*), one per move slot. Illegal
  moves are removed (*masked*), and the *softmax* function turns the rest
  into probabilities that add up to 1: π(a | s).
- **value head**: one number in [-1, 1], an estimate of V(s) for the side to move

The network does not learn an *action value* Q(s, a), "how good is this move
here". The Q-values come from the search instead. Each child in the PUCT tree
keeps the average of the results passed up (*backed up*) to it from deeper
in the tree (`PUCTNode.q_value()` in `search/puct.py`). The search builds
this Q(s, a) estimate new for each position. (Q-learning, the other classic
family, learns Q(s, a) directly and never searches; see "Where AlphaZero sits" below.)

## Ideas you meet along the way

**Value function by hand (Module 1).** `engine/evaluator.py::evaluate()` is a
value function written by a person: position in, "how good" out. Its unit is
centipawns (hundredths of a pawn) from White's view, where the value head
uses [-1, 1] from the view of the side to move. It is the same idea that the
value head later *learns*.

**Bellman backup (Module 2).** The value of a position is the best value
reachable in one move, seen from the opponent's side: V(s) = max over a of
-V(s after a). That one line is minimax, written as the *Bellman optimality
equation* for a two-player game. Minimax applies it exactly, to a fixed depth,
then uses `evaluate()` at the leaves.

**Exploration and exploitation (Module 3).** At each choice the agent can
play the move that looks best so far (*exploit*), or try one it knows less
about, in case it is better (*explore*). The simplest form of this choice is
the *multi-armed bandit*: several slot machines with unknown payouts and a
limited number of pulls. UCT (Upper Confidence bound for Trees), the
selection rule in `search/mcts.py`, applies the bandit algorithm UCB1 (Upper
Confidence Bound) at every node of the tree. UCB1 picks the option with the
highest average result plus a bonus that shrinks as that option is tried more.

**Monte Carlo evaluation (Module 3).** Estimate how good a position is by
playing it out to the end, many times, and averaging the results. One such
play-out is a *rollout*, and the rule that picks its moves is the *rollout
policy*. It needs no chess knowledge. Its average is *unbiased* (no systematic
error) for the rollout policy itself, but it is *noisy* (single results vary
a lot) and slow.

**Imitation learning (Module 4).** Training on human games is *supervised*
learning: predict the move a strong player made (behaviour cloning) and the
result of the game. No trial and error is involved, so it is not RL. It
can at best play as well as the humans in the data. It is still a useful
first step: it checks that the training pipeline works, and gives RL a
network that already knows something. (AlphaGo, AlphaZero's predecessor for
the game of Go, started this way; AlphaZero did not.)

**Function approximation and generalisation (Module 4).** A table with one
entry per position works for tic-tac-toe (5,478 legal positions, 4,520 of
them where the game isn't over yet). Chess has about
10^44, so the agent must *generalise*: give sensible answers in positions it
has never seen. The network is a *function approximator*: it computes an
answer for any input instead of looking one up. The cost is its errors in
unseen positions. Those errors are the reason for the *gate* (an arena
match a new network must win before it is used; see "Why a gate" below).
Module 6, exercise 7 has you replace the tic-tac-toe table with a small
network and measure both the benefit and the cost.

**Search as policy improvement (Module 5).** The visit counts from a PUCT
search are a better policy than the network's own output, because search
looks ahead. An
operation that turns a policy into a better one is called *policy
improvement*. Here it's `puct_search`.

**The AlphaZero loop is policy iteration (Module 6).** Classic *policy
iteration* alternates two steps until nothing changes:

1. **policy evaluation**: work out V for the current policy
2. **policy improvement**: make a better policy using V

AlphaZero does an approximate, sampled version of both in every iteration.
The *champion* is the network that plays the self-play games; the
*candidate* is a copy trained to replace it.

| Step | In `selfplay/` |
|---|---|
| improve: search gives a better policy than the raw network | `self_play.py`: every move comes from `puct_search` with the champion network |
| evaluate: learn V of the policy that played | value head trained on game results (`train_loop.py::soft_target_loss`, mean squared error part: the average squared difference between predicted value and game result) |
| write the improvement back into the network | policy head trained toward the visit counts (cross-entropy: a measure of how far the predicted probabilities are from the visit counts) |
| keep only real improvements (the gate, from AlphaGo Zero) | `arena.py`: the candidate becomes the new champion (is *promoted*) only if it scores > 55% |

**Monte Carlo and temporal-difference targets (Module 6).** The value target
for every position is the final game result. This is a *Monte Carlo*
target: a real return with no guessing. The alternative is a
*temporal-difference (TD)* target: use the network's own estimate of the
*next* position, V(s'), as the target for V(s). This is called
*bootstrapping*. TD can learn before the game ends, and its targets vary less from game to
game (lower *variance*). But it learns from its own possibly wrong guesses,
which gives a systematic error (*bias*).
Monte Carlo has no bias but is noisy: one lucky blunder in a won game labels
every position in that game +1. TD-Gammon (a 1992 backgammon program) used
TD; AlphaZero uses Monte Carlo.

**Credit assignment (Module 6).** A game is won at the end, and the learner
must decide which of its 40 or so moves deserve the credit. Monte Carlo
targets give every position the same label, and averaging over many games
separates good moves from bad ones. Search also helps: the policy target is
the search's opinion of each position, which does not depend on the game
result.

**Exploration in self-play (Module 6).** Without exploration, self-play
repeats one game and learns nothing (the tic-tac-toe lab shows this). This project explores at three levels:

| Mechanism | Level | Where |
|---|---|---|
| exploration bonus of PUCT / UCT (the part of the selection score that favours moves with few visits) | inside the search, at every node | `PUCTNode.select_child` |
| Dirichlet noise (random amounts added to the network's move probabilities) | the root of each self-play search: which moves get examined at all | `add_dirichlet_noise`, `add_root_noise=True` |
| temperature (at 1, moves are sampled in proportion to visits; at 0, the most-visited one is played) | which move is played, in the opening plies of each game | `select_move(temperature=1.0)` while `ply < temperature_moves` |

**On-policy and off-policy data (Module 6).** *On-policy* learning trains only
on data from the current policy. *Off-policy* learning uses data from other
policies, such as older versions of itself. The *replay buffer* (the store
of recent self-play positions that training samples from) mixes games from
several recent champions, so training is slightly off-policy. That helps up
to a point: there is more data, and each training *batch* (the group of
positions used for one update) mixes positions from many games. But a game
from a much weaker champion records how that weaker player's game ended.
That result is a poor target for today's policy. The buffer's capacity
sets this balance (`selfplay/replay_buffer.py`).

**Why a gate, and why AlphaZero dropped it (Module 6).** In exact policy
iteration each step is guaranteed to be at least as good as the last. With a
neural network trained on a few noisy games, it isn't. A candidate can be
worse, and if a worse network generates the next data, each iteration can
make the next one worse. AlphaGo Zero (the Go version that came just before
AlphaZero) therefore promoted a candidate only if it won 55% of an arena
match. AlphaZero (2018) dropped the gate and always trained the latest
network: at its scale the risk was small and the gate slowed progress.
The tic-tac-toe lab shows the cost of a gate: once both sides draw every
game, no candidate can score above 55%, and a gated loop stops learning.

**Measuring an RL agent (Module 7).** Game results are partly random, so a
short match gives only a rough estimate of strength, and every score needs
a 95% interval (the range that very likely contains the true score;
see Module 7). Also, "beats its previous self" says little about absolute
strength.

## Going deeper: questions a practitioner would ask

These questions add detail to the ideas above. Read them after Module 6, once
you've run the tic-tac-toe lab.

**Episodes, and no discount.** A game always ends, so this is an *episodic*
problem. Many RL methods shrink future rewards by a discount γ < 1 per step.
Here γ = 1 and nothing is discounted, so the return from any position is the
final result. (Discounting matters in problems that never end, or when
earlier rewards are worth more than later ones.)

**Which policy the value head measures.** It learns the expected result of
the games as they were played, which includes:

- the champion's search
- its Dirichlet noise and temperature
- an average over several recent champions, because the buffer mixes them

This *behaviour policy* is a little weaker and more random than the *greedy* policy (always play the
most-visited move) that arena matches test. The opponent is also the agent, so
the "environment" changes whenever the agent does: self-play is
*non-stationary*, unlike learning against a fixed opponent.

**Generalised policy iteration, and its condition.** Search improves the
policy and training on results evaluates it, a little at a time: this is
*generalised policy iteration*. Its condition is that search only improves on the network if the values it backs up are roughly
right. If the value head is badly wrong, search prefers the moves the wrong
values favour, training copies those choices, and the loop reinforces its
own mistakes. The corrections come from the rules: terminal
positions (game over) that search reaches, and game results.

**What each head contributes to search.** With random values, backed-up
Q-values are random, so visits end up following the priors (the policy head's
move probabilities) and more simulations add nothing. With good values and
uniform priors, more simulations keep helping, because each one adds real
information (Module 5, Q3). So the value head is what makes extra search
useful.

**Why both kinds of exploration.** Dirichlet noise changes which moves the
search *looks at*. But with temperature 0 the most-visited move is still
played, so games stay nearly deterministic. Temperature changes which move is
*played*. But without Dirichlet noise the search never examines moves the
network gives low probability. So each mechanism adds exploration that the
other does not provide. Arena matches turn both off because they measure the
greedy player you would use in real games. Either one would add variance and
measure a weaker player.

**Off-policy data, in more detail.** The *value* target from an old game is
the result of an older, weaker behaviour policy, so it is a biased label for
today's policy. The *policy* target is different. It is the visit
distribution of an older search, and the network learns to copy it, as in
supervised learning. The game result is not part of it, so the weaker
player's results do not distort it; it is only out of date. Off-policy
methods often correct for old data with *importance sampling* (reweighting
each example by how likely today's policy was to produce it). This project
works without it because the buffer only holds recent champions, and the
gate keeps each change small, so the data is nearly on-policy.

**What keeps the loop stable.** The network trains on data it generated itself.
Several parts of the design stop errors from growing through this cycle:

- the champion is *frozen* (not changed) while the candidate trains, so the
  source of data stays fixed. This is similar to the "target network" in
  deep Q-learning (Q-learning with a neural network), a fixed copy of the
  network used to compute targets.
- the replay buffer mixes generations, so each batch mixes many games
- `train_steps`, the learning rate (how far each update moves the network's *weights*, its adjustable numbers) and
  gradient clipping (a cap on the size of an update;
  `clip_grad_norm_` in `train_candidate`) keep each update small
- the gate, with the cost you saw in the lab

**Credit assignment gets worse with distance.** A label from the end of the
game is nearly right for positions close to the end. It says little about
positions in the opening, where many later moves decided the result. Draws
add a further problem: a draw labels every position 0 and says almost
nothing about which moves were good.

**How much a gate can tell you.** A candidate of exactly equal strength
needs 4 wins out of 6 (with no draws) to beat 55%, which happens about 34%
of the time. So a 6-game gate promotes a no-better network about one time in
three. Arena games are also nearly deterministic, so six games may be fewer than
six independent samples. And strength isn't always transitive:
B can beat A and C beat B while C still loses to A. A chain of "beats its
predecessor" results can go round in a cycle with no real progress. This is
why Module 7 measures against a fixed, *calibrated* opponent (one whose
strength has been measured).

**A game stopped at the move cap has not ended.** The self-play loop scores a game that
hits `max_moves` as a draw, as if the game had ended. But the game was
stopped early (*truncated*); it did not end. The state the network sees has
no move counter that would show the cap coming (`board_to_tensor` has a
fifty-move counter, but nothing counts plies). So the same position gets
labelled 0 near the cap and its real result elsewhere. Two positions that
look identical to the network but need different answers are an example of
*state aliasing*. It means the state is not fully *Markov*: it does not
contain everything needed to predict what happens next. There are two
standard fixes:

1. Treat the cap as truncation and bootstrap the target from the network's
   value of the last position. This is a TD target, used exactly where Monte
   Carlo has no result to give.
2. Add the number of remaining plies to the input, so the state contains what it needs.

**Changing the reward changes the goal.** You could score games cut off at
the move cap by material instead of calling them draws. That is
*reward shaping*, and it changes what the agent is optimising: it may learn
to win material instead of games, and turn down a sacrifice that wins. If
you shape, keep it to clearly decided positions, or phase it out as real
results start arriving.

**The played move isn't the search's opinion.** The policy target is the
visit distribution at temperature 1, even when the move played was chosen
greedily or at random. Which move gets played is an *exploration* decision
about what data to collect, so the target ignores it.

**Self-play has shared weaknesses.** In self-play the opponent is the same
network. A blunder the network does not recognise is one its opponent, the
same network, also fails to exploit. Such a
"mistake" wins in the data game after game, and averaging won't remove it.
Exploration and games against other opponents reveal it.

**Alternatives to the gate.** The gate decides which network *generates
data*. You can train one network continuously and gate only the data
generator, or drop the gate and always generate with the latest network
(AlphaZero). A data source that changes slowly has a role like the target
network in deep Q-learning: the learner's target stays the same between
updates, instead of changing after every update.

## Where AlphaZero sits among RL methods

| Family | Learns | Example | Compared with AlphaZero |
|---|---|---|---|
| Value-based | Q(s, a) directly, by TD | Q-learning, DQN (Deep Q-Network, which learned Atari video games) | no search; acts by picking the highest Q |
| Policy gradient | π directly, by raising the probability of actions that led to high return | REINFORCE, PPO (Proximal Policy Optimisation) | no search; learns from returns only |
| Model-based and search | π and V, improved by planning with a model | AlphaZero, MuZero | search turns a weak network into a strong policy, and the network learns to imitate the search |

## Check yourself

Each module's questions test its own RL connection. These test the whole picture.

<details>
<summary><b>R1.</b> Self-play with no exploration at all (no Dirichlet noise, temperature 0). Predict what the replay buffer contains after 10 iterations.</summary>

The same game, many times over. With no randomness, the same network plays
the same moves from the start position every game. It never sees a position
outside that one line, so it can't learn anything about the rest of the game.
Run `python3 selfplay/tictactoe_lab.py --no-root-noise --temperature-moves 0`:
coverage stays below 1% of positions.
</details>

<details>
<summary><b>R2.</b> Why can't supervised training on grandmaster games ever produce a superhuman engine on its own?</summary>

Its target is "the move the human played". A perfect score means playing
exactly like the humans in the data, mistakes included. To play better than
those humans, the learner needs a signal that does not come from them.
In AlphaZero that signal is search and game results: the search can find
moves no human in the data played, and the result says whether they worked.
</details>

<details>
<summary><b>R3.</b> Name the policy-evaluation step and the policy-improvement step in this repository's loop.</summary>

Evaluation: training the value head on the results of the champion's
self-play games. Improvement: PUCT search with the champion network, whose
visit counts are a better policy than the network's raw output. The policy
head is then trained toward those visit counts, which writes the improvement
back into the network.
</details>

<details>
<summary><b>R4.</b> Swap the Monte Carlo value target for a TD target. What would you change in <code>selfplay/self_play.py</code>, and what risk do you take on?</summary>

Instead of labelling each position with the final result, label it with the
current network's value of the *next* position, with the sign flipped for the
side to move (and the final result for the last position). You would learn
from your own estimates, so early on the network trains toward its own random
guesses. That is bias, which can slow learning or stop it from working. In
exchange you get lower variance, because one lucky game no longer labels 40
positions.
</details>

<details>
<summary><b>R5.</b> Someone sets <code>max_moves=40</code> to speed up self-play. What happens to the value targets, and why can't the network learn the right answer even with infinite data?</summary>

Most games hit the cap, so their positions are labelled 0. In addition, the
network can't see how close the cap is (the input has no ply counter), so
the same position is labelled 0 in one game and ±1 in another: state
aliasing. Infinite data would only average those labels; it would not
separate them. Fix it by bootstrapping from the network's value at the
cut-off instead of calling it a draw, or by putting the number of remaining
plies into the input.
</details>

<details>
<summary><b>R6.</b> Candidates keep beating their predecessors in the arena, but you suspect the engine isn't improving. Give two reasons that can happen, and the evaluation that would settle it.</summary>

A 6-game gate promotes an equal network about a third of the time, so some
"wins" are luck. Also, strength can be non-transitive, so a chain of
head-to-head wins can form a cycle. Settle it against a *fixed* reference:
the initial checkpoint, or better a calibrated opponent such as Stockfish at
a set rating, with enough games for a narrow 95% interval (Module 7).
</details>

<details>
<summary><b>R7.</b> Why does this project's loop not reweight old replay-buffer examples with importance sampling, as many off-policy methods do?</summary>

The buffer only holds games from recent champions, and a champion changes
only when a candidate passes the gate. So the policies that produced the
data are close to today's: the data is nearly on-policy. The policy targets
are also visit distributions from search, which contain no returns, so the
usual correction applies to them less. A much larger buffer would make this
argument weaker.
</details>

<details>
<summary><b>R8.</b> The value head is badly inaccurate early in training. Why can search make things worse rather than better, and what stops that going on forever?</summary>

Search relies on the values it backs up. If they're wrong, it concentrates
visits on moves that only look good to the inaccurate value head. Training
then copies those visit counts into the policy, so the loop reinforces its
own error. The correction comes from information outside the network:
exact values at terminal positions the search reaches, and the real game
results used as value targets.
</details>

Back to [the course](README.md)
