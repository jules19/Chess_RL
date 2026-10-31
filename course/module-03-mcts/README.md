# Module 3: Monte Carlo Tree Search

> **Where you are:** the second search, the one AlphaZero is built on. You'll find the two parts of it that a neural network will later replace.
> **Time:** 4-6 hours, plus matches that run in the background (Monte Carlo tree search, MCTS, is slow in Python; times below are measured).
> **You'll be able to:**
> - walk through one MCTS simulation (one pass of its four steps: select, expand, roll out, backpropagate) and say what each step is for
> - explain the exploration/exploitation trade-off and what the exploration constant C controls
> - estimate a position's value by Monte Carlo rollouts (games played out to the end), and say why that estimate is noisy
> - name the two parts of `mcts.py` that Module 5 replaces with the network

## Concepts

Minimax searches every move to a fixed depth. Monte Carlo tree search (MCTS)
grows its tree selectively, to no fixed depth, and spends most of its work on
the most promising moves. In the tree, each *node* is a position: the
*root* is the current position, and a node's *children* are the positions
one move later. Each pass of the search's four steps is a *simulation* (see
`search/mcts.py`):

1. **Selection.** Walk down from the root, choosing one child at each node.
   *Exploitation* favours children with good results so far; *exploration*
   favours children with few visits, whose results are still uncertain. UCT
   (Upper Confidence bound for Trees) combines both in one score, and the
   search picks the highest:
   `Q + C·√(ln N_parent / N_child)`.
   Q is the child's average result. N_child and N_parent are the visit
   counts of the child and its parent (ln is the natural logarithm), so the
   second term is large for a little-visited child. C, the *exploration
   constant*, sets the size of that bonus. This is the bandit trade-off (see
   "RL connection" below), applied at every node.
2. **Expansion.** When selection reaches a node that still has untried
   moves, pick one of them and add its position to the tree as a new child.
3. **Rollout.** Estimate the new node's value by playing the game out from
   it. Random rollouts give weak estimates, so this repository uses
   *evaluator-guided* rollouts (`simulate_with_evaluator`). A *ply* is one
   move by one side. The rollout uses *1-ply greedy* play: from a sample of
   legal moves, it plays the one whose resulting position the Module 1
   evaluator scores best.
4. **Backpropagation.** Add the result to every node on the path back to
   the root. Flip the sign at each ply, because a result that is good for
   White is bad for Black. Sign-flipping is a frequent source of MCTS bugs,
   and `tests/test_mcts.py` tests it.

**Why visit counts choose the final move.** The average value of a rarely
visited move is noisy. A move only collects many visits if UCT keeps
choosing it, and UCT keeps choosing it only while its results stay good. So
the visit count measures how confident the search is in a move.

**What the blunder filter is.** With `filter_blunders=True` (the default), a
node skips any move that leaves a piece worth a knight or more attacked and
undefended (`hangs_material()` at the top of `search/mcts.py`, a one-ply
count of attackers and defenders). This is hand-written chess knowledge in a
search that is supposed to need none, so exercise 3 measures it.

**Where this is heading.** Step 3 only needs an estimate of who is winning,
and step 1 needs to know which moves are promising. The network you build in
[Module 4](../module-04-neural-network/README.md) provides both: its *value
head* estimates who is winning, and its *policy head* gives a probability for
each move. Module 5 puts it into the search and removes rollouts completely.

## RL connection: exploration, bandits, and Monte Carlo evaluation

Three reinforcement learning (RL) ideas are part of this one algorithm.

**Exploration and exploitation.** At each node MCTS spends the next
simulation either on the move that looks best so far (exploit) or on one it
knows little about, in case it is better (explore). The simplest form of
this choice is the *multi-armed bandit*: several slot machines, unknown
payouts, a limited number of pulls. UCT's formula is the classic bandit
algorithm UCB1 (Upper Confidence Bound), applied at every node.

**Monte Carlo evaluation.** A rollout estimates how good a position is by
playing it to the end and recording the result. The average of many
rollouts estimates V(s), the value of position s (the expected result for
the side to move), with no chess knowledge. Its weakness is noise: one game
is a single, very random sample. Better rollout policies (the rules that pick
the rollout's moves; exercise 1) reduce that noise.

**Action values in the tree.** Each child's average result is an estimate
of Q(s, a): how good it is to play move a here. MCTS never stores a learned
Q; it builds these estimates new for each position. AlphaZero keeps this
part and learns V and the move *priors* (the policy head's probability for
each move before any search).

## Read

- `search/mcts.py`: full teaching docstrings throughout
- `tests/test_mcts.py`: how the tests check the sign flip and other rules
  that must always hold

## Do

**About time.** MCTS here is pure Python. At 200 simulations a move takes
about 10 seconds with evaluator-guided rollouts and 1.4 seconds with random
rollouts (measured on an 8-core laptop CPU), so a 10-game match at 200
simulations takes hours. The matches below use 50 simulations (sims= in the
commands). Start one, then read `search/mcts.py` while it runs. First,
check your setup in about a minute with a small match:
`python3 cli/match.py mcts:sims=10 mcts:sims=10,rollouts=random --games 2 --max-plies 20 --adjudicate 3`.
Two games check that the command works; comparing engines needs many more.

At 50 simulations MCTS can win material but rarely checkmates, so most
games would reach the move limit as draws and tell you nothing. `--max-plies 120 --adjudicate 3` sets the move
limit to 120 plies (60 moves) and then *adjudicates* the game (decides the
result without playing on): a side at least 3 pawns of material ahead wins.
Engine tournaments stop very long games in the same way.

1. **Exercise: measure how much rollout quality matters.** Play
   evaluator-guided rollouts against random rollouts:
   ```bash
   python3 cli/match.py mcts:sims=50 mcts:sims=50,rollouts=random --games 6 \
       --max-plies 120 --adjudicate 3
   ```
   **Expected (measured, about 15 minutes):** evaluator-guided rollouts
   scored 3.5/6 (58%). The 95% interval (the range that very likely contains
   the true score; see [Module 7](../module-07-measure-and-scale/README.md))
   is 24% to 86%. `match.py` reports that the match can't tell the engines
   apart. Six games can only detect a very large difference; to measure a
   gap this size, run `--games 40` overnight. The purpose here is to read
   the interval before you believe the score.
   The gap is the value of rollout knowledge. Remember it for Module 5,
   where a network replaces the rollout completely.
2. **Exercise: exploration constant sweep.** First look, then play. On one
   position, compare the visit distributions:
   ```python
   import chess
   from search.mcts import best_move_mcts
   board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
   for c in (0.5, 1.41, 3.0):
       print("C =", c)
       best_move_mcts(board, simulations=200, exploration_constant=c, verbose=True)
   ```
   **Expected (measured, about 9 s per run):** with C = 0.5 the top three
   moves took 32, 31 and 31 of the 200 visits: the search concentrates on a
   few moves early. With C = 3.0 the top move got only 12, close to an even
   spread over 27 legal moves. At C = 0.5 the poor move Ke2 was among the
   leaders, because Monte Carlo values are still noisy at 200 simulations.
   Then play `mcts:sims=50,c=0.5` against `mcts:sims=50,c=3.0` with
   `cli/match.py` (same flags as exercise 1). Write one paragraph on how
   too-low and too-high C behave and which won, with its 95% interval.
   There is no reference result for this match.
3. **Exercise: measure how much the blunder filter helps.**
   ```bash
   python3 cli/match.py mcts:sims=50 mcts:sims=50,filter=0 --games 6 \
       --max-plies 120 --adjudicate 3
   ```
   **Expected (measured, about 50 minutes, because both sides use
   evaluator-guided rollouts):** 3.5/6 for the side with the filter, 95%
   interval 24% to 86%. Again, there are too few games to say which is
   stronger. Report the score with its 95% interval and without an Elo
   figure (Elo is a rating scale for playing strength; six games are too few
   to estimate it, see Module 7). Remember the result for when the network makes the filter unnecessary.
4. **Exercise: make the search reproducible.** First write a test asserting
   that MCTS (200 simulations, evaluator-guided rollouts) finds mate-in-1
   from `6k1/5ppp/8/8/8/8/5PPP/4R1K1 w`. Run it 10 times. Expected: it
   passes on some runs and fails on others, because the search is random.
   To fix this, add an optional `rng: random.Random` parameter to
   `mcts_search`/`best_move_mcts`. Pass it down to every place where
   `search/mcts.py` uses the global `random` module.
   Calling `random.seed(7)` would also make the test pass, but it changes
   the behaviour of all other code in the process that uses `random`, with
   no warning. An explicit random number generator avoids that.
   Your checkpoint is `tests/course/test_module03_mcts.py`. Delete the skip
   line above its test (the line that tells pytest to skip the test), and
   make it pass.

## Check your understanding

<details>
<summary><b>Q1 (explain it in your own words).</b> Which two pieces of <code>mcts.py</code> will Module 5 replace with neural network calls?</summary>

The first is the rollout step (`simulate_with_evaluator`): one call to the
value head replaces the whole rollout. The second is the equal treatment of
untried moves in expansion and selection, which policy-head priors replace
in the PUCT formula. (PUCT, Predictor + Upper Confidence bounds for Trees, is
AlphaZero's selection rule: it combines the tree's results with the
network's move probabilities.) The selection walk, expansion and
backpropagation with sign flips move into `puct.py` almost unchanged.
</details>

<details>
<summary><b>Q2.</b> Predict: you delete the <code>value = -value</code> sign flip in <code>backpropagate</code>. What does the engine now do?</summary>

The engine chooses moves that are good for its opponent. Each result is
credited to a node and its parent with the same sign, so moves good for the
side that just moved look good for the side choosing. The engine then plays
moves that let its opponent checkmate it.
Make the change and run
`tests/test_mcts.py::test_backpropagation_updates_and_alternates_perspective`.
Expected: it fails within milliseconds. You will meet the same bug again in
the neural network version in Module 5.
</details>

<details>
<summary><b>Q3.</b> Why is the final move chosen by <em>visit count</em> instead of best average value?</summary>

A move visited 3 times with a lucky 0.9 average tells you very little. A
move visited 150 times with 0.6 kept that value over many visits, because
UCT only keeps visiting a child whose results stay good. So a visit count
combines value and confidence in one number. In Module 6 the visit
distribution becomes the training target.
</details>

<details>
<summary><b>Q4 (RL question).</b> A rollout from a position returns +1. Another from the same position returns -1. What does that tell you about Monte Carlo value estimates, and what are the two ways to make them better?</summary>

That a single rollout is one noisy sample of the position's value: the same
position can be won or lost depending on random choices later. You improve
the estimate by averaging more rollouts (more simulations: slower), or by
making each rollout less random, with a better rollout policy (exercise
1). AlphaZero uses a third method: it replaces the rollout with a learned
value function.
</details>

## Checkpoint

```bash
pytest tests/test_mcts.py -q                  # regression suite still green
pytest tests/course/test_module03_mcts.py -q  # un-skipped and passing
```

- [ ] Both commands pass
- [ ] Q1 answered correctly before looking at the answer

Previous: [Module 2: Minimax Search](../module-02-minimax-search/README.md) · Next: [Module 4: The Neural Network](../module-04-neural-network/README.md)
