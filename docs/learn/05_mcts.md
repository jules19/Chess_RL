# Unit 5: Monte Carlo Tree Search (MCTS)

**You'll be able to:**
- describe the four MCTS steps: select, expand, simulate, backpropagate
- calculate a UCT score by hand and explain the exploration term
- explain why this engine's MCTS needs guided rollouts, and why 200 simulations are too few

**Before this:** [Unit 2](02_minimax.md) (unit 4 is optional here). **Next:** [Unit 6, The network](06_network.md). **Time:** ~60–90 min.

## The idea

Minimax searches *every* move to a fixed depth. MCTS takes another approach: **sample**, and
spend more effort on the moves that look promising.

MCTS grows a tree of positions one **simulation** at a time. Each simulation does four steps:

1. **Select:** start at the root and walk down the tree, at each node picking the child with
   the best **UCT** score (below), until you reach a node with untried moves.
2. **Expand:** add one untried move as a new child node.
3. **Simulate** (also called a "rollout" or "playout"): play the game forward from that new
   position quickly, and turn the outcome into a number between −1 and +1.
4. **Backpropagate:** walk back up to the root, adding that number to every node on the path.
   **Flip its sign at each level**, because a result that's good for White is bad for Black.

After all the simulations, play the root move that was **visited most**. Visits measure how
much the search trusted a move, which is more robust than its average score.

### UCT: balancing "known good" against "not yet explored"

```
UCT(child) = average value  +  C × sqrt( ln(parent visits) / child visits )
             └ exploitation ┘   └──────────── exploration ─────────────┘
```

The first term favours moves that have done well. The second is large for moves that have
rarely been tried, so nothing is ignored forever. `C` (here 1.41 ≈ √2) sets the balance.
An unvisited child scores infinity, so every move gets tried once.

## In the code

| Where | What to look for |
|---|---|
| `search/mcts.py`, `MCTSNode` | `visit_count`, `total_value`, `children`, `untried_moves` (shuffled, and blunders removed if `filter_blunders`) |
| `search/mcts.py`, `MCTSNode.uct_value` | The formula above, line for line |
| `search/mcts.py`, `mcts_search` | The four steps, labelled `# 1. SELECTION` … `# 4. BACKPROPAGATION` |
| `search/mcts.py`, `simulate_random` | Pure random rollout |
| `search/mcts.py`, `simulate_with_evaluator` | Guided rollout: up to 50 plies, each picking the best of ~10 sampled moves by `evaluate`, then `evaluate / 1000` clamped to [−1, 1] |
| `search/mcts.py`, `backpropagate` | `value = -value` at each step up |
| `search/mcts.py`, `hangs_material` | The blunder filter used when creating nodes |

## Try it

```bash
python -c "import chess, random; from search.mcts import mcts_search; random.seed(0); mcts_search(chess.Board(), simulations=200, verbose=True)"
```

It takes about 10 seconds and ends with:

```
Top moves by visit count:
  1. d2d3: 18 visits, avg value: +0.326, win rate: 66.3%
  2. a2a3: 16 visits, avg value: +0.276, win rate: 63.8%
  3. h2h4: 14 visits, avg value: +0.218, win rate: 60.9%
  4. e2e4: 13 visits, avg value: +0.098, win rate: 54.9%
  5. e2e3: 12 visits, avg value: +0.075, win rate: 53.7%
```

(`random.seed(0)` makes it repeatable; with another seed you'll see a different order.)

## Exercises

**1. Predict:** before looking at the output above, which first move would you expect a good
search to prefer? Then explain why this MCTS likes `d2d3` and `a2a3`, moves no strong player
would choose.

<details><summary>Answer</summary>

There are 20 root moves and 200 simulations, so each move gets about 10 rollouts. Each
rollout is a noisy, 50-ply guided game, so 10 of them can't tell `e2e4` from `a2a3`. The
differences in "avg value" (0.1 vs 0.3) are well within the noise. MCTS needs
**many more simulations**, or a much better rollout or evaluation (unit 9), before its
visit counts mean much. Search quality is a matter of *statistics*, not just of the algorithm.
</details>

**2. UCT by hand:** after 200 simulations at the root, child A has average value 0.3 from 10
visits, and child B has average value 0.1 from 2 visits. With C = 1.41, which does UCT pick next?

<details><summary>Answer</summary>

- A: 0.3 + 1.41 × √(ln 200 / 10) = 0.3 + 1.03 = **1.33**
- B: 0.1 + 1.41 × √(ln 200 / 2) = 0.1 + 2.29 = **2.39**

**B**: its exploration bonus outweighs its lower average. Check with
`python -c "import math; print(0.3 + 1.41 * math.sqrt(math.log(200) / 10), 0.1 + 1.41 * math.sqrt(math.log(200) / 2))"`.
</details>

**3. Why not random rollouts?** Run 100 purely random playouts from the starting position:

```bash
python -c "import chess, random; from search.mcts import simulate_random; random.seed(1); r = [simulate_random(chess.Board()) for _ in range(100)]; print(sum(1 for x in r if x != 0), 'decisive of 100')"
```

Expected: `9 decisive of 100`. What does that tell you about random rollouts as a signal?

<details><summary>Answer</summary>

91% of random games return 0, so a random rollout says almost nothing about whether the
starting move was good. That's why this project uses `simulate_with_evaluator`: shorter,
guided playouts that end with an evaluation. The trade-off is speed. Guided rollouts call
`evaluate` hundreds of times each, which is why MCTS here manages only ~20–25 simulations per second.
</details>

**4.** Why must `backpropagate` flip the sign at every level?

<details><summary>Answer</summary>

Each node's statistics are from the point of view of the player who chose the move into
it. Players alternate at each level, so a result worth +0.5 to one player is worth −0.5 to
the other. Without the flip, both players would be "trying to win for White" and the search
would be meaningless for Black's moves.
</details>

**5. Stretch:** minimax at depth 3 (unit 2) visited 2,470 nodes and took a fraction of a second.
MCTS with 200 simulations took about 10 seconds. Where does MCTS spend its time?

<details><summary>Answer</summary>

Almost all of it goes on rollouts: each one is up to 50 plies × ~10 calls to `evaluate`, so
up to ~500 evaluations per simulation and ~100,000 per move. The tree itself is tiny.
Remember this number; unit 9 shows that the neural version of MCTS has the same problem,
only with an expensive network instead of `evaluate`.
</details>

## Check yourself

1. Name the four steps of an MCTS simulation.
2. What do the two terms of UCT encourage?
3. Which root move does MCTS finally play, and why that one?

<details><summary>Answers</summary>

1. Select, expand, simulate (rollout), backpropagate.
2. Exploitation (choosing moves that have done well) and exploration (trying moves that have
   rarely been visited).
3. The most-visited move. Visits reflect where the search kept going back to, which is
   more robust than a noisy average.
</details>
