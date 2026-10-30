# Unit 2: Looking ahead with minimax and alpha-beta

**You'll be able to:**
- explain minimax as "assume the opponent plays their best reply"
- explain how alpha-beta pruning skips work without changing the answer
- measure how much pruning saves, using the real search

**Before this:** [Unit 1](01_evaluation.md). **Next:** [Unit 3, Quiescence](03_quiescence.md). **Time:** ~60–90 min.
**Longer reading:** `minimax_tutorial.html`, section "Day 3–4: Looking Ahead".

## The idea

The greedy player from unit 1 looks one move ahead and assumes the opponent then does
nothing. **Minimax** looks several plies ahead and assumes the opponent always picks
the reply that's worst for us:

- White **maximises** the evaluation, Black **minimises** it
- at the search's depth limit, call the evaluation function
- pass the scores back up: at White's turns take the max, at Black's the min

The number of positions explodes: about 20 at depth 1, 400 at depth 2, 8,902 at depth 3,
and 197,281 at depth 4. **Alpha-beta pruning** cuts this down. If you already know one move
guarantees you +50, and you find that another move lets the opponent force −200, you can stop
looking at that second move. You don't need to know *exactly* how bad it is.

- **alpha** = the best score White is already guaranteed
- **beta** = the best score Black is already guaranteed
- when `beta <= alpha`, the rest of this branch can't affect the result, so it's **pruned**

Pruning works best when the strongest moves are tried first. That's why the code **orders
moves** (captures and checks first) before searching them.

## In the code

| Where | What to look for |
|---|---|
| `search/minimax.py`, `minimax` | The recursion: `maximizing` flips at every ply; `if beta <= alpha: break` is the prune |
| `search/minimax.py`, `order_moves` | Move ordering: captures (most valuable victim first), checks, promotions |
| `search/minimax.py`, `best_move_minimax` | Runs `minimax` for each root move and returns the best |
| `search/minimax.py`, `minimax`, `if depth == 0:` | At the leaves it calls `quiescence_search`, not `evaluate`. That's unit 3 |

## Try it

Count how many positions the search visits at depth 3 from the starting position:

```bash
python -c "import chess; from search.minimax import minimax; n = [0]; minimax(chess.Board(), 3, float('-inf'), float('inf'), True, n); print(n[0])"
```

Expected output: `2470`

Compare it with the number of positions a search **without pruning** would reach (the
"perft" numbers, a standard count of legal move sequences):

| Depth | Positions without pruning | This search visits |
|---|---|---|
| 1 | 20 | 41 |
| 2 | 400 | 250 |
| 3 | 8,902 | 2,470 |

## Exercises

**1. Predict before reading on:** at depth 1 the search visits **41** nodes, more than the 20
legal moves. Where do the extra 21 come from?

<details><summary>Answer</summary>

`nodes_searched` counts quiescence nodes too (unit 3). At depth 1, every leaf goes into
`quiescence_search`, which counts itself, and the root counts as one node. So it's 1 root
+ 20 children + 20 quiescence entries = 41. The count also has to be read carefully: it measures
"function calls", not "distinct positions".
</details>

**2.** At depth 3 the search visits 2,470 nodes instead of 8,902. Why does the saving get
**bigger** as depth increases?

<details><summary>Answer</summary>

Each prune removes a whole subtree, and deeper searches have bigger subtrees to remove.
With perfect move ordering, alpha-beta visits roughly the *square root* of the minimax
node count, so the gap widens exponentially with depth.
</details>

**3. Hands-on:** how much does move ordering matter? This position is the Scholar's Mate setup,
where White can play `Qxf7#`. Count nodes at depth 3 with ordering:

```bash
python -c "import chess; from search.minimax import minimax; n = [0]; minimax(chess.Board('r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4'), 3, float('-inf'), float('inf'), True, n); print(n[0])"
```

Then, in `order_moves`, temporarily make it `return moves` on its first line (no ordering)
and run the command again. Predict first: up or down, and by roughly how much? Undo the
change afterwards with `git checkout search/minimax.py`.

<details><summary>Answer</summary>

With ordering: **2,886** nodes. Without: **4,033**, about 40% more work. Pruning only kicks
in once a good move has been found, and without ordering good moves (here, the capture
`Qxf7`) are found later. Both versions still choose `h5f7`: ordering changes the speed,
not the answer. (From the quiet starting position the difference is tiny, 2,470 vs 2,521,
because there are no captures to put first.)
</details>

**4.** Minimax assumes the opponent plays perfectly. When might that be a bad assumption in a
real game?

<details><summary>Answer</summary>

Against a weak or tricky opponent, a line that's "worse against perfect play" can be
practically better (setting traps, for example). Minimax also only sees as far as its depth
and as well as its evaluation, so its "perfect opponent" is really just "an opponent who
thinks exactly like me".
</details>

## Check yourself

1. Which side maximises and which minimises?
2. What condition triggers a prune?
3. Does alpha-beta ever change which move is chosen?

<details><summary>Answers</summary>

1. White maximises, Black minimises (scores are from White's view).
2. `beta <= alpha`.
3. No. It returns the same result as plain minimax, just with fewer nodes searched.
</details>
