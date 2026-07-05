# Module 6 — Self-Play Reinforcement Learning

**Goal:** close the loop. Remove the human games: the engine now generates
its own curriculum, and you run the full AlphaZero cycle end-to-end.

## Concepts

The loop (see the diagram in `selfplay/__init__.py`):

```
self-play (champion + exploration) → replay buffer → train candidate
        ↑                                                   │
        └────────── arena gate: promote only if >55% ───────┘
```

**The training target is the search, not the move** (`selfplay/self_play.py`).
Each position's policy target is the full PUCT *visit distribution*. The
network learns to predict what search-amplified-network concludes — and since
search(net) is stronger than net, imitating it is self-improvement. This one
idea is AlphaZero.

**Outcome labels are noisy and that's fine.** Every position in a won game
gets +1 (from the winner's perspective) — including the winner's mistakes.
Averaged over thousands of games, signal beats noise. Don't try to be clever
per-position; be statistical.

**The replay buffer is a window** (`selfplay/replay_buffer.py`): big enough
to decorrelate batches and span several generations, small enough that
ancient weak-network games age out. It's a real hyperparameter.

**The arena is a gate, not a scoreboard** (`selfplay/arena.py`): a bad
training iteration can make the network worse, and if the worse network
generates the next data, the loop can spiral. Candidates must *demonstrate*
superiority (>55%, colors alternated, no exploration noise) to take over.

**Exploration is load-bearing.** Root Dirichlet noise + early-game
temperature are what stop self-play from repeating one deterministic game
forever. Remove either (exercise 3) and watch the buffer fill with clones.

**Soft-target loss** (`selfplay/train_loop.py::soft_target_loss`): the
supervised loss was cross-entropy against a class index; now the target is a
distribution, so it's `-Σ π(a)·log p(a)`. The test suite proves the two
coincide for one-hot targets.

## Read

All four files in `selfplay/` — they're short and heavily documented.

## Do

1. **Run the toy loop** (~1 minute on CPU — it proves plumbing, not strength):
   ```bash
   python3 selfplay/train_loop.py --iterations 2 --games-per-iter 2 \
       --simulations 16 --train-steps 20 --arena-games 2
   ```
2. **Run a real (small) loop overnight.** Warm-start from your Module 4
   supervised checkpoint — this is what AlphaGo did before AlphaZero went
   tabula rasa:
   ```bash
   python3 selfplay/train_loop.py --init-checkpoint checkpoints/best_model.pt \
       --iterations 10 --games-per-iter 20 --simulations 64 --train-steps 300
   ```
   Track promotions. Then arena the final champion against the *initial*
   checkpoint — did the loop actually gain strength?
3. **Exercise: sabotage study.** Three runs: (a) no Dirichlet noise,
   (b) temperature always 0, (c) promotion threshold 0.30. Predict each
   failure mode *before* running, then check. Write three sentences.
4. **Exercise: buffer-size sweep.** Capacity 2,000 vs 50,000 at fixed games
   per iteration. Which promotes more often early? Later?
5. **Exercise: resumable loops.** The loop already saves
   `champion.pt`/`replay_buffer.npz`; add `--resume` so a killed run
   continues (load buffer + champion, keep iteration numbering). Long RL
   runs die; checkpointing discipline is part of the craft.

## Checkpoint

```bash
pytest tests/test_selfplay.py -q
```

- [ ] Green
- [ ] A completed multi-iteration loop with at least one promotion
- [ ] Final champion beats the starting checkpoint in your own arena match

Next: [Module 7 — Measure & Scale](../module-07-measure-and-scale/README.md)
