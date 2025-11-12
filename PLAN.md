# Goal

Design a Python chess program that **improves via reinforcement learning** (self-play) and stays maintainable, reproducible, and fast. No code—just the architecture.

# High-Level System

* **Players (Actors / Self-Play Workers):** Generate games by playing against themselves using the current policy (optionally guided by MCTS). Output game trajectories.
* **Replay Store:** Large, prioritized buffer of recent self-play positions, actions, search stats, and outcomes.
* **Trainer:** Samples from the replay store to update the neural network (policy + value). Periodically emits new checkpoints.
* **Evaluator (League / Arena):** Compares new checkpoints vs. the current best (Elo gates). Promotes only if better.
* **Inference Service:** Serves the latest “accepted” model to self-play workers with versioning.
* **Orchestrator:** Coordinates processes (start/stop workers, push/pull model, rollouts quotas), tracks metrics, and schedules training.
* **Artifact Registry:** Stores checkpoints, training configs, replay shards, and experiment metadata.

# Core Data Structures

* **State Encoding:**

  * 8×8×C tensor planes (pieces by color, side-to-move, castling rights, repetition counters, move count).
  * Optional history planes (last N boards) to disambiguate repetition and momentum.
  * Auxiliary scalars (ply count, no-capture/no-pawn clock).
* **Action Encoding:**

  * 4672-dim move head (from-to + underpromotions + special moves), or policy over legal moves only with a masking layer.
* **Trajectory Record (per move):**

  ```
  s_t, π_t (MCTS-improved policy), a_t, r_T (game result from the terminal), v̂_t (bootstrapped value), meta (FEN, legality mask)
  ```
* **Replay Entry Metadata:** game id, model version used, MCTS stats (visits, Q), game quality flags (resign, timeouts), and priority.

# Learning Algorithms (choose one; keep pluggable)

* **AlphaZero-style (recommended):**

  * **Network:** shared trunk → policy head + value head.
  * **Target:** minimize cross-entropy(π_t, pθ(s_t)) + MSE(r_T, vθ(s_t)) + L2.
  * **Search:** MCTS with PUCT using pθ as prior and vθ for leaf evaluation.
  * Pros: strong sample efficiency for board games.
* **PPO (alternative):**

  * Actors generate games on-policy; optional shallow search.
  * Pros: simpler infra; Cons: usually weaker than strong MCTS-guided self-play.
* **MuZero-style (advanced):**

  * Learn a dynamics model; higher complexity; future-proof if you plan to generalize beyond chess.

# Neural Network

* **Backbone:** Residual CNN stack (e.g., 10–40 ResBlocks; scale with compute).
* **Policy Head:** 1×1 conv → softmax over action space (or logits for all moves with legal-move mask).
* **Value Head:** 1×1 conv → global pooling → MLP → tanh in [−1,1].
* **Optional:** Squeeze-and-Excitation / CoordConv; Half-precision inference (fp16/bf16).

# Self-Play + MCTS

* **Search Parameters:** simulations per move (e.g., 200–800), Dirichlet noise at root, temperature τ>0 in opening (anneal to 0).
* **Exploration:** PUCT with c_puct; virtual loss for parallel tree search.
* **Move Selection:** sample from π early (diversity), argmax late (strength).
* **Resignation & Draw Logic:** threshold on running value; 3-fold and 50-move handled by rules engine.

# Training Loop

1. **Generate:** Actors play games using model v_k; store (s_t, π_t, r_T).
2. **Ingest:** Append to replay; shard to disk (e.g., LMDB/Parquet) + in-mem ring buffer.
3. **Sample:** Weighted by recency and surprise (|vθ−r_T| or low visit counts).
4. **Optimize:** SGD/AdamW with cosine decay + warmup; gradient clipping; EMA weights for eval.
5. **Checkpoint:** Save model, optimizer, config, and data fingerprint.
6. **Evaluate:** Run matches vs. current best; promote if win-rate > threshold with CI bounds.
7. **Rollout:** Orchestrator pushes promoted checkpoint to inference service; actors reload.

# Rules / Environment

* **Chess Engine Core:**

  * Legal move generator (bitboards), repetition detection, clocks, castling, en passant.
  * FEN/PGN I/O for debugging and analysis.
  * Deterministic, validated against reference suites (perft tests).
* **APIs:**

  * `Env.reset(fen=None) → state`
  * `Env.step(move) → (state', reward, done, info)`
  * `Model.predict(states, mask) → (policy_logits, value)`
  * `Search.run(state, model) → π_t`

# Evaluation & Curriculum

* **Elo System:** Glicko or SPRT gate to accept models.
* **Opening Book for Diversity:** Sample ECO lines up to depth d; or stochastic playouts from random legal openings; periodically refresh.
* **Curriculum Levers:**

  * Early: low simulations, high temperature, shorter games (move cap).
  * Later: more simulations, temperature→0, endgame tablebase guidance (optional for evaluation only).

# Persistence & Reproducibility

* **Artifacts:** versioned checkpoints (`v1.2.3`), immutable replay shards with checksums.
* **Config & Seeds:** Hydra/JSON configs; log seeds and hashes.
* **Experiment Tracking:** Metrics (losses, KL, Elo, promotion history, draw rate, resignation errors, node visits) with a run dashboard.

# Performance & Scaling

* **Parallelism:**

  * Many CPU self-play workers (legal moves + MCTS) + batched GPU inference server.
  * Trainer on 1–N GPUs with gradient accumulation; DDP if multi-GPU.
* **Throughput Tricks:**

  * Async RPC for inference; batch by timeout or size.
  * Mixed precision; on-device masks; quantized int8 inference for actors if needed.
  * Replay sampling in separate process with pinned memory.
* **Sharding:**

  * Replay split per day/epoch; compaction jobs remove oldest shards beyond horizon (e.g., last 1–4 weeks).

# Safety, Debuggability, and Quality

* **Sanity Tests:** perft, mate-in-n suites, random playout stress, invariants (policy mass only on legal moves).
* **Overfit Guardrails:** keep a small, fixed evaluation set of held-out positions (no backprop—just score drift over time).
* **Blunder Analysis:** track top-k logits for illegal moves; surprise positions where policy changes sharply; store critical positions for regression tests.
* **Cheat Resistance (optional):** verify no tablebases/books during training if you want “pure” RL.

# Processes & Services (modules)

* **`engine/`**: bitboards, movegen, rules, FEN/PGN.
* **`net/`**: model definitions, checkpoint I/O, inference server.
* **`search/`**: MCTS implementation, UCT stats, transposition tables.
* **`rl/`**: replay buffer, dataset, loss functions, optimizer, trainer.
* **`selfplay/`**: actor loop, opening sampling, resignation logic.
* **`eval/`**: matchmaker, Elo, promotion gate, reports.
* **`infra/`**: orchestrator, configs, logging, metrics, artifact store bindings.
* **`cli/`**: `train`, `selfplay`, `eval`, `serve`, `promote`, `analyze`.

# Extensibility

* **Variants:** Chess960 support via alternative initializers; same network/loop.
* **Search Ablations:** Plug-in interfaces for different exploration constants, rollout policies, or NN heads.
* **Human Play / UI:** UCI/USI adaptor and a minimalist GUI; can play against current best checkpoint.

# Minimum Viable Starting Plan

1. Implement **rules + perft-validated movegen**.
2. Stand up **policy-value net** + **inference**.
3. Wire **MCTS** and **self-play** with a tiny model to prove the loop.
4. Add **replay buffer + trainer**; run short cycles (e.g., 50k positions) and confirm loss/elo trends.
5. Introduce **evaluation gates** and promotion flow.
6. Scale actors and model size; add experiment tracking; iterate.

Awesome—here’s a super-granular, **smallest-possible-delta** roadmap. Every step ends with **What you can see/test** so you’ve always got something to poke.

---

# Phase 0 — Project skeleton & board I/O

1. **Create repo + CLI stub**

* Make `engine/`, `rl/`, `search/`, `net/`, `eval/`, `infra/`, `cli/`.
* Minimal `cli/chess.py --version`.
* **See/test:** `--version` prints `0.0.1`.

2. **Board state container**

* `engine/state.py`: 8×8 array + side-to-move + castling flags + halfmove/fullmove counters.
* **See/test:** `cli/chess.py print-startpos` dumps ASCII board.

3. **FEN parse/emit (startpos only)**

* Parse starting FEN; emit same FEN back.
* **See/test:** `print-fen` → starting FEN; round-trip equals input.

4. **Piece placement parser for any legal FEN (no legality checks)**

* Support arbitrary piece maps and side-to-move.
* **See/test:** Feed random valid FENs; round-trip equals input.

---

# Phase 1 — Move generation (bit-by-bit)

5. **Legal targets: pseudo-moves for pawns only**

* Forward, double push, captures (no en passant), promotions.
* **See/test:** From startpos, white pawns have 16 pseudo-moves.

6. **Add knights**

* L-shaped jumps, board edges respected.
* **See/test:** Knight perft from positions with only knights.

7. **Add bishops + sliding ray attacks (no pins)**

* First sliding rays with blockers.
* **See/test:** Bishop on d4 reports correct diagonals with a few blockers.

8. **Add rooks (sliding)**

* Horizontal/vertical rays.
* **See/test:** Rook mobility test positions.

9. **Add queen (combine bishop+rook)**

* **See/test:** Queen mobility equals union of rook+bishop.

10. **Add king (no castling)**

* One-square neighborhood.
* **See/test:** King mobility from center vs corner.

11. **Check detection + legal filtering (no pins yet)**

* Reject moves that leave own king in check (basic).
* **See/test:** Simple cases where making a capture reveals check.

12. **Pins & discovered checks**

* Compute pinned pieces using attack maps; filter pinned moves.
* **See/test:** Classic pin positions (e.g., rook pinning knight to king).

13. **Castling rights & moves**

* All rules: empty squares, not in check, pass-through squares not attacked.
* **See/test:** Known castle-available FENs; generate correct castle moves.

14. **En passant**

* Track ep-square; ensure EP doesn’t expose your king (discovered-check rule).
* **See/test:** EP scenarios incl. illegal EP due to pin.

15. **Draw rules (3-fold, 50-move)**

* Counters + repetition tracking (position hash map).
* **See/test:** Force 50-move draw; detect triple repetition in a scripted loop.

16. **Perft validation**

* Implement perft; compare against standard perft suites (depth 1–5).
* **See/test:** Perft numbers match reference for at least 10 test FENs.

---

# Phase 2 — Minimal game loop & logs

17. **Environment API**

* `Env.reset(fen) → obs`, `Env.step(move) → (obs, reward, done, info)`.
* Reward only at terminal: win=+1/−1, draw=0.
* **See/test:** Random playout to terminal from startpos; PGN-ish move list printed.

18. **Legality mask**

* `Env.legal_mask()` over a fixed action space index mapping (placeholder 4k–5k).
* **See/test:** Mask matches move list size; illegal indices zeroed.

19. **CLI “play random vs random”**

* Deterministic seed; prints outcome.
* **See/test:** 10 games finish; mix of draws/decisions.

---

# Phase 3 — Network & inference skeleton (no training yet)

20. **Model interface only**

* `Model.predict(batch_states, mask) → (policy_logits, value)`, returns zeros.
* **See/test:** Call returns tensors with correct shapes.

21. **Policy softmax on legal moves**

* Apply mask; renormalize probability mass.
* **See/test:** Sum of π over legal moves = 1.0 for random states.

22. **Batching queue for inference**

* Async queue; micro-batch by timeout or size.
* **See/test:** Self-play loop requests get batched (log batch sizes).

---

# Phase 4 — MCTS wiring (stub to full)

23. **PUCT tree with placeholder priors**

* Use uniform prior from model; backprop Q-values.
* **See/test:** From a midgame FEN, show visit count histogram per move.

24. **Root Dirichlet noise**

* Add α noise; temperature schedule (τ=1 opening, →0 late).
* **See/test:** Opening variety across 50 games; log KL vs. no-noise.

25. **Virtual loss for parallel sims**

* Thread-safe node updates.
* **See/test:** Speed-up vs single-thread; no race conditions (stable visit counts).

26. **MCTS-improved policy output**

* Save π_t (visit distribution) per move to trajectory.
* **See/test:** Inspect stored π_t arrays in a replay JSON/Parquet row.

---

# Phase 5 — Replay & trainer scaffolding

27. **Replay buffer (in-mem ring)**

* Stores (s_t, π_t, r_T, meta).
* **See/test:** Buffer size grows during self-play; can sample batches.

28. **Disk sharding (e.g., Parquet/LMDB)**

* Append-only shards; checksum + schema versioning.
* **See/test:** Files appear per N games; `ls` matches runs.

29. **Loss functions wired (no learning yet)**

* Policy CE with π_t; value MSE to r_T; L2 reg placeholders.
* **See/test:** Compute loss on a random sample; print scalar.

30. **Optimizer & single step**

* AdamW; one gradient step on a tiny batch.
* **See/test:** Loss decreases after a few manual steps on fixed batch.

---

# Phase 6 — Close the self-play → train → evaluate loop (tiny)

31. **Tiny model (toy CNN)**

* 2–4 residual blocks to start.
* **See/test:** `train_one_epoch` runs; training/val losses logged.

32. **Self-play workers (1–2 processes)**

* Use current checkpoint; generate K games, then sleep.
* **See/test:** Trajectories accumulate; ETA per game printed.

33. **Trainer samples from latest data**

* Mixed recent+older ratio (e.g., 0.7/0.3).
* **See/test:** Tensorboard: policy CE and value MSE trending down.

34. **Evaluator: head-to-head 50 games**

* New vs current-best; SPRT or simple CI on win-rate.
* **See/test:** Report `+X Elo (±CI)`; promotion decision printed.

35. **Orchestrator script**

* Round-robin: self-play → train → eval → (optional promote).
* **See/test:** One full cycle completes end-to-end with logs.

---

# Phase 7 — Make it less toy, still tiny deltas

36. **Temperature anneal + resign threshold**

* Resign if running value < −0.9 after move 30.
* **See/test:** Average game length drops; resign-accuracy metric logged (false resigns).

37. **Opening diversity**

* Random ECO seeds or shuffle startpos with a few random legal moves.
* **See/test:** Opening histogram shows spread over top 50 lines.

38. **Prioritized replay (simple)**

* Priority ∝ |v̂−r_T| or low-visit states.
* **See/test:** Sample probability heatmap differs from uniform.

39. **EMA (teacher) weights for eval**

* Maintain EMA of θ for arena matches.
* **See/test:** EMA outperforms raw weights in evaluator.

40. **Checkpoint versioning & artifact registry**

* Save `model.pt`, `optimizer.pt`, `config.json`, `data_fingerprint.txt`.
* **See/test:** `eval/compare --a v12 --b v9` prints match result.

---

# Phase 8 — Scale knobs without changing the recipe

41. **Increase MCTS sims gradually**

* 64 → 128 → 200 sims.
* **See/test:** Elo vs sims plotted; diminishing returns noted.

42. **Increase model depth a bit**

* 4 → 8 residual blocks.
* **See/test:** Throughput vs Elo trade-off graphed; keep within GPU budget.

43. **Batched inference server (separate process)**

* Actors RPC to `net/serve`.
* **See/test:** GPU utilization rises; latency histogram healthy.

44. **Add held-out test positions (no backprop)**

* Fixed 1k positions; track value drift and move quality over time.
* **See/test:** Stability chart; catch regressions.

---

# Phase 9 — Quality, robustness, and analysis

45. **Perft & legality regression suite in CI**

* Run at PR time up to depth 4 on 5 FENs.
* **See/test:** CI green/red with precise counts.

46. **Blunder detector**

* Track illegal-move logit mass and top-k volatility.
* **See/test:** Metric trends toward zero illegal mass.

47. **Draw behavior checks**

* Detect early draw drift (too many draws) and adjust resign/temp.
* **See/test:** Draw rate between 30–55% (tunable); alarms outside band.

48. **Crash-proofing & autosave**

* Resume from last good checkpoint after failures.
* **See/test:** Kill a worker mid-game; orchestrator recovers.

---

# Phase 10 — Usability

49. **UCI adaptor**

* Play engine vs human in a GUI (e.g., Arena) using promoted checkpoint.
* **See/test:** You play a game; engine responds within target time.

50. **One-command demo run**

* `./run_miniloop.sh` spins up 2 actors + trainer + evaluator for 1 hour.
* **See/test:** After 1 hour, a small Elo bump and a promoted model.

---

## Notes on pacing and scope

* Try to ship **1–3 steps per day**; stop after each step and run the “See/test”.
* Don’t scale (sims/model size) until Step 35 loop is stable.
* Keep per-step diffs tiny: one file or one feature at a time.

If you want, tell me your target hardware (CPU/GPU) and I’ll annotate steps 31–44 with concrete batch sizes, sims-per-move, and expected wall-clock timings.

Totally possible—this project *loves* compounding progress. You can start on a Mac mini M4 and scale up in short bursts on cloud GPUs without throwing anything away. Here’s a staged plan with what to run where, rough settings, expected throughput, and how your work carries forward.

---

# Stage 0 — **Mac mini M4 only** (baseline, zero cost)

**Goal:** Prove the loop works and start accumulating skill.

* **What you run**

  * Self-play actors (CPU)
  * Tiny policy-value net training (GPU via Apple Metal / `torch.mps`)
  * Evaluator (CPU)
* **Suggested settings**

  * Net: 4–6 residual blocks, 128 channels
  * MCTS: 64 sims/move (early), 96 later
  * Batch size: 64 (train), FP16/bf16 on `mps`
  * Self-play workers: 4–8 Python processes
  * Replay window: last 150k–300k positions
* **Throughput (ballpark)**

  * **300–800 games/day** (depends on sims & move time)
  * First useful strength in **3–7 days** (basic tactics, fewer blunders)
* **Storage**

  * Keep **checkpoints** (`.pt`), **replay shards** (Parquet/LMDB), **config.json**, and **Elo history** in a `runs/` folder.
* **What “persists”**

  * Everything. Checkpoints and replay shards are portable across machines; you’ll keep training from exactly where you left off.

---

# Stage 1 — **Mac mini + short cloud “bursts”** (Colab/RunPod/Vast), <$50/mo

**Goal:** Use the Mac for self-play all week; rent a GPU for a few hours to train faster.

* **What you run**

  * **Mac mini**: self-play 24/7 producing replay data
  * **Cloud GPU** (2–6 hours per burst): run the trainer on your replay shards, then download the new checkpoint back to the Mac
* **Cheap GPU picks** (whichever is easiest that day)

  * NVIDIA **L4** / **A10G** / **RTX 4090** single-GPU
* **Settings**

  * Net: 10–12 blocks, 160–192 channels
  * MCTS (actors on Mac): 96–128 sims/move
  * Trainer (cloud): batch 256–512, AMP on
* **Throughput**

  * 3–5× faster training during bursts; weekly “promotion” more likely
* **Workflow**

  1. Mac uploads latest **replay shard** to cloud storage (e.g., Drive/S3).
  2. Spin up GPU, **train for N steps**, save **checkpoint vX**.
  3. Pull vX down to the Mac; actors switch to vX; repeat.
* **What “persists”**

  * Same artifacts; checkpoints keep incrementing (v1, v2, …). No retraining from scratch.

---

# Stage 2 — **Single rented GPU most evenings** (hobbyist serious mode), ~$100–200/mo

**Goal:** Daily progress without running a home GPU.

* **What you run**

  * Mac mini: actors all day
  * Cloud: trainer **1–3 hours nightly**, evaluator **weekly 200–400 games** (can also schedule actors on cloud for evaluation bursts)
* **GPU**

  * **RTX 4090** or **A100 40GB** (if affordable)
* **Settings**

  * Net: 16–20 blocks, 192–224 channels
  * Trainer batch: 512–1024
  * Replay window: last 0.5–1.0M positions
* **Throughput**

  * **Thousands of training steps/day**, steady Elo climb to strong club level in weeks/months
* **Extras**

  * Add **EMA (teacher) weights** and **prioritized replay**.
  * Keep nightly **tensorboard logs** in cloud storage.

---

# Stage 3 — **Occasional multi-GPU “sprints”** (once/twice a month), pay-per-sprint

**Goal:** Jump in capability by scaling training briefly.

* **What you run**

  * 2–4× GPUs for **6–24 hours** (DDP), only when you’ve banked lots of fresh self-play
* **GPU**

  * 2–4× A100 40/80GB (or 4090s on a cheaper provider)
* **Settings**

  * Net: 24–32 blocks, 256 channels (if you’ve got the data to feed it)
  * Global batch: 2048–4096, cosine LR schedule
* **Throughput**

  * 10–20× a Stage 0 day; big Elo jumps after each sprint
* **What “persists”**

  * The bigger model becomes your new line; actors will still *use* it on the Mac (inference is cheap compared to training).

---

# Stage 4 — **Add cloud actors when needed** (CPU-heavy), flexible spend

**Goal:** Generate more diverse games when your Mac becomes the bottleneck.

* **What you run**

  * Burst **actor fleets** (cheap CPU instances) for 2–6 hours to fill the replay quickly.
* **Settings**

  * Keep MCTS sims modest (64–96) when scaling actors; volume beats depth for exploration.
* **Throughput**

  * 5–20k games in a day is doable on a small fleet; then train overnight on a single GPU.

---

## Will progress carry over as time & compute grow?

**Yes—by design.** You keep:

* **Checkpoints** (model weights): keep training from the latest.
* **Replay shards**: older data can be sampled less, but it remains useful for stability.
* **Configs & seeds**: makes runs reproducible and comparable.
* **Elo history & evaluation sets**: you can see long-term improvement and catch regressions.

You can pause for weeks, come back, rent a GPU for a few hours, and **continue exactly where you left off**.

---

## Practical tips to make scaling painless

* **Artifact discipline**

  * Version everything: `vNNN/model.pt`, `optimizer.pt`, `config.json`, `data_fingerprint.txt`.
  * Store to Drive/S3/Backblaze; write a tiny `sync_up` / `sync_down` script.
* **Determinism**

  * Log seeds, git commit hash, dataset window (shard ids), and training step.
* **Interchangeable backends**

  * Train on CUDA; run inference on **Metal (`torch.mps`)** on the Mac—same checkpoint format.
* **Throughput knobs (when time is tight)**

  * Lower sims/move (64) and keep temperature >0 longer to increase diversity.
  * Shorten games with a **move cap** during early phases.
* **Budget control**

  * Prefer **single-GPU nightly bursts** (best $/improvement).
  * Reserve multi-GPU sprints for when the replay is saturated with fresh data.

---

## Example weekly cadence (Stage 1–2)

* **Mon–Fri**

  * Mac mini runs **actors 24/7** (64–96 sims, 6-block net)
  * Nightly **trainer burst 90 minutes** on a 4090 (download new vX)
* **Saturday**

  * Extra **cloud actors** for 3 hours → big replay refresh
  * **Trainer 3–4 hours** → vX+1
* **Sunday**

  * **Evaluator 200–400 games** (accept if win-rate >55% with CI)
  * Archive week’s logs; bump network size every ~2–4 weeks

---

## Rough performance landmarks (with the plan above)

* **Week 1–2 (Mac only + small bursts):** Stops hanging pieces, basic mates, low 1400–1600 rapid strength vs casual humans.
* **Month 1–2 (daily 4090 bursts):** Strong club play; recognisable strategy; punishes simple tactics.
* **Month 3+ (occasional multi-GPU sprints):** Expert/master-ish depending on total games & model size.

---


