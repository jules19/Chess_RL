# Chess_RL
Incremental development of a Chess program using Reinforcement Learning

---

## **🚀 Getting Started (Start Here!)**

**New to the project? Start with baby steps:**

1. **Quick Start** (5 minutes): See [`QUICKSTART.md`](QUICKSTART.md) to play your first chess game
2. **Risk Reduction Strategy**: Read [`RISK_REDUCTION.md`](RISK_REDUCTION.md) to understand the incremental approach
3. **Development Plan**: Check [`PLAN.md`](PLAN.md) for the full roadmap

**Current Status:** ✅ Phases 0-2 Complete! 🚧 Phase 3a (Neural Network Training) in Planning

**What's Working:**
- ✅ Phase 0 (Day 1): Random move engine (~600 Elo)
- ✅ Phase 1 (Days 2-7): Minimax engine (~1200-1600 Elo)
  - Material evaluation (P=1, N=3, B=3, R=5, Q=9)
  - Alpha-beta pruning with quiescence search
  - Positional evaluation (center, development, king safety, pawns)
  - UCI interface (works with Cute Chess, Arena, PyChess)
  - UCI logging and PGN export
- ✅ Phase 2 (Days 8-14): MCTS engine (~1400-1600 Elo)
  - Monte Carlo Tree Search with UCT
  - Evaluator-guided rollouts
  - Tactical blunder filtering
  - Smart move prioritization
- ✅ Phase 3a Planning (Day 8): Neural Network Ready
  - ResNet architecture designed (4 blocks, 128 channels, ~830K params)
  - Training pipeline implemented (`training/train.py`, `training/dataset.py`)
  - Evaluation framework ready (`training/evaluate.py`)
  - Comprehensive documentation (`PHASE_3A_PLAN.md`, `DAY_8_SUMMARY.md`)

**Quick Start:**
- **Play in terminal**: `python3 cli/play.py random` (or `minimax`, `mcts`)
- **Play in GUI**: See [`UCI_SETUP_MAC.md`](UCI_SETUP_MAC.md) for Mac setup
- **Test UCI**: `python3 chess_rl_uci.py`
- **UCI Logging**: `python3 uci/engine.py --uci-log debug.log --pgn-log games.pgn`
- **Training (pending)**: `python3 training/train.py --pgn data/games.pgn --epochs 10`

**Next Steps:**
- Phase 3a: Train neural network on expert games (~1400-1600 Elo)
- Phase 3b: Integrate NN with MCTS (~1700-1800 Elo)
- Phase 4: Self-play reinforcement learning (1800+ Elo)

**Why this approach?** Build a progression of playable chess engines, each independently useful, while learning and validating as you go. No cloud costs for the first month.

---

## **Long-Term Vision**

Here's the **plain-English vision** for the final chess program, how it works, and the kind of computer you'd need to run it.

---

## **Vision**

We’re building a chess program that **starts out knowing almost nothing** about good play, then improves by **playing millions of games against itself**, learning from mistakes, and gradually becoming stronger.
It’s not given opening books or endgame tables—it discovers strategies by trial, error, and reinforcement. Over time, it will:

* Learn basic checkmates,
* Discover tactics like forks and pins,
* Develop positional understanding,
* Eventually play at strong club or master level depending on how much compute you feed it.

Think of it like teaching a child chess—but instead of you showing moves, it experiments, remembers what worked, and slowly polishes its game.

---

## **How it Works (big picture)**

1. **Game Loop**

   * The engine plays games against itself (self-play).
   * Each move is chosen using **Monte Carlo Tree Search (MCTS)**, which explores possible future moves and uses the neural network’s guidance to prune bad branches.

2. **Learning from Games**

   * Every move is recorded with:

     * The board position,
     * The move probabilities from MCTS,
     * The final result of the game.
   * These game records go into a **replay buffer** (a giant history of recent games).

3. **Training the Brain**

   * A **neural network** takes board positions and learns to:

     * Predict which moves are likely to be good (policy),
     * Predict who will win from that position (value).
   * The network is trained on the replay buffer data—learning to imitate MCTS and match actual outcomes.

4. **Evaluation & Promotion**

   * The new version of the network plays matches against the current “best” version.
   * If it consistently wins more, it becomes the new champion model.
   * This cycle repeats endlessly—self-play → training → evaluation → promotion.

5. **Over Time**

   * Early games are random and messy.
   * Gradually, openings stabilize, blunders drop, and strategic play emerges.
   * With enough games, it develops strong, creative play—completely self-taught.

---

## **Compute Resources Needed**

### **Small-Scale Prototype (good for testing the loop)**

* **CPU:** Quad-core (for basic move generation + small MCTS)
* **GPU:** Entry-level CUDA GPU (e.g., NVIDIA GTX 1650, RTX 3050) for small neural nets
* **RAM:** 8–16 GB
* **Performance:**

  * Can run a tiny network (e.g., 4 residual blocks)
  * Maybe hundreds of self-play games per day
  * Will learn basic tactics in days/weeks

### **Medium Setup (serious hobbyist)**

* **CPU:** 8–16 cores for parallel self-play
* **GPU:** RTX 3080/4080 or similar
* **RAM:** 32 GB+
* **Performance:**

  * Larger networks (20–40 blocks)
  * Thousands of self-play games per day
  * Strong club-level play within weeks/months

### **Large Setup (research lab style)**

* **CPU Cluster:** 100s of cores across many nodes
* **GPUs:** Multiple A100s / H100s or similar high-end cards
* **RAM:** 128 GB+
* **Performance:**

  * Billions of positions trained
  * Grandmaster-level play in weeks/months

