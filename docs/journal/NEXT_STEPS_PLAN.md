# Chess RL: Next Steps - Strategic Roadmap

**Date:** 2025-11-14
**Current Status:** Phase 2 Complete (MCTS Engine)
**Project Vision:** Build toward AlphaZero-style self-play reinforcement learning

---

## 🎯 Executive Summary: Where We Are

### ✅ What's Been Built (Phases 0-2)

You've successfully built a **progression of increasingly sophisticated chess engines**:

1. **Random Player** - Baseline (Phase 0) ✅
2. **Material Evaluator** - Greedy piece counting ✅
3. **Minimax Engine** (depth-3) - Tactical lookahead with:
   - Alpha-beta pruning
   - Quiescence search (horizon effect fix)
   - Position evaluation (center, development, king safety)
   - **Strength:** ~1200-1400 Elo ✅

4. **MCTS Engine** (200 sims) - Monte Carlo Tree Search with:
   - UCT (Upper Confidence bounds for Trees)
   - Evaluator-guided rollouts (smart simulations)
   - Move sampling optimization (3-5x speedup)
   - Tactical blunder filtering
   - Smart move prioritization (captures/checks/promotions)
   - **Strength:** ~1400-1600 Elo ✅

**Key Achievements:**
- Zero cloud costs (all local development)
- Every phase independently playable
- UCI interface for chess GUIs
- Comprehensive test suites
- ~1400-1600 Elo strength without neural networks

---

## 🎯 The Two Next Logical Steps

Based on your risk-reduction philosophy and the natural progression toward AlphaZero:

### **STEP 1: Neural Network Evaluation (Phase 3a)**
*Supervised learning baseline - Train a policy-value network on existing data*

### **STEP 2: NN-Guided MCTS Integration (Phase 3b)**
*Replace hand-crafted evaluator with neural network in MCTS*

**Why NOT jump straight to self-play RL (Phase 4)?**
1. Self-play requires a working NN-MCTS system first
2. Need baseline NN strength before training it on self-play
3. Validates NN architecture and training pipeline separately
4. Maintains incremental validation approach (your core philosophy)

---

## 📋 STEP 1: Neural Network Evaluation (Phase 3a)

### **Goal**
Build and train a policy-value neural network using **supervised learning** on existing chess games, creating a strong baseline before attempting self-play RL.

### **Duration:** 1-2 weeks
### **Hardware:** Mac mini M4 (with Apple Metal/MPS for training)
### **Target Strength:** Network should match or exceed hand-crafted evaluator (~1200-1400 Elo)

---

### **1.1 Core Components to Build**

#### **A. Neural Network Architecture**
**File:** `net/model.py`

```python
# Policy-Value Network (AlphaZero-style)
#
# Input: 8x8xN board representation
#   - Piece planes (12 planes: 6 piece types x 2 colors)
#   - Repetition counters (2 planes)
#   - Side to move (1 plane)
#   - Castling rights (4 planes)
#   - Move count / 50-move rule (1 plane)
#   Total: ~20 input planes
#
# Architecture:
#   - Shared ResNet backbone (4-8 residual blocks to start)
#   - Policy head → 4672-dim move probabilities
#   - Value head → scalar in [-1, 1] (win/loss/draw)
#
# Start small: 4-6 ResBlocks, 128 channels
# Can scale up after validating the pipeline
```

**Key Design Decisions:**
- **Residual blocks:** Essential for deep networks (gradient flow)
- **Shared trunk:** One backbone for both policy and value (AlphaZero approach)
- **Small initial size:** 4-6 blocks to start (can train on Mac mini)
- **Metal/MPS support:** Leverage Apple Silicon GPU

#### **B. Board State Encoding**
**File:** `net/encoding.py`

```python
# Convert chess.Board → tensor representation
#
# Functions needed:
# - board_to_tensor(board) → (20, 8, 8) tensor
# - move_to_index(move) → int in [0, 4671]
# - index_to_move(idx, board) → chess.Move
# - legal_move_mask(board) → binary mask over 4672 moves
```

**Critical:** Encoding must be consistent and reversible

#### **C. Dataset Pipeline**
**File:** `net/dataset.py`

```python
# Load and preprocess chess games for training
#
# Data Sources (choose one to start):
# 1. Lichess database (millions of rated games)
# 2. Self-play from your MCTS engine (smaller, domain-specific)
# 3. Hybrid: Start with self-play, add Lichess later
#
# What to store per position:
# - Board state (tensor)
# - Move played (policy target)
# - Game outcome (value target: +1/0/-1)
# - Legal move mask
#
# Start with 50k-100k positions from MCTS self-play games
```

#### **D. Training Loop**
**File:** `net/trainer.py`

```python
# Supervised learning training loop
#
# Loss = Policy Loss + Value Loss + L2 Regularization
#   Policy Loss: Cross-entropy(predicted_policy, actual_move)
#   Value Loss: MSE(predicted_value, game_outcome)
#   L2: Weight regularization (prevent overfitting)
#
# Training procedure:
# 1. Load batch of positions
# 2. Forward pass through network
# 3. Compute combined loss
# 4. Backprop and update weights
# 5. Track metrics (accuracy, value MSE, loss)
# 6. Save checkpoints every N batches
#
# Hyperparameters (starting values):
# - Batch size: 64-128 (Mac mini M4 can handle this)
# - Learning rate: 0.001 with cosine decay
# - Optimizer: AdamW
# - Epochs: 10-20 (until convergence)
# - Train/Val split: 90/10
```

---

### **1.2 Implementation Plan**

#### **Week 1: Architecture + Data Pipeline**

**Day 1-2: Network Architecture**
- [ ] Define ResNet blocks in PyTorch
- [ ] Implement policy head (conv → flatten → softmax)
- [ ] Implement value head (conv → pool → dense → tanh)
- [ ] Test forward pass with dummy input
- [ ] Verify output shapes: policy (4672,), value (scalar)

**Validation:**
```python
# Test script
board = chess.Board()
tensor = board_to_tensor(board)  # (20, 8, 8)
policy_logits, value = model(tensor)
assert policy_logits.shape == (4672,)
assert -1 <= value <= 1
print("✅ Architecture works!")
```

**Day 3-4: Board Encoding**
- [ ] Implement `board_to_tensor()` - encode all features
- [ ] Implement `move_to_index()` - UCI → flat index
- [ ] Implement `index_to_move()` - flat index → UCI
- [ ] Create legal move masking
- [ ] Unit tests for 20+ positions

**Validation:**
```python
# Round-trip test
board = chess.Board()
tensor = board_to_tensor(board)
reconstructed = tensor_to_board(tensor)
assert board.fen() == reconstructed.fen()

# Move encoding test
move = chess.Move.from_uci("e2e4")
idx = move_to_index(move)
recovered = index_to_move(idx, board)
assert move == recovered
print("✅ Encoding bijective!")
```

**Day 5-7: Dataset Creation**
- [ ] Generate 100 games of MCTS self-play (200 sims/move)
- [ ] Extract all positions (5000-10000 positions)
- [ ] Store as Parquet/HDF5 files
- [ ] Implement DataLoader with batching
- [ ] Split train/val (90/10)

**Validation:**
```python
# Dataset sanity checks
dataset = ChessDataset("data/mcts_games.parquet")
print(f"Total positions: {len(dataset)}")
print(f"Train: {len(train_set)}, Val: {len(val_set)}")

# Sample batch
batch = next(iter(train_loader))
states, moves, outcomes = batch
assert states.shape == (batch_size, 20, 8, 8)
assert moves.shape == (batch_size,)  # Move indices
assert outcomes.shape == (batch_size,)  # Game results
print("✅ Dataset pipeline works!")
```

#### **Week 2: Training + Evaluation**

**Day 8-10: Training Loop**
- [ ] Implement combined loss function
- [ ] Setup optimizer (AdamW)
- [ ] Add learning rate scheduler
- [ ] Implement training loop with metrics
- [ ] Add checkpoint saving
- [ ] TensorBoard logging

**Validation:**
```python
# Training metrics to track:
# - Policy accuracy (top-1 and top-5)
# - Value MSE
# - Combined loss
# - Validation performance

# After 1 epoch:
print(f"Policy accuracy: {policy_acc:.2%}")
print(f"Value MSE: {value_mse:.4f}")
print(f"Val loss: {val_loss:.4f}")

# Overfitting check:
# Train loss should decrease
# Val loss should decrease (not increase)
print("✅ Network is learning!")
```

**Day 11-12: Evaluation Suite**
- [ ] Implement NN evaluation function for positions
- [ ] Compare NN eval vs hand-crafted eval on test set
- [ ] Measure policy prediction accuracy
- [ ] Test on tactical positions (mate-in-1, tactics)

**Day 13-14: Integration Testing**
- [ ] Create `nn_evaluator.py` wrapper
- [ ] Test NN in minimax search (depth-1)
- [ ] Play NN-minimax vs hand-crafted-minimax
- [ ] Measure strength difference

---

### **1.3 Validation Criteria (Step 1 Complete)**

#### **Must Pass (GO/NO-GO Gates):**

✅ **Gate 1: Architecture Works**
- Network trains without errors
- Loss decreases over 10 epochs
- No NaN/Inf in gradients

✅ **Gate 2: Learns Chess Knowledge**
- Policy accuracy >40% on validation set (top-1)
- Policy accuracy >70% on validation set (top-5)
- Value MSE <0.3 on game outcomes
- Network prefers legal moves over illegal (>95% mass on legal)

✅ **Gate 3: Better than Random**
- NN-guided minimax (depth-1) beats random >90%
- NN-guided minimax finds mate-in-1 >70%

✅ **Gate 4: Competitive with Hand-Crafted**
- NN evaluator correlates with hand-crafted eval (R² >0.6)
- In practice games, NN-minimax (depth-1) draws or wins vs material-only

#### **Success Metrics:**
- **Training time:** <2 hours per epoch on Mac mini M4
- **Model size:** <50MB (small enough for quick iteration)
- **Inference speed:** >100 positions/sec on Mac (fast enough for search)
- **Policy entropy:** 2-4 bits (balanced between confident and exploratory)

#### **Deliverables:**
1. ✅ Trained policy-value network checkpoint
2. ✅ Encoding/decoding utilities
3. ✅ Training scripts and configs
4. ✅ Evaluation metrics dashboard
5. ✅ Documentation: architecture diagram, hyperparameters, results

---

### **1.4 Common Pitfalls & Solutions**

| **Pitfall** | **Symptom** | **Solution** |
|-------------|-------------|-------------|
| **Encoding bugs** | Policy predicts illegal moves | Rigorous unit tests; mask illegal moves in loss |
| **Overfitting** | Val loss increases, train loss decreases | More data; L2 regularization; dropout |
| **Mode collapse** | Network always predicts same move | Check data diversity; reduce LR; add noise |
| **Slow training** | Hours per epoch | Reduce batch size; simplify model; use mixed precision |
| **Poor policy accuracy** | <20% top-1 | More data; longer training; check data quality |
| **Value doesn't learn** | MSE stays >0.5 | Balance wins/draws/losses in data; check encoding |

---

### **1.5 Optional Enhancements (If Time Permits)**

🌟 **Add more training data:**
- Download Lichess database (1M games of 2000+ Elo players)
- Filter to rated games only
- Increases generalization

🌟 **Augment with opening variety:**
- Sample different opening positions
- Prevents network from memorizing e4/d4 only

🌟 **Piece-square table initialization:**
- Initialize value head with hand-crafted knowledge
- Speeds up convergence

🌟 **Attention mechanisms:**
- Add self-attention layers (Transformer-style)
- Can improve tactical vision (future work)

---

## 📋 STEP 2: NN-Guided MCTS Integration (Phase 3b)

### **Goal**
Replace the hand-crafted evaluator in MCTS with the neural network, creating a full **NN-MCTS** system like AlphaZero (but not self-play yet—that's Phase 4).

### **Duration:** 1 week
### **Hardware:** Mac mini M4 (inference only, no training)
### **Target Strength:** NN-MCTS should match or beat hand-crafted MCTS (~1400-1600 Elo)

---

### **2.1 Core Components to Build**

#### **A. NN Inference Server**
**File:** `net/inference.py`

```python
# Batched inference for MCTS
#
# Problem: MCTS needs to evaluate 100s of positions per move
# Solution: Batch requests together for GPU efficiency
#
# Architecture:
# - Async request queue
# - Batch positions every 10ms or when batch size reaches 32
# - Single forward pass for entire batch
# - Return results to individual MCTS threads
#
# Speedup: 10-50x vs sequential inference
```

#### **B. Modified MCTS with NN Evaluation**
**File:** `search/mcts_nn.py`

```python
# Changes to MCTS:
#
# OLD (hand-crafted):
#   Leaf evaluation: evaluate(board) → score
#   Rollout: Play 30 moves with greedy eval
#
# NEW (neural network):
#   Leaf evaluation: nn.predict(board) → (policy, value)
#   Use policy for PUCT prior: P(a) from network
#   Use value for leaf: V(s) from network
#   NO MORE ROLLOUTS (network value replaces simulation)
#
# This is the AlphaZero MCTS algorithm!
```

#### **C. PUCT with Neural Priors**
```python
# UCT formula (old):
#   score = Q(a) + C * sqrt(log(N_parent) / N(a))
#
# PUCT formula (new):
#   score = Q(a) + C * P(a) * sqrt(N_parent) / (1 + N(a))
#
# Where P(a) comes from the neural network policy
# This dramatically improves exploration efficiency
```

---

### **2.2 Implementation Plan**

#### **Day 1-2: Batched Inference**
- [ ] Implement async inference queue
- [ ] Add batching logic (timeout + size threshold)
- [ ] Test throughput: aim for 200+ positions/sec
- [ ] Profile: ensure GPU is saturated, not CPU

**Validation:**
```python
# Throughput test
positions = [random_board() for _ in range(1000)]
start = time.time()
results = batch_inference(positions)
elapsed = time.time() - start
print(f"Throughput: {len(positions)/elapsed:.0f} pos/sec")
# Target: >200 pos/sec on Mac mini M4
```

#### **Day 3-4: NN-MCTS Algorithm**
- [ ] Replace evaluator with NN in leaf nodes
- [ ] Use NN policy for PUCT priors
- [ ] Remove rollout phase (NN value is the estimate)
- [ ] Test on simple positions

**Validation:**
```python
# Correctness test
board = chess.Board()
move_nn = mcts_nn_search(board, simulations=100)
move_eval = mcts_eval_search(board, simulations=100)
print(f"NN-MCTS: {move_nn}")
print(f"Eval-MCTS: {move_eval}")
# Both should be reasonable opening moves (e4, d4, Nf3, etc.)
```

#### **Day 5: Optimization & Tuning**
- [ ] Tune PUCT constant (try 1.0, 1.4, 2.0)
- [ ] Tune number of simulations (100, 200, 400)
- [ ] Add Dirichlet noise at root for exploration
- [ ] Profile search speed

**Validation:**
```python
# Speed test
time_per_move = measure_search_time(simulations=200)
print(f"Time per move: {time_per_move:.2f}s")
# Target: <2 seconds per move at 200 sims
```

#### **Day 6-7: Comprehensive Testing**
- [ ] NN-MCTS vs Random (should win 100%)
- [ ] NN-MCTS vs Material-only (should win >90%)
- [ ] NN-MCTS vs Hand-crafted MCTS (competitive)
- [ ] NN-MCTS vs Minimax depth-3 (competitive)
- [ ] Tactical puzzle suite

---

### **2.3 Validation Criteria (Step 2 Complete)**

#### **Must Pass (GO/NO-GO Gates):**

✅ **Gate 1: NN Integration Works**
- NN-MCTS completes full games without crashes
- Inference batching provides >5x speedup vs sequential
- Search tree statistics look reasonable (visit counts, Q-values)

✅ **Gate 2: Plays Legal, Sensible Chess**
- NN-MCTS makes legal moves 100% of the time
- Opens with principled moves (e4, d4, Nf3, c4, etc.)
- Doesn't hang pieces in opening
- Castles when appropriate

✅ **Gate 3: Beats Weak Baselines**
- NN-MCTS vs Random: >95% win rate
- NN-MCTS vs Material-only: >80% win rate
- NN-MCTS finds mate-in-1: >60% success rate

✅ **Gate 4: Competitive with Phase 2**
- NN-MCTS vs Hand-crafted MCTS (200 sims): 40-60% win rate
  - **Why 40-60%?** NN might be slightly weaker initially (needs more training data), but architecture is correct for future learning
- NN-MCTS tactical understanding: finds 50%+ of tactical puzzles
- Strength estimate: 1300-1500 Elo

#### **Success Metrics:**
- **Search speed:** 100-200 sims/move in <2 seconds
- **Inference throughput:** 200+ positions/sec (batched)
- **Policy agreement:** NN policy top-5 includes MCTS best move >60% of time
- **Value accuracy:** NN value correlates with game outcome (R² >0.5)

#### **Deliverables:**
1. ✅ NN-MCTS implementation (`search/mcts_nn.py`)
2. ✅ Batched inference server (`net/inference.py`)
3. ✅ UCI mode for NN-MCTS (playable in chess GUIs)
4. ✅ Comparison report: NN-MCTS vs all previous engines
5. ✅ Performance benchmarks (speed, strength, tactics)

---

### **2.4 Common Pitfalls & Solutions**

| **Pitfall** | **Symptom** | **Solution** |
|-------------|-------------|-------------|
| **Inference too slow** | >5 seconds per move | Batch more aggressively; reduce network size; use FP16 |
| **NN policy overconfident** | Exploration collapses | Add Dirichlet noise; tune temperature; increase PUCT constant |
| **Search doesn't converge** | Visit counts evenly spread | Check NN policy quality; ensure value is informative |
| **Weaker than hand-crafted MCTS** | <40% win rate | More training data; longer training; check for bugs |
| **Memory issues** | OOM during batch inference | Reduce batch size; use gradient checkpointing |

---

## 📊 Evaluation Framework for Both Steps

### **Continuous Evaluation Metrics**

#### **1. Engine Strength Ladder**
Test each new system against ALL previous engines:

| Opponent | Expected Win Rate | What it Tests |
|----------|------------------|---------------|
| Random | >99% | Basic functionality |
| Material-only | >90% | Tactical awareness |
| Minimax depth-3 | >60% | Positional understanding |
| Hand-crafted MCTS | 45-55% | Search quality |
| **Self-comparison** | Baseline | Elo tracking over time |

Run **20-game matches** for each pairing (statistical significance).

#### **2. Tactical Test Suite**
Curated positions testing specific skills:

- **Mate-in-1** (10 positions): Should solve >70%
- **Mate-in-2** (10 positions): Should solve >40%
- **Tactics** (20 positions): Forks, pins, skewers >60%
- **Endgames** (10 positions): Basic checkmates >80%

Track improvement over time.

#### **3. Self-Play Quality Metrics**
Once NN-MCTS is working, generate self-play games and measure:

- **Game length:** 40-80 moves (realistic)
- **Decisive rate:** 50-70% decisive, 30-50% draws
- **Blunder rate:** <5% moves hang material
- **Opening diversity:** 20+ unique openings in 100 games
- **Position complexity:** Non-trivial middle games (not just tactical shootouts)

#### **4. Training Diagnostics**
Monitor neural network health:

- **Policy accuracy:** Should increase with more data
- **Value MSE:** Should decrease with better predictions
- **Illegal move probability:** Should stay <1%
- **Generalization gap:** Val loss should track train loss

---

## 🚨 Decision Points & Contingencies

### **Decision Point 1: After Step 1 (NN Training)**

**IF Policy Accuracy <30%:**
- ❌ **STOP** - Data quality issue
- 🔧 **Fix:** Regenerate dataset; verify encoding; check for bugs
- 🔧 **Alternative:** Use Lichess database instead of self-play

**IF Value MSE >0.5:**
- ❌ **STOP** - Network not learning game outcomes
- 🔧 **Fix:** Balance win/draw/loss ratio; check outcome encoding
- 🔧 **Alternative:** Train policy-only network first (simpler)

**IF Training too slow (>5 hours per epoch):**
- ⚠️ **CONSIDER** - Simplify model (fewer blocks, smaller channels)
- 🔧 **Fix:** Use mixed precision (FP16); reduce batch size
- 🔧 **Alternative:** Cloud GPU for training bursts ($5-10 for full training)

### **Decision Point 2: After Step 2 (NN-MCTS)**

**IF NN-MCTS weaker than hand-crafted MCTS (<30% win rate):**
- ❌ **STOP** - NN not strong enough yet
- 🔧 **Fix:** More training data (100k → 500k positions)
- 🔧 **Fix:** Longer training (20 → 50 epochs)
- 🔧 **Fix:** Tune PUCT constant, temperature, noise

**IF NN-MCTS too slow (>5 sec per move):**
- ⚠️ **ADJUST** - Reduce simulations (200 → 100)
- 🔧 **Fix:** Optimize batching; use model distillation (smaller network)
- 🔧 **Alternative:** Use NN for leaf eval only, keep rollouts (hybrid)

**IF NN-MCTS plays nonsensical chess:**
- ❌ **STOP** - Bug in integration
- 🔧 **Fix:** Check value perspective (negamax); verify policy masking
- 🔧 **Fix:** Validate on simple positions (starting position should give e4/d4/Nf3)

---

## 🎯 Success Criteria Summary

### **Step 1 Success = Neural Network Baseline**
- ✅ Trained policy-value network
- ✅ Policy accuracy >40% (top-1), >70% (top-5)
- ✅ Value MSE <0.3
- ✅ NN-minimax competitive with material-only
- ✅ Clean, reproducible training pipeline

### **Step 2 Success = NN-MCTS Engine**
- ✅ NN-MCTS plays complete games
- ✅ Competitive with hand-crafted MCTS (45-55% win rate)
- ✅ Strength: 1300-1500 Elo
- ✅ Ready for self-play training (Phase 4)
- ✅ Fast enough for iterative development

### **Combined Success = Ready for Phase 4 (Self-Play RL)**
- ✅ Working NN-MCTS system (the "actor" for self-play)
- ✅ Training pipeline validated (can improve network over time)
- ✅ Evaluation framework in place (can measure Elo growth)
- ✅ All previous engines still playable (baselines for comparison)

---

## 📈 What Comes After (Preview of Phase 4)

Once Steps 1 & 2 are complete, you'll be ready for **self-play reinforcement learning**:

### **Phase 4: Self-Play RL Loop**
1. **Generate:** NN-MCTS plays 1000 games against itself
2. **Learn:** Train network on self-play data (policy + value targets from MCTS)
3. **Evaluate:** New network plays 100 games vs old network
4. **Promote:** If new network wins >55%, it becomes the new "best"
5. **Repeat:** Loop forever, network gets progressively stronger

**Expected Outcome:**
- First generation: ~1400 Elo (starting from supervised baseline)
- After 10 generations: ~1600 Elo
- After 50 generations: ~1800+ Elo
- After 100+ generations: Master level (if you have compute)

**Why this works:**
- Network learns from its own mistakes
- MCTS provides high-quality training signal (better than raw game outcomes)
- Self-play creates increasingly challenging opponents
- AlphaZero reached superhuman strength this way (with more compute)

---

## 🛠️ Practical Tips for Implementation

### **Development Workflow**
1. **Start small:** 4-block network, 10k positions, 5 epochs
2. **Validate continuously:** Test after each component
3. **Save everything:** Checkpoints, configs, metrics, logs
4. **Version control:** Git commit after each working feature
5. **Document:** Write down hyperparameters, design decisions, results

### **Debugging Strategy**
1. **Unit test everything:** Encoding, decoding, losses, search
2. **Visualize outputs:** Print sample policies, values, search trees
3. **Overfit intentionally:** Train on 100 positions to verify network CAN learn
4. **Compare to baseline:** NN predictions vs hand-crafted eval
5. **Isolate components:** Test NN separately from MCTS

### **Resource Management (Mac mini M4)**
- **Training:** 2-4 hours for full pipeline (10-20 epochs)
- **Data generation:** 2-3 hours for 100 games of MCTS self-play
- **Evaluation:** 1 hour for full test suite (20 games × 5 opponents)
- **Total time:** 1-2 weeks for both steps (part-time work)

### **When to Consider Cloud Compute**
- **NOT YET** - Mac mini M4 can handle Steps 1 & 2
- **Phase 4 (self-play):** Consider cloud for:
  - Training bursts (nightly GPU for 1-3 hours)
  - Large-scale self-play (1000s of games/day)
  - Scaling up network (20-40 blocks)
- **Cost estimate:** $50-200/month for serious self-play

---

## 🎓 Learning Resources

### **Neural Networks for Chess**
- AlphaZero paper: *Mastering Chess and Shogi by Self-Play with a General RL Algorithm*
- Leela Chess Zero (open-source AlphaZero): https://lczero.org/
- PyTorch ResNet tutorial
- Policy-value network architecture explanations

### **MCTS with Neural Networks**
- PUCT algorithm explanation
- AlphaGo/AlphaZero MCTS modifications
- Dirichlet noise for exploration
- Temperature-based move selection

### **Practical Chess RL**
- Lichess database: https://database.lichess.org/
- Chess programming wiki: https://www.chessprogramming.org/
- Training tips: Data augmentation, balancing, filtering

---

## ✅ Final Checklist: Ready to Start Step 1?

Before beginning, ensure you have:

- [x] ✅ Working MCTS engine (Phase 2 complete)
- [x] ✅ Python environment with PyTorch
- [x] ✅ Mac mini M4 with Metal/MPS support
- [x] ✅ ~50GB disk space (for datasets and checkpoints)
- [x] ✅ Cleared schedule for 1-2 weeks focused work
- [x] ✅ Understanding of the plan and validation criteria
- [x] ✅ Commitment to incremental validation (don't skip gates!)

---

## 🎯 TL;DR: The Next Two Steps

### **STEP 1: Neural Network Baseline (1-2 weeks)**
Build and train a policy-value network on supervised data (MCTS self-play games).
**Goal:** Network matches hand-crafted evaluator strength.
**Validation:** Policy accuracy >40%, Value MSE <0.3, NN-minimax competitive.

### **STEP 2: NN-MCTS Integration (1 week)**
Replace hand-crafted evaluator with neural network in MCTS.
**Goal:** AlphaZero-style NN-MCTS system.
**Validation:** Competitive with hand-crafted MCTS, ready for self-play.

### **Then:** Phase 4 (Self-Play RL) - The AlphaZero loop! 🚀

---

**Next Action:** Start with Step 1, Day 1-2 → Build the neural network architecture.

Let me know when you're ready to dive into the code!
