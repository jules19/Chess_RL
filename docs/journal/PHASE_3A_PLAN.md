# Phase 3a: Supervised Learning - Implementation Plan

**Status:** Ready to implement (dependencies pending installation)
**Timeline:** 1-2 weeks
**Goal:** Train a neural network on expert chess games to match/exceed minimax baseline (~1400-1600 Elo)

---

## Overview

Phase 3a trains our policy-value network using **supervised learning** on high-quality chess games. This provides:
1. A strong baseline before self-play RL
2. Validation that the NN architecture works
3. Experience with the training pipeline
4. A checkpoint to compare against later self-play improvements

---

## Architecture Review ✅

**Network:** `net/model.py` - PolicyValueNetwork
**Encoding:** `net/encoding.py` - board_to_tensor, move encoding
**Parameters:** ~830K (4 ResBlocks, 128 channels)
**Model Size:** ~3.3MB (FP32)

**Status:** Architecture complete and tested. Ready for training.

---

## Baby Steps Implementation Plan

### Step 1: Environment Setup (Day 1)

**Goal:** Get PyTorch and dependencies installed

```bash
# Option A: CPU-only PyTorch (fast, no GPU needed)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Option B: CUDA PyTorch (for GPU training, larger download)
pip3 install torch torchvision

# Other dependencies
pip3 install python-chess numpy tqdm tensorboard
```

**Validation:**
```python
import torch
import chess
from net.model import create_model
from net.encoding import board_to_tensor

# Test model creation
model = create_model(num_res_blocks=4, num_channels=128)
model.summary()

# Test encoding
board = chess.Board()
tensor = board_to_tensor(board)
print(f"Board tensor shape: {tensor.shape}")  # Should be (20, 8, 8)
```

**Decision Gate:** Only proceed if model creation and encoding work correctly.

---

### Step 2: Data Collection (Day 1-2)

**Goal:** Download and prepare training data from Lichess

#### Option A: Lichess Database (Recommended)

Download high-quality games (2000+ Elo):
- Source: https://database.lichess.org/
- Download: Standard rated games (PGN format)
- Filter: Elo 2000+, time control ≥10+0 (avoid bullet games)
- Size: Start small (10K-100K games), scale up later

```bash
# Download January 2024 games (example)
wget https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst

# Decompress
unzstd lichess_db_standard_rated_2024-01.pgn.zst

# Filter high-Elo games (using python-chess)
python3 training/filter_games.py \
  --input lichess_db_standard_rated_2024-01.pgn \
  --output data/filtered_games.pgn \
  --min-elo 2000 \
  --max-moves 80
```

#### Option B: Use Pre-Processed Dataset

If bandwidth is limited, use smaller curated datasets:
- CCRL games
- Top GM games
- FICS database

**Expected Output:**
- `data/filtered_games.pgn` - Cleaned PGN file
- 10K-100K high-quality games
- ~50-100MB file size

**Validation:**
```bash
# Count games
grep -c "Event" data/filtered_games.pgn

# Sample a few games
head -100 data/filtered_games.pgn
```

**Decision Gate:** Only proceed if you have ≥10K quality games.

---

### Step 3: Dataset Creation (Day 2-3)

**Goal:** Convert PGN games → training examples (board states + targets)

Create `training/dataset.py`:

```python
"""
Chess Dataset for Supervised Learning

Converts PGN games into training examples:
- Input: Board state (20, 8, 8) tensor
- Target Policy: Move played (one-hot 4672-dim)
- Target Value: Game outcome (+1 White win, 0 draw, -1 Black win)
"""

import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from net.encoding import board_to_tensor, move_to_index

class ChessDataset(Dataset):
    """
    Dataset of (board, policy, value) tuples from PGN games.

    Each example:
    - board: (20, 8, 8) float32 tensor
    - policy: int (move index 0-4671)
    - value: float in {-1, 0, +1}
    """

    def __init__(self, pgn_path, max_games=None):
        """
        Args:
            pgn_path: Path to PGN file
            max_games: Maximum games to load (None = all)
        """
        self.examples = []
        self.load_games(pgn_path, max_games)

    def load_games(self, pgn_path, max_games):
        """Parse PGN and extract training examples."""
        print(f"Loading games from {pgn_path}...")

        with open(pgn_path) as pgn_file:
            game_count = 0

            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                # Extract game result
                result = game.headers.get("Result", "*")
                if result == "1-0":
                    value = 1.0  # White wins
                elif result == "0-1":
                    value = -1.0  # Black wins
                elif result == "1/2-1/2":
                    value = 0.0  # Draw
                else:
                    continue  # Skip unfinished games

                # Extract positions and moves
                board = game.board()
                for move in game.mainline_moves():
                    # Encode current position
                    board_tensor = board_to_tensor(board)

                    # Encode move played (policy target)
                    move_index = move_to_index(move)

                    # Value from current player's perspective
                    game_value = value if board.turn == chess.WHITE else -value

                    self.examples.append({
                        'board': board_tensor,
                        'policy': move_index,
                        'value': game_value
                    })

                    board.push(move)

                game_count += 1
                if game_count % 100 == 0:
                    print(f"Loaded {game_count} games, {len(self.examples)} positions")

                if max_games and game_count >= max_games:
                    break

        print(f"Dataset created: {len(self.examples)} training examples from {game_count} games")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return (
            torch.from_numpy(example['board']),
            torch.tensor(example['policy'], dtype=torch.long),
            torch.tensor(example['value'], dtype=torch.float32)
        )


def create_dataloaders(pgn_path, batch_size=256, train_split=0.9, max_games=None):
    """
    Create train and validation dataloaders.

    Args:
        pgn_path: Path to PGN file
        batch_size: Batch size for training
        train_split: Fraction for training (rest for validation)
        max_games: Maximum games to load

    Returns:
        train_loader, val_loader
    """
    # Load full dataset
    dataset = ChessDataset(pgn_path, max_games=max_games)

    # Split into train/val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Train set: {train_size} examples")
    print(f"Val set: {val_size} examples")

    return train_loader, val_loader


if __name__ == "__main__":
    """Test dataset creation."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 training/dataset.py <pgn_path>")
        sys.exit(1)

    pgn_path = sys.argv[1]

    # Test with small dataset
    print("Testing dataset creation...")
    dataset = ChessDataset(pgn_path, max_games=10)

    print(f"\nDataset size: {len(dataset)}")

    # Check first example
    board, policy, value = dataset[0]
    print(f"\nExample 0:")
    print(f"  Board shape: {board.shape}")
    print(f"  Policy target: {policy} (move index)")
    print(f"  Value target: {value} (game outcome)")

    # Test dataloader
    print("\nTesting dataloader...")
    train_loader, val_loader = create_dataloaders(
        pgn_path, batch_size=32, max_games=100
    )

    # Check one batch
    boards, policies, values = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Boards: {boards.shape}")
    print(f"  Policies: {policies.shape}")
    print(f"  Values: {values.shape}")

    print("\n✅ Dataset creation successful!")
```

**Usage:**
```bash
# Test dataset creation
python3 training/dataset.py data/filtered_games.pgn

# Expected output:
# - Dataset size: ~40-80 positions per game
# - Batch shapes: boards (32, 20, 8, 8), policies (32,), values (32,)
```

**Decision Gate:** Dataset loads correctly, batches have right shapes.

---

### Step 4: Training Loop (Day 3-5)

**Goal:** Implement supervised learning training loop

Create `training/train.py`:

```python
"""
Supervised Learning Training Loop

Train policy-value network on expert games using:
- Policy loss: Cross-entropy (predicted moves vs actual moves)
- Value loss: MSE (predicted outcome vs actual outcome)
- Combined loss: policy_loss + value_loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import argparse
from datetime import datetime

from net.model import create_model
from training.dataset import create_dataloaders


class Trainer:
    """Supervised learning trainer for chess neural network."""

    def __init__(self, model, device, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device

        # Loss functions
        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.5
        )

        # Tensorboard
        log_dir = f"runs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir)

        print(f"Logging to {log_dir}")

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, (boards, target_policies, target_values) in enumerate(pbar):
            # Move to device
            boards = boards.to(self.device)
            target_policies = target_policies.to(self.device)
            target_values = target_values.to(self.device).unsqueeze(1)

            # Forward pass
            policy_logits, pred_values = self.model(boards)

            # Compute losses
            policy_loss = self.policy_criterion(policy_logits, target_policies)
            value_loss = self.value_criterion(pred_values, target_values)
            loss = policy_loss + value_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'policy': f"{policy_loss.item():.4f}",
                'value': f"{value_loss.item():.4f}"
            })

        # Average losses
        avg_loss = total_loss / len(train_loader)
        avg_policy_loss = total_policy_loss / len(train_loader)
        avg_value_loss = total_value_loss / len(train_loader)

        return avg_loss, avg_policy_loss, avg_value_loss

    def validate(self, val_loader, epoch):
        """Validate on validation set."""
        self.model.eval()

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        correct_moves = 0
        total_moves = 0

        with torch.no_grad():
            for boards, target_policies, target_values in tqdm(val_loader, desc="Validation"):
                boards = boards.to(self.device)
                target_policies = target_policies.to(self.device)
                target_values = target_values.to(self.device).unsqueeze(1)

                # Forward pass
                policy_logits, pred_values = self.model(boards)

                # Compute losses
                policy_loss = self.policy_criterion(policy_logits, target_policies)
                value_loss = self.value_criterion(pred_values, target_values)
                loss = policy_loss + value_loss

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

                # Policy accuracy (top-1)
                pred_moves = policy_logits.argmax(dim=1)
                correct_moves += (pred_moves == target_policies).sum().item()
                total_moves += target_policies.size(0)

        # Average metrics
        avg_loss = total_loss / len(val_loader)
        avg_policy_loss = total_policy_loss / len(val_loader)
        avg_value_loss = total_value_loss / len(val_loader)
        policy_accuracy = correct_moves / total_moves

        return avg_loss, avg_policy_loss, avg_value_loss, policy_accuracy

    def train(self, train_loader, val_loader, num_epochs, checkpoint_dir="checkpoints"):
        """Full training loop."""
        os.makedirs(checkpoint_dir, exist_ok=True)

        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")

            # Train
            train_loss, train_policy_loss, train_value_loss = self.train_epoch(
                train_loader, epoch
            )

            # Validate
            val_loss, val_policy_loss, val_value_loss, val_accuracy = self.validate(
                val_loader, epoch
            )

            # Learning rate step
            self.scheduler.step()

            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Policy Loss/train', train_policy_loss, epoch)
            self.writer.add_scalar('Policy Loss/val', val_policy_loss, epoch)
            self.writer.add_scalar('Value Loss/train', train_value_loss, epoch)
            self.writer.add_scalar('Value Loss/val', val_value_loss, epoch)
            self.writer.add_scalar('Policy Accuracy/val', val_accuracy, epoch)
            self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Print summary
            print(f"\nTrain Loss: {train_loss:.4f} (policy: {train_policy_loss:.4f}, value: {train_value_loss:.4f})")
            print(f"Val Loss: {val_loss:.4f} (policy: {val_policy_loss:.4f}, value: {val_value_loss:.4f})")
            print(f"Val Policy Accuracy: {val_accuracy:.2%}")

            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save(self.model.state_dict(), best_path)
                print(f"✓ New best model! (val_loss: {val_loss:.4f})")

        self.writer.close()
        print(f"\n✅ Training complete! Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train chess neural network")
    parser.add_argument('--pgn', type=str, required=True, help='Path to PGN file')
    parser.add_argument('--max-games', type=int, default=None, help='Max games to load')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--res-blocks', type=int, default=4, help='Number of residual blocks')
    parser.add_argument('--channels', type=int, default=128, help='Number of channels')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    print(f"\nCreating model ({args.res_blocks} ResBlocks, {args.channels} channels)...")
    model = create_model(num_res_blocks=args.res_blocks, num_channels=args.channels)
    model.summary()

    # Create dataloaders
    print(f"\nLoading data from {args.pgn}...")
    train_loader, val_loader = create_dataloaders(
        args.pgn,
        batch_size=args.batch_size,
        max_games=args.max_games
    )

    # Create trainer
    trainer = Trainer(model, device, learning_rate=args.lr)

    # Train
    trainer.train(train_loader, val_loader, args.epochs, args.checkpoint_dir)


if __name__ == "__main__":
    main()
```

**Usage:**
```bash
# Start training (small test run)
python3 training/train.py \
  --pgn data/filtered_games.pgn \
  --max-games 1000 \
  --batch-size 256 \
  --epochs 10 \
  --lr 0.001

# Monitor with tensorboard
tensorboard --logdir runs/
```

**Expected Metrics:**
- **Epoch 1:** Val loss ~6-7, Policy accuracy ~10-15%
- **Epoch 5:** Val loss ~3-4, Policy accuracy ~25-35%
- **Epoch 10:** Val loss ~2-3, Policy accuracy ~35-45%

**Decision Gate:** Training converges, validation loss decreases, policy accuracy improves.

---

### Step 5: Model Evaluation (Day 5-6)

**Goal:** Test trained model strength against existing engines

Create `training/evaluate.py`:

```python
"""
Evaluate trained neural network against baseline engines.

Tests:
1. Policy accuracy on held-out test set
2. Value prediction accuracy
3. Engine vs Engine matches (NN vs minimax/MCTS)
4. Tactical puzzle solving
"""

import torch
import chess
import chess.engine
from tqdm import tqdm

from net.model import create_model
from net.encoding import board_to_tensor, legal_moves_mask, index_to_move


class NeuralNetworkPlayer:
    """Chess player using trained neural network."""

    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)

        # Load model
        self.model = create_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded model from {model_path}")

    def select_move(self, board):
        """Select best move using neural network."""
        with torch.no_grad():
            # Encode board
            board_tensor = board_to_tensor(board)
            board_tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(self.device)

            # Get legal moves mask
            legal_mask = legal_moves_mask(board)
            legal_mask = torch.from_numpy(legal_mask).unsqueeze(0).to(self.device)

            # Forward pass
            policy_probs, value = self.model.get_policy_value(board_tensor, legal_mask)

            # Select move with highest probability
            policy_probs = policy_probs.squeeze(0).cpu().numpy()

            # Convert top move index to chess.Move
            for move_idx in policy_probs.argsort()[::-1]:
                move = index_to_move(move_idx, board)
                if move and move in board.legal_moves:
                    return move, value.item()

        # Fallback: random legal move
        return list(board.legal_moves)[0], 0.0


def play_game(white_player, black_player, max_moves=100, verbose=False):
    """
    Play a game between two players.

    Returns:
        result: "1-0", "0-1", "1/2-1/2", or "*" (unfinished)
    """
    board = chess.Board()
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        # Select move
        if board.turn == chess.WHITE:
            move, value = white_player.select_move(board)
        else:
            move, value = black_player.select_move(board)

        if verbose:
            print(f"{move_count+1}. {move} (value: {value:.3f})")

        board.push(move)
        move_count += 1

    # Determine result
    if board.is_checkmate():
        result = "1-0" if board.turn == chess.BLACK else "0-1"
    elif board.is_stalemate() or board.is_insufficient_material():
        result = "1/2-1/2"
    elif move_count >= max_moves:
        result = "1/2-1/2"  # Draw by move limit
    else:
        result = "*"

    return result, move_count


def tournament(player1, player2, num_games=10):
    """
    Run a tournament between two players.

    Returns:
        (wins, losses, draws) from player1's perspective
    """
    wins = 0
    losses = 0
    draws = 0

    for i in tqdm(range(num_games), desc="Tournament"):
        # Alternate colors
        if i % 2 == 0:
            result, moves = play_game(player1, player2)
            if result == "1-0":
                wins += 1
            elif result == "0-1":
                losses += 1
            else:
                draws += 1
        else:
            result, moves = play_game(player2, player1)
            if result == "1-0":
                losses += 1
            elif result == "0-1":
                wins += 1
            else:
                draws += 1

    return wins, losses, draws


if __name__ == "__main__":
    import sys
    from search.minimax import MinimaxPlayer
    from search.mcts import MCTSPlayer

    if len(sys.argv) < 2:
        print("Usage: python3 training/evaluate.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]

    # Load NN player
    nn_player = NeuralNetworkPlayer(model_path)

    # Test vs Minimax (depth 3)
    print("\n" + "="*60)
    print("Tournament: Neural Network vs Minimax (depth 3)")
    print("="*60)

    minimax_player = MinimaxPlayer(depth=3)
    wins, losses, draws = tournament(nn_player, minimax_player, num_games=20)

    print(f"\nResults:")
    print(f"  NN wins: {wins}")
    print(f"  NN losses: {losses}")
    print(f"  Draws: {draws}")
    print(f"  Win rate: {wins / (wins + losses + draws):.1%}")

    # Test vs MCTS (200 simulations)
    print("\n" + "="*60)
    print("Tournament: Neural Network vs MCTS (200 sims)")
    print("="*60)

    mcts_player = MCTSPlayer(num_simulations=200)
    wins, losses, draws = tournament(nn_player, mcts_player, num_games=20)

    print(f"\nResults:")
    print(f"  NN wins: {wins}")
    print(f"  NN losses: {losses}")
    print(f"  Draws: {draws}")
    print(f"  Win rate: {wins / (wins + losses + draws):.1%}")

    print("\n✅ Evaluation complete!")
```

**Usage:**
```bash
# Evaluate best model
python3 training/evaluate.py checkpoints/best_model.pt
```

**Success Criteria:**
- **vs Minimax (depth 3):** Win rate ≥40% (competitive)
- **vs MCTS (200 sims):** Win rate ≥35% (reasonable)
- **Policy accuracy:** ≥35% on test set

**Decision Gate:** If NN is competitive (~1400-1600 Elo), proceed to Phase 3b. Otherwise, train longer or adjust hyperparameters.

---

## Expected Timeline

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Environment setup + data collection | PyTorch installed, 10K+ games downloaded |
| 2 | Dataset creation | Training pipeline working |
| 3-4 | Training (first run) | Model checkpoints, tensorboard logs |
| 5 | Evaluation | NN vs engines results |
| 6 | Hyperparameter tuning | Improved model |
| 7 | Documentation | DAY_8_SUMMARY.md, updated README |

---

## Success Metrics

**Training Metrics:**
- ✅ Validation loss < 2.5 after 10 epochs
- ✅ Policy accuracy > 35% on validation set
- ✅ Value prediction error < 0.3

**Playing Strength:**
- ✅ Beats Random 100% of the time
- ✅ Beats Material (~1000 Elo) ≥90%
- ✅ Competitive with Minimax depth 3 (~1400 Elo) ≥40% win rate

---

## Risk Mitigation

**Risk 1: Overfitting**
- Symptom: Train loss << val loss
- Solution: Add dropout, reduce model size, or get more data

**Risk 2: Slow convergence**
- Symptom: Loss plateaus early
- Solution: Adjust learning rate, try different optimizer (SGD, AdamW)

**Risk 3: Poor playing strength**
- Symptom: NN loses to weak engines
- Solution: Train longer, use higher-quality games (2200+ Elo), increase model capacity

**Risk 4: Memory issues**
- Symptom: OOM errors during training
- Solution: Reduce batch size, use gradient accumulation, or use CPU training

---

## Next Steps After Phase 3a

**Phase 3b: NN-Guided MCTS**
- Replace evaluator in MCTS with neural network
- Use policy head for move prioritization (PUCT)
- Expected improvement: +200-300 Elo

**Phase 4: Self-Play RL**
- Generate training data from self-play
- Iterative improvement loop
- Scale to 20+ ResBlocks, 256 channels

---

## File Structure

```
Chess_RL/
├── training/
│   ├── dataset.py          # Dataset class for PGN → tensors
│   ├── train.py            # Training loop
│   ├── evaluate.py         # Evaluation against engines
│   └── filter_games.py     # PGN filtering utilities
├── data/
│   ├── filtered_games.pgn  # Training data
│   └── test_games.pgn      # Held-out test set
├── checkpoints/
│   ├── model_epoch_1.pt
│   ├── model_epoch_2.pt
│   └── best_model.pt       # Best validation loss
└── runs/                   # Tensorboard logs
```

---

## Recommended Starting Configuration

**For Fast Iteration (CPU):**
- Model: 4 ResBlocks, 128 channels (~830K params)
- Data: 10K games (400K-800K positions)
- Batch size: 128
- Epochs: 10
- Time: ~2-4 hours on modern CPU

**For Better Results (GPU):**
- Model: 6 ResBlocks, 256 channels (~3.3M params)
- Data: 100K games (4M-8M positions)
- Batch size: 512
- Epochs: 20
- Time: ~4-8 hours on consumer GPU

---

## References

- AlphaZero paper: https://arxiv.org/abs/1712.01815
- Lichess database: https://database.lichess.org/
- PyTorch tutorials: https://pytorch.org/tutorials/
- Tensorboard guide: https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html

---

**Status:** Documentation complete. Ready to implement when dependencies are installed.
