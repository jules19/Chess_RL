"""
Supervised Learning Training Loop

Train the policy-value neural network on expert chess games using supervised learning.

The network learns two tasks simultaneously:
1. Policy: Predict which move expert players would make (cross-entropy loss)
2. Value: Predict the game outcome from a position (MSE loss)

Training Process:
- Load games from PGN file and create training examples
- Train for multiple epochs with train/validation split
- Track metrics: loss, policy accuracy, value error
- Save checkpoints and best model
- Log to Tensorboard for visualization

Usage:
    python3 training/train.py \
        --pgn data/filtered_games.pgn \
        --batch-size 256 \
        --epochs 10 \
        --lr 0.001
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import argparse
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net.model import create_model
from training.dataset import create_dataloaders


class Trainer:
    """
    Supervised learning trainer for chess neural network.

    The trainer manages:
    - Forward/backward passes
    - Loss computation (policy + value)
    - Optimization with learning rate scheduling
    - Checkpointing
    - Tensorboard logging
    """

    def __init__(self, model, device, learning_rate=0.001, weight_decay=1e-4):
        """
        Initialize trainer.

        Args:
            model: PolicyValueNetwork instance
            device: torch.device ('cpu' or 'cuda')
            learning_rate: Initial learning rate
            weight_decay: L2 regularization strength
        """
        self.model = model.to(device)
        self.device = device

        # Loss functions
        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()

        # Optimizer with weight decay for regularization
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler (reduce LR every 5 epochs)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.5
        )

        # Tensorboard logging
        log_dir = f"runs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir)
        print(f"Tensorboard logging to: {log_dir}")
        print(f"  View with: tensorboard --logdir runs/")

    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number (for logging)

        Returns:
            avg_loss: Average total loss
            avg_policy_loss: Average policy loss
            avg_value_loss: Average value loss
        """
        self.model.train()

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

        for batch_idx, (boards, target_policies, target_values) in enumerate(pbar):
            # Move batch to device (CPU or GPU)
            boards = boards.to(self.device)
            target_policies = target_policies.to(self.device)
            target_values = target_values.to(self.device).unsqueeze(1)

            # Forward pass through network
            policy_logits, pred_values = self.model(boards)

            # Compute losses
            policy_loss = self.policy_criterion(policy_logits, target_policies)
            value_loss = self.value_criterion(pred_values, target_values)

            # Combined loss (equal weighting)
            loss = policy_loss + value_loss

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'policy': f"{policy_loss.item():.4f}",
                'value': f"{value_loss.item():.4f}"
            })

        # Compute averages
        avg_loss = total_loss / num_batches
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches

        return avg_loss, avg_policy_loss, avg_value_loss

    def validate(self, val_loader, epoch):
        """
        Validate on validation set.

        Args:
            val_loader: DataLoader for validation data
            epoch: Current epoch number (for logging)

        Returns:
            avg_loss: Average total loss
            avg_policy_loss: Average policy loss
            avg_value_loss: Average value loss
            policy_accuracy: Top-1 policy prediction accuracy
        """
        self.model.eval()

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        correct_moves = 0
        total_moves = 0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")

            for boards, target_policies, target_values in pbar:
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
                num_batches += 1

                # Compute policy accuracy (top-1)
                pred_moves = policy_logits.argmax(dim=1)
                correct_moves += (pred_moves == target_policies).sum().item()
                total_moves += target_policies.size(0)

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{correct_moves/total_moves:.2%}"
                })

        # Compute averages
        avg_loss = total_loss / num_batches
        avg_policy_loss = total_policy_loss / num_batches
        avg_value_loss = total_value_loss / num_batches
        policy_accuracy = correct_moves / total_moves

        return avg_loss, avg_policy_loss, avg_value_loss, policy_accuracy

    def train(self, train_loader, val_loader, num_epochs, checkpoint_dir="checkpoints"):
        """
        Full training loop.

        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        best_val_loss = float('inf')
        best_epoch = 0

        print("\n" + "="*70)
        print("Starting Training")
        print("="*70)

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*70}")

            # Train for one epoch
            train_loss, train_policy_loss, train_value_loss = self.train_epoch(
                train_loader, epoch
            )

            # Validate
            val_loss, val_policy_loss, val_value_loss, val_accuracy = self.validate(
                val_loader, epoch
            )

            # Step learning rate scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Policy_Loss/train', train_policy_loss, epoch)
            self.writer.add_scalar('Policy_Loss/val', val_policy_loss, epoch)
            self.writer.add_scalar('Value_Loss/train', train_value_loss, epoch)
            self.writer.add_scalar('Value_Loss/val', val_value_loss, epoch)
            self.writer.add_scalar('Metrics/policy_accuracy', val_accuracy, epoch)
            self.writer.add_scalar('Metrics/learning_rate', current_lr, epoch)

            # Print summary
            print(f"\nResults:")
            print(f"  Train - Loss: {train_loss:.4f} (policy: {train_policy_loss:.4f}, value: {train_value_loss:.4f})")
            print(f"  Val   - Loss: {val_loss:.4f} (policy: {val_policy_loss:.4f}, value: {val_value_loss:.4f})")
            print(f"  Val   - Policy Accuracy: {val_accuracy:.2%}")
            print(f"  Learning Rate: {current_lr:.6f}")

            # Save checkpoint every epoch.
            # Checkpoints are SELF-DESCRIBING: they carry the architecture
            # config, so loaders never need to guess num_res_blocks/channels
            # (see net.model.load_model for why this matters).
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                # Architecture metadata (also inferable from the weights)
                'input_channels': self.model.input_channels,
                'num_res_blocks': self.model.num_res_blocks,
                'num_channels': self.model.num_channels,
            }
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"  Saved: {checkpoint_path}")

            # Save best model (same self-describing format, not a bare state_dict)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_path = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save(checkpoint, best_path)
                print(f"  ✓ New best model! (val_loss: {val_loss:.4f})")

        # Training complete
        self.writer.close()

        print("\n" + "="*70)
        print("✅ Training Complete!")
        print("="*70)
        print(f"Best model: Epoch {best_epoch} (val_loss: {best_val_loss:.4f})")
        print(f"Saved to: {os.path.join(checkpoint_dir, 'best_model.pt')}")
        print(f"\nNext steps:")
        print(f"  1. Evaluate: python3 training/evaluate.py {os.path.join(checkpoint_dir, 'best_model.pt')}")
        print(f"  2. View logs: tensorboard --logdir runs/")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train chess neural network with supervised learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument('--pgn', type=str, required=True,
                        help='Path to PGN file with training games')
    parser.add_argument('--max-games', type=int, default=None,
                        help='Maximum number of games to load (None = all)')

    # Model arguments
    parser.add_argument('--res-blocks', type=int, default=4,
                        help='Number of residual blocks')
    parser.add_argument('--channels', type=int, default=128,
                        help='Number of channels in residual blocks')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='L2 regularization weight')

    # Other arguments
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, auto-detected if not specified)')

    args = parser.parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("Chess Neural Network - Supervised Learning")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  PGN file: {args.pgn}")
    print(f"  Max games: {args.max_games or 'all'}")
    print(f"  Model: {args.res_blocks} ResBlocks, {args.channels} channels")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")

    # Create model
    print(f"\nCreating model...")
    model = create_model(num_res_blocks=args.res_blocks, num_channels=args.channels)
    model.summary()

    # Create dataloaders
    print(f"\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        args.pgn,
        batch_size=args.batch_size,
        max_games=args.max_games,
        num_workers=args.num_workers
    )

    # Create trainer and start training
    trainer = Trainer(model, device, learning_rate=args.lr, weight_decay=args.weight_decay)
    trainer.train(train_loader, val_loader, args.epochs, args.checkpoint_dir)


if __name__ == "__main__":
    main()
