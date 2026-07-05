"""
Chess Dataset for Supervised Learning

Converts PGN games into training examples for the neural network:
- Input: Board state encoded as (20, 8, 8) tensor
- Target Policy: Move index (0-4671) representing the move played
- Target Value: Game outcome from current player's perspective

This dataset enables supervised learning from expert games, teaching
the network to predict both good moves (policy) and position evaluation (value).

Usage:
    # Create dataset from PGN file
    dataset = ChessDataset('games.pgn', max_games=10000)

    # Create train/val dataloaders
    train_loader, val_loader = create_dataloaders(
        'games.pgn', batch_size=256, max_games=10000
    )
"""

import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net.encoding import board_to_tensor, move_to_index


class ChessDataset(Dataset):
    """
    PyTorch Dataset of chess positions from PGN games.

    Each training example consists of:
    - board: (20, 8, 8) tensor encoding the board state
    - policy: int (0-4671) representing the move played
    - value: float {-1, 0, +1} representing game outcome

    The value is from the perspective of the player to move, so:
    - +1.0 means current player won
    -  0.0 means draw
    - -1.0 means current player lost
    """

    def __init__(self, pgn_path, max_games=None, skip_games=0):
        """
        Load chess games from PGN file and create training examples.

        Args:
            pgn_path: Path to PGN file containing chess games
            max_games: Maximum number of games to load (None = all)
            skip_games: Number of games to skip at start (for test split)
        """
        self.examples = []
        self.load_games(pgn_path, max_games, skip_games)

    def load_games(self, pgn_path, max_games, skip_games):
        """
        Parse PGN file and extract training examples.

        Each position in the game becomes a training example, where:
        - The board state is the input
        - The move actually played is the policy target
        - The final game result is the value target
        """
        print(f"Loading games from {pgn_path}...")

        if not os.path.exists(pgn_path):
            raise FileNotFoundError(f"PGN file not found: {pgn_path}")

        with open(pgn_path) as pgn_file:
            game_count = 0
            skipped = 0

            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                # Skip games if requested
                if skipped < skip_games:
                    skipped += 1
                    continue

                # Extract game result
                result = game.headers.get("Result", "*")
                if result == "1-0":
                    value = 1.0  # White wins
                elif result == "0-1":
                    value = -1.0  # Black wins
                elif result == "1/2-1/2":
                    value = 0.0  # Draw
                else:
                    continue  # Skip games without result

                # Extract positions and moves from the game
                board = game.board()
                move_count = 0

                for move in game.mainline_moves():
                    # Skip very short games (likely resignations/timeouts)
                    if move_count < 5:
                        board.push(move)
                        move_count += 1
                        continue

                    # Encode current board position
                    try:
                        board_tensor = board_to_tensor(board)
                        move_index = move_to_index(move)

                        # The encoding guarantees indices < 4672. The old code
                        # silently skipped out-of-range moves here, which hid a
                        # promotion-encoding bug and quietly removed every queen
                        # promotion from the training data. Fail loudly instead.
                        assert move_index < 4672, f"Move {move} encoded out of range: {move_index}"

                    except Exception as e:
                        print(f"Warning: Failed to encode position in game {game_count + 1}: {e}")
                        board.push(move)
                        move_count += 1
                        continue

                    # Value from current player's perspective
                    # If White to move and White won -> +1
                    # If Black to move and Black won -> +1
                    game_value = value if board.turn == chess.WHITE else -value

                    self.examples.append({
                        'board': board_tensor,
                        'policy': move_index,
                        'value': game_value
                    })

                    board.push(move)
                    move_count += 1

                game_count += 1
                if game_count % 100 == 0:
                    positions = len(self.examples)
                    avg_pos = positions / game_count if game_count > 0 else 0
                    print(f"Loaded {game_count} games, {positions:,} positions (avg {avg_pos:.1f} pos/game)")

                if max_games and game_count >= max_games:
                    break

        print(f"Dataset created: {len(self.examples):,} training examples from {game_count} games")

    def __len__(self):
        """Return the number of training examples."""
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Get a training example.

        Returns:
            board: (20, 8, 8) float32 tensor
            policy: int64 scalar (move index)
            value: float32 scalar (game outcome)
        """
        example = self.examples[idx]
        return (
            torch.from_numpy(example['board']),
            torch.tensor(example['policy'], dtype=torch.long),
            torch.tensor(example['value'], dtype=torch.float32)
        )


def create_dataloaders(pgn_path, batch_size=256, train_split=0.9, max_games=None, num_workers=4):
    """
    Create train and validation dataloaders from a PGN file.

    Args:
        pgn_path: Path to PGN file
        batch_size: Batch size for training
        train_split: Fraction of data to use for training (rest for validation)
        max_games: Maximum number of games to load (None = all)
        num_workers: Number of worker processes for data loading

    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
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
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\nDataset split:")
    print(f"  Train: {train_size:,} examples ({train_split:.0%})")
    print(f"  Val: {val_size:,} examples ({1-train_split:.0%})")

    return train_loader, val_loader


if __name__ == "__main__":
    """Test dataset creation on a small PGN file."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 training/dataset.py <pgn_path> [max_games]")
        print("\nExample:")
        print("  python3 training/dataset.py data/filtered_games.pgn 100")
        sys.exit(1)

    pgn_path = sys.argv[1]
    max_games = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    print("="*70)
    print("Testing Chess Dataset Creation")
    print("="*70)

    # Test dataset creation
    print(f"\n1. Creating dataset from {pgn_path}...")
    print(f"   Loading first {max_games} games...")
    dataset = ChessDataset(pgn_path, max_games=max_games)

    print(f"\n2. Dataset size: {len(dataset):,} positions")

    # Check first example
    if len(dataset) > 0:
        board, policy, value = dataset[0]
        print(f"\n3. Example training instance:")
        print(f"   Board shape: {board.shape} (expected: torch.Size([20, 8, 8]))")
        print(f"   Policy target: {policy.item()} (move index in range 0-4671)")
        print(f"   Value target: {value.item():.1f} (game outcome: -1, 0, or +1)")

        # Validate shapes
        assert board.shape == torch.Size([20, 8, 8]), f"Wrong board shape: {board.shape}"
        assert 0 <= policy.item() < 4672, f"Policy index out of range: {policy.item()}"
        assert value.item() in [-1.0, 0.0, 1.0], f"Invalid value: {value.item()}"
        print("   ✓ Shapes and values validated")

    # Test dataloader
    print(f"\n4. Testing DataLoader creation...")
    train_loader, val_loader = create_dataloaders(
        pgn_path, batch_size=32, max_games=max_games
    )

    # Check one batch
    print(f"\n5. Testing batch loading...")
    boards, policies, values = next(iter(train_loader))
    print(f"   Batch shapes:")
    print(f"     Boards: {boards.shape} (expected: [batch_size, 20, 8, 8])")
    print(f"     Policies: {policies.shape} (expected: [batch_size])")
    print(f"     Values: {values.shape} (expected: [batch_size])")

    assert boards.shape[0] <= 32, "Batch size too large"
    assert boards.shape[1:] == torch.Size([20, 8, 8]), "Wrong board dimensions"
    print("   ✓ Batch shapes validated")

    print("\n" + "="*70)
    print("✅ Dataset creation test PASSED!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Download larger PGN dataset (10K+ games)")
    print("  2. Run: python3 training/train.py --pgn data/filtered_games.pgn")
