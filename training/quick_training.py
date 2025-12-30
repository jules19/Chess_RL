#!/usr/bin/env python3
"""
Quick Neural Network Training Pipeline

A faster version of the training pipeline that uses minimax instead of MCTS
for self-play data generation. Minimax at depth 2-3 is much faster than MCTS.

Usage:
    python3 training/quick_training.py
"""

import os
import sys
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_header(title):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def generate_training_data(num_games, output_path, depth=3):
    """Generate training games via self-play using minimax."""
    print_header("STEP 1: Generating Training Data")

    import chess
    import chess.pgn
    from search.minimax import best_move_minimax

    print(f"  Generating {num_games} self-play games...")
    print(f"  Output: {output_path}")
    print(f"  Using Minimax depth {depth}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    start_time = time.time()
    total_positions = 0

    with open(output_path, 'w') as f:
        for i in range(num_games):
            game_start = time.time()

            board = chess.Board()
            game = chess.pgn.Game()
            game.headers["Event"] = "Neural Network Training"
            game.headers["Site"] = "Chess RL Self-Play"
            game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            game.headers["Round"] = str(i + 1)
            game.headers["White"] = f"Minimax-{depth}"
            game.headers["Black"] = f"Minimax-{depth}"
            game.headers["WhiteElo"] = "1500"
            game.headers["BlackElo"] = "1500"

            node = game
            move_count = 0
            max_moves = 100

            while not board.is_game_over() and move_count < max_moves:
                move = best_move_minimax(board, depth=depth)

                if move is None:
                    break

                node = node.add_variation(move)
                board.push(move)
                move_count += 1
                total_positions += 1

            # Set game result
            if board.is_checkmate():
                result = "0-1" if board.turn == chess.WHITE else "1-0"
            elif board.is_stalemate() or board.is_insufficient_material():
                result = "1/2-1/2"
            elif board.is_fifty_moves() or board.is_repetition():
                result = "1/2-1/2"
            else:
                result = "1/2-1/2"

            game.headers["Result"] = result

            print(game, file=f, end="\n\n")

            game_time = time.time() - game_start
            elapsed = time.time() - start_time
            games_done = i + 1
            eta = (elapsed / games_done) * (num_games - games_done)

            print(f"\r  Game {games_done}/{num_games} - {move_count} moves, "
                  f"result: {result} ({game_time:.1f}s) - ETA: {eta:.0f}s",
                  end='', flush=True)

    total_time = time.time() - start_time
    print(f"\n\n  Completed {num_games} games in {total_time:.1f}s")
    print(f"  Total positions: {total_positions}")
    print(f"  Saved to: {output_path}")

    return output_path, total_positions


def train_network(pgn_path, epochs, batch_size, device, checkpoint_dir):
    """Train the neural network on game data."""
    print_header("STEP 2: Training Neural Network")

    import torch
    from net.model import create_model
    from training.dataset import create_dataloaders
    from training.train import Trainer

    print(f"  PGN file: {pgn_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")

    # Create model
    print("\n  Creating model...")
    model = create_model(num_res_blocks=4, num_channels=128)
    model.summary()

    # Create dataloaders
    print("\n  Loading training data...")
    train_loader, val_loader = create_dataloaders(
        pgn_path,
        batch_size=batch_size,
        max_games=None,
        num_workers=0
    )

    if len(train_loader.dataset) < 10:
        print(f"\n  Error: Only {len(train_loader.dataset)} training examples!")
        print("  Need at least 10 positions.")
        return None

    # Create trainer and train
    print("\n  Starting training...")
    trainer = Trainer(
        model,
        torch.device(device),
        learning_rate=0.001,
        weight_decay=1e-4
    )

    trainer.train(
        train_loader,
        val_loader,
        num_epochs=epochs,
        checkpoint_dir=checkpoint_dir
    )

    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    return best_model_path


def quick_evaluate(model_path, device, num_games=5):
    """Quick evaluation against random and minimax."""
    print_header("STEP 3: Quick Evaluation")

    import torch
    import chess
    import random
    from net.model import create_model
    from net.encoding import board_to_tensor, legal_moves_mask, index_to_move
    from search.minimax import best_move_minimax

    print(f"  Model: {model_path}")
    print(f"  Games per matchup: {num_games}")

    # Load model
    model = create_model(num_res_blocks=4, num_channels=128)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    def nn_move(board):
        with torch.no_grad():
            board_tensor = board_to_tensor(board)
            board_tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(device)
            legal_mask = legal_moves_mask(board)
            legal_mask = torch.from_numpy(legal_mask).unsqueeze(0).to(device)
            policy_probs, _ = model.get_policy_value(board_tensor, legal_mask)
            policy_probs = policy_probs.squeeze(0).cpu().numpy()

            for move_idx in policy_probs.argsort()[::-1]:
                move = index_to_move(move_idx, board)
                if move and move in board.legal_moves:
                    return move
            return list(board.legal_moves)[0]

    def play_game(white_fn, black_fn, max_moves=80):
        board = chess.Board()
        while not board.is_game_over() and board.fullmove_number < max_moves:
            move = white_fn(board) if board.turn == chess.WHITE else black_fn(board)
            board.push(move)
        if board.is_checkmate():
            return "1-0" if board.turn == chess.BLACK else "0-1"
        return "1/2-1/2"

    # NN vs Random
    print("\n  Match 1: Neural Network vs Random")
    nn_wins, draws, losses = 0, 0, 0
    random_fn = lambda b: random.choice(list(b.legal_moves))

    for i in range(num_games):
        if i % 2 == 0:
            result = play_game(nn_move, random_fn)
            if result == "1-0": nn_wins += 1
            elif result == "0-1": losses += 1
            else: draws += 1
        else:
            result = play_game(random_fn, nn_move)
            if result == "0-1": nn_wins += 1
            elif result == "1-0": losses += 1
            else: draws += 1
        print(f"\r    Game {i+1}/{num_games}...", end='', flush=True)

    score = (nn_wins + 0.5*draws) / num_games * 100
    print(f"\r    vs Random: {nn_wins}W-{draws}D-{losses}L ({score:.0f}%)")

    # NN vs Minimax depth 2
    print("\n  Match 2: Neural Network vs Minimax (depth 2)")
    nn_wins, draws, losses = 0, 0, 0
    mm_fn = lambda b: best_move_minimax(b, depth=2)

    for i in range(num_games):
        if i % 2 == 0:
            result = play_game(nn_move, mm_fn)
            if result == "1-0": nn_wins += 1
            elif result == "0-1": losses += 1
            else: draws += 1
        else:
            result = play_game(mm_fn, nn_move)
            if result == "0-1": nn_wins += 1
            elif result == "1-0": losses += 1
            else: draws += 1
        print(f"\r    Game {i+1}/{num_games}...", end='', flush=True)

    score = (nn_wins + 0.5*draws) / num_games * 100
    print(f"\r    vs Minimax-2: {nn_wins}W-{draws}D-{losses}L ({score:.0f}%)")

    print("\n  Evaluation complete!")


def main():
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Quick neural network training")
    parser.add_argument('--games', type=int, default=50,
                       help='Number of self-play games')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--depth', type=int, default=2,
                       help='Minimax search depth for self-play')
    parser.add_argument('--eval-games', type=int, default=5,
                       help='Games per evaluation matchup')
    parser.add_argument('--skip-data-gen', action='store_true',
                       help='Skip data generation')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training')
    parser.add_argument('--data-path', type=str, default='data/training_games.pgn')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')

    args = parser.parse_args()

    print("="*70)
    print(" CHESS RL - QUICK TRAINING PIPELINE")
    print(" Phase 3a: Supervised Learning")
    print("="*70)

    # Device selection
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f"\n  Device: {device}")

    # Step 1: Generate data
    if not args.skip_data_gen:
        generate_training_data(args.games, args.data_path, args.depth)
    else:
        print(f"\n  Using existing data: {args.data_path}")

    # Step 2: Train
    if not args.skip_training:
        model_path = train_network(
            args.data_path,
            args.epochs,
            args.batch_size,
            device,
            args.checkpoint_dir
        )
    else:
        model_path = os.path.join(args.checkpoint_dir, "best_model.pt")

    # Step 3: Evaluate
    if model_path and os.path.exists(model_path):
        quick_evaluate(model_path, device, args.eval_games)

    print_header("TRAINING COMPLETE")
    print(f"\n  Model: {model_path}")
    print(f"  Data: {args.data_path}")


if __name__ == "__main__":
    main()
