#!/usr/bin/env python3
"""
Fast Neural Network Training Pipeline

Uses simple greedy material-based move selection for fast data generation.
This is a proof-of-concept to validate the training pipeline.

Usage:
    python3 training/fast_training.py
"""

import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_header(title):
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def generate_fast_games(num_games, output_path):
    """Generate training games using fast greedy selection."""
    print_header("STEP 1: Generating Training Data (Fast Mode)")

    import chess
    import chess.pgn
    import random
    from engine.evaluator import evaluate_material

    print(f"  Generating {num_games} games...")
    print(f"  Using greedy material evaluation (fast)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    start_time = time.time()
    total_positions = 0

    def greedy_move(board):
        """Pick move with best material outcome, with randomness."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Mix of random and greedy for variety
        if random.random() < 0.3:
            return random.choice(legal_moves)

        # Greedy: pick best material outcome
        best_move = None
        best_score = float('-inf') if board.turn == chess.WHITE else float('inf')

        for move in legal_moves:
            board.push(move)
            score = evaluate_material(board)
            board.pop()

            if board.turn == chess.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        return best_move or legal_moves[0]

    with open(output_path, 'w') as f:
        for i in range(num_games):
            board = chess.Board()
            game = chess.pgn.Game()
            game.headers["Event"] = "Fast Training Data"
            game.headers["Site"] = "Chess RL"
            game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            game.headers["Round"] = str(i + 1)
            game.headers["White"] = "Greedy"
            game.headers["Black"] = "Greedy"
            game.headers["WhiteElo"] = "1200"
            game.headers["BlackElo"] = "1200"

            node = game
            move_count = 0

            while not board.is_game_over() and move_count < 80:
                move = greedy_move(board)
                if move is None:
                    break
                node = node.add_variation(move)
                board.push(move)
                move_count += 1
                total_positions += 1

            if board.is_checkmate():
                result = "0-1" if board.turn == chess.WHITE else "1-0"
            else:
                result = "1/2-1/2"

            game.headers["Result"] = result
            print(game, file=f, end="\n\n")

            if (i + 1) % 10 == 0:
                print(f"\r  Generated {i+1}/{num_games} games...", end='', flush=True)

    elapsed = time.time() - start_time
    print(f"\n\n  Completed {num_games} games in {elapsed:.1f}s")
    print(f"  Total positions: {total_positions}")
    print(f"  Speed: {num_games/elapsed:.1f} games/sec")

    return total_positions


def train_network(pgn_path, epochs, batch_size, device, checkpoint_dir):
    """Train the neural network."""
    print_header("STEP 2: Training Neural Network")

    import torch
    from net.model import create_model
    from training.dataset import create_dataloaders
    from training.train import Trainer

    print(f"  Data: {pgn_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Device: {device}")

    model = create_model(num_res_blocks=4, num_channels=128)
    model.summary()

    print("\n  Loading data...")
    train_loader, val_loader = create_dataloaders(
        pgn_path, batch_size=batch_size, max_games=None, num_workers=0
    )

    print(f"  Training examples: {len(train_loader.dataset)}")
    print(f"  Validation examples: {len(val_loader.dataset)}")

    if len(train_loader.dataset) < 100:
        print("  Warning: Very few training examples!")

    trainer = Trainer(model, torch.device(device), learning_rate=0.001)
    trainer.train(train_loader, val_loader, num_epochs=epochs, checkpoint_dir=checkpoint_dir)

    return os.path.join(checkpoint_dir, "best_model.pt")


def quick_eval(model_path, device):
    """Quick sanity check evaluation."""
    print_header("STEP 3: Quick Evaluation")

    import torch
    import chess
    import random
    from net.model import create_model
    from net.encoding import board_to_tensor, legal_moves_mask, index_to_move

    model = create_model(num_res_blocks=4, num_channels=128)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Test on starting position
    board = chess.Board()
    with torch.no_grad():
        board_tensor = board_to_tensor(board)
        board_tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(device)
        legal_mask = legal_moves_mask(board)
        legal_mask = torch.from_numpy(legal_mask).unsqueeze(0).to(device)
        policy, value = model.get_policy_value(board_tensor, legal_mask)

    print(f"  Starting position value: {value.item():.3f}")

    # Get top moves
    policy = policy.squeeze(0).cpu().numpy()
    top_indices = policy.argsort()[::-1][:5]
    print("  Top predicted moves:")
    for idx in top_indices:
        move = index_to_move(idx, board)
        if move:
            print(f"    {move}: {policy[idx]:.3f}")

    # Play a quick game vs random
    print("\n  Playing test game vs random...")
    wins = 0
    for game_num in range(3):
        board = chess.Board()
        moves = 0
        while not board.is_game_over() and moves < 60:
            if board.turn == chess.WHITE:
                # NN move
                with torch.no_grad():
                    bt = board_to_tensor(board)
                    bt = torch.from_numpy(bt).unsqueeze(0).to(device)
                    lm = legal_moves_mask(board)
                    lm = torch.from_numpy(lm).unsqueeze(0).to(device)
                    p, _ = model.get_policy_value(bt, lm)
                    p = p.squeeze(0).cpu().numpy()
                    for idx in p.argsort()[::-1]:
                        m = index_to_move(idx, board)
                        if m and m in board.legal_moves:
                            board.push(m)
                            break
            else:
                # Random move
                board.push(random.choice(list(board.legal_moves)))
            moves += 1

        if board.is_checkmate() and board.turn == chess.BLACK:
            wins += 1
            print(f"    Game {game_num+1}: WIN (checkmate)")
        elif board.is_checkmate():
            print(f"    Game {game_num+1}: LOSS (checkmated)")
        else:
            print(f"    Game {game_num+1}: Draw ({moves} moves)")

    print(f"\n  Result: {wins}/3 wins vs random")


def main():
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--skip-gen', action='store_true')
    parser.add_argument('--skip-train', action='store_true')
    args = parser.parse_args()

    print("="*70)
    print(" CHESS RL - FAST TRAINING PIPELINE")
    print("="*70)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(f"\n  Device: {device}")

    data_path = 'data/training_games.pgn'
    checkpoint_dir = 'checkpoints'

    if not args.skip_gen:
        generate_fast_games(args.games, data_path)

    if not args.skip_train:
        model_path = train_network(data_path, args.epochs, args.batch_size, device, checkpoint_dir)
    else:
        model_path = os.path.join(checkpoint_dir, "best_model.pt")

    if os.path.exists(model_path):
        quick_eval(model_path, device)

    print_header("COMPLETE")
    print(f"  Model saved to: {model_path}")


if __name__ == "__main__":
    main()
