#!/usr/bin/env python3
"""
Complete Neural Network Training Pipeline

This script runs the entire Phase 3a training pipeline:
1. Generate training data (self-play or download)
2. Filter and prepare games
3. Train neural network
4. Evaluate trained model against baselines

Usage:
    python3 training/run_training_pipeline.py

    # With options:
    python3 training/run_training_pipeline.py --games 200 --epochs 10
"""

import argparse
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


def check_dependencies():
    """Check that all required packages are installed."""
    print_header("STEP 0: Checking Dependencies")

    missing = []

    try:
        import torch
        print(f"  [OK] PyTorch {torch.__version__}")

        # Check for GPU
        if torch.cuda.is_available():
            print(f"  [OK] CUDA available: {torch.cuda.get_device_name(0)}")
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"  [OK] Apple MPS (Metal) available")
            device = 'mps'
        else:
            print(f"  [OK] CPU mode (no GPU detected)")
            device = 'cpu'

    except ImportError:
        missing.append("torch")
        device = 'cpu'

    try:
        import chess
        print(f"  [OK] python-chess {chess.__version__}")
    except ImportError:
        missing.append("python-chess")

    try:
        import numpy as np
        print(f"  [OK] NumPy {np.__version__}")
    except ImportError:
        missing.append("numpy")

    try:
        from tqdm import tqdm
        print(f"  [OK] tqdm available")
    except ImportError:
        print(f"  [WARN] tqdm not installed (optional)")

    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print(f"  Install with: pip install {' '.join(missing)}")
        return None

    print(f"\n  Device selected: {device}")
    return device


def generate_training_data(num_games, output_path):
    """Generate training games via self-play."""
    print_header("STEP 1: Generating Training Data")

    import chess
    import chess.pgn
    from search.mcts import mcts_search

    print(f"  Generating {num_games} self-play games...")
    print(f"  Output: {output_path}")
    print(f"  Using MCTS with 100 simulations per move")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    start_time = time.time()

    with open(output_path, 'w') as f:
        for i in range(num_games):
            game_start = time.time()

            board = chess.Board()
            game = chess.pgn.Game()
            game.headers["Event"] = "Neural Network Training"
            game.headers["Site"] = "Chess RL Self-Play"
            game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
            game.headers["Round"] = str(i + 1)
            game.headers["White"] = "MCTS-100"
            game.headers["Black"] = "MCTS-100"
            game.headers["WhiteElo"] = "2000"
            game.headers["BlackElo"] = "2000"

            node = game
            move_count = 0
            max_moves = 120

            while not board.is_game_over() and move_count < max_moves:
                # Use MCTS to select move (uses built-in evaluator)
                move = mcts_search(board, simulations=100, use_evaluator=True)

                if move is None:
                    break

                node = node.add_variation(move)
                board.push(move)
                move_count += 1

            # Set game result
            if board.is_checkmate():
                result = "0-1" if board.turn == chess.WHITE else "1-0"
            elif board.is_stalemate() or board.is_insufficient_material():
                result = "1/2-1/2"
            elif board.is_fifty_moves() or board.is_repetition():
                result = "1/2-1/2"
            else:
                result = "1/2-1/2"  # Max moves reached

            game.headers["Result"] = result

            # Write game to file
            print(game, file=f, end="\n\n")

            game_time = time.time() - game_start
            elapsed = time.time() - start_time
            games_done = i + 1
            eta = (elapsed / games_done) * (num_games - games_done)

            print(f"\r  Game {games_done}/{num_games} - {move_count} moves, "
                  f"result: {result} ({game_time:.1f}s) - ETA: {eta/60:.1f}min",
                  end='', flush=True)

    total_time = time.time() - start_time
    print(f"\n\n  Completed {num_games} games in {total_time/60:.1f} minutes")
    print(f"  Saved to: {output_path}")

    return output_path


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
    print(f"  Checkpoint dir: {checkpoint_dir}")

    # Create model
    print("\n  Creating model...")
    model = create_model(num_res_blocks=4, num_channels=128)
    model.summary()

    # Create dataloaders
    print("\n  Loading training data...")
    try:
        train_loader, val_loader = create_dataloaders(
            pgn_path,
            batch_size=batch_size,
            max_games=None,  # Use all games
            num_workers=0  # Avoid multiprocessing issues
        )
    except Exception as e:
        print(f"\n  Error loading data: {e}")
        print("  Trying with fewer games...")
        train_loader, val_loader = create_dataloaders(
            pgn_path,
            batch_size=batch_size,
            max_games=100,
            num_workers=0
        )

    # Check if we have enough data
    if len(train_loader.dataset) < 100:
        print(f"\n  Warning: Only {len(train_loader.dataset)} training examples!")
        print("  Consider generating more games.")

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
    print(f"\n  Best model saved to: {best_model_path}")

    return best_model_path


def evaluate_model(model_path, device, num_games=10):
    """Evaluate trained model against baselines."""
    print_header("STEP 3: Evaluating Trained Model")

    import torch
    import chess
    from net.model import create_model
    from net.encoding import board_to_tensor, legal_moves_mask, index_to_move
    from search.minimax import best_move_minimax
    from engine.evaluator import evaluate

    print(f"  Model: {model_path}")
    print(f"  Games per matchup: {num_games}")

    # Load model
    print("\n  Loading trained model...")
    model = create_model(num_res_blocks=4, num_channels=128)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    def nn_select_move(board):
        """Select move using neural network."""
        with torch.no_grad():
            board_tensor = board_to_tensor(board)
            board_tensor = torch.from_numpy(board_tensor).unsqueeze(0).to(device)

            legal_mask = legal_moves_mask(board)
            legal_mask = torch.from_numpy(legal_mask).unsqueeze(0).to(device)

            policy_probs, value = model.get_policy_value(board_tensor, legal_mask)
            policy_probs = policy_probs.squeeze(0).cpu().numpy()

            # Select highest probability legal move
            for move_idx in policy_probs.argsort()[::-1]:
                move = index_to_move(move_idx, board)
                if move and move in board.legal_moves:
                    return move

            # Fallback
            return list(board.legal_moves)[0]

    def minimax_select_move(board, depth=3):
        """Select move using minimax."""
        return best_move_minimax(board, depth=depth)

    def random_select_move(board):
        """Select random legal move."""
        import random
        return random.choice(list(board.legal_moves))

    def play_game(white_fn, black_fn, max_moves=100):
        """Play a game between two move selection functions."""
        board = chess.Board()

        while not board.is_game_over() and board.fullmove_number < max_moves:
            if board.turn == chess.WHITE:
                move = white_fn(board)
            else:
                move = black_fn(board)
            board.push(move)

        if board.is_checkmate():
            return "1-0" if board.turn == chess.BLACK else "0-1"
        return "1/2-1/2"

    def run_match(player1_fn, player2_fn, player1_name, player2_name, games):
        """Run a match between two players."""
        p1_wins = 0
        p2_wins = 0
        draws = 0

        for i in range(games):
            # Alternate colors
            if i % 2 == 0:
                result = play_game(player1_fn, player2_fn)
                if result == "1-0":
                    p1_wins += 1
                elif result == "0-1":
                    p2_wins += 1
                else:
                    draws += 1
            else:
                result = play_game(player2_fn, player1_fn)
                if result == "1-0":
                    p2_wins += 1
                elif result == "0-1":
                    p1_wins += 1
                else:
                    draws += 1

            print(f"\r    Game {i+1}/{games}...", end='', flush=True)

        print(f"\r    {player1_name} vs {player2_name}: "
              f"{p1_wins}W-{draws}D-{p2_wins}L "
              f"({(p1_wins + 0.5*draws)/(p1_wins+p2_wins+draws)*100:.0f}%)")

        return p1_wins, draws, p2_wins

    # Run evaluation matches
    results = {}

    print("\n  Running evaluation matches...")

    # NN vs Random
    print("\n  Match 1: Neural Network vs Random")
    w, d, l = run_match(nn_select_move, random_select_move, "NN", "Random", num_games)
    results['vs_random'] = (w, d, l)

    # NN vs Minimax depth 2
    print("\n  Match 2: Neural Network vs Minimax (depth 2)")
    w, d, l = run_match(nn_select_move, lambda b: minimax_select_move(b, 2),
                        "NN", "Minimax-2", num_games)
    results['vs_minimax2'] = (w, d, l)

    # NN vs Minimax depth 3
    print("\n  Match 3: Neural Network vs Minimax (depth 3)")
    w, d, l = run_match(nn_select_move, lambda b: minimax_select_move(b, 3),
                        "NN", "Minimax-3", num_games)
    results['vs_minimax3'] = (w, d, l)

    # Print summary
    print("\n" + "-"*50)
    print("  EVALUATION SUMMARY")
    print("-"*50)

    for opponent, (w, d, l) in results.items():
        total = w + d + l
        score = (w + 0.5 * d) / total * 100
        print(f"  vs {opponent:15s}: {w}W-{d}D-{l}L ({score:.0f}%)")

    # Overall assessment
    print("\n  ASSESSMENT:")
    vs_random = results['vs_random']
    vs_mm3 = results['vs_minimax3']

    random_score = (vs_random[0] + 0.5 * vs_random[1]) / sum(vs_random)
    mm3_score = (vs_mm3[0] + 0.5 * vs_mm3[1]) / sum(vs_mm3)

    if random_score > 0.9:
        print("  [OK] Beats random consistently (>90%)")
    elif random_score > 0.7:
        print("  [OK] Beats random most of the time (>70%)")
    else:
        print("  [!!] Struggles against random - needs more training")

    if mm3_score > 0.4:
        print("  [OK] Competitive with Minimax depth 3 (>40%)")
    else:
        print("  [!!] Weaker than Minimax - needs more training data")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run complete neural network training pipeline"
    )
    parser.add_argument('--games', type=int, default=50,
                       help='Number of self-play games to generate')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Training batch size')
    parser.add_argument('--eval-games', type=int, default=10,
                       help='Games per evaluation matchup')
    parser.add_argument('--skip-data-gen', action='store_true',
                       help='Skip data generation (use existing data)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training (evaluate existing model)')
    parser.add_argument('--data-path', type=str, default='data/training_games.pgn',
                       help='Path to training data')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory for model checkpoints')

    args = parser.parse_args()

    print("="*70)
    print(" CHESS RL - NEURAL NETWORK TRAINING PIPELINE")
    print(" Phase 3a: Supervised Learning on Self-Play Games")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Games to generate: {args.games}")
    print(f"  Training epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Evaluation games: {args.eval_games}")
    print(f"  Data path: {args.data_path}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")

    # Check dependencies and get device
    device = check_dependencies()
    if device is None:
        print("\n[ERROR] Missing dependencies. Please install and retry.")
        sys.exit(1)

    # Step 1: Generate training data
    if not args.skip_data_gen:
        if os.path.exists(args.data_path):
            print(f"\n  Existing data found at {args.data_path}")
            print("  Use --skip-data-gen to use existing data")
        generate_training_data(args.games, args.data_path)
    else:
        if not os.path.exists(args.data_path):
            print(f"\n[ERROR] No data found at {args.data_path}")
            print("  Remove --skip-data-gen to generate new data")
            sys.exit(1)
        print(f"\n  Using existing data: {args.data_path}")

    # Step 2: Train network
    if not args.skip_training:
        best_model_path = train_network(
            args.data_path,
            args.epochs,
            args.batch_size,
            device,
            args.checkpoint_dir
        )
    else:
        best_model_path = os.path.join(args.checkpoint_dir, "best_model.pt")
        if not os.path.exists(best_model_path):
            print(f"\n[ERROR] No model found at {best_model_path}")
            print("  Remove --skip-training to train a new model")
            sys.exit(1)
        print(f"\n  Using existing model: {best_model_path}")

    # Step 3: Evaluate model
    results = evaluate_model(best_model_path, device, args.eval_games)

    # Final summary
    print_header("TRAINING PIPELINE COMPLETE")
    print(f"\n  Model saved to: {best_model_path}")
    print(f"  Training data: {args.data_path}")
    print(f"\n  Next steps:")
    print(f"    1. Review model performance against baselines")
    print(f"    2. Generate more training data if needed")
    print(f"    3. Proceed to Phase 3b: NN-MCTS integration")
    print(f"\n  To retrain with more data:")
    print(f"    python3 training/run_training_pipeline.py --games 200 --epochs 20")
    print(f"\n  To evaluate existing model:")
    print(f"    python3 training/evaluate.py {best_model_path}")


if __name__ == "__main__":
    main()
