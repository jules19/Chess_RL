"""
Self-Play Training Loop - Phase 4's main entry point

Orchestrates the full AlphaZero cycle:

    for each iteration:
        1. GENERATE  - champion plays games against itself (with exploration)
        2. STORE     - positions go into the replay buffer
        3. TRAIN     - a candidate (clone of champion) trains on random
                       batches from the buffer
        4. GATE      - candidate plays champion in the arena;
                       promote only if it scores above the threshold

The LOSS here differs from supervised training in one important way:
supervised targets were a single expert move (class index → CrossEntropyLoss),
but self-play targets are full DISTRIBUTIONS over moves (PUCT visit
proportions). Cross-entropy against a soft target is
    L_policy = -Σ π(a) · log p(a)
which we compute manually with log_softmax. The value loss is MSE, as before.

Start tiny. This runs end-to-end on a laptop CPU in a few minutes:

    python3 selfplay/train_loop.py --iterations 2 --games-per-iter 2 \
        --simulations 16 --train-steps 20 --arena-games 2

It will NOT get strong at that scale — the point is to verify the plumbing.
Real improvement needs orders of magnitude more games and simulations
(see course/module-07 for the scaling discussion).
"""

import argparse
import copy
import os
import random
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from net.model import create_model, load_model, auto_device
from search.puct import network_policy_value
from selfplay.replay_buffer import ReplayBuffer
from selfplay.self_play import generate_games
from selfplay.arena import play_match


def soft_target_loss(policy_logits, target_policies, pred_values, target_values):
    """
    AlphaZero loss: soft-target cross-entropy (policy) + MSE (value).

    Args:
        policy_logits: (batch, 4672) raw network outputs
        target_policies: (batch, 4672) PUCT visit distributions
        pred_values: (batch, 1) network value predictions
        target_values: (batch, 1) game outcomes

    Returns:
        (total_loss, policy_loss, value_loss)
    """
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(target_policies * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(pred_values, target_values)
    return policy_loss + value_loss, policy_loss, value_loss


def train_candidate(model, buffer, device, train_steps=100, batch_size=64,
                    learning_rate=1e-3, weight_decay=1e-4, rng=None,
                    verbose=True):
    """
    Train a candidate network on random batches from the replay buffer.

    Note: STEPS, not epochs. The buffer is a moving window, so "one pass
    over the data" isn't meaningful the way it is in supervised learning —
    we just take N gradient steps on freshly sampled batches.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)

    for step in range(1, train_steps + 1):
        states, policies, values = buffer.sample(batch_size, rng=rng)
        states = torch.from_numpy(states).to(device)
        policies = torch.from_numpy(policies).to(device)
        values = torch.from_numpy(values).unsqueeze(1).to(device)

        policy_logits, pred_values = model(states)
        loss, policy_loss, value_loss = soft_target_loss(
            policy_logits, policies, pred_values, values
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if verbose and (step % max(1, train_steps // 5) == 0 or step == 1):
            print(f"    step {step}/{train_steps}: loss {loss.item():.4f} "
                  f"(policy {policy_loss.item():.4f}, value {value_loss.item():.4f})")

    model.eval()


def save_checkpoint(model, path, iteration):
    """Save a self-describing checkpoint (see net.model.load_model)."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_channels': model.input_channels,
        'num_res_blocks': model.num_res_blocks,
        'num_channels': model.num_channels,
        'selfplay_iteration': iteration,
    }, path)


def load_resume_state(checkpoint_dir):
    """
    Reference solution — course Module 6, exercise 5.

    Restore everything a killed run needs to continue, from the artifacts
    the loop already writes every iteration:
      champion.pt        -> the current best network + iteration number
      replay_buffer.npz  -> the training-data window

    Returns:
        (champion_model, replay_buffer, last_completed_iteration)
    """
    champion, checkpoint = load_model(os.path.join(checkpoint_dir, "champion.pt"))
    buffer = ReplayBuffer.load(os.path.join(checkpoint_dir, "replay_buffer.npz"))
    return champion, buffer, int(checkpoint.get('selfplay_iteration', 0))


def run_training_loop(iterations=5,
                      games_per_iter=10,
                      simulations=50,
                      train_steps=200,
                      batch_size=64,
                      arena_games=6,
                      arena_simulations=None,
                      promote_threshold=0.55,
                      buffer_capacity=50_000,
                      temperature_moves=20,
                      max_moves=200,
                      learning_rate=1e-3,
                      res_blocks=4,
                      channels=128,
                      init_checkpoint=None,
                      checkpoint_dir="selfplay_checkpoints",
                      device=None,
                      seed=None,
                      resume=False,
                      verbose=True):
    """
    Run the full self-play RL loop. Returns the champion model.

    See module docstring for the algorithm; every parameter is a knob worth
    experimenting with (that's the course exercise for module 6).
    """
    device = torch.device(device) if device else auto_device()
    rng = random.Random(seed) if seed is not None else None
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # The CHAMPION generates data; a CANDIDATE clone trains and challenges
    start_iteration = 0
    if resume:
        champion, buffer, start_iteration = load_resume_state(checkpoint_dir)
        champion.to(device)
        champion.eval()
        print(f"Resumed from {checkpoint_dir}: iteration {start_iteration}, "
              f"{len(buffer):,} buffered positions")
    elif init_checkpoint:
        champion, _ = load_model(init_checkpoint, device)
        buffer = ReplayBuffer(capacity=buffer_capacity)
        print(f"Initialized champion from {init_checkpoint}")
    else:
        champion = create_model(num_res_blocks=res_blocks, num_channels=channels)
        champion.to(device)
        champion.eval()
        buffer = ReplayBuffer(capacity=buffer_capacity)
        print("Initialized champion with RANDOM weights (tabula rasa, like AlphaZero)")

    promotions = 0
    end_iteration = start_iteration + iterations

    for iteration in range(start_iteration + 1, end_iteration + 1):
        print(f"\n{'=' * 70}\nIteration {iteration}/{end_iteration}\n{'=' * 70}")

        # 1. GENERATE - champion self-play with exploration noise
        print(f"Self-play: {games_per_iter} games @ {simulations} sims/move")
        champion_fn = network_policy_value(champion, device)
        examples = generate_games(
            champion_fn, games_per_iter,
            simulations=simulations,
            temperature_moves=temperature_moves,
            max_moves=max_moves,
            rng=rng,
            verbose=verbose,
        )

        # 2. STORE
        buffer.add_game(examples)
        print(f"Replay buffer: {len(buffer):,} positions")

        # 3. TRAIN a candidate clone
        print(f"Training candidate: {train_steps} steps, batch {batch_size}")
        candidate = copy.deepcopy(champion)
        train_candidate(candidate, buffer, device,
                        train_steps=train_steps, batch_size=batch_size,
                        learning_rate=learning_rate, rng=rng, verbose=verbose)

        # 4. GATE - candidate must earn promotion in the arena
        print(f"Arena: {arena_games} games, promote at >{promote_threshold:.0%}")
        candidate_fn = network_policy_value(candidate, device)
        score = play_match(candidate_fn, champion_fn,
                           num_games=arena_games,
                           simulations=arena_simulations or simulations,
                           max_moves=max_moves,
                           verbose=verbose)
        print(f"Candidate score: {score:.0%}")

        if score > promote_threshold:
            champion = candidate
            promotions += 1
            print("✓ PROMOTED: candidate is the new champion")
        else:
            print("✗ Rejected: champion stays (candidate not clearly better)")

        save_checkpoint(champion, os.path.join(checkpoint_dir, "champion.pt"), iteration)
        save_checkpoint(candidate, os.path.join(checkpoint_dir, f"candidate_iter_{iteration}.pt"), iteration)
        buffer.save(os.path.join(checkpoint_dir, "replay_buffer.npz"))

    print(f"\n{'=' * 70}")
    print(f"✅ Loop complete: {iterations} iterations, {promotions} promotions")
    print(f"Champion saved to {os.path.join(checkpoint_dir, 'champion.pt')}")
    print(f"{'=' * 70}")
    return champion


def main():
    parser = argparse.ArgumentParser(
        description="AlphaZero-style self-play training loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--games-per-iter', type=int, default=10)
    parser.add_argument('--simulations', type=int, default=50,
                        help='PUCT simulations per move during self-play')
    parser.add_argument('--train-steps', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--arena-games', type=int, default=6)
    parser.add_argument('--arena-simulations', type=int, default=None,
                        help='Sims per move in arena games (defaults to --simulations)')
    parser.add_argument('--promote-threshold', type=float, default=0.55)
    parser.add_argument('--buffer-capacity', type=int, default=50_000)
    parser.add_argument('--temperature-moves', type=int, default=20)
    parser.add_argument('--max-moves', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--res-blocks', type=int, default=4)
    parser.add_argument('--channels', type=int, default=128)
    parser.add_argument('--init-checkpoint', type=str, default=None,
                        help='Warm-start from a supervised checkpoint (Phase 3a)')
    parser.add_argument('--resume', action='store_true',
                        help='Continue a killed run from checkpoint-dir '
                             '(champion.pt + replay_buffer.npz)')
    parser.add_argument('--checkpoint-dir', type=str, default='selfplay_checkpoints')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    run_training_loop(
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        simulations=args.simulations,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        arena_games=args.arena_games,
        arena_simulations=args.arena_simulations,
        promote_threshold=args.promote_threshold,
        buffer_capacity=args.buffer_capacity,
        temperature_moves=args.temperature_moves,
        max_moves=args.max_moves,
        learning_rate=args.lr,
        res_blocks=args.res_blocks,
        channels=args.channels,
        init_checkpoint=args.init_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        seed=args.seed,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
