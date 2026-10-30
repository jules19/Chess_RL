"""
Evaluate a supervised checkpoint on held-out games, against baselines.

train.py prints a validation accuracy, but a bare number can't tell you
whether the network learned anything: is 9% good? This script answers that
by putting the network next to two baselines that know nothing about chess
positions, and by checking each head separately.

It rebuilds exactly the validation split train.py used (same PGN, same
--max-games, same split_by_game), but keeps the boards, so it can do what
the engine does: pick only among LEGAL moves.

Report sections:
    Policy  - does it predict the move a strong player made, in games it
              never saw? Compared with a random legal move and with "the
              legal move that was most common in the training set".
    Value   - does it know who is winning? Compared with always predicting
              0 ("even"). Also its correlation with material balance, which
              it was never told about: a positive number means it learned it.
    Smell tests - six fixed positions with known answers. Expect the raw
              network (no search) to fail the tactics: that gap is what
              PUCT search adds in Module 5.

Usage:
    python3 training/eval_supervised.py checkpoints/best_model.pt \\
        --pgn data/filtered.pgn --max-games 1000
"""

import argparse
import collections
import os
import sys
from types import SimpleNamespace

import chess
import chess.pgn
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net.encoding import board_to_tensor, index_to_move, legal_moves_mask, move_to_index
from net.model import load_model
from training.dataset import split_by_game

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

SMELL_TESTS = [
    # (name, FEN, moves that pass; None = report the value only)
    ("start position", chess.STARTING_FEN, {"e4", "d4", "Nf3", "c4"}),
    ("1.e4 - Black's reply", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
     {"e5", "c5", "e6", "c6", "d5", "d6", "Nf6", "g6"}),
    ("free queen on d5", "rnb1kbnr/ppp1pppp/8/3q4/8/2N5/PPPP1PPP/R1BQKBNR w KQkq - 0 4", {"Nxd5"}),
    ("back-rank mate in 1", "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", {"Rd8#"}),
    ("White a queen up: value > 0?", "4k3/8/8/8/8/8/3QPPPP/4K3 w - - 0 1", None),
    ("White a queen down: value < 0?", "3qk3/8/8/8/8/8/4PPPP/4K3 w - - 0 1", None),
]


def load_positions(pgn_path, max_games=None):
    """
    Parse games exactly as training.dataset.ChessDataset does, keeping the board.

    Returns a list of dicts: fen, move (policy index), value (side-to-move
    outcome), game (index used by split_by_game), ply.
    """
    examples = []
    with open(pgn_path) as f:
        game_count = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            value = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}.get(game.headers.get("Result", "*"))
            if value is None:
                continue
            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                if ply >= 5:  # ChessDataset skips the first five plies
                    examples.append({
                        'fen': board.fen(), 'move': move_to_index(move), 'ply': ply,
                        'value': value if board.turn == chess.WHITE else -value,
                        'game': game_count,
                    })
                board.push(move)
            game_count += 1
            if max_games and game_count >= max_games:
                break
    return examples


def material_balance(board):
    """Material from the side to move's point of view (P=1 N=B=3 R=5 Q=9)."""
    return sum(PIECE_VALUES.get(p.piece_type, 0) * (1 if p.color == board.turn else -1)
               for p in board.piece_map().values())


def evaluate_checkpoint(checkpoint, pgn_path, max_games=None, batch_size=512):
    """Return a dict of metrics for the checkpoint on the held-out games."""
    examples = load_positions(pgn_path, max_games)
    train_sub, val_sub = split_by_game(SimpleNamespace(examples=examples))
    train = [examples[i] for i in train_sub.indices]
    val = [examples[i] for i in val_sub.indices]

    # Baselines that know nothing about the position in front of them
    freq = collections.Counter(e['move'] for e in train)
    random_acc = freq_hits = 0.0
    for e in val:
        legal = [move_to_index(m) for m in chess.Board(e['fen']).legal_moves]
        random_acc += 1 / len(legal)
        freq_hits += max(legal, key=lambda i: freq[i]) == e['move']

    model, _ = load_model(checkpoint, torch.device('cpu'))
    model.eval()
    top1 = top5 = unmasked = illegal = 0
    phase = collections.defaultdict(lambda: [0, 0])
    values, outcomes, material = [], [], []
    with torch.no_grad():
        for i in range(0, len(val), batch_size):
            chunk = val[i:i + batch_size]
            boards = [chess.Board(e['fen']) for e in chunk]
            x = torch.from_numpy(np.stack([board_to_tensor(b) for b in boards]))
            mask = torch.from_numpy(np.stack([legal_moves_mask(b) for b in boards]))
            logits, v = model(x)
            raw = logits.argmax(1)
            best5 = logits.masked_fill(mask == 0, -1e9).topk(5, dim=1).indices
            for j, e in enumerate(chunk):
                hit = best5[j, 0].item() == e['move']
                top1 += hit
                top5 += e['move'] in best5[j].tolist()
                unmasked += raw[j].item() == e['move']
                illegal += mask[j, raw[j]].item() == 0
                name = 'opening (ply < 20)' if e['ply'] < 20 else \
                       'middlegame (20-59)' if e['ply'] < 60 else 'endgame (60+)'
                phase[name][0] += hit
                phase[name][1] += 1
                values.append(v[j, 0].item())
                outcomes.append(e['value'])
                material.append(material_balance(boards[j]))

    n = len(val)
    values, outcomes, material = map(np.array, (values, outcomes, material))
    decisive = outcomes != 0
    return {
        'train_positions': len(train), 'val_positions': n,
        'val_games': len({e['game'] for e in val}),
        'random_legal': random_acc / n, 'most_frequent': freq_hits / n,
        'top1': top1 / n, 'top5': top5 / n, 'top1_unmasked': unmasked / n,
        'argmax_illegal': illegal / n,
        'by_phase': {k: h / c for k, (h, c) in phase.items()},
        'value_mse': float(np.mean((values - outcomes) ** 2)),
        'value_mse_always_0': float(np.mean(outcomes ** 2)),
        'value_sign_correct': float(np.mean(np.sign(values[decisive]) == outcomes[decisive]))
                              if decisive.any() else float('nan'),
        'value_material_corr': float(np.corrcoef(values, material)[0, 1])
                               if values.std() > 0 and material.std() > 0 else 0.0,
        'model': model,
    }


def smell_test(model, fen):
    board = chess.Board(fen)
    with torch.no_grad():
        logits, v = model(torch.from_numpy(board_to_tensor(board)).unsqueeze(0))
    logits = logits[0].masked_fill(torch.from_numpy(legal_moves_mask(board)) == 0, -1e9)
    top = torch.softmax(logits, 0).topk(3)
    moves = [board.san(index_to_move(i.item(), board)) for i in top.indices]
    return list(zip(moves, top.values.tolist())), v.item()


def main():
    parser = argparse.ArgumentParser(description="Evaluate a supervised checkpoint against baselines")
    parser.add_argument('checkpoint')
    parser.add_argument('--pgn', required=True, help='the PGN you trained on')
    parser.add_argument('--max-games', type=int, default=None, help='same value you gave train.py')
    a = parser.parse_args()

    r = evaluate_checkpoint(a.checkpoint, a.pgn, a.max_games)
    print(f"Held out: {r['val_positions']:,} positions from {r['val_games']} games the network never saw "
          f"(trained on {r['train_positions']:,})")

    print("\nPOLICY: does it pick the move the player actually made?")
    print(f"  random legal move          {r['random_legal']:6.1%}")
    print(f"  most common legal move     {r['most_frequent']:6.1%}   (move counts only, no board)")
    print(f"  network, best legal move   {r['top1']:6.1%}   <- must beat the line above")
    print(f"  network, in its top 5      {r['top5']:6.1%}")
    print(f"  train.py's figure          {r['top1_unmasked']:6.1%}   (argmax over all 4,672 moves; "
          f"{r['argmax_illegal']:.0%} of those picks are illegal)")
    for k in ('opening (ply < 20)', 'middlegame (20-59)', 'endgame (60+)'):
        if k in r['by_phase']:
            print(f"    {k:<20} {r['by_phase'][k]:6.1%}")

    print("\nVALUE: does it know who is winning?")
    print(f"  error if always 'even' (0) {r['value_mse_always_0']:.3f}")
    print(f"  network's error            {r['value_mse']:.3f}   <- must be lower than the line above")
    print(f"  right winner, decisive     {r['value_sign_correct']:6.1%}   (50% = coin flip)")
    print(f"  correlation with material  {r['value_material_corr']:+.2f}   (it was never told material)")

    print("\nSMELL TESTS (raw network, no search)")
    for name, fen, passing in SMELL_TESTS:
        top, value = smell_test(r['model'], fen)
        verdict = '' if passing is None else ('  PASS' if top[0][0] in passing else '  fail')
        print(f"  {name:<32} " + ", ".join(f"{m} {p:.0%}" for m, p in top)
              + f"   value {value:+.2f}{verdict}")


if __name__ == '__main__':
    main()
