"""
Engine-vs-engine matches with an honest error bar.

Every module asks "play N games of A vs B". This is the tool for it. Each
engine is a spec string, colours alternate, and each random opening is played
twice with colours swapped, so neither side gets the lucky opening. The
report gives the score, a 95% interval and the Elo difference that implies.

Engine specs (options after ':' separated by ','):
    random                              uniform random legal move
    material                            greedy 1-ply, material only
    minimax:depth=3                     alpha-beta (Module 2)
    minimax:depth=2,eval=pkg.mod:fn     ...with a different evaluate(board) function
    mcts:sims=200                       rollout MCTS (Module 3)
    mcts:sims=200,rollouts=random,c=3.0,filter=0
    puct:sims=100                       PUCT with uniform priors (no knowledge)
    puct:sims=100,net=checkpoints/best_model.pt,c=1.5
    puct:sims=100,net=material          PUCT with material_policy_value (Module 5 exercise)

Examples:
    python3 cli/match.py minimax:depth=2 random --games 10
    python3 cli/match.py mcts:sims=50 mcts:sims=50,rollouts=random --games 6 \\
                         --max-plies 120 --adjudicate 3
    python3 cli/match.py puct:sims=64,net=selfplay_checkpoints/champion.pt \\
                         puct:sims=64,net=checkpoints/best_model.pt --games 20
"""

import argparse
import importlib
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import chess.pgn


def _parse_spec(spec):
    name, _, rest = spec.partition(':')
    opts = dict(kv.split('=', 1) for kv in rest.split(',') if kv)
    return name, opts


def make_engine(spec):
    """Turn a spec string into a move function: board -> move."""
    name, o = _parse_spec(spec)

    if name == 'random':
        return lambda board: random.choice(list(board.legal_moves))

    if name == 'material':
        from engine.evaluator import best_move_material
        return best_move_material

    if name == 'minimax':
        import search.minimax as mm
        depth = int(o.get('depth', 3))
        if 'eval' not in o:
            return lambda board: mm.best_move_minimax(board, depth=depth)
        module, _, fn = o['eval'].partition(':')
        eval_fn = getattr(importlib.import_module(module), fn)

        def move(board):
            # minimax looks up `evaluate` in its own module at call time, so
            # swapping it for the duration of this player's move gives this
            # player a different evaluator without touching the other one.
            saved = mm.evaluate
            mm.evaluate = eval_fn
            try:
                return mm.best_move_minimax(board, depth=depth)
            finally:
                mm.evaluate = saved
        return move

    if name == 'mcts':
        from search.mcts import best_move_mcts
        sims = int(o.get('sims', 200))
        use_eval = o.get('rollouts', 'evaluator') != 'random'
        c = float(o.get('c', 1.41))
        filt = o.get('filter', '1') != '0'
        return lambda board: best_move_mcts(board, simulations=sims, use_evaluator=use_eval,
                                            exploration_constant=c, filter_blunders=filt)

    if name == 'puct':
        from search import puct
        sims = int(o.get('sims', 100))
        c = float(o.get('c', 1.5))
        net = o.get('net', 'uniform')
        if net == 'uniform':
            fn = puct.uniform_policy_value
        elif net == 'material':
            fn = puct.material_policy_value  # exists once you've done Module 5, exercise 1
        else:
            from net.model import load_model
            model, _ = load_model(net)
            model.eval()
            fn = puct.network_policy_value(model)
        return lambda board: puct.best_move_puct(board, fn, simulations=sims, c_puct=c)

    raise SystemExit(f"Unknown engine spec '{spec}' (see python3 cli/match.py --help)")


PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material(board):
    """White's material minus Black's, in pawns."""
    return sum(PIECE_VALUES.get(p.piece_type, 0) * (1 if p.color == chess.WHITE else -1)
               for p in board.piece_map().values())


def play_game(white, black, opening, max_plies, adjudicate=None, fen=chess.STARTING_FEN):
    """
    Play one game. A game that reaches max_plies is a draw, unless `adjudicate`
    is set and one side is at least that many pawns of material ahead: then
    it's scored as a win for that side, the way engine tournaments end games
    that would otherwise run forever.
    """
    board = chess.Board(fen)
    for move in opening:
        board.push(move)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        board.push((white if board.turn == chess.WHITE else black)(board))
    result = board.result(claim_draw=True)
    if result == '*' and adjudicate is not None and abs(material(board)) >= adjudicate:
        return ('1-0' if material(board) > 0 else '0-1'), board
    return (result if result != '*' else '1/2-1/2'), board


def random_opening(plies, rng):
    board = chess.Board()
    moves = []
    for _ in range(plies):
        move = rng.choice(list(board.legal_moves))
        board.push(move)
        moves.append(move)
    return moves


def summarize(scores):
    """
    Score fraction, its 95% interval, and the Elo difference both imply.

    The interval is the Wilson score interval for a binomial proportion,
    counting a draw as half a win. Its rough size is 1.96 * sqrt(p(1-p)/n)
    either side: with 20 games near 50% that is about +-22 percentage points
    (+-4.4 points out of 20), which is why 11.5/20 proves nothing. Unlike the
    plain formula, Wilson stays honest at 4/4 (it does not claim 100%-100%).
    Draws make the true interval somewhat narrower, so this errs on the
    cautious side.
    """
    n = len(scores)
    mean = sum(scores) / n
    z = 1.96
    centre = (mean + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(mean * (1 - mean) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)

    def elo(s):
        if s <= 0:
            return -math.inf
        if s >= 1:
            return math.inf
        return -400 * math.log10(1 / s - 1)
    return mean, lo, hi, elo(mean), elo(lo), elo(hi)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('engine_a')
    parser.add_argument('engine_b')
    parser.add_argument('--games', type=int, default=10, help='total games (rounded up to even)')
    parser.add_argument('--opening-plies', type=int, default=2,
                        help='random opening plies, each opening played with both colours (0 = start position)')
    parser.add_argument('--max-plies', type=int, default=200, help='game length cap, scored as a draw')
    parser.add_argument('--adjudicate', type=int, default=None, metavar='PAWNS',
                        help='at the cap, award the game to a side this many pawns of material ahead')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pgn', type=str, default=None, help='save the games here')
    args = parser.parse_args()

    random.seed(args.seed)
    rng = random.Random(args.seed)
    a, b = make_engine(args.engine_a), make_engine(args.engine_b)
    pairs = (args.games + 1) // 2
    scores, wins, draws, losses = [], 0, 0, 0
    pgn_file = open(args.pgn, 'w') if args.pgn else None
    start = time.time()

    print(f"A = {args.engine_a}\nB = {args.engine_b}\n{pairs * 2} games\n")
    for pair in range(pairs):
        opening = random_opening(args.opening_plies, rng)
        for a_is_white in (True, False):
            white, black = (a, b) if a_is_white else (b, a)
            result, board = play_game(white, black, opening, args.max_plies, args.adjudicate)
            capped = board.ply() >= args.max_plies and not board.is_game_over(claim_draw=True)
            s = {'1-0': 1.0, '0-1': 0.0, '1/2-1/2': 0.5}[result]
            s = s if a_is_white else 1 - s
            scores.append(s)
            wins += s == 1
            draws += s == 0.5
            losses += s == 0
            print(f"  game {len(scores):>3}: A as {'White' if a_is_white else 'Black'}  "
                  f"{result:<7}  ({board.ply()} plies{', capped' if capped else ''})  "
                  f"running A score {sum(scores)}/{len(scores)}")
            if pgn_file:
                game = chess.pgn.Game.from_board(board)
                game.headers['White'] = args.engine_a if a_is_white else args.engine_b
                game.headers['Black'] = args.engine_b if a_is_white else args.engine_a
                print(game, file=pgn_file, end='\n\n')
    if pgn_file:
        pgn_file.close()

    mean, lo, hi, e, elo_lo, elo_hi = summarize(scores)
    print(f"\nA: +{wins} ={draws} -{losses}  score {sum(scores)}/{len(scores)} = {mean:.1%}"
          f"   (95% interval {lo:.0%} to {hi:.0%})")
    print(f"Elo difference A-B: {e:+.0f}   (95% interval {elo_lo:+.0f} to {elo_hi:+.0f})")
    if lo < 0.5 < hi:
        print("The interval contains 50%: this match cannot tell the engines apart. Play more games.")
    print(f"Time: {time.time() - start:.0f}s")


if __name__ == '__main__':
    main()
