"""
AlphaZero in miniature: the Module 6 loop on tic-tac-toe, small enough to watch learn.

Self-play on chess at CPU scale proves the plumbing but rarely shows learning:
the network is 1.8M parameters and the arena gate is mostly noise. This lab
runs the SAME algorithm on a game small enough to finish in seconds,
and it can be scored against perfect play, because tic-tac-toe is solved.

What is reused unchanged from the chess engine:
    search.puct.puct_search       - the PUCT search, line for line
    search.puct.select_move       - temperature-based move selection
    search.puct.visits_to_policy  - visit counts -> policy training target
    selfplay.replay_buffer        - the replay buffer window

puct_search only ever calls four board methods: is_game_over(), copy(),
push(move) and is_checkmate(). TicTacToeBoard below provides those four, so
the search has no idea it isn't playing chess. That is the answer to "what
would I change to run this loop on another game?": the rules and the
representation, not the algorithm.

What is different, on purpose:
    The "network" is a TABLE: one (prior vector, value) entry per position,
    updated by moving it a step toward each training target. There are only
    5,478 legal positions, so a table can hold them all. Chess has ~10^44,
    which is why chess needs a network: it must GENERALISE to positions it
    has never seen. A table cannot generalise at all - an unseen position
    keeps its uniform prior and zero value. That limitation is visible in
    the "coverage" column of the learning curve.

The RL view (course Module 6 names each of these):
    agent        - PUCT search guided by the table, playing both sides
    environment  - the rules + the opponent's replies (the agent itself)
    reward       - 0 on every move; +1 / 0 / -1 only when the game ends
    policy eval  - value[s] moves toward the game outcome z (a Monte Carlo target)
    policy impr. - search(table) plays better than table alone; the policy
                   entry moves toward the search's visit distribution
    exploration  - Dirichlet noise at the root + temperature for early moves

Run (about 7 seconds on a laptop CPU):
    python3 selfplay/tictactoe_lab.py
    python3 selfplay/tictactoe_lab.py --no-root-noise --temperature-moves 0   # sabotage
    python3 selfplay/tictactoe_lab.py --td                                    # TD value targets
    python3 selfplay/tictactoe_lab.py --csv runs/ttt.csv                      # save the curve
"""

import argparse
import copy
import csv
import os
import random
import sys
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.puct import puct_search, select_move, visits_to_policy
from selfplay.replay_buffer import ReplayBuffer

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


# ---------------------------------------------------------------------------
# The environment: rules, with the four methods puct_search needs
# ---------------------------------------------------------------------------

class TicTacToeBoard:
    """
    Board with the same interface puct_search uses on chess.Board.

    cells: 9 ints, +1 = X, -1 = O, 0 = empty (index 0 is the top-left square)
    turn:  +1 when X is to move, -1 when O is to move
    Moves are square indices 0-8.
    """

    def __init__(self, cells=None, turn=1):
        self.cells = list(cells) if cells is not None else [0] * 9
        self.turn = turn

    @property
    def legal_moves(self) -> List[int]:
        if self.winner() != 0:
            return []
        return [i for i, c in enumerate(self.cells) if c == 0]

    def push(self, move: int):
        assert self.cells[move] == 0, f"square {move} is occupied"
        self.cells[move] = self.turn
        self.turn = -self.turn

    def copy(self) -> 'TicTacToeBoard':
        return TicTacToeBoard(self.cells, self.turn)

    def winner(self) -> int:
        """+1 if X has three in a row, -1 if O has, 0 otherwise."""
        for a, b, c in LINES:
            if self.cells[a] != 0 and self.cells[a] == self.cells[b] == self.cells[c]:
                return self.cells[a]
        return 0

    def is_game_over(self, claim_draw: bool = False) -> bool:
        return self.winner() != 0 or all(c != 0 for c in self.cells)

    def is_checkmate(self) -> bool:
        """The chess name for 'the side to move has lost': the last mover made a line."""
        return self.winner() == -self.turn

    def key(self) -> Tuple[int, ...]:
        """Position as seen by the side to move (+1 = my stones), so X and O share a table."""
        return tuple(c * self.turn for c in self.cells)

    def __str__(self):
        sym = {1: 'X', -1: 'O', 0: '.'}
        return '\n'.join(' '.join(sym[c] for c in self.cells[r:r + 3]) for r in (0, 3, 6))


# ---------------------------------------------------------------------------
# Perfect play, for scoring only (the agent never sees it)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def solve(key: Tuple[int, ...]) -> int:
    """True game value for the side to move under perfect play: +1 win, 0 draw, -1 loss."""
    board = TicTacToeBoard(key, 1)  # keys are side-to-move relative, so "me" is +1
    if board.is_checkmate():
        return -1
    if board.is_game_over():
        return 0
    best = -1
    for m in board.legal_moves:
        child = board.copy()
        child.push(m)
        best = max(best, -solve(child.key()))
    return best


def optimal_moves(board: TicTacToeBoard) -> List[int]:
    best = solve(board.key())
    out = []
    for m in board.legal_moves:
        child = board.copy()
        child.push(m)
        if -solve(child.key()) == best:
            out.append(m)
    return out


def reachable_positions() -> List[TicTacToeBoard]:
    """Every non-terminal position reachable from the empty board."""
    seen, out, stack = set(), [], [TicTacToeBoard()]
    while stack:
        b = stack.pop()
        if b.key() in seen or b.is_game_over():
            continue
        seen.add(b.key())
        out.append(b)
        for m in b.legal_moves:
            c = b.copy()
            c.push(m)
            stack.append(c)
    return out


# ---------------------------------------------------------------------------
# The "network": a table of (prior over 9 squares, value)
# ---------------------------------------------------------------------------

class TablePolicyValue:
    """
    A lookup-table stand-in for the policy-value network.

    Unseen positions get the same answer an untrained network gives: a
    uniform prior and value 0. Training moves an entry a fraction `lr` of
    the way toward its target, which is gradient descent on squared error
    for a model with one free parameter per entry.
    """

    def __init__(self):
        self.policy: Dict[Tuple[int, ...], np.ndarray] = {}
        self.value: Dict[Tuple[int, ...], float] = {}

    def __call__(self, board: TicTacToeBoard):
        """policy_value_fn(board) -> (priors, value for side to move), as puct_search expects."""
        legal = board.legal_moves
        p = self.policy.get(board.key())
        if p is None:
            priors = {m: 1.0 / len(legal) for m in legal}
        else:
            total = sum(p[m] for m in legal) or 1.0
            priors = {m: p[m] / total for m in legal}
        return priors, self.value.get(board.key(), 0.0)

    def train(self, states, policies, values, lr: float):
        for s, pi, z in zip(states, policies, values):
            key = tuple(int(x) for x in s)
            p = self.policy.get(key, np.full(9, 1 / 9, dtype=np.float32))
            self.policy[key] = p + lr * (pi - p)
            v = self.value.get(key, 0.0)
            self.value[key] = v + lr * (float(z) - v)

    def raw_move(self, board: TicTacToeBoard) -> int:
        """Play the prior's favourite move with no search: what the 'network' alone knows."""
        priors, _ = self(board)
        return max(priors, key=priors.get)


# ---------------------------------------------------------------------------
# Self-play, arena and evaluation
# ---------------------------------------------------------------------------

def selfplay_game(net, simulations, temperature_moves, root_noise, rng, td=False):
    """
    One game of the agent against itself. Returns (state, pi, value target) examples.

    Value target, Monte Carlo (default): the game result z, from the side to
    move's point of view. A real return; noisy, but no guessing.
    Value target, TD (td=True): minus the table's own value of the NEXT
    position (the opponent's view, flipped), and the real result for the last
    move. It bootstraps from the agent's own estimates: less noise, some bias.
    """
    board = TicTacToeBoard()
    history = []
    while not board.is_game_over():
        _, visits = puct_search(board, net, simulations=simulations,
                                add_root_noise=root_noise, rng=rng)
        target = visits_to_policy(visits, temperature=1.0)
        pi = np.zeros(9, dtype=np.float32)
        for m, p in target.items():
            pi[m] = p
        history.append((np.array(board.key(), dtype=np.float32), pi, board.turn))
        temperature = 1.0 if len(history) <= temperature_moves else 0.0
        board.push(select_move(visits, temperature=temperature, rng=rng))
    winner = board.winner()  # the reward arrives only now, at the end
    if not td:
        return [(s, pi, float(winner * turn)) for s, pi, turn in history]
    examples = []
    for i, (s, pi, turn) in enumerate(history):
        if i + 1 < len(history):
            next_key = tuple(int(x) for x in history[i + 1][0])
            target = -net.value.get(next_key, 0.0)
        else:
            target = float(winner * turn)
        examples.append((s, pi, target))
    return examples


def search_move(net, board, simulations):
    _, visits = puct_search(board, net, simulations=simulations)
    return select_move(visits, temperature=0.0)


def play(x_move, o_move) -> int:
    """Play one game; returns the winner (+1 X, -1 O, 0 draw)."""
    board = TicTacToeBoard()
    while not board.is_game_over():
        board.push((x_move if board.turn == 1 else o_move)(board))
    return board.winner()


def arena(candidate, champion, simulations, games, rng) -> float:
    """Candidate's score vs champion, colours alternated; one random opening move for variety."""
    score = 0.0
    for g in range(games):
        board = TicTacToeBoard()
        board.push(rng.randrange(9))
        cand_is_x = g % 2 == 0
        while not board.is_game_over():
            to_move_is_cand = (board.turn == 1) == cand_is_x
            net = candidate if to_move_is_cand else champion
            board.push(search_move(net, board, simulations))
        w = board.winner()
        score += 0.5 if w == 0 else float((w == 1) == cand_is_x)
    return score / games


def evaluate(net, positions, rng, games=100) -> Dict[str, float]:
    """
    Score the table against ground truth. These numbers are the learning curve.

    value_mse  : mean squared error of value[s] vs the true game value, over
                 all reachable positions (policy evaluation quality)
    best_move  : % of reachable positions where the prior's favourite move
                 is a perfect-play move (what the 'network' alone knows)
    coverage   : % of reachable positions the table has ever been trained on
    vs_perfect : raw policy (no search) vs a perfect player, % of games NOT
                 lost (a perfect player can't be beaten; 100% = always draws)
    vs_random  : raw policy vs a random mover, score %
    """
    se = hit = covered = 0
    for b in positions:
        k = b.key()
        se += (net.value.get(k, 0.0) - solve(k)) ** 2
        hit += net.raw_move(b) in optimal_moves(b)
        covered += k in net.value

    def perfect(b):
        return rng.choice(optimal_moves(b))

    def randmove(b):
        return rng.choice(b.legal_moves)

    not_lost = rand_score = 0.0
    for g in range(games):
        if g % 2 == 0:
            w = play(net.raw_move, perfect); not_lost += w != -1
            w = play(net.raw_move, randmove); rand_score += {1: 1, 0: 0.5, -1: 0}[w]
        else:
            w = play(perfect, net.raw_move); not_lost += w != 1
            w = play(randmove, net.raw_move); rand_score += {-1: 1, 0: 0.5, 1: 0}[w]
    n = len(positions)
    return {'value_mse': se / n, 'best_move': 100 * hit / n, 'coverage': 100 * covered / n,
            'vs_perfect': 100 * not_lost / games, 'vs_random': 100 * rand_score / games}


def run(iterations=20, games_per_iter=100, simulations=25, temperature_moves=2,
        root_noise=True, buffer_capacity=4000, train_steps=20, batch_size=256,
        lr=0.3, arena_games=20, promote_threshold=0.55, gate=False, td=False,
        seed=0, csv_path=None, verbose=True) -> List[Dict[str, float]]:
    """Run the loop; returns one metrics row per iteration (iteration 0 = untrained)."""
    rng = random.Random(seed)
    positions = reachable_positions()
    champion = TablePolicyValue()
    buffer = ReplayBuffer(capacity=buffer_capacity)
    rows = []

    def record(it, promoted, arena_score):
        row = {'iteration': it, **evaluate(champion, positions, random.Random(seed + it)),
               'buffer': len(buffer), 'arena': arena_score, 'promoted': promoted}
        rows.append(row)
        if verbose:
            print(f"{it:>4} {row['value_mse']:>9.3f} {row['best_move']:>9.1f} "
                  f"{row['coverage']:>9.1f} {row['vs_perfect']:>10.0f} {row['vs_random']:>9.0f} "
                  f"{row['buffer']:>7} {'' if arena_score is None else f'{arena_score:.0%}':>6} "
                  f"{'yes' if promoted else '':>4}")

    if verbose:
        print(f"Tic-tac-toe self-play: {iterations} iterations x {games_per_iter} games, "
              f"{simulations} sims/move, root noise {'on' if root_noise else 'OFF'}, "
              f"temperature for {temperature_moves} moves, "
              f"{'TD' if td else 'Monte Carlo'} value targets")
        print(f"{'iter':>4} {'value_mse':>9} {'best_move':>9} {'coverage':>9} "
              f"{'vs_perfect':>10} {'vs_random':>9} {'buffer':>7} {'arena':>6} {'prom':>4}")
    record(0, False, None)

    for it in range(1, iterations + 1):
        for _ in range(games_per_iter):                       # 1. generate
            buffer.add_game(selfplay_game(champion, simulations, temperature_moves,
                                          root_noise, rng, td=td))  # 2. store
        candidate = copy.deepcopy(champion)                   # 3. train
        for _ in range(train_steps):
            candidate.train(*buffer.sample(batch_size, rng=rng), lr=lr)
        if gate:                                              # 4. gate
            score = arena(candidate, champion, simulations, arena_games, rng)
            promoted = score > promote_threshold
        else:
            score, promoted = None, True
        if promoted:
            champion = candidate
        record(it, promoted, score)

    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if verbose:
            print(f"Learning curve saved to {csv_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--games-per-iter', type=int, default=100)
    parser.add_argument('--simulations', type=int, default=25)
    parser.add_argument('--temperature-moves', type=int, default=2,
                        help='moves per game played with temperature 1 (0 = always greedy)')
    parser.add_argument('--no-root-noise', action='store_true',
                        help='turn off Dirichlet noise at the search root')
    parser.add_argument('--buffer-capacity', type=int, default=4000)
    parser.add_argument('--train-steps', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.3)
    parser.add_argument('--arena-games', type=int, default=20)
    parser.add_argument('--promote-threshold', type=float, default=0.55)
    parser.add_argument('--gate', action='store_true',
                        help='promote only if the candidate beats the champion in an arena match')
    parser.add_argument('--td', action='store_true',
                        help='bootstrapped TD value targets instead of the game result')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--csv', type=str, default=None, help='write the learning curve here')
    a = parser.parse_args()
    run(iterations=a.iterations, games_per_iter=a.games_per_iter, simulations=a.simulations,
        temperature_moves=a.temperature_moves, root_noise=not a.no_root_noise,
        buffer_capacity=a.buffer_capacity, train_steps=a.train_steps, lr=a.lr,
        arena_games=a.arena_games, promote_threshold=a.promote_threshold,
        gate=a.gate, td=a.td, seed=a.seed, csv_path=a.csv)


if __name__ == '__main__':
    main()
