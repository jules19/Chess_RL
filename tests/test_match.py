"""Tests for the engine-vs-engine match tool (cli/match.py)."""

import random

import chess

from cli.match import make_engine, play_game, summarize


def test_interval_is_honest_for_small_matches():
    mean, lo, hi, *_ = summarize([1.0] * 4)
    assert mean == 1.0 and lo < 0.6          # 4/4 does not prove 100%
    mean, lo, hi, *_ = summarize([1.0] * 11 + [0.5] + [0.0] * 8)
    assert lo < 0.5 < hi                      # 11.5/20 is inside the noise


def test_every_engine_spec_returns_legal_moves():
    board = chess.Board()
    for spec in ['random', 'material', 'minimax:depth=1',
                 'minimax:depth=1,eval=engine.evaluator:evaluate_material',
                 'mcts:sims=8', 'mcts:sims=8,rollouts=random,c=3.0,filter=0',
                 'puct:sims=8', 'puct:sims=8,c=0.5']:
        assert make_engine(spec)(board) in board.legal_moves, spec


_calls = []


def counting_evaluate(board):
    from engine.evaluator import evaluate
    _calls.append(1)
    return evaluate(board)


def test_eval_override_applies_only_to_its_own_player():
    import search.minimax as mm
    original = mm.evaluate
    a = make_engine('minimax:depth=1,eval=tests.test_match:counting_evaluate')
    b = make_engine('minimax:depth=1')
    random.seed(0)
    play_game(a, b, opening=[], max_plies=4)
    assert _calls and mm.evaluate is original


def test_adjudication_scores_a_capped_game_by_material():
    first_legal = lambda board: next(iter(board.legal_moves))
    queen_up = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
    assert play_game(first_legal, first_legal, [], max_plies=0, fen=queen_up)[0] == '1/2-1/2'
    assert play_game(first_legal, first_legal, [], max_plies=0, fen=queen_up,
                     adjudicate=3)[0] == '1-0'
    assert play_game(first_legal, first_legal, [], max_plies=0, adjudicate=3)[0] == '1/2-1/2'
