"""Learners can use familiar notation and navigate without starting a long game."""
import chess
import pytest
from cli import play


@pytest.mark.parametrize('text,uci', [('e4','e2e4'), ('e2e4','e2e4'), ('Nf3','g1f3'), ('  e4  ','e2e4')])
def test_move_formats(text, uci):
    board = chess.Board()
    assert play.parse_human_move(board, text).uci() == uci
    assert board.ply() == 0


@pytest.mark.parametrize('text', ['e5','e2e5','Nf6','--','0000','nonsense'])
def test_invalid_or_null_moves_do_not_mutate_board(text):
    board = chess.Board()
    with pytest.raises(ValueError):
        play.parse_human_move(board, text)
    assert board.fen() == chess.STARTING_FEN


def test_special_moves_and_ambiguity():
    castle = chess.Board('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1')
    assert play.parse_human_move(castle, 'O-O').uci() == 'e1g1'
    promotion = chess.Board('7k/P7/8/8/8/8/8/4K3 w - - 0 1')
    assert play.parse_human_move(promotion, 'a8=Q+').uci() == 'a7a8q'
    assert play.parse_human_move(promotion, 'a7a8n').uci() == 'a7a8n'
    ambiguous = chess.Board('4k3/8/8/8/8/8/8/1N2KN2 w - - 0 1')
    with pytest.raises(ValueError):
        play.parse_human_move(ambiguous, 'Nd2')
    assert play.parse_human_move(ambiguous, 'Nbd2').uci() == 'b1d2'


@pytest.mark.parametrize('choices,expected', [(['q'],None), (['bad','q'],None),
    (['1','1'],'human'), (['2'],'think'), (['3'],'learn'), (['4'],'course'),
    (['a','minimax'],'minimax'), (['1','b','q'],None)])
def test_menu_choices(monkeypatch, choices, expected):
    answers=iter(choices)
    monkeypatch.setattr('builtins.input', lambda _: next(answers))
    assert play.choose_mode() == expected
