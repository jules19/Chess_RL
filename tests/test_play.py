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


def test_last_move_is_named_in_standard_notation():
    board = chess.Board()
    move = chess.Move.from_uci('g1f3')
    board.push(move)
    assert play.last_move_text(board, move) == 'Nf3'
    assert board.move_stack == [move]          # the board is not changed
    # Not the latest move: fall back to square-to-square notation
    assert play.last_move_text(board, chess.Move.from_uci('e2e4')) == 'e2e4'


ROWS = [{'vs_perfect': 0.0, 'coverage': 0.0}, {'vs_perfect': 89.0, 'coverage': 32.3}]
FROZEN = [{'vs_perfect': 0.0, 'coverage': 0.0}, {'vs_perfect': 50.0, 'coverage': 0.5}]


def test_learning_summary_uses_the_actual_numbers():
    text = ' '.join(play.summarise_learning(ROWS))
    assert 'avoided losing 89% of test games, up from 0%' in text
    assert 'without search' in text and 'draw' in text
    assert '32% of the reachable positions' in text
    compared = play.compare_exploration(ROWS, FROZEN)
    assert '32.3%' in compared[0] and '89%' in compared[0]
    assert '0.5%' in compared[1] and '50%' in compared[1]


def _script(monkeypatch, answers):
    answers = iter(answers)
    monkeypatch.setattr('builtins.input', lambda _='': next(answers))
    monkeypatch.setattr(play.sys, 'argv', ['play.py'])


def test_menu_learn_then_exploration_experiment_then_finish(monkeypatch, capsys):
    import selfplay.tictactoe_lab as lab
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return FROZEN if kwargs.get('root_noise') is False else ROWS
    monkeypatch.setattr(lab, 'run', fake_run)
    # menu: 3 = learn; Enter to run; 1 = exploration experiment; Enter; q = finish
    _script(monkeypatch, ['3', '', '1', '', 'q'])
    play.main()
    out = capsys.readouterr().out
    assert calls == [{'seed': 0}, {'seed': 0, 'root_noise': False, 'temperature_moves': 0}]
    assert 'avoided losing 89%' in out and 'Without exploration' in out
    assert out.index('Predict: what happens') < out.index('Without exploration')


def test_menu_return_loops_back(monkeypatch):
    import selfplay.tictactoe_lab as lab
    monkeypatch.setattr(lab, 'run', lambda **kw: ROWS)
    # learn, Enter, m = back to menu, q = quit from the menu
    _script(monkeypatch, ['3', '', 'm', 'q'])
    play.main()                                  # returns instead of raising StopIteration


def test_direct_command_does_not_ask_what_next(monkeypatch, capsys):
    import selfplay.tictactoe_lab as lab
    monkeypatch.setattr(lab, 'run', lambda **kw: ROWS)
    answers = iter([''])                         # only the "Press Enter" prompt
    monkeypatch.setattr('builtins.input', lambda _='': next(answers))
    monkeypatch.setattr(play.sys, 'argv', ['play.py', 'learn'])
    play.main()
    assert 'Return to menu' not in capsys.readouterr().out
