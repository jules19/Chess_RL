"""Protect the teaching claim, not exact floating-point output or move ties."""

import hashlib

import chess
import pytest

torch = pytest.importorskip("torch")

from selfplay.mind_change_lab import (
    BAIT, CHECKPOINT, FEN, REPLY, SAFE_FEN, SIMILAR_FEN,
    policy_distance, run_lab,
)


def has_mate_in_one(board):
    for move in board.legal_moves:
        following = board.copy()
        following.push(move)
        if following.is_checkmate():
            return True
    return False


def test_prepared_positions_really_distinguish_mate_from_escape():
    for fen, expected_mate in ((FEN, True), (SIMILAR_FEN, True), (SAFE_FEN, False)):
        board = chess.Board(fen)
        assert board.is_valid()
        assert BAIT in board.legal_moves
        assert board.piece_at(BAIT.to_square).piece_type == chess.PAWN
        board.push(BAIT)
        assert REPLY in board.legal_moves
        board.push(REPLY)
        assert board.is_checkmate() == expected_mate


def test_shipped_lab_changes_decision_and_fits_search_without_overwriting(capsys):
    original_hash = hashlib.sha256(CHECKPOINT.read_bytes()).digest()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        result = run_lab()
    finally:
        torch.set_num_threads(previous_threads)
    assert hashlib.sha256(CHECKPOINT.read_bytes()).digest() == original_hash
    assert result["before"][BAIT] > 0.9
    assert max(result["before"], key=result["before"].get) == BAIT
    first, last = result["snapshots"][0], result["snapshots"][-1]
    assert first["best"] == BAIT
    assert last["best"] != BAIT
    assert last["mate"] and last["line"] == ["Rxa7", "Re1#"]
    for row in result["snapshots"]:
        assert sum(row["visits"].values()) == row["budget"]
    # A different preference is only useful if it avoids the demonstrated trap.
    for move in (last["best"], max(result["after"], key=result["after"].get)):
        board = chess.Board(FEN)
        board.push(move)
        assert not has_mate_in_one(board)
    assert policy_distance(result["after"], result["target"]) < 0.1
    assert policy_distance(result["after"], result["target"]) < policy_distance(result["before"], result["target"]) / 2
    assert result["after"][BAIT] < 0.3
    # These measurements are probes, not a requirement that generalisation succeeds.
    assert [probe["mate"] for probe in result["probes"]] == [True, False]
    output = capsys.readouterr().out
    assert "not an increase in overall chess strength" in output
