import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # 01 · Checkpoint reality check

        **Question:** what did Phase 3a actually train, and how strong is it?

        Until now the project's success criteria (policy accuracy > 35%, ≥40% vs minimax)
        were targets, never measured. This notebook measures them:

        1. What the checkpoints recorded during training
        2. What the network is made of
        3. What data it learned from
        4. How often it agrees with the moves humans played
        5. How it does in actual games

        This notebook goes with [unit 8 of the learning path](../docs/learn/08_measuring.md), which
        has the exercises. Before each section runs, **predict** the number you expect, then compare.

        Run with `marimo edit notebooks/01_checkpoint_reality_check.py --host 0.0.0.0 --port 8080`.
        """
    )
    return (mo,)


@app.cell
def _():
    import random
    import sys
    from pathlib import Path

    import chess
    import chess.pgn
    import chess.svg
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch

    # Same import convention as the rest of the project
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))

    from engine.evaluator import best_move_material
    from engine.nn_evaluator import NeuralNetworkEvaluator
    from search.minimax import best_move_minimax
    return (
        NeuralNetworkEvaluator,
        ROOT,
        best_move_material,
        best_move_minimax,
        chess,
        pd,
        plt,
        random,
        torch,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 1 · What the checkpoints recorded

    `training/train.py` saves loss and validation accuracy inside every `model_epoch_N.pt`,
    so no TensorBoard is needed.

    **Predict first:** the plan's target was 35% validation accuracy. After 5 epochs on 411
    games, what do you expect? Write a number down before scrolling.
    """)
    return


@app.cell
def _(ROOT, pd, torch):
    _rows = []
    for _path in sorted((ROOT / "checkpoints").glob("model_epoch_*.pt")):
        _ckpt = torch.load(_path, map_location="cpu", weights_only=True)
        _rows.append({
            "epoch": _ckpt["epoch"],
            "train_loss": _ckpt["train_loss"],
            "val_loss": _ckpt["val_loss"],
            "val_accuracy": _ckpt["val_accuracy"],
        })
    history = pd.DataFrame(_rows).set_index("epoch")
    history
    return (history,)


@app.cell
def _(history, plt):
    _fig, (_ax_loss, _ax_acc) = plt.subplots(1, 2, figsize=(10, 3.5))
    history[["train_loss", "val_loss"]].plot(ax=_ax_loss, marker="o", title="Loss (policy + value)")
    (history["val_accuracy"] * 100).plot(ax=_ax_acc, marker="o", title="Validation top-1 accuracy (%)")
    _ax_acc.axhline(35, color="grey", linestyle="--", label="Phase 3a target (35%)")
    _ax_acc.axhline(100 / 30, color="tab:red", linestyle=":", label="Random legal move (~3%)")
    _ax_acc.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(history, mo):
    mo.md(f"""
    Validation accuracy ends at **{history['val_accuracy'].iloc[-1]:.1%}**: about the same as
    picking a random legal move (a typical position has ~30), and an order of magnitude
    below the 35% target. Loss is still falling, so training stopped long before convergence.

    Caveat: the train/val split in `training/dataset.py` is by *position*, not by game,
    so even these numbers are flattered by positions from the same games on both sides.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2 · What the network is made of
    """)
    return


@app.cell
def _(NeuralNetworkEvaluator, ROOT):
    nn_eval = NeuralNetworkEvaluator(str(ROOT / "checkpoints" / "best_model.pt"), device="auto", verbose=False)
    model = nn_eval.model
    return model, nn_eval


@app.cell
def _(model, pd):
    _groups = {}
    for _name, _param in model.named_parameters():
        _group = _name.split(".")[0]
        if _group == "policy_head":
            _group = "policy_head." + _name.split(".")[1]
        _groups[_group] = _groups.get(_group, 0) + _param.numel()
    params = pd.Series(_groups, name="parameters").sort_values(ascending=False).to_frame()
    params["share"] = (params["parameters"] / params["parameters"].sum()).map("{:.0%}".format)
    params
    return (params,)


@app.cell
def _(mo, model, params):
    mo.md(f"""
    **{model.num_res_blocks} residual blocks × {model.num_channels} channels,
    {model.num_parameters():,} parameters.** The largest single piece is the policy head's
    fully-connected layer ({params['parameters'].iloc[0]:,} parameters, {params['share'].iloc[0]}),
    which maps a small feature map straight onto 4,672 move slots. The "thinking" part,
    the residual trunk, is small by comparison.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3 · What it learned from
    """)
    return


@app.cell
def _(ROOT, chess, pd):
    def load_games(path):
        games = []
        with open(path) as f:
            while (game := chess.pgn.read_game(f)) is not None:
                games.append(game)
        return games

    games = load_games(ROOT / "data" / "sample_lichess.pgn")
    game_info = pd.DataFrame([{
        "white_elo": int(g.headers.get("WhiteElo", 0)),
        "black_elo": int(g.headers.get("BlackElo", 0)),
        "result": g.headers.get("Result"),
        "event": g.headers.get("Event"),
        "plies": len(list(g.mainline_moves())),
    } for g in games])
    game_info.describe().round(0)
    return game_info, games


@app.cell
def _(game_info, mo):
    _avg_elo = (game_info["white_elo"] + game_info["black_elo"]).mean() / 2
    mo.md(f"""
    **{len(game_info)} games, ~{game_info['plies'].sum():,} positions.** Average player rating
    **{_avg_elo:.0f}**; the most common event is *{game_info['event'].mode()[0]}*.
    The plan called for 10,000 games from 2000+ players at slow time controls; this is about
    4% of that volume, from weaker players at blitz.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4 · Does it predict the moves humans played?

    For every position in a sample of games: rank the legal moves by the network's policy
    and check where the human move landed. Also check whether the value head's sign
    matches the eventual result.

    These are (mostly) games the network trained on, so this is an **upper bound**.
    There is no held-out set yet.

    **Predict first:** a random legal move matches the human about 1 time in 16. How much
    better than that do you expect the network to be: 2×, 5×, 10×?
    """)
    return


@app.cell
def _(mo):
    sample_size = mo.ui.slider(10, 200, value=50, step=10, label="Games to score")
    sample_size
    return (sample_size,)


@app.cell
def _(chess, games, nn_eval, pd, random, sample_size):
    _rng = random.Random(0)
    _sample = _rng.sample(games, min(sample_size.value, len(games)))

    _rows = []
    for _game in _sample:
        _result = {"1-0": 1, "0-1": -1}.get(_game.headers.get("Result"), 0)
        _board = _game.board()
        for _move in _game.mainline_moves():
            _value, _probs = nn_eval.evaluate_with_policy(_board)
            _ranked = sorted(_probs, key=_probs.get, reverse=True)
            _rows.append({
                "rank": _ranked.index(_move) + 1,
                "legal_moves": len(_ranked),
                "value_white": _value,
                "result": _result,
                "ply": _board.ply(),
            })
            _board.push(_move)
    agreement = pd.DataFrame(_rows)
    return (agreement,)


@app.cell
def _(agreement, mo, pd):
    _random_top1 = (1 / agreement["legal_moves"]).mean()
    _random_top3 = (3 / agreement["legal_moves"]).clip(upper=1).mean()
    _decisive = agreement[agreement["result"] != 0]
    _value_sign = ((_decisive["value_white"] > 0) == (_decisive["result"] > 0)).mean()

    agreement_table = pd.DataFrame({
        "network": [(agreement["rank"] == 1).mean(), (agreement["rank"] <= 3).mean(), _value_sign],
        "random baseline": [_random_top1, _random_top3, 0.5],
    }, index=["top-1 move match", "top-3 move match", "value sign = result"]).map("{:.1%}".format)

    mo.vstack([
        mo.md(f"Scored **{len(agreement):,} positions**."),
        agreement_table,
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Look at individual positions

    Green arrow: the move the human played. Blue arrows: the network's top 3.
    """)
    return


@app.cell
def _(games, mo):
    game_pick = mo.ui.number(0, len(games) - 1, value=0, label="Game #")
    game_pick
    return (game_pick,)


@app.cell
def _(game_pick, games, mo):
    picked_moves = list(games[game_pick.value].mainline_moves())
    ply_pick = mo.ui.slider(0, len(picked_moves) - 1, value=10, label="Ply", full_width=True)
    ply_pick
    return picked_moves, ply_pick


@app.cell
def _(chess, game_pick, games, mo, nn_eval, picked_moves, ply_pick):
    _board = games[game_pick.value].board()
    for _move in picked_moves[: ply_pick.value]:
        _board.push(_move)
    _human = picked_moves[ply_pick.value]
    _value, _probs = nn_eval.evaluate_with_policy(_board)
    _top = sorted(_probs, key=_probs.get, reverse=True)[:3]

    _arrows = [chess.svg.Arrow(m.from_square, m.to_square, color="#3b82f680") for m in _top]
    _arrows.append(chess.svg.Arrow(_human.from_square, _human.to_square, color="#16a34acc"))
    _svg = chess.svg.board(_board, arrows=_arrows, size=360)

    _lines = "\n".join(
        f"| {_board.san(m)} | {_probs[m]:.1%} |" for m in _top
    )
    mo.hstack([
        mo.Html(_svg),
        mo.md(f"""
    **{'White' if _board.turn else 'Black'} to move.** Human played **{_board.san(_human)}**
    (network gave it {_probs[_human]:.1%}).

    Value head (White's view): **{_value:+.2f}**

    | Network top 3 | prob |
    |---|---|
    {_lines}
        """),
    ], justify="start", gap=2)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5 · How does it play?

    The network plays by picking its highest-policy legal move (no search).
    Each engine is deterministic, so every pair of games starts from a different random
    4-ply opening, played once with each colour.

    This is the slow part, so it only runs when you press the button.

    **Predict first:** what score do you expect against the *random* mover? Against minimax
    at depth 1? Commit to both numbers, then press the button.
    """)
    return


@app.cell
def _(mo):
    openings = mo.ui.slider(2, 20, value=5, label="Openings (2 games each)")
    run_matches = mo.ui.run_button(label="Play matches")
    mo.hstack([openings, run_matches], justify="start")
    return openings, run_matches


@app.cell
def _(best_move_material, best_move_minimax, chess, nn_eval, random):
    def nn_policy(board):
        return nn_eval.get_best_move(board)

    def random_mover(board):
        return random.choice(list(board.legal_moves))

    opponents = {
        "random": random_mover,
        "material (1-ply)": best_move_material,
        "minimax depth 1": lambda b: best_move_minimax(b, depth=1),
        "minimax depth 2": lambda b: best_move_minimax(b, depth=2),
    }

    def play_from(opening, white, black, max_plies=200):
        """Play out a game from an opening; returns the result string and ply count."""
        board = chess.Board()
        for move in opening:
            board.push(move)
        while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
            board.push((white if board.turn == chess.WHITE else black)(board))
        outcome = board.outcome(claim_draw=True)
        return (outcome.result() if outcome else "1/2-1/2"), board.ply()

    def random_opening(rng, plies=4):
        board = chess.Board()
        for _ in range(plies):
            board.push(rng.choice(list(board.legal_moves)))
        return list(board.move_stack)
    return nn_policy, opponents, play_from, random_opening


@app.cell
def _(
    mo,
    nn_policy,
    openings,
    opponents,
    pd,
    play_from,
    random,
    random_opening,
    run_matches,
):
    mo.stop(not run_matches.value, mo.md("*Press **Play matches** to run.*"))

    _rng = random.Random(42)
    _openings = [random_opening(_rng) for _ in range(openings.value)]

    _rows = []
    with mo.status.progress_bar(total=len(opponents) * len(_openings) * 2, title="Playing") as _bar:
        for _name, _opponent in opponents.items():
            _score = {"win": 0, "draw": 0, "loss": 0}
            for _opening in _openings:
                for _nn_white in (True, False):
                    _white, _black = (nn_policy, _opponent) if _nn_white else (_opponent, nn_policy)
                    _result, _ = play_from(_opening, _white, _black)
                    _nn_won = (_result == "1-0") == _nn_white
                    if _result == "1/2-1/2":
                        _score["draw"] += 1
                    elif _nn_won:
                        _score["win"] += 1
                    else:
                        _score["loss"] += 1
                    _bar.update()
            _games = sum(_score.values())
            _rows.append({"opponent": _name, **_score,
                          "NN score": (_score["win"] + 0.5 * _score["draw"]) / _games})
    match_results = pd.DataFrame(_rows).set_index("opponent")
    match_results.style.format({"NN score": "{:.0%}"})
    return


@app.cell
def _(mo):
    mo.md("""
    ## Takeaways

    On the checkpoint measured on 2026-10-27 (numbers also in `STATUS.md`):
    the network ranks the human move first 13% of the time (random: 6%), its value head is
    barely better than a coin flip, and with no search it won none of 40 games against
    random, material-only and minimax opponents. The architecture works; the training data
    and procedure are the bottleneck.

    What would actually improve the network, roughly in order of payoff:
    1. **More and better data:** thousands of games from stronger players, not 411 blitz games.
    2. **Split train/val by game**, so validation numbers mean something.
    3. **Train to convergence:** loss was still falling at epoch 5.
    4. Then measure again with this notebook.
    """)
    return


if __name__ == "__main__":
    app.run()
