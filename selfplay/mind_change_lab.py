"""Watch search revise a move preference, then write it back into a network.

The bundled checkpoint is a deliberately biased teaching fixture, not a model
trained on human games. See assets/mind_change/README.md for its full recipe.
"""

import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import torch

from net.encoding import NUM_MOVES, board_to_tensor, move_to_index
from net.model import create_model, load_model
from search.puct import network_policy_value, puct_search, visits_to_policy
from selfplay.train_loop import soft_target_loss

FEN = "4r1k1/p4ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
SIMILAR_FEN = "4r2k/p4ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
SAFE_FEN = "4r1k1/p4ppp/8/8/8/6P1/5P1P/R5K1 w - - 0 1"
BAIT = chess.Move.from_uci("a1a7")
REPLY = chess.Move.from_uci("e8e1")
BUDGETS = (10, 25, 50, 100, 200)
CHECKPOINT = Path(__file__).resolve().parent / "assets/mind_change/early.pt"


def encode(board):
    return torch.from_numpy(board_to_tensor(board)).unsqueeze(0)


def build_checkpoint(path):
    """Rebuild the prepared learner using only the escape-square position.

    Teach one capture with 48 supervised updates; the value target of zero is
    a preparation convention, not an established chess evaluation. All weights
    train, with BatchNorm statistics held fixed for this single-example lab.
    """
    torch.manual_seed(7)
    model = create_model(num_res_blocks=1, num_channels=8).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    state = encode(chess.Board(SAFE_FEN))
    target = torch.tensor([move_to_index(BAIT)])
    for _ in range(48):
        logits, value = model(state)
        loss = torch.nn.functional.cross_entropy(logits, target) + value.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_channels": 20, "num_res_blocks": 1, "num_channels": 8,
        "purpose": "Prepared capture-biased teaching fixture; not a chess-strength checkpoint",
        "preparation_fen": SAFE_FEN, "seed": 7, "steps": 48,
        "torch_version": str(torch.__version__),
    }, path)


def searched_bait_line(root, board):
    """Return the most-visited continuation under the bait, from this tree.

    An empty line means the bait has not been visited. This is a sampled
    continuation, not a general proof that every alternative reply loses.
    """
    child = root.children[BAIT]
    if not child.visit_count:
        return [], False
    position = board.copy()
    line = [position.san(BAIT)]
    position.push(BAIT)
    while child.children and not position.is_game_over() and len(line) < 8:
        move, next_child = max(child.children.items(), key=lambda item: item[1].visit_count)
        if not next_child.visit_count:
            break
        line.append(position.san(move))
        position.push(move)
        child = next_child
    return line, position.is_checkmate()


def fit_search_policy(model, board, target, steps=60):
    """Fit just the search policy; no completed game means no outcome label.

    Reuse the real loss, selecting its policy term. BatchNorm stays in eval
    mode so a single board does not rewrite running population statistics.
    Gradients still update the network; eval() does not disable autograd.
    """
    candidate = copy.deepcopy(model).eval()
    optimizer = torch.optim.Adam(candidate.parameters(), lr=0.001)
    policies = torch.zeros(1, NUM_MOVES)
    for move, probability in target.items():
        policies[0, move_to_index(move)] = probability
    state = encode(board)
    for _ in range(steps):
        logits, values = candidate(state)
        _, policy_loss, _ = soft_target_loss(logits, policies, values, values.detach())
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
    return candidate


def policy_distance(policy, target):
    """Total variation: zero means identical legal-move distributions."""
    return sum(abs(policy[m] - target[m]) for m in target) / 2


def run_lab(checkpoint=CHECKPOINT, pause=False):
    model, _ = load_model(checkpoint, device="cpu")
    board = chess.Board(FEN)
    evaluate = network_policy_value(model, "cpu")
    before, _ = evaluate(board)
    print("PREDICT: White to move. Would you take the pawn on a7?")
    for rank, row in zip(range(8, 0, -1), str(board).splitlines()):
        print(f"{rank} {row}")
    print("  a b c d e f g h (White at the bottom; uppercase pieces are White)")
    print(f"Raw network: {board.san(max(before, key=before.get))}; Rxa7 {before[BAIT]:.1%}")
    print("Prepared, capture-biased checkpoint: trained on one OTHER board, not human games.")
    if pause:
        input("Predict Black's reply, then press Enter to search. ")
    print("\nWATCH: independent searches from the same board; no exploration noise.")
    print(" Sims   Preferred   Rxa7 visits   Most-visited line under Rxa7")
    snapshots = []
    for budget in BUDGETS:
        root, visits = puct_search(board, evaluate, simulations=budget, add_root_noise=False)
        line, mate = searched_bait_line(root, board)
        best = max(visits, key=visits.get)
        print(f" {budget:4d}   {board.san(best):9s}   {visits[BAIT]:3d}/{budget:<3d}       {' '.join(line)}")
        snapshots.append({"budget": budget, "best": best, "visits": visits,
                          "line": line, "mate": mate})
    target = visits_to_policy(visits)
    print("The # marks checkmate: that leaf is scored by chess rules, not the value head.")
    print("Visit counts record search effort; they are not probabilities of winning.")
    print("\nPREDICT: after fitting this search, will the first instinct change?")
    print("Will it also change with Black's king on h8? What if White has an escape square?")
    if pause:
        input("Make your predictions, then press Enter to train on ONLY the original board. ")
    candidate = fit_search_policy(model, board, target)
    after_fn = network_policy_value(candidate, "cpu")
    after, _ = after_fn(board)
    print("\nEXPLAIN: 60 updates to a copy; the shipped checkpoint is unchanged.")
    print("No game was completed: fit the policy term of soft_target_loss, with no value target.")
    print(f"Original board: Rxa7 {before[BAIT]:.1%} -> {after[BAIT]:.1%}; search target {target[BAIT]:.1%}")
    print(f"Distance from search policy: {policy_distance(before, target):.3f} -> {policy_distance(after, target):.3f} (0 = match)")
    print(f"New first choice, WITHOUT search: {board.san(max(after, key=after.get))}")
    probes = []
    for label, fen in (("Unseen similar board (black king h8)", SIMILAR_FEN),
                       ("Escape-square control (white pawn g3)", SAFE_FEN)):
        probe = chess.Board(fen)
        old, _ = evaluate(probe)
        new, _ = after_fn(probe)
        probe.push(BAIT)
        probe.push(REPLY)
        mate = probe.is_checkmate()
        print(f"{label}: Rxa7 {old[BAIT]:.1%} -> {new[BAIT]:.1%}; Rxa7 Re1 is {'mate' if mate else 'NOT mate'}")
        probes.append({"before": old[BAIT], "after": new[BAIT], "mate": mate})
    print("The control was used to PREPARE the checkpoint, but neither probe was used in these 60 updates.")
    print("A shift on a similar board alone does not prove useful generalisation.")
    print("Did the model learn the danger, or mostly learn to dislike this move?")
    print("Search changed the decision using lookahead. Training changed the weights.")
    print("This demonstrates fitting one lesson, not an increase in overall chess strength.")
    return {"before": before, "after": after, "target": target,
            "snapshots": snapshots, "probes": probes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pause", action="store_true", help="pause for predictions at each stage")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--build-checkpoint", type=Path, metavar="PATH",
                        help="reproduce the prepared fixture at PATH, then exit")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.build_checkpoint:
        build_checkpoint(args.build_checkpoint)
        print(f"Wrote prepared checkpoint to {args.build_checkpoint}")
    else:
        run_lab(args.checkpoint, pause=args.pause)


if __name__ == "__main__":
    main()
