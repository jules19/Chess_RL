"""
PUCT Search - True AlphaZero-style MCTS (Phase 3b/4)

This is the search algorithm AlphaZero actually uses, and it differs from the
rollout MCTS in search/mcts.py in one fundamental way:

    THERE ARE NO ROLLOUTS.

Classic MCTS estimates a leaf's value by playing a (semi-)random game to the
end. AlphaZero replaces that entire simulation step with a single neural
network call: the VALUE HEAD is the estimate. The POLICY HEAD replaces
uniform exploration: moves the network believes in get explored first.

One search iteration:
    1. SELECT     - walk down the tree, at each node picking the child that
                    maximizes Q + U (the PUCT formula, below)
    2. EXPAND     - at a leaf, ask the network for (move priors, value);
                    create one child per legal move, storing its prior
    3. BACKPROPAGATE - push the value up the path, flipping sign each ply
                    (a good position for White is bad for Black)

The PUCT formula (Predictor + Upper Confidence bounds for Trees):

    score(child) = Q(child) + c_puct * P(child) * sqrt(N(parent)) / (1 + N(child))

    Q = average value of the child from the parent's perspective (exploitation)
    P = the network's prior probability for the move (the "predictor")
    N = visit counts; the U term shrinks as a child is visited more

Compare with UCT in mcts.py: UCT explores every untried move once (uniform
prior); PUCT lets the network say "these 3 moves are promising" so 800
simulations go deep on promising lines instead of wide on everything.

Design note — the search is decoupled from the network:
    puct_search() takes a `policy_value_fn(board) -> (priors, value)`.
    Any function with that signature works, which means the search logic can
    be unit-tested with a fake network (see uniform_policy_value) without
    torch installed or a model trained. Decoupling "algorithm" from "learned
    model" is what makes both testable.

Perspective convention (the #1 source of sign bugs in AlphaZero code):
    - policy_value_fn returns the value from the perspective of the SIDE TO
      MOVE at that board (+1 = the player about to move is winning).
    - node.total_value accumulates values from the perspective of the side
      to move AT THAT NODE.
    - Therefore the parent (who is choosing between children) wants children
      with LOW value-for-the-child: selection uses -Q(child).
"""

import math
import random
from typing import Callable, Dict, Optional, Tuple

import chess


class PUCTNode:
    """
    Node in the PUCT search tree.

    Attributes:
        prior: The network's prior probability P(move) for the move that
               leads to this node (set by the parent's expansion).
        visit_count: N — number of times this node was on a search path.
        total_value: W — sum of backed-up values, from the perspective of
                     the side to move at THIS node.
        children: Dict[chess.Move, PUCTNode]; empty until expanded.
    """

    __slots__ = ('prior', 'visit_count', 'total_value', 'children')

    def __init__(self, prior: float):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.children: Dict[chess.Move, 'PUCTNode'] = {}

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def q_value(self) -> float:
        """Average value from this node's side-to-move perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def select_child(self, c_puct: float) -> Tuple[chess.Move, 'PUCTNode']:
        """
        Pick the child maximizing the PUCT score.

        Note the minus sign on q_value(): the child's value is stored from
        the CHILD's side-to-move perspective, and what's good for the child's
        mover is bad for us (the parent choosing the move).
        """
        sqrt_parent_visits = math.sqrt(self.visit_count)

        def puct_score(item):
            move, child = item
            exploitation = -child.q_value()
            exploration = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            return exploitation + exploration

        return max(self.children.items(), key=puct_score)


def uniform_policy_value(board: chess.Board) -> Tuple[Dict[chess.Move, float], float]:
    """
    A fake "network" with no chess knowledge: uniform priors, value 0.

    With this, PUCT degrades to something UCT-like — useful for two things:
    1. Unit-testing the search machinery without torch or a trained model
    2. A baseline: any trained network should beat this in an arena match
    """
    legal_moves = list(board.legal_moves)
    prior = 1.0 / len(legal_moves)
    return {move: prior for move in legal_moves}, 0.0


def network_policy_value(model, device=None) -> Callable:
    """
    Adapt a trained PolicyValueNetwork into a policy_value_fn for puct_search.

    Args:
        model: PolicyValueNetwork (already on the right device, in eval mode)
        device: torch device for input tensors (auto-detected if None)

    Returns:
        policy_value_fn(board) -> (Dict[move, prior], value)
        Value is from the side-to-move perspective — which is exactly what
        the network was trained to predict (the dataset stores outcomes from
        the perspective of the player to move).
    """
    import torch
    from net.encoding import board_to_tensor, legal_moves_mask, move_to_index
    from net.model import auto_device

    if device is None:
        device = auto_device()

    def policy_value_fn(board: chess.Board):
        with torch.no_grad():
            board_tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)
            mask = torch.from_numpy(legal_moves_mask(board)).unsqueeze(0).to(device)
            policy_probs, value = model.get_policy_value(board_tensor, mask)
            policy_probs = policy_probs.squeeze(0).cpu().numpy()

        priors = {move: float(policy_probs[move_to_index(move)])
                  for move in board.legal_moves}
        # Renormalize over legal moves (masking makes this near-1 already)
        total = sum(priors.values())
        if total > 0:
            priors = {m: p / total for m, p in priors.items()}
        else:
            # Untrained/degenerate network: fall back to uniform
            priors = {m: 1.0 / len(priors) for m in priors}

        return priors, float(value.item())

    return policy_value_fn


def add_dirichlet_noise(priors: Dict[chess.Move, float],
                        alpha: float = 0.3,
                        epsilon: float = 0.25,
                        rng: Optional[random.Random] = None) -> Dict[chess.Move, float]:
    """
    Mix Dirichlet noise into root priors: p ← (1-ε)·p + ε·noise.

    Why: during SELF-PLAY the engine must occasionally try moves the network
    currently dislikes, or it can never discover they're actually good — the
    classic exploration/exploitation dilemma at the move level. Noise is only
    added at the ROOT (we want variety in what we play, not corrupted search
    deeper in the tree) and only during training data generation, never in
    serious play.

    alpha controls the noise shape: small alpha (0.3 for chess) concentrates
    the noise on a few random moves rather than spreading it thinly.
    """
    rng = rng or random
    moves = list(priors.keys())
    # Sample Dirichlet(alpha) via normalized Gamma draws
    # (avoids a numpy dependency for one function)
    gammas = [rng.gammavariate(alpha, 1.0) for _ in moves]
    total = sum(gammas) or 1.0
    noise = [g / total for g in gammas]
    return {
        move: (1 - epsilon) * priors[move] + epsilon * n
        for move, n in zip(moves, noise)
    }


def puct_search(board: chess.Board,
                policy_value_fn: Callable,
                simulations: int = 100,
                c_puct: float = 1.5,
                add_root_noise: bool = False,
                dirichlet_alpha: float = 0.3,
                dirichlet_epsilon: float = 0.25,
                rng: Optional[random.Random] = None) -> Tuple[PUCTNode, Dict[chess.Move, int]]:
    """
    Run PUCT search from a position.

    Args:
        board: Position to search from
        policy_value_fn: (board) -> (Dict[move, prior], value_for_side_to_move)
        simulations: Number of search iterations
        c_puct: Exploration constant (higher = trust priors/exploration more)
        add_root_noise: Mix Dirichlet noise into root priors (self-play only)
        dirichlet_alpha / dirichlet_epsilon: Noise parameters
        rng: Optional random.Random for reproducible noise

    Returns:
        (root_node, visit_counts) where visit_counts maps each root move to
        the number of simulations that went through it. The visit counts ARE
        the output of the search: they drive both move selection and the
        policy training target during self-play.
    """
    if board.is_game_over():
        raise ValueError("puct_search called on a finished game")

    root = PUCTNode(prior=1.0)

    # Expand the root immediately (so noise can be applied to its priors)
    priors, _ = policy_value_fn(board)
    if add_root_noise:
        priors = add_dirichlet_noise(priors, dirichlet_alpha, dirichlet_epsilon, rng)
    for move, prior in priors.items():
        root.children[move] = PUCTNode(prior=prior)

    for _ in range(simulations):
        node = root
        search_board = board.copy()
        path = [root]

        # 1. SELECT — walk down until we reach an unexpanded node
        while node.is_expanded():
            move, node = node.select_child(c_puct)
            search_board.push(move)
            path.append(node)

        # 2. EXPAND + EVALUATE — one network call replaces a whole rollout
        if search_board.is_game_over():
            # Terminal positions need no network: the game rules give the
            # exact value. Side to move is checkmated -> -1; otherwise draw.
            value = -1.0 if search_board.is_checkmate() else 0.0
        else:
            priors, value = policy_value_fn(search_board)
            for move, prior in priors.items():
                node.children[move] = PUCTNode(prior=prior)

        # 3. BACKPROPAGATE — flip the sign each ply going up: `value` is
        # from the leaf's side-to-move perspective, and perspectives
        # alternate as we walk toward the root.
        for path_node in reversed(path):
            path_node.visit_count += 1
            path_node.total_value += value
            value = -value

    visit_counts = {move: child.visit_count for move, child in root.children.items()}
    return root, visit_counts


def visits_to_policy(visit_counts: Dict[chess.Move, int],
                     temperature: float = 1.0) -> Dict[chess.Move, float]:
    """
    Convert visit counts to a move probability distribution.

    π(a) ∝ N(a)^(1/τ)

    Temperature τ controls how sharply we commit to the most-visited move:
        τ → 0: deterministic argmax (competitive play)
        τ = 1: proportional to visits (early self-play moves, for diversity)

    In AlphaZero self-play, τ=1 for the first ~30 plies then τ→0: explore
    openings, but convert winning positions cleanly.
    """
    if temperature <= 1e-6:
        best = max(visit_counts, key=visit_counts.get)
        return {move: (1.0 if move == best else 0.0) for move in visit_counts}

    powered = {m: n ** (1.0 / temperature) for m, n in visit_counts.items()}
    total = sum(powered.values()) or 1.0
    return {m: p / total for m, p in powered.items()}


def select_move(visit_counts: Dict[chess.Move, int],
                temperature: float = 0.0,
                rng: Optional[random.Random] = None) -> chess.Move:
    """Pick a move from the visit distribution at the given temperature."""
    policy = visits_to_policy(visit_counts, temperature)
    if temperature <= 1e-6:
        return max(policy, key=policy.get)
    rng = rng or random
    moves = list(policy.keys())
    return rng.choices(moves, weights=[policy[m] for m in moves], k=1)[0]


def best_move_puct(board: chess.Board,
                   policy_value_fn: Callable = None,
                   simulations: int = 100,
                   c_puct: float = 1.5,
                   temperature: float = 0.0,
                   verbose: bool = False) -> Optional[chess.Move]:
    """
    Convenience wrapper matching the interface of the other engines.

    Args:
        board: Current position
        policy_value_fn: Network adapter; uniform_policy_value if None
        simulations: Search iterations
        c_puct: Exploration constant
        temperature: Move selection temperature (0 = strongest play)
        verbose: Print top moves by visit count

    Returns:
        Selected move, or None if the game is over
    """
    if board.is_game_over():
        return None

    policy_value_fn = policy_value_fn or uniform_policy_value
    root, visit_counts = puct_search(board, policy_value_fn,
                                     simulations=simulations, c_puct=c_puct)

    if verbose:
        print(f"\n=== PUCT Search ({simulations} simulations) ===")
        top = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for move, visits in top:
            child = root.children[move]
            print(f"  {move}: {visits} visits, prior {child.prior:.3f}, "
                  f"Q {-child.q_value():+.3f}")

    return select_move(visit_counts, temperature)


if __name__ == "__main__":
    """Smoke test with the uniform fake network (no torch needed)."""
    print("Testing PUCT search with uniform priors...")

    # Mate in 1: even a knowledge-free value function finds it, because
    # terminal nodes return exact values and PUCT concentrates visits there.
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    print(board)
    move = best_move_puct(board, simulations=300, verbose=True)
    print(f"\nSelected: {move} (expected e1e8)")
    assert str(move) == "e1e8", "PUCT failed to find mate in 1"
    print("✅ PUCT smoke test passed")
