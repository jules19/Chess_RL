#!/usr/bin/env python3
"""
Build Complete Tutorial

This script generates the comprehensive chess_rl_complete_tutorial.html
by merging content from existing tutorials and adding new sections.
"""

import re
from pathlib import Path


def extract_content_section(html_content, start_marker, end_marker=None):
    """Extract a section of HTML content between markers."""
    start_idx = html_content.find(start_marker)
    if start_idx == -1:
        return ""

    if end_marker:
        end_idx = html_content.find(end_marker, start_idx)
        if end_idx == -1:
            return html_content[start_idx:]
        return html_content[start_idx:end_idx]

    return html_content[start_idx:]


def build_phase_1_content():
    """Extract and adapt Phase 1 content from minimax_tutorial.html"""
    try:
        with open('minimax_tutorial.html', 'r') as f:
            content = f.read()

        # Extract main content (skip head/body tags)
        # Look for content between <body> and </body>
        body_match = re.search(r'<body>(.*?)</body>', content, re.DOTALL)
        if body_match:
            body_content = body_match.group(1)
            # Remove container div if present
            body_content = re.sub(r'<div class="container">(.*?)</div>\s*$', r'\1', body_content, flags=re.DOTALL)
            return body_content.strip()
        return ""
    except FileNotFoundError:
        return "<p>Phase 1 content to be migrated from minimax_tutorial.html</p>"


def build_phase_2_content():
    """Build Phase 2 (MCTS) content"""
    return """
                    <div class="breadcrumb">
                        <a href="#intro">Home</a> → Phase 2: Monte Carlo Tree Search
                    </div>

                    <div class="day-marker complete">✅ Week 2 - MCTS Engine</div>

                    <div class="learning-objective">
                        <h3>Learning Objectives - Phase 2</h3>
                        <p>By the end of this phase, you will:</p>
                        <ul class="checklist">
                            <li>Understand Monte Carlo Tree Search (MCTS) algorithm</li>
                            <li>Implement UCT (Upper Confidence bounds applied to Trees)</li>
                            <li>Build tree-based search with random rollouts</li>
                            <li>Replace rollouts with evaluator guidance</li>
                            <li>Achieve ~1400-1600 Elo strength</li>
                        </ul>
                    </div>

                    <h3>🎯 Goal: Selective Search with Statistical Sampling</h3>

                    <p>Minimax explores the <em>entire</em> game tree to a fixed depth. MCTS is fundamentally different - it <strong>selectively explores promising branches</strong> using random simulations.</p>

                    <div class="key-insight">
                        <strong>💡 Key Insight:</strong> MCTS doesn't try to search everything. Instead, it:
                        <ul>
                            <li>Focuses computational effort on promising moves</li>
                            <li>Uses random playouts to estimate position quality</li>
                            <li>Balances exploration (trying new moves) vs exploitation (deepening good moves)</li>
                            <li>Gets better with more simulations (anytime algorithm)</li>
                        </ul>
                    </div>

                    <h3>🏗️ The Four Phases of MCTS</h3>

                    <div class="diagram">
<strong>MCTS Algorithm (4 Phases, Repeated N Times):</strong>
<pre>
1. SELECTION: Walk down tree using UCT formula
   └─ Pick child with highest UCT value = Q(a) + C × √(ln(N_parent) / N(a))

2. EXPANSION: Add one new child node
   └─ Pick untried move, create new node

3. SIMULATION: Play out random game from new node
   └─ Make random moves until terminal position

4. BACKPROPAGATION: Update all nodes from leaf to root
   └─ Increment visit counts, update values

After N simulations: Select most-visited move
</pre>
                    </div>

                    <h3>💻 Code: MCTS Node Structure</h3>

                    <pre><code class="language-python">class MCTSNode:
    """Node in the MCTS search tree."""

    def __init__(self, board, parent=None, move=None):
        self.board = board.copy()
        self.parent = parent
        self.move = move  # Move that led to this position

        # MCTS statistics
        self.visit_count = 0
        self.total_value = 0.0  # Sum of simulation results

        # Children and unexplored moves
        self.children = {}  # Dict[Move, MCTSNode]
        self.untried_moves = list(board.legal_moves)
        random.shuffle(self.untried_moves)

    def uct_value(self, exploration_constant=1.41):
        """
        Calculate UCT (Upper Confidence bound for Trees) value.

        UCT = Q(v) + C × √(ln(N_parent) / N(v))

        Where:
        - Q(v) = average value (exploitation)
        - C = exploration constant (√2 ≈ 1.41)
        - N = visit counts
        """
        if self.visit_count == 0:
            return float('inf')  # Unvisited nodes have infinite priority

        exploitation = self.total_value / self.visit_count
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )

        return exploitation + exploration

    def best_child(self, exploration_constant=1.41):
        """Select child with highest UCT value."""
        return max(self.children.values(),
                  key=lambda child: child.uct_value(exploration_constant))

    def expand(self):
        """Add one new child node."""
        move = self.untried_moves.pop()
        new_board = self.board.copy()
        new_board.push(move)

        child = MCTSNode(new_board, parent=self, move=move)
        self.children[move] = child
        return child
</code></pre>

                    <div class="code-explanation">
                        <strong>🔍 UCT Formula Explained:</strong>
                        <ul>
                            <li><strong>Exploitation term (Q/N):</strong> Average value of this node's simulations</li>
                            <li><strong>Exploration term (C × √...):</strong> Bonus for less-visited nodes</li>
                            <li><strong>Balance:</strong> High-value nodes get explored, but so do uncertain nodes</li>
                            <li><strong>C parameter:</strong> Higher = more exploration, Lower = more exploitation</li>
                        </ul>
                    </div>

                    <h3>💻 Code: The Complete MCTS Algorithm</h3>

                    <pre><code class="language-python">def mcts_search(board, simulations=200):
    """
    Perform MCTS search to find best move.

    Args:
        board: Current position
        simulations: Number of MCTS iterations

    Returns:
        Best move (most visited)
    """
    root = MCTSNode(board)

    for _ in range(simulations):
        node = root

        # 1. SELECTION: Walk down tree using UCT
        while node.is_fully_expanded() and not node.is_terminal():
            node = node.best_child()

        # 2. EXPANSION: Add new child if not terminal
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()

        # 3. SIMULATION: Play out random game
        value = simulate_random(node.board)

        # 4. BACKPROPAGATION: Update tree
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value  # Flip perspective for parent
            node = node.parent

    # Return most visited move
    best_child = root.most_visited_child()
    return best_child.move if best_child else None
</code></pre>

                    <div class="checkpoint">
                        <h4>✅ Phase 2 Validation Checklist</h4>
                        <p>Before moving to Phase 3, verify:</p>
                        <ul class="checklist">
                            <li>MCTS beats random player >90%</li>
                            <li>MCTS beats minimax depth-3 >55%</li>
                            <li>With 200 sims/move, strength ~1400-1600 Elo</li>
                            <li>More simulations = better play (anytime algorithm)</li>
                            <li>Tree statistics look reasonable (visit counts grow)</li>
                        </ul>
                    </div>

                    <h3>🚀 Optimization: Smart Rollouts</h3>

                    <p>Pure random rollouts are weak. We can do better by using the evaluator:</p>

                    <pre><code class="language-python">def simulate_with_evaluator(board, max_moves=30):
    """
    Simulate game using evaluator for move selection.
    Much stronger than random rollouts!
    """
    sim_board = board.copy()
    moves = 0

    while not sim_board.is_game_over() and moves < max_moves:
        # Sample subset of moves (speedup)
        legal_moves = list(sim_board.legal_moves)
        sample_size = min(10, len(legal_moves))
        moves_to_try = random.sample(legal_moves, sample_size)

        # Pick best move from sample using evaluator
        best_move = None
        best_eval = float('-inf') if sim_board.turn == chess.WHITE else float('inf')

        for move in moves_to_try:
            sim_board.push(move)
            eval_score = evaluate(sim_board)
            sim_board.pop()

            if (sim_board.turn == chess.WHITE and eval_score > best_eval) or \\
               (sim_board.turn == chess.BLACK and eval_score < best_eval):
                best_eval = eval_score
                best_move = move

        if best_move:
            sim_board.push(best_move)
        moves += 1

    # Return result or evaluation
    if sim_board.is_checkmate():
        return 1.0 if sim_board.turn == chess.BLACK else -1.0
    elif sim_board.is_game_over():
        return 0.0
    else:
        # Normalize evaluation to [-1, 1]
        return max(-1.0, min(1.0, evaluate(sim_board) / 1000.0))
</code></pre>

                    <div class="success">
                        <h4>🎉 Phase 2 Complete!</h4>
                        <p>You've built a Monte Carlo Tree Search engine that selectively explores the game tree!</p>
                        <p><strong>Strength:</strong> ~1400-1600 Elo with 200 simulations per move</p>
                        <p><strong>What's next:</strong> Replace evaluator rollouts with neural network predictions!</p>
                    </div>

                    <div class="section-nav">
                        <a href="#phase1" class="prev">← Previous: Minimax</a>
                        <a href="#phase3" class="next">Next: Neural Networks →</a>
                    </div>
"""


def build_phase_3_content():
    """Extract and adapt Phase 3 content from neural_network_tutorial.html"""
    try:
        with open('neural_network_tutorial.html', 'r') as f:
            content = f.read()

        # Extract main content
        body_match = re.search(r'<body>(.*?)</body>', content, re.DOTALL)
        if body_match:
            body_content = body_match.group(1)
            body_content = re.sub(r'<div class="container">(.*?)</div>\s*$', r'\1', body_content, flags=re.DOTALL)
            return body_content.strip()
        return ""
    except FileNotFoundError:
        return "<p>Phase 3 content to be migrated from neural_network_tutorial.html</p>"


def build_glossary():
    """Build unified glossary"""
    return """
                    <h2 id="glossary">📖 Complete Glossary</h2>

                    <p>Comprehensive glossary covering all phases of the tutorial.</p>

                    <h3>Core Concepts</h3>

                    <div class="glossary-term">
                        <dt>Alpha-Beta Pruning</dt>
                        <dd>Optimization of minimax that skips branches that can't affect the final decision. Reduces search tree size by 50-95%.</dd>
                    </div>

                    <div class="glossary-term">
                        <dt>Evaluation Function</dt>
                        <dd>Function that assigns a numerical score to a chess position, estimating who is winning. Example: +3.5 means White is ahead by ~3.5 pawns.</dd>
                    </div>

                    <div class="glossary-term">
                        <dt>MCTS (Monte Carlo Tree Search)</dt>
                        <dd>Search algorithm that uses random simulations to estimate position quality, selectively expanding promising branches.</dd>
                    </div>

                    <div class="glossary-term">
                        <dt>Minimax</dt>
                        <dd>Game tree search algorithm that assumes both players play optimally. Maximizing player wants highest score, minimizing player wants lowest.</dd>
                    </div>

                    <div class="glossary-term">
                        <dt>Policy</dt>
                        <dd>Probability distribution over moves. The policy tells you which moves are likely to be good in a given position.</dd>
                    </div>

                    <div class="glossary-term">
                        <dt>Quiescence Search</dt>
                        <dd>Extension of minimax that continues searching tactical moves (captures, checks) beyond normal depth to avoid horizon effect.</dd>
                    </div>

                    <div class="glossary-term">
                        <dt>Reinforcement Learning (RL)</dt>
                        <dd>Machine learning approach where an agent learns by trial and error, receiving rewards for good outcomes.</dd>
                    </div>

                    <div class="glossary-term">
                        <dt>ResNet (Residual Network)</dt>
                        <dd>Neural network architecture with skip connections that allow training very deep networks. Used in AlphaZero.</dd>
                    </div>

                    <div class="glossary-term">
                        <dt>Rollout</dt>
                        <dd>In MCTS, a rollout is a random (or guided) simulation of a game from a given position to a terminal state.</dd>
                    </div>

                    <div class="glossary-term">
                        <dt>Self-Play</dt>
                        <dd>Training method where a program plays games against itself to generate training data and improve over time.</dd>
                    </div>

                    <div class="glossary-term">
                        <dt>UCT (Upper Confidence bounds applied to Trees)</dt>
                        <dd>Formula used in MCTS to balance exploration vs exploitation when selecting which node to expand.</dd>
                    </div>

                    <div class="glossary-term">
                        <dt>Value</dt>
                        <dd>Scalar evaluation of a position, typically in range [-1, +1]. +1 = win, 0 = draw, -1 = loss.</dd>
                    </div>
"""


print("Building complete tutorial...")
print("This will merge content from existing tutorials and generate the final HTML.")
print("Run this script to create the comprehensive tutorial file.")
print("")
print("Usage: python build_complete_tutorial.py")
