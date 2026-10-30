# Chess_RL comprehension quiz

Answer each question about THIS project's code and state (not chess engines in general). 1–3 sentences each.
If you couldn't find out, say "don't know". Don't guess silently.

1. What does this project's `evaluate()` return for the standard starting position, and why isn't it 0?
2. How many nodes does this project's minimax visit at depth 3 from the starting position (as counted by its `nodes_searched`), and why can the count at depth 1 be larger than the 20 legal moves?
3. What problem does quiescence search solve, and which kinds of moves does THIS project's quiescence search consider?
4. In this project's MCTS, which root move is finally played, and what value of the UCT exploration constant is used by default?
5. With the default 200 simulations from the starting position, why can this MCTS prefer odd moves such as a2a3 or d2d3?
6. What are the shape of the neural network's input tensor and the number of policy outputs?
7. Which pawn promotions can the move encoding NOT represent correctly, and why?
8. What are the two training targets for each position, and which loss function is used for each?
9. Why are this project's validation accuracy numbers optimistic?
10. How strong is the trained network, as actually measured? Give at least one concrete number.
11. About how many neural-network calls does one simulation of this project's NN-MCTS make, and how does AlphaZero avoid this cost?
12. Has self-play reinforcement learning been implemented in this project?
13. What does the engine's UCI `info ... score cp` value actually report, and how does that differ from what the UCI spec expects?
14. Were the Elo ratings quoted for each phase (e.g. "~1400-1600 Elo" for MCTS) measured? How would you know?
