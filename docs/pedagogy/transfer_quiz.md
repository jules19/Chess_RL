# Chess_RL transfer quiz

These questions ask you to **apply** what you learned about THIS project to situations it doesn't
describe directly. Reason from how the project works. 2–4 sentences each. Say "don't know" if
you can't reason it out.

1. Suppose `evaluate()`'s mobility term were changed to (White's legal moves − Black's legal moves), whoever is to move. What would `evaluate()` return for the starting position, and why?
2. If you deleted the move ordering from this project's minimax (searching moves in generation order), would the engine choose a different move at the same depth? What *would* change?
3. You raise this project's MCTS from 200 to 2,000 simulations from the starting position. What do you expect to happen to the visit counts, and to how sensible the chosen move is? Why?
4. The engine is playing **Black** in a GUI and prints `info depth 3 score cp 30`. What does that 30 actually tell you, and how would a GUI user probably misread it?
5. You want queen promotions to work in the network's policy. Name at least two things that must change in the project, and one consequence for the existing trained model.
6. A friend retrains this network and reports 40% validation top-1 accuracy using the project's training code unchanged. What one check would you insist on before believing the network has really improved, and why?
7. Design a fair 20-game match between two deterministic engines from this project. Give two specific design rules and the problem each one prevents.
8. One network call takes about 2 ms. Roughly how long would this project's NN-MCTS take per move at 200 simulations, and roughly how long would an AlphaZero-style search with the same network take?
9. In `backpropagate`, suppose someone removed the `value = -value` line. How would the search behave when it is choosing moves for Black?
10. The project's trained network loses to minimax at depth 1. Give the two most likely reasons from what you know about how it was trained, and say which you would fix first.
