# Frozen course assessment v1

Audience: a comfortable Python programmer who knows chess rules but is new to
engines and reinforcement learning. Answer from the supplied course snapshot.
Cite the file supporting each answer. Say "not established" when the material
cannot support a claim. Do not execute code, inspect solutions, or use the web.

## Recall

1. What two outputs does the chess network learn? What supplies their targets
   in human-game training, and what supplies them in self-play?
2. Which program lets a newcomer observe a complete learning loop without
   downloading chess games? What does its `vs_perfect` score mean?
3. In the change-of-mind chess lab, where did the starting checkpoint come
   from, and which part of the normal training loss is used during the lab?
4. What reward does chess self-play assign to a game that reaches its length
   limit unfinished? Why can a very small value loss then be misleading?
5. Give the exact steps needed to observe the initial failure of the Module 2
   time-budget exercise. Can the test fail while its skip marker is present?
6. How can a fresh clone show just the reference implementation for the
   evaluator exercise without switching the learner's working branch?

## Transfer

7. Search finds a refutation of a tempting move. Explain what changes during
   search, what changes during training, and why training can change the next
   choice even when no search is run.
8. After fitting one position, the same move loses probability on an unseen
   similar board. Does that establish useful generalisation? Describe a
   control and the result that would weaken that interpretation.
9. A six-game match scores 3.5/6 for a modified engine. What can you defend?
   Design a quick check of the mechanism separately from a strength experiment.
10. You have completed a long self-play run and want to rerun the smoke test.
    Explain how to preserve the long run, what the current program does if you
    reuse its directory, and whether recording a configuration alone protects it.
11. Training stops halfway through an iteration. What must a resume feature
    restore? Explain a consistent completed iteration, whether Adam state must
    survive that boundary in this implementation, and whether specifying the
    same seed guarantees an identical continuation.
12. Before testing your own change, what will you record so your final report
    can distinguish a prediction from an explanation invented after the result?
    Explain where you would keep it and what you would add after the experiment.
