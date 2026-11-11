1. **Policy and value.** The policy predicts move probabilities; the value predicts the eventual result for the player to move. Human-game training uses the actual played move and final game result. Self-play uses PUCT’s visit distribution and the eventual self-play result, signed for the player to move. Sources: `course/module-04-neural-network/README.md`, `training/dataset.py`, `course/module-06-selfplay-rl/README.md`.

2. Run `python3 selfplay/tictactoe_lab.py`. It needs no game download or torch. `vs_perfect` measures the percentage of games the learned table **without search** does not lose against a perfect opponent. Since that opponent cannot lose, these are draws; 100% means always drawing. Source: `course/module-00-orientation/README.md`.

3. The shipped `early.pt` is a deliberately prepared teaching fixture: a randomly initialised network trained for 48 Adam steps on one escape-square position to favour `Rxa7`. Preparation used move cross-entropy plus a conventional zero value target; it did not use human games. During the lab, 60 updates fit only the **policy component** of `soft_target_loss()` to search visits on the original threat position. There is no completed game and no value target. Source: `selfplay/assets/mind_change/README.md`.

4. An unfinished length-limited game becomes a draw with reward **0**. If nearly every game is truncated, predicting zero gives very small value loss without demonstrating useful evaluation of wins and losses. Sources: `selfplay/self_play.py`, `course/module-06-selfplay-rl/README.md`.

5. After setup, open `tests/course/test_module02_search.py`, remove the **entire multiline** `@pytest.mark.skip(...)` decorator, and run:
   ```bash
   pytest tests/course/test_module02_search.py -q
   ```
   Do this before implementing the timeout. The test requests a six-second search, requires a legal move and elapsed time below eight seconds, and is expected to fail after roughly 18 seconds on the reference machine. Timing varies. With its skip marker present, the test body does not run and cannot expose this failure. Sources: `course/module-02-minimax-search/README.md`, `tests/course/test_module02_search.py`.

6. From the clone:
   ```bash
   git fetch origin course-solutions:refs/remotes/origin/course-solutions
   git diff main origin/course-solutions -- engine/evaluator.py
   ```
   This displays only the evaluator difference without switching branches or modifying working files. The current exercise test remains the contract because the reference branch may lag. Source: `course/README.md`.

7. Search updates its tree’s visits and backed-up values, changing the preferred move while network weights stay fixed. Training then updates weights to make the policy resemble the search distribution. A later network-only choice can change because the network has learned different move probabilities. In the lab it chooses `f4` without searching after fitting. Source: `course/module-05-puct/README.md`.

8. **No: useful generalisation is not established.** Compare the similar threat board with an escape-square control where `Rxa7 Re1+` is not mate. If capture probability falls similarly on both, the network may simply have suppressed that move broadly. That is what the lab reports. The escape-square control appeared during checkpoint preparation; the king-on-h8 probe was unseen during both preparation and fitting. Sources: `course/module-05-puct/README.md`, `selfplay/assets/mind_change/README.md`.

9. I can defend the observed **3.5/6, about 58%**, under those match conditions. A strength improvement is **not established**: six games have broad uncertainty compatible with equal strength. For a quick mechanism check, use controlled positions; for example, verify a rook bonus’s expected size and that swapping colours reverses its sign. Separately, compare baseline and modified engines with fixed budgets, varied openings or suitable randomness, enough games chosen in advance, and a reported 95% interval. Sources: `course/module-01-board-and-evaluation/README.md`, `course/module-07-measure-and-scale/README.md`.

10. Keep the long run’s directory intact and give the smoke test a new directory, such as `--checkpoint-dir runs/m6_smoke_02`. The current unmodified program refuses an occupied directory; there is no destructive override. Recording configuration alone does not preserve weights, buffers or logs. The program’s directory refusal and immutable completed snapshots provide protection. Source: `course/module-06-selfplay-rl/README.md`.

11. Resume must validate the snapshot referenced by `completed.json`, load its champion, replay buffer and iteration, preserve its committed log and original configuration, and restore saved scientific settings and architecture. Missing, corrupt or inconsistent state must be rejected. A completed iteration has champion, candidate, buffer and log written before the completion marker changes; top-level convenience copies may be stale. Resume repeats incomplete iteration work. Adam state need not survive this boundary because `train_candidate` creates a new optimiser each iteration. The same seed does **not** guarantee identical continuation: Python, NumPy and torch RNG states are not saved. Resume itself is an exercise, not already available in the unmodified program. Source: `course/module-06-selfplay-rl/README.md`.

12. Before running, record the prediction and its reason, the comparison, the measurement, and what result would count against the prediction. Include commands, settings, seed, results directory and code version. Keep this in root `JOURNAL.md`, created from `course/JOURNAL_TEMPLATE.md`, preserving the original entry even if wrong. Afterwards add observations with uncertainty, the conclusion supported, what remains unestablished, and the next question. Bring those entries into `REPORT.md` so the original prediction remains distinguishable from retrospective explanation. Sources: `course/JOURNAL_TEMPLATE.md`, `course/module-00-orientation/README.md`, `course/module-09-capstone/README.md`.
