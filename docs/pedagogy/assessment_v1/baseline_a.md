1. The network learns a **policy** (move probabilities) and a **value** (expected game result for the side to move). Human-game training uses the human’s played move and the game’s final result. Self-play uses the full search visit distribution and the final self-play outcome, signed for the side to move. Sources: `course/module-04-neural-network/README.md`, `course/module-06-selfplay-rl/README.md`.

2. Run `python3 selfplay/tictactoe_lab.py`. Its `vs_perfect` is the percentage of games the learned table **without search** does not lose against perfect play. Since perfect play cannot be beaten, 100% means all draws. The measurement uses 100 games and is noisy. Source: `course/module-06-selfplay-rl/README.md`.

3. The shipped `early.pt` is a deliberately prepared teaching fixture, not a human-game checkpoint. A randomly initialized one-block, eight-channel network was trained for 48 Adam steps on one escape-square position to favour `a1a7`, with a zero value target. The lab then performs 60 updates using **only the policy component** of `soft_target_loss`, fitting the original threat position’s search distribution. There is no game outcome to supply a value target. Source: `selfplay/assets/mind_change/README.md`.

4. An unfinished game at the length limit receives **0**, treated as a draw. A value head predicting near zero can therefore have tiny loss even though the run produces no decisive outcomes and teaches little about winning or losing. Low value loss alone does not establish strength. Source: `course/module-06-selfplay-rl/README.md`.

5. In the configured learner checkout, before implementing the abort:
   - Open `tests/course/test_module02_search.py`.
   - Remove the entire multiline `@pytest.mark.skip(...)` decorator.
   - Run `pytest tests/course/test_module02_search.py::test_time_budget_is_a_hard_promise -q`.
   
   The test requests six seconds and asserts elapsed time below eight seconds; the documented baseline takes about eighteen seconds. It cannot execute and fail its timing assertion while the skip marker remains. The module’s instruction to run before removing the marker is inconsistent with that mechanism. Sources: `tests/course/test_module02_search.py`, `course/module-02-minimax-search/README.md`.

6. The course establishes the solution branch and evaluator file, but does **not establish a complete fresh-clone command sequence**. A targeted Git workflow inferred from those references is:
   ```bash
   git fetch origin
   git diff origin/main origin/course-solutions -- engine/evaluator.py
   ```
   This displays only the evaluator changes without switching branches. To display the complete reference file instead:
   ```bash
   git show origin/course-solutions:engine/evaluator.py
   ```
   Sources identifying the branch and file: `course/README.md`, `tests/course/test_module01_evaluator.py`. I did not inspect solutions.

7. Search changes the tree’s visit counts and backed-up value averages, so the selected move can change while network weights remain fixed. Training changes network weights to fit the search distribution. Afterwards, a forward pass alone produces different move probabilities, so the next choice can change without search. In the bridge lab, the raw network switches from `Rxa7` to `f4`. Source: `course/module-05-puct/README.md`.

8. No. That shows transfer of an update, not necessarily useful discrimination. Use an escape-square control where the same capture does not permit the mating refutation. If capture probability falls similarly there, the network may simply have learned a broad aversion to that move. The lab reports just that: both the unseen threat board and escape-square control shift to roughly 17%. The control was seen during checkpoint preparation, so it is not an unseen test set. Sources: `course/module-05-puct/README.md`, `selfplay/assets/mind_change/README.md`.

9. I can defend the observed score, **3.5/6 = 58.3%**, but not a reliable strength improvement. Six games have large uncertainty, and identical deterministic games can reduce the effective sample further. For a quick mechanism check, choose positions where the proposed change predicts a specific effect and compare before/after move probabilities, visits, evaluations, or node counts while holding other settings fixed. Include a control position where the effect should be absent. Separately, measure strength through a substantially larger match with varied openings, balanced colours, fixed budgets, and a reported interval; small improvements may need hundreds of games. Sources: `course/module-05-puct/README.md`, `course/module-07-measure-and-scale/README.md`.

10. Preserve the long run’s directory and run the smoke test with a new explicit directory, for example `--checkpoint-dir smoke_after_long_run`. Back up the long run if needed. Reusing its directory currently starts a new champion and empty buffer, truncates `run_log.csv`, restarts numbering at 1, and overwrites `champion.pt`, `replay_buffer.npz`, and matching candidate filenames as iterations finish. Higher-numbered old candidate files can remain, creating a misleading mixture. Recording configuration does not preserve overwritten artifacts. These behaviours follow from `selfplay/train_loop.py`; separate run directories are also illustrated in `course/module-06-selfplay-rl/README.md`.

11. For restarting from the last **completed iteration**, restore the champion including architecture and model state, replay contents and capacity, completed iteration number, and compatible settings; continue at the next iteration and preserve the matching log history. A consistent boundary means generation, buffer insertion, candidate training, arena decision, and saved champion/buffer/log all belong to the same completed iteration. Current saves are sequential, so crash-consistent atomic recovery is **not established**.

    Adam state need not survive that boundary in this implementation: `train_candidate()` constructs a fresh Adam optimizer each iteration. To continue from a partially completed training phase instead of restarting the iteration, candidate weights, Adam state, step count, random-generator state, and the relevant partial iteration data would also be needed. Using the same seed does not restore an already advanced random stream and does not guarantee identical continuation. The existing resume exercise checks only champion, buffer, and iteration restoration; exact continuation is not established. Sources: `selfplay/train_loop.py`, `selfplay/replay_buffer.py`, `tests/course/test_module06_resume.py`.

12. Before running, I would record the hypothesis, predicted direction and mechanism, the single variable changed, baseline/control, commands and settings, seeds, measurement, sample size, and what result would weaken the hypothesis. I would commit this dated prediction in `REPORT.md` on my learner branch before collecting results. Afterwards, I would add observations, uncertainty, deviations, failures, and the conclusion the evidence supports, preserving the original prediction. The course explicitly requires a hypothesis before the run, one isolated variable, reproducible settings, and honest reporting; it specifies `REPORT.md` in the learner’s fork as the final artifact. Source: `course/module-09-capstone/README.md`.
