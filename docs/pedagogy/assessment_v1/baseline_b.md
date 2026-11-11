1. The network learns a **policy** (move probabilities, produced from 4,672 logits) and a **value** (expected result in `[-1, 1]` for the side to move). Human-game training uses the human’s played move and eventual game result. Self-play uses the PUCT visit distribution as policy target and the eventual self-play result as value target, with the sign appropriate to each position’s player. Sources: `course/RL_CONCEPTS.md`, `course/module-06-selfplay-rl/README.md`.

2. Run `python3 selfplay/tictactoe_lab.py`. It needs no downloaded chess games. `vs_perfect` measures the percentage of games **not lost** by the learned table playing without search against a perfect opponent. Since perfect play cannot lose, 100% means all draws. It is measured over 100 games and is noisy. Source: `course/module-06-selfplay-rl/README.md`.

3. The shipped `early.pt` is a deliberately prepared teaching checkpoint, not a human-game model. A randomly initialized one-block, eight-channel network was trained on one escape-square board, with seed 7, for 48 Adam steps at learning rate 0.001. Preparation targeted `Rxa7` and a conventional value of zero. During the lab, a copy receives 60 updates using **only the policy term** of `soft_target_loss`, targeting the original threat board’s search visits. There is no completed game and therefore no value target. Sources: `selfplay/assets/mind_change/README.md`, `course/module-05-puct/README.md`.

4. An unfinished game at the move cap receives draw reward **0**. If nearly all games hit that cap, predicting zero yields a tiny value loss without learning who would actually win. The input also lacks the remaining-ply counter, so it cannot distinguish cap-related zero labels from otherwise identical positions with real outcomes. Sources: `course/module-06-selfplay-rl/README.md`, `course/RL_CONCEPTS.md`.

5. In the installed learner clone, open `tests/course/test_module02_search.py` and remove the **entire multiline `@pytest.mark.skip(...)` decorator**, leaving the test intact. Before implementing the abort, run:
   ```bash
   pytest tests/course/test_module02_search.py -q
   ```
   It searches the supplied middlegame with `max_depth=6` and a six-second budget, requires a legal move, and asserts elapsed time below eight seconds. The documented baseline takes approximately 18 seconds and fails; timing varies by machine. With the skip marker present, this test body does not run and cannot produce that failure. The module’s instruction to run before removing the marker is inconsistent with the actual test. Sources: `tests/course/test_module02_search.py`, `course/module-02-minimax-search/README.md`.

6. The course identifies the reference branch and evaluator path, but **does not establish an exact fresh-clone, single-file retrieval recipe**. Using Git’s remote branch reference, I would inspect just the evaluator without checking out another branch:
   ```bash
   git fetch origin
   git show origin/course-solutions:engine/evaluator.py
   ```
   For only its changes relative to the reference baseline:
   ```bash
   git diff origin/main origin/course-solutions -- engine/evaluator.py
   ```
   These are proposed Git commands, not commands documented verbatim by the course, and I did not execute them or inspect solutions. Sources for branch and file: `course/README.md`, `course/module-01-board-and-evaluation/README.md`.

7. Search changes the temporary tree: visits and backed-up value averages change as it discovers the refutation. That can change the most-visited move without changing network weights. Training then changes weights to make the network’s policy resemble that search distribution. The next forward pass can therefore prefer the safer move even with no search: the result of earlier planning has been fitted into the policy. The lab’s weights remain fixed during search, while its later training updates change them. Sources: `course/module-05-puct/README.md`, `course/RL_CONCEPTS.md`.

8. No. A probability shift on an unseen board establishes transfer, but not that the transfer reflects the relevant chess distinction. Use a control where the tempting capture has an escape square and the alleged refutation is only check, not mate. If capture probability drops similarly there, the model may have broadly suppressed that move rather than learned when it is dangerous. That is what the lab reports. Its escape-square control was seen during checkpoint preparation; the king-on-h8 probe was unseen during both preparation and fitting. Sources: `course/module-05-puct/README.md`, `selfplay/assets/mind_change/README.md`.

9. I can defend the observed score, **58.3% in those six games**, but not a reliable strength improvement or absolute rating. A suitable interval will be wide; repeated deterministic games would further reduce the effective evidence.

   For a quick mechanism check, isolate a position with a known expected consequence and compare the changed and baseline engines under identical settings. For example, a rook-file bonus should reward an open file and reverse sign under colour reversal; a search improvement should preserve the answer while reducing node visits. These test the proposed mechanism directly.

   Separately, run a larger strength experiment with fixed budgets, balanced colours, varied openings, a fixed opponent and a reported confidence interval. Decide the required game count before looking at results. Sources: `course/module-01-board-and-evaluation/README.md`, `course/module-02-minimax-search/README.md`, `course/module-07-measure-and-scale/README.md`.

10. Keep the long run’s directory intact and use a new, unused directory for the smoke test, for example:
    ```bash
    python3 selfplay/train_loop.py --iterations 2 --games-per-iter 2 \
      --simulations 16 --train-steps 20 --arena-games 2 \
      --checkpoint-dir smoke_run_new
    ```
    Reusing the old directory does not resume it. The current program initializes a new champion unless given an initialization checkpoint, creates an empty buffer, restarts numbering at 1, and immediately truncates `run_log.csv`. Completed iterations overwrite `champion.pt`, `replay_buffer.npz`, and matching candidate filenames. Higher-numbered old candidate files can remain, mixing runs. A recorded configuration helps reproduction but cannot preserve overwritten artifacts. Source: `selfplay/train_loop.py`.

11. The exercise requires restoring the champion, replay buffer and completed iteration number, then continuing at the next iteration. The model checkpoint includes architecture and model state; the buffer file includes its contents and capacity. A consistent completed iteration means generation, storage, candidate training and the arena decision have finished, and the champion, buffer and log all represent that same iteration.

    The current files are written separately, so an interruption during saving could leave inconsistent artifacts; crash-consistent recovery is **not established** by the exercise’s round-trip test. If interrupted midway through training, restarting from the previous consistent iteration discards that partial work. Exact continuation inside the interrupted iteration would additionally need the candidate, training-step position, Adam state, relevant intermediate state and random-generator states.

    Adam need not survive a **completed iteration boundary** in this implementation: every `train_candidate` call constructs a fresh optimizer. Merely specifying the same seed does not restore the generators’ positions after prior random draws, nor establish identical execution across platforms. The current checkpoints do not save RNG states. Sources: `selfplay/train_loop.py`, `selfplay/replay_buffer.py`, `tests/course/test_module06_resume.py`, `selfplay/assets/mind_change/README.md`.

12. Before running, I would record the hypothesis, predicted direction and mechanism, isolated change, baseline, commands/settings, seeds, workload, primary metric, sample size, and what outcome would support or weaken the prediction. I would commit that dated plan in my fork’s `REPORT.md` before collecting results, preserving the original wording.

    Afterwards I would add measured results and uncertainty, artifact locations, deviations from the planned procedure, surprises, and the conclusion the evidence supports. Any explanation developed after seeing the outcome would be explicitly labelled as such. The course requires a hypothesis before the run and a committed `REPORT.md`, but a detailed pre-registration template or mandatory intermediate storage location is **not established**. Sources: `course/module-09-capstone/README.md`, `course/module-07-measure-and-scale/README.md`, `course/module-01-board-and-evaluation/README.md`.
