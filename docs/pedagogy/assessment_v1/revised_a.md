1. The network learns a **policy** (move probabilities) and a **value** (expected game result for the side to move). Human-game training uses the human’s played move and the game’s final result. Self-play uses the full PUCT root visit distribution and the eventual game result, signed for the side to move. Sources: `course/module-04-neural-network/README.md`, `course/module-06-selfplay-rl/README.md`.

2. Run `python3 selfplay/tictactoe_lab.py`. Its `vs_perfect` score counts games **not lost** by the learned table playing without search against a perfect opponent. Since that opponent cannot lose, these are draws; the reported evaluation uses 100 games. Source: `course/module-00-orientation/README.md`.

3. The shipped `early.pt` is a deliberately prepared teaching fixture, not a human-game-trained model. A randomly initialized one-block, eight-channel network received 48 Adam updates on one escape-square position, targeting `Rxa7` and a preparation value of zero. The lab then makes 60 updates on the original threat position using **only the policy component of `soft_target_loss()`**, targeting search visits. There is no completed game and therefore no value target. Source: `selfplay/assets/mind_change/README.md`.

4. An unfinished game at the length limit is scored as a draw, reward **0**. If nearly every game hits that limit, predicting zero produces tiny value loss without learning to distinguish wins from losses. It demonstrates fitting uninformative targets, not strong chess evaluation. Source: `course/module-06-selfplay-rl/README.md`, Part B.

5. With the course environment installed, open `tests/course/test_module02_search.py` and remove the **entire multiline** `@pytest.mark.skip(...)` decorator from `test_time_budget_is_a_hard_promise`. Before implementing the fix, run from the repository root:
   ```bash
   pytest tests/course/test_module02_search.py -q
   ```
   It searches the supplied middlegame with `max_depth=6`, `time_limit=6.0`, and requires a legal move in under eight seconds. The documented initial failure takes roughly 18 seconds on the reference machine; timing varies. **The test cannot expose this failure while skipped**, because its body does not run. Sources: `course/module-02-minimax-search/README.md`, `tests/course/test_module02_search.py`.

6. Fetch the reference branch and inspect only the evaluator diff:
   ```bash
   git fetch origin course-solutions:refs/remotes/origin/course-solutions
   git diff main origin/course-solutions -- engine/evaluator.py
   ```
   Neither command switches the learner’s working branch or replaces working files. The current exercise test remains the contract because the reference branch may lag. Source: `course/README.md`.

7. Search changes its tree: visits, backed-up values, and resulting move preference. It leaves network weights unchanged. Training adjusts the weights to make the network’s policy resemble the search distribution. Consequently, a later forward pass can prefer a different move without searching: the lesson is now encoded in the learned function. Sources: `course/module-05-puct/README.md`, especially the bridge lab.

8. **Useful generalisation is not established.** The change proves that fitting one board affected another, but it may simply suppress the capture everywhere. Use an otherwise similar control with an escape square, where the anticipated reply is not checkmate. If capture probability falls similarly there, the interpretation that the network learned the relevant danger is weakened. That is the documented lab result. Its escape-square control appeared in checkpoint preparation, so it is not an unseen test set. Sources: `course/module-05-puct/README.md`, `selfplay/assets/mind_change/README.md`.

9. I can defend the observed score, **3.5/6 ≈ 58.3%**, under that match’s conditions. Superior underlying strength is **not established** by six games. Separately, I would test the proposed mechanism on controlled positions: for example, compare before/after probabilities and search visits on a known tactical threat and an escape-square control, with identical weights where appropriate, seeds, and search budgets. For strength, predefine a larger comparison, balance colours, vary openings or randomness to avoid repeated identical games, keep computing budgets comparable, and report the score with its interval. Sources: `course/module-05-puct/README.md`, `course/module-07-measure-and-scale/README.md`.

10. Keep the long run in its existing directory and use a new, empty directory for each smoke run, such as `runs/m6_smoke_02`. The current program refuses an occupied output directory; there is no destructive override. A configuration record documents settings but does **not itself protect checkpoints or replay data**. Protection comes from separate run directories and the program’s refusal, with immutable completed snapshots supporting recovery. Source: `course/module-06-selfplay-rl/README.md`, Part B.

11. Resume must follow `completed.json`, validate the snapshot hashes and agreement with `config.json`, and restore the champion, replay buffer, completed iteration number, saved scientific settings, and committed log history. It must preserve configuration and prior snapshots, reject inconsistent state, and avoid trusting possibly stale top-level exports.

    A consistent completed iteration has its champion, candidate, buffer, and log written before the completion marker advances. An interrupted iteration is repeated from the preceding completed boundary. **Adam state need not survive that boundary here**, because `train_candidate` creates a new optimizer each iteration. Specifying the same seed does **not guarantee identical continuation**: Python, NumPy, and torch RNG states are not saved. Resume is an exercise to implement, not an already available CLI capability. Sources: `course/module-06-selfplay-rl/README.md`, `tests/course/test_module06_resume.py`.

12. Before running, record the prediction, its reason, the comparison, the measurement, and what result would count against it. Also record command/settings/seed, results directory, and code version. Keep this in `JOURNAL.md`, initially copied without overwriting from `course/JOURNAL_TEMPLATE.md`. Preserve the original prediction even if wrong. Afterwards, add observations and uncertainty, what they support and do not establish, and the next question; carry these entries into the final report. Sources: `course/JOURNAL_TEMPLATE.md`, `course/module-00-orientation/README.md`, `course/README.md`.
