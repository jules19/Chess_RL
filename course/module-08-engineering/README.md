# Module 8: Engineering Extras

> **Where you are:** an optional module. The reinforcement learning (RL) loop works; this module is about keeping a long-running machine learning (ML) project reliable.
> **Time:** open-ended. Each exercise is 1-4 hours; the src-layout refactor (exercise 3) takes about a weekend.
> **You'll be able to:**
> - recognise the bug patterns that let ML code run without errors while measuring the wrong thing
> - keep a test suite fast enough that people run it
> - refactor a whole codebase safely, using the tests to tell you when something breaks
> - profile code (measure where its running time goes) before you optimise it

This module covers the software engineering around the ML work, using
real problems from this project. Every long-running ML project has the same
kinds of problem.

## Case studies from this repository

1. **The self-test that could never pass.** The original `__main__` block in
   `net/encoding.py` encoded the move e7e5 and decoded it again, using the
   starting position, where e7e5 is illegal. The test would have failed every
   time, but nothing ever executed it.
   Lesson: a test that doesn't run in continuous integration (CI, a service
   that runs the tests automatically on every change) is only documentation,
   and is probably wrong.
2. **A guard that skipped bad data.** A *guard* is a check that stops bad
   input from reaching the rest of the code. The legal-move mask (which sets
   every illegal move's probability to zero, see
   [Module 4](../module-04-neural-network/README.md)) had an
   `if index < 4672` guard. The promotion-encoding bug gave queen promotions
   indices beyond the end of the 4,672-number output. The guard skipped those
   moves without any error or warning, so for weeks the network never saw a
   queen promotion. Nothing crashed, so no test failed.
   Lesson: a guard should stop the program when it finds bad data. Prefer
   `assert`.
3. **The same setting stored in three places.** The architecture constants
   (2 residual blocks of 64 channels in one file, 4 blocks of 128 in
   another) disagreed across three files. No test compared the files. The
   problem ended when each checkpoint started storing its own architecture
   settings.
   Lesson: store each value in one place, because copies eventually differ.
4. **Documents that went out of date.** The old README called Phase 3b
   (network-guided search, now Module 5) "future work" after its code was
   merged. This course was also published claiming the network had "~830K
   parameters", when the actual figure was 1.8 million. The number came
   from an untested planning document.
   Lesson: keep fewer documents, derive numbers from the code, and keep old
   versions in git history (`docs/history/` preserves the originals).
5. **A measuring tool with no tests.** `training/evaluate.py` was published
   importing `MinimaxPlayer` and `MCTSPlayer`, classes that never existed.
   The imports were inside a try/except that printed a warning and skipped
   the baseline matches (the matches against the minimax and MCTS engines
   that every new network is compared with). For months the script produced
   no comparisons, and `training/` had no tests to notice. An external
   review found the problem. The fix and its regression
   tests (tests that check an old bug has not come back) are in
   `training/evaluate.py` (see the LESSON LEARNED docstring) and
   `tests/test_training.py`.
   Lesson: find the tools you rely on that have no tests, and test them.
6. **A score with the wrong meaning.** After every minimax move, the engine's
   Universal Chess Interface (UCI) code (the text protocol chess programs
   use to run an engine) printed `info ... score cp N`, where cp means
   centipawns (hundredths of a pawn). N was `evaluate()` of the position
   before the search: a static score, from White's point of view. When the
   engine played Black a queen up, it reported about −900, and a graphical
   interface showed it as losing.
   The protocol tests passed, because they only checked that a legal
   `bestmove` arrived. The fix reports the search score from the side to move
   (`best_move_minimax(..., return_score=True)`). The test
   `tests/test_uci.py::test_info_score_is_search_score_from_side_to_move`
   now checks the value of the score.
   Lesson: test what an output means, as well as that it exists.
7. **A validation set that wasn't held out.** A validation set is *held out*
   when none of its data is used for training. `training/dataset.py` split
   the data into training and validation sets with `random_split` over
   individual positions. Consecutive positions from one game are almost
   identical, so every game ended up in both sets, and validation accuracy
   partly measured memorisation. No test checked how the split was made.
   Now `split_by_game()` keeps each game in one set, and
   `tests/test_training.py::test_train_val_split_keeps_games_whole` checks
   that no game appears in both sets.
   Lesson: in ML, check most carefully the bugs that would make your
   metrics look better than they are.
8. **Instructions that nobody had run.** Module 4 used to tell learners to
   filter the January 2013 Lichess file with `--min-elo 2000` and a minimum
   time control of 10 minutes. Run as written, that keeps 3 games out of
   121,332, because hardly anyone was rated 2000 in 2013.
   The same review found two "Check your understanding" answers that the
   code contradicted (`evaluate()` on the start position returns 20, not 0,
   and training does not mask illegal moves). Nobody had run these steps.
   The fix was to run every step and quote measured output.
   `training/get_lichess.py` now does the filtering in code that tests can
   run.
   Lesson: run every instruction you publish, and treat unrun instructions
   as probably wrong.

## What's already in place (study it)

- `pyproject.toml`: packaging, extras (optional groups of dependencies:
  `train`, `dev`), and pytest configuration
- `tests/`: the fast suite (count the tests yourself: `pytest --collect-only -q`);
  every course module has a checkpoint here
- `.github/workflows/ci.yml`: CI that installs the CPU-only build of PyTorch
  and runs the tests on every push and every pull request

## Exercises

1. **Keep CI fast.**
   1. Add a `pytest --durations=5` step to CI. It lists the five slowest
      tests.
   2. Set a time budget of 60 s for the suite.
   3. If the suite takes longer, mark the slowest tests `@pytest.mark.slow`
      and exclude them from the default run.
   4. One test is already marked. Find it, and check that
      `pytest -m "not slow"` deselects it (leaves it out of the run).

   A fast suite matters because people run it often.
   **Check:** `pytest -m "not slow" -q` ends with `1 deselected` today; the
   suite takes about 15-20 s on a laptop CPU.
2. **Remove the path workarounds.** Every entry-point script calls
   `sys.path.insert(0, ...)`. With the package installed (`pip install -e .`)
   these calls are redundant.
   1. Remove them.
   2. Convert the scripts' `__main__` blocks to console entry points in
      `pyproject.toml`. A console entry point is a command that pip installs
      on your PATH, for example `chess-rl-play` or `chess-rl-selfplay`.
   3. Update the documents.

   **Check:** `grep -rln "sys.path.insert" --include=*.py . | grep -v ^./experiments/`
   lists 28 files today and should list none when you're done. `pytest -q`
   still passes, and each new command (for example `chess-rl-play --help`)
   runs.
3. **The src-layout refactor.** In a *src layout*, all of a project's code
   sits in one package inside a folder named src. Here the top-level packages are
   named `net`, `engine` and `search`. Once installed, they sit in
   site-packages (the folder where Python keeps installed packages), where
   such common names can clash with other packages (a comment in the project
   file warns about this). Restructure the code into
   `src/chess_rl/{engine,net,search,selfplay,...}`, fix all imports, and let
   the test suite tell you when you're done. This is the biggest exercise in
   the course, and the skill transfers to other projects.
   **Check:** the full test suite passes, and no import anywhere in the code
   still uses the old top-level package names.
4. **Profile before optimising.** Run a 100-simulation PUCT search (the
   network-guided search from Module 5) under `python -m cProfile -o prof.out`,
   which records the time spent in each function. View the results
   with `snakeviz` (a browser view) or `pstats` (a text report). Where does
   the time go? Compare it with what you expected. In Module 2, the first
   transposition-table version turned out slower than no table at all, and
   only timing showed it. Expect at least one result here to differ from
   your prediction.
   **Check (measured with the one-month network from Module 4, start
   position, CPU):** about 3.5 s in total. The network's forward passes
   (running a position through the network to get its output) take 3.4 s
   (96%). Most of that is in the convolutions (the filters that scan the
   board, see [Module 4](../module-04-neural-network/README.md)) of the four
   residual blocks. Board encoding and the legal-move mask together take about
   1%, and walking the search tree takes very little time. Optimising the
   Python search code would gain almost nothing; batching the network calls
   (Module 5, exercise 5) helps.
5. **Property-based tests for encoding and decoding.** A property-based test
   states a rule that must hold for every input, and a library checks it on
   many generated inputs. Rewrite the encode-then-decode test with
   `hypothesis` generating random legal positions (random legal move
   sequences from the start). The rule: decoding an encoded move gives back
   the same move. A test like this would have found the promotion bug as soon
   as it generated a queen promotion.
   **Check:** the property must fail when you point it at the encoder from
   before the fix (Module 4, exercise 2 shows how to get it), on the first
   queen promotion it generates. If it does not fail, it is not testing
   promotions.

## Check your understanding

<details>
<summary><b>Q1.</b> Case studies 6, 7 and 8 have something in common that a crash never has. What is it, and what kind of test catches it?</summary>

In all three, the code ran without errors and the output looked plausible,
but it meant something different from what everyone assumed: a score from
the wrong side, a validation set that wasn't held out, and a filter that
produced almost no data. A crash shows up immediately, so someone fixes it.
The test that catches these bugs checks the meaning of an output against a
case where you know the answer (a position where the score must be
negative, no game in both splits, a filter that must keep thousands of
games).
</details>

<details>
<summary><b>Q2.</b> Why is <code>if index &lt; 4672: ...</code> (case study 2) worse than no check at all?</summary>

Without the check, a bad index crashes on the first promotion and someone
fixes the encoder that day. With it, the bad moves are skipped without any
error or warning, and training carries on with a dataset missing every
queen promotion. Nobody notices for weeks. An `assert` would have stopped
the program at the first bad index.
</details>

<details>
<summary><b>Q3.</b> Your suite takes 4 minutes, so you've started running only the tests for the file you're editing. Which of this repository's bugs would that habit have let through?</summary>

The cross-file ones: case study 3 (architecture constants that disagreed
across three files) and case study 5 (`evaluate.py` importing classes that
didn't exist). A fast full suite, run every time, catches breakage in files
you weren't editing. Exercise 1 keeps the suite fast enough for that.
</details>

## Checkpoint

- [ ] CI passes on your fork (your own copy of the repository on GitHub) for
      a pull request that changes real code
- [ ] At least one exercise above merged, with tests

## Where to go next

- Endgame tablebases: precomputed databases of perfect play for positions
  with few pieces left. python-chess reads the Syzygy tablebase format
  directly.
- An opening book (a table of good first moves that the engine plays
  without searching), built from your own self-play games
- A Lichess bot account (an account played by an engine instead of a
  person): put your engine online and get a public rating. This is the
  strongest external check of the Module 7 checkpoint.
- Read the AlphaZero paper (Silver et al. 2018). After this course, you can
  match each part of it to a file you've edited.

Previous: [Module 7: Measure and Scale](../module-07-measure-and-scale/README.md) · Next: [Module 9: Capstone: The Experiment Report](../module-09-capstone/README.md)
