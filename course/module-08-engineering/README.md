# Module 8 — Engineering Extras

> **Where you are:** a side road. The RL loop works; this module is about keeping a long-running ML project trustworthy.
> **Time:** open-ended. Each exercise is 1-4 hours; the src-layout refactor is a weekend.
> **You'll be able to:**
> - recognise the bug patterns that let ML code run happily while measuring the wrong thing
> - keep a test suite fast enough that it actually gets run
> - refactor a whole codebase safely, with the tests as your rope
> - profile before you optimise, and say where the time really goes

The software craft around the science. This module works through the
engineering debt this project actually accumulated — because every
long-running ML project accumulates the same kinds.

## Case studies from this repo (read the scars)

1. **The self-test that could never pass.** `net/encoding.py`'s original
   `__main__` block round-tripped the move e7e5 *against the starting
   position, where it's illegal* — the test would fail every time. It
   survived because nothing ever executed it. Moral: a test that doesn't run
   in CI is documentation, and probably wrong documentation.
2. **Silent guards hide bugs.** The `if index < 4672` "sanity check" in the
   legal-move mask quietly swallowed the promotion-encoding bug for weeks
   (see Module 4). Guards that *skip* instead of *crash* convert loud bugs
   into silent data corruption. Prefer `assert`.
3. **Config drift.** Architecture constants (2/64 vs 4/128) disagreed across
   three files until checkpoints became self-describing. Any value written
   twice will eventually differ.
4. **Docs rot faster than code.** The old README declared Phase 3b "future
   work" after its code was merged, and this course itself shipped claiming
   the network had "~830K parameters" (actual: 1.8M — the number was copied
   from a planning doc nobody ran). The fix isn't more docs — it's *fewer,
   load-bearing* docs, numbers derived from code, and history in git where
   it can't lie (`docs/history/` preserves the originals).
5. **Untested tools break silently — even the measuring instruments.**
   `training/evaluate.py` shipped importing `MinimaxPlayer` and
   `MCTSPlayer` — classes that never existed — inside a try/except that
   printed a warning and skipped the baseline matches. The script that was
   supposed to measure everything else measured nothing, for months,
   because `training/` had zero test coverage. It was found by an external
   review, not by a user; the fix and its regression tests are in
   `training/evaluate.py` (see the LESSON LEARNED docstring) and
   `tests/test_training.py`. Coverage isn't about percentages — it's about
   which tools you're trusting blind.
6. **A number with the wrong meaning looks fine.** The UCI engine printed
   `info ... score cp N` after every minimax move, and the protocol tests
   passed, because they only checked that a legal `bestmove` arrived. But N
   was `evaluate()` of the position *before* the search: a static score, from
   White's point of view. Playing Black a queen up, the engine reported
   about −900, and a GUI showed the engine "losing". Nothing crashed, so
   nothing flagged it. The fix reports the search score from the side to
   move (`best_move_minimax(..., return_score=True)`), and
   `tests/test_uci.py::test_info_score_is_search_score_from_side_to_move`
   pins the *meaning* of the number, not just its presence. Moral: test what
   an output means, not only that it exists.
7. **A validation set that wasn't held out.** `training/dataset.py` split
   train and validation with `random_split` over individual *positions*.
   Consecutive positions from one game are near-duplicates, so every game ended up
   on both sides and validation accuracy partly measured memorisation. The
   code ran, the numbers looked plausible, and no test could tell. Now
   `split_by_game()` holds out whole games, and
   `tests/test_training.py::test_train_val_split_keeps_games_whole` checks
   that no game appears in both sets. Moral: in ML, the most dangerous bugs
   are the ones that make your metrics look *better*.
8. **A recipe nobody had run.** Module 4 used to tell learners to filter the
   January 2013 Lichess file with `--min-elo 2000` and the default 10-minute
   minimum. Run as written, that keeps **3 games out of 121,332**: hardly
   anyone was rated 2000 in 2013. The same review found two "Check your
   understanding" answers that the code contradicted (`evaluate()` on the
   start position returns 20, not 0; training does *not* mask illegal
   moves). Each read plausibly; none had been executed. The fix was to run
   every step and quote measured output, and `training/get_lichess.py` now
   encodes the recipe as code, which tests can run. Moral: instructions are
   code too. If nobody has run them, assume they're wrong.

## What's already in place (study it)

- `pyproject.toml` — packaging, extras (`train`, `dev`), pytest config
- `tests/` — the fast suite (count them yourself: `pytest --collect-only -q`);
  every course module has a checkpoint here
- `.github/workflows/ci.yml` — CPU-torch CI on every push/PR

## Exercises

1. **Keep CI honest.** Add a `pytest --durations=5` step to CI and set a
   budget: if the suite exceeds 60s, mark the slowest tests
   `@pytest.mark.slow` and exclude them from the default run (one test is
   already marked — find it, and check `pytest -m "not slow"` actually
   deselects it). Fast suites get run; slow ones get skipped by humans.
   **Check:** `pytest -m "not slow" -q` ends with `1 deselected` today; the
   suite takes about 15-20 s on a laptop CPU.
2. **Delete the sys.path hacks.** Every entry-point script does
   `sys.path.insert(0, ...)`. With the package installed (`pip install -e .`)
   they're redundant. Remove them, convert the scripts' `__main__` blocks to
   console entry points in `pyproject.toml` (e.g. `chess-rl-play`,
   `chess-rl-selfplay`), and update the docs.
   **Check:** `grep -rln "sys.path.insert" --include=*.py . | grep -v ^./test/`
   lists 28 files today and should list none when you're done; `pytest -q`
   stays green; and each new command (e.g. `chess-rl-play --help`) runs.
3. **The src-layout refactor.** Top-level packages named `net`, `engine`,
   `search` are collision bait in site-packages (the pyproject even warns
   about it). Restructure into `src/chess_rl/{engine,net,search,selfplay,...}`,
   fix all imports, and let the test suite tell you when you're done. This
   is the biggest exercise in the course and the most transferable: safe
   large-scale refactoring with a test harness as your rope.
4. **Profile before optimizing.** `python -m cProfile -o prof.out` a 100-sim
   PUCT search, then look at it with `snakeviz` or `pstats`. Where does the
   time actually go? Compare with what you *expected* (Module 2's TT story
   suggests you'll be wrong at least once).
   **Check (measured, one-month network, start position, CPU):** about
   3.5 s in total, of which the network's forward passes are 3.4 s (96%),
   most of that in the four residual blocks' convolutions. Board encoding
   and the legal-move mask together are about 1%, and the tree walk is
   barely visible. So optimising the Python search code would gain almost
   nothing; batching the network calls (Module 5, exercise 5) is the lever.
5. **Property-based round-trips.** Rewrite the encoding round-trip test with
   `hypothesis` generating random legal positions (random legal move
   sequences from the start). One property test would have caught the
   promotion bug on day one.
   **Check:** the property must fail when you point it at the pre-fix
   encoder (Module 4, exercise 2 shows how to get it), on the first queen
   promotion it generates. If it doesn't, it isn't testing the edge that
   mattered.

## Check your understanding

<details>
<summary><b>Q1.</b> Case studies 6, 7 and 8 have something in common that a crash never has. What is it, and what kind of test catches it?</summary>

Nothing failed. The code ran, the output looked plausible, and the number
meant something different from what everyone assumed: a score from the
wrong side, a validation set that wasn't held out, a recipe that produced
almost no data. A crash announces itself; these don't. The test that catches
them checks the *meaning* of an output against a case where you know the
answer (a position where the score must be negative, no game in both splits,
a filter that must keep thousands of games).
</details>

<details>
<summary><b>Q2.</b> Why is <code>if index &lt; 4672: ...</code> (case study 2) worse than no check at all?</summary>

Without the check, a bad index crashes on the first promotion and someone
fixes the encoder that day. With it, bad moves are silently skipped and
training carries on with a dataset missing every queen promotion, which
nobody notices for weeks. A guard that skips turns a loud bug into silent
data corruption. Use `assert`.
</details>

<details>
<summary><b>Q3.</b> Your suite takes 4 minutes, so you've started running only the tests for the file you're editing. Which of this repo's bugs would that habit have let through?</summary>

The cross-file ones: case study 3 (architecture constants that disagreed
across three files) and case study 5 (`evaluate.py` importing classes that
didn't exist). A fast full suite, run every time, is what catches breakage
somewhere you weren't looking. That's the point of exercise 1.
</details>

## Checkpoint

- [ ] Your fork's CI is green on a PR that changes real code
- [ ] At least one exercise above merged, with tests

## Where to go next

- Endgame tablebases (python-chess reads Syzygy natively)
- An opening book from your own self-play games
- Lichess bot account: put your engine online and get a *public* rating —
  the ultimate Module 7 checkpoint
- Read the AlphaZero paper (Silver et al. 2018) — after this course, every
  paragraph maps onto a file you've edited

Previous: [Module 7 — Measure & Scale](../module-07-measure-and-scale/README.md) · Next: [Module 9 — Capstone: The Experiment Report](../module-09-capstone/README.md)
