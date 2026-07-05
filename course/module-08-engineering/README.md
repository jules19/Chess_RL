# Module 8 — Engineering Extras

**Goal:** the software craft around the science. This module works through
the engineering debt this project actually accumulated — because every
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

## What's already in place (study it)

- `pyproject.toml` — packaging, extras (`train`, `dev`), pytest config
- `tests/` — 53 fast tests; every course module has a checkpoint here
- `.github/workflows/ci.yml` — CPU-torch CI on every push/PR

## Exercises

1. **Keep CI honest.** Add a `pytest --durations=5` step to CI and set a
   budget: if the suite exceeds 60s, mark the slowest tests
   `@pytest.mark.slow` and exclude them from the default run (one test is
   already marked — find it, and check `pytest -m "not slow"` actually
   deselects it). Fast suites get run; slow ones get skipped by humans.
2. **Delete the sys.path hacks.** Every entry-point script does
   `sys.path.insert(0, ...)`. With the package installed (`pip install -e .`)
   they're redundant. Remove them, convert the scripts' `__main__` blocks to
   console entry points in `pyproject.toml` (e.g. `chess-rl-play`,
   `chess-rl-selfplay`), and update the docs.
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
5. **Property-based round-trips.** Rewrite the encoding round-trip test with
   `hypothesis` generating random legal positions (random legal move
   sequences from the start). One property test would have caught the
   promotion bug on day one.

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
