# Course exercises: checks, hints and recovery

The project regression tests pass before you start. Course tests are skipped
until you choose their exercise. For one exercise at a time:

1. Read its test docstring: it specifies the function and behaviour to build.
2. Remove its **entire** `@pytest.mark.skip(...)` decorator, including any
   continuation lines. For the resume exercise, remove the module-level
   `pytestmark` line instead; it enables that exercise's whole group of tests.
3. Run just that file before implementing. It may fail with a missing import,
   an unsupported argument, or a behaviour assertion. A skip is not a failure
   and is not evidence that your implementation works.
4. Implement the change and rerun that file. Then run the regression suite.
   Leave unrelated exercises skipped until you reach them.

## One hint before the solution

| Exercise test | First thing to inspect | Reference source file |
|---|---|---|
| `test_module01_evaluator.py` | Reward open files for both colours with opposite signs | `engine/evaluator.py` |
| `test_module02_search.py` | Check the clock inside recursion; retain the last completed depth | `search/minimax.py` |
| `test_module02_killers.py` | A killer changes order, never the minimax score | `search/minimax.py` |
| `test_module03_mcts.py` | Pass one RNG through every random choice, not just the root | `search/mcts.py` |
| `test_module04_augmentation.py` | Mirror both board and move; transform twice to recover the original | `net/encoding.py` |
| `test_module04_first_network.py` | Compare held-out material error with always predicting the mean | `training/first_network.py` |
| `test_module05_puct.py` | The value is for the side to move, not always White | `search/puct.py` |
| `test_module06_nstep.py` | Negate a bootstrapped value when the number of plies is odd | `selfplay/tictactoe_lab.py` |
| `test_module06_network.py` | The replacement must satisfy the table's interface, including caching | `selfplay/tictactoe_lab.py` |
| `test_module06_resume.py` | Use the supplied loader; reconnect the champion, experience and log | Local patch below |

The time-budget test is intentionally slow when it fails. The course's
18-second estimate is a reference measurement, not a deadline that every
machine must reproduce. It cannot demonstrate the failure until enabled.

## Inspect one reference without changing your branch

From the repository root:

```bash
git fetch origin course-solutions:refs/remotes/origin/course-solutions
git diff main origin/course-solutions -- engine/evaluator.py
```

Substitute the source file from the table for the exercise you are doing.
For a complete file, use `git show origin/course-solutions:engine/evaluator.py`.
These commands do not switch branches or replace your work. Fetch needs network
access; if the remote branch is already present, the read commands work offline.
The remote branch may lag behind this course. Treat it as an explanation of an
approach; the current test remains the contract. Avoid copying an entire older
file over other exercises you have already completed.

## Revised resume reference

Module 6 exercise 9 has a local reference patch using the supplied `load_run()` helper: [module06_resume.patch](../../course/solutions/module06_resume.patch).
Read the patch first. To try it in a disposable copy after saving your work:

```bash
git apply --check course/solutions/module06_resume.patch
git apply course/solutions/module06_resume.patch
```

The first command verifies that it applies without changing files. If it fails,
your learner edits differ; inspect the patch and adapt the idea instead of
forcing it. The patch does not enable exercise tests for you. Remove that
file's `pytestmark` and run `pytest tests/course/test_module06_resume.py -q`.
It restores saved scientific settings; `--iterations` means additional
iterations, while device/verbosity can change. It does not restore RNG state.
The original `config.json` is preserved. By default the last complete save
replaces older saves, while the cumulative log is kept. Run only one writer
per experiment directory. File validation belongs to the supplied helper,
not the learner's implementation.

If you disagree with a test's requirements, explain and test your alternative.
That is a design decision, not a reason to hide a failure behind a skip marker.
