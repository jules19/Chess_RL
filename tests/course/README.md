# Course exercise tests: how your exercises are checked

Every file here is a test for one course exercise, and each test is skipped
until you do that exercise. The workflow:

1. Read the exercise in the module README. Each test's skip reason (the
   text in the skip line above the test) names the module and exercise.
2. Implement it in the real codebase.
3. Delete the entire `@pytest.mark.skip(...)` decorator; for resume, remove
   the module-level `pytestmark` instead.
4. Run `pytest tests/course/ -q`. A passing test means your implementation
   works.

These tests are different from the tests in `tests/`. Those check the
existing project code, and they pass on a fresh clone. The tests here fail
until you do the work, because they check your code. That is why they work
as checkpoints.

Each test's docstring states the requirements for its exercise: the exact
function signature and the behaviour you are implementing. If you disagree
with the requirements, you may change them. That is a design decision, and
explaining your reasons for it is part of the exercise.

Earlier reference solutions are on the
`course-solutions` branch: `git diff main course-solutions` shows them
all as one readable diff, and on that branch every test passes with no
skips. Use the local patch below for the revised resume exercise. The module pages also give measured results, so you can check your
numbers without looking at the solutions.


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
