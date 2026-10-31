# Course exercise tests: how your exercises are checked

Every file here is a test for one course exercise, and each test is skipped
until you do that exercise. The workflow:

1. Read the exercise in the module README. Each test's skip reason (the
   text in the skip line above the test) names the module and exercise.
2. Implement it in the real codebase.
3. Delete the `@pytest.mark.skip(...)` line.
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

Reference solutions for every exercise here are on the
`course-solutions` branch: `git diff main course-solutions` shows them
all as one readable diff, and on that branch every test passes with no
skips. The module pages also give measured results, so you can check your
numbers without looking at the solutions.
