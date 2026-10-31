# Course exercise tests — the real autograder

Every file here is a **deliberately skipped** test for one course exercise.
The workflow:

1. Read the exercise in the module README (each test's skip reason names it).
2. Implement it in the real codebase.
3. Delete the `@pytest.mark.skip(...)` line.
4. `pytest tests/course/ -q` — green means your implementation works.

These are different from the tests in `tests/`: those protect the
maintainer's code and pass on a fresh clone. These **fail until you do the
work** — they verify *your* code, which is what makes them checkpoints
rather than decoration.

Each test's docstring is the exercise contract: the exact function
signature and behavior you're implementing. If you disagree with the
contract, changing it is allowed — that's a design decision, and defending
it is part of the exercise.

Reference solutions live on the `course-solutions` branch for the
exercises of Modules 1-6 that predate October 2026. The two newest
(Module 6 exercises 6 and 7: `test_module06_nstep.py` and
`test_module06_network.py`) have no published solution yet; their
module text quotes measured results from a reference solution, so you can
still check your numbers.
