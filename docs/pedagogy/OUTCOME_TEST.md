# Outcome test: does the material actually teach?

The rubric (`RUBRIC.md`) judges whether material *looks like* good teaching. This test
measures the outcome: after studying the repository, does a newcomer understand **this project**
correctly?

## Method (A/B, blind)

1. **Two copies.** Copy A is the version to compare against (e.g. `git archive <old-commit>`).
   Copy B is the current version. Leave out `docs/pedagogy/` (this folder), `checkpoints/`, `runs/`, logs and `.git`.
2. **Two learners.** Give each a fresh agent with the *same* prompt. The persona is a Python
   programmer who knows the chess rules but has never built an engine. It starts at the README,
   has a budget of 25 tool calls, may read but not execute, and answers "don't know" rather than
   filling gaps with general knowledge.
3. **The quiz** (`quiz.md`): 14 questions that only this project can answer. They cover numbers,
   quirks, limitations and the measured status, so general knowledge of chess engines scores
   nothing.
4. **Blind grading.** A separate grader marks both answer sets against `key.md` (0 / 0.5 / 1 per
   question) without knowing which copy is which.

## Caveats

- The quiz was written by the same person who wrote the new learning units, so they cover the same
  ground. The test shows whether the material conveys the facts and ideas the author judged
  important, not every possible learning goal.
- One learner per arm is a small sample. Treat differences of a point or two as noise.
- Everything the quiz asks is also answerable from the code alone, so copy A is not set up to
  fail. It only lacks guidance.

## Results

| Date | A (version) | B (version) | A score | B score | Reading used (tool calls) A / B |
|---|---|---|---|---|---|
| 2026-10-30 | commit a1908f8 (before the learning path) | working tree with `LEARN.md` + units | 12.5 / 14 | **14 / 14** | 8 / 4 |

**What the 2026-10-30 run showed:**
- Learner A lost points on **Q10**, the trained network's measured strength. The original docs held only
  targets, and it reported "training is awaiting dependencies". It also lost half a point on **Q5**, why MCTS picks odd
  moves: no material explained it.
- A answered most other questions by **reading the code directly**. The project's code is readable, so
  a capable reader can recover the facts without the docs. The learning path's value showed up as
  *the right overall picture* (measured status, why the search behaves as it does) at **half the reading effort**.
- **Ceiling effect:** the simulated learner is a strong model, not a real beginner, and B hit 14/14. A harder
  quiz, or human learners, would separate the versions more. Treat this as a check that the new
  material is correct and complete, not as an effect size.
- Learner A also surfaced a measured result that the new STATUS.md had left out (the minimax
  tactical-puzzle pass rate). It has been re-measured and added.

### Run 2 (2026-10-30): transfer quiz, weaker learners, 2 per arm

To reduce the ceiling effect: `transfer_quiz.md` (10 "what would happen if…" and design questions,
key in `transfer_key.md`) answered by **Haiku** learners, **two per arm**. Same copies, same rules
(20 tool calls, no code execution). The four answer sets were anonymised and shuffled before blind grading.

| Arm | Learner 1 | Learner 2 | Total |
|---|---|---|---|
| A: original | 9 / 10 | 9 / 10 | **18 / 20** |
| B: learning path | 10 / 10 | 10 / 10 | **20 / 20** |

The **same two areas** separated the arms in both runs:
- **MCTS statistics** (Q3 here, Q5 in run 1). Original-repo learners gave generic "visits concentrate"
  answers. Learning-path learners reasoned from ~10 visits per root move to ~100, and from the cost.
- **The project's real state** (Q10 here, Q10 in run 1). One original-repo learner relied on the old
  STATUS.md ("training pending; data size unknown"), which is exactly the misinformation the rewrite removed.

Everything else was answered equally well from the code by both arms. The code is readable, so
the material's added value is concentrated in these two areas: understanding why the search behaves as it does, and
an accurate picture of the project's state. It also halves the reading effort. The key for Q2 was
corrected before grading: every learner rightly noted that ordering can change the move chosen *among
tied scores*.

**Remaining limitation:** model learners can reconstruct much from source code. Only human
beginners would show how much the units' scaffolding (intuition first, exercises) helps someone who
can't.
