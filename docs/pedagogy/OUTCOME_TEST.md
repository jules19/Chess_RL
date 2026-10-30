# Outcome test: does the material actually teach?

The rubric (`RUBRIC.md`) judges whether material *looks like* good teaching. An outcome test
measures whether a newcomer who studies it ends up understanding **this project** correctly.

## Method (A/B, blind)

1. **Two copies** of the repository: A is the version to compare against (e.g.
   `git archive <old-commit>`), B is the current one. Leave out `docs/pedagogy/` (it holds the
   answer keys), `checkpoints/`, `runs/`, logs and `.git`.
2. **Learners.** Fresh agents with *identical* prompts (or people). Persona: a Python programmer
   who knows the chess rules but has never built an engine. They start at the README, get a fixed
   reading budget (e.g. 20–25 tool calls), may read but not execute code, and must answer
   "don't know" instead of using general knowledge. Use at least two learners per arm, and use
   smaller models to stay closer to a real beginner, since strong models can reconstruct a lot from
   source code alone.
3. **Two quizzes**, written for this project so that general knowledge of chess engines scores nothing:
   - *recall*: project-specific facts, numbers, quirks and the measured state;
   - *transfer*: "what would happen if…" and "design this" questions that need understanding.
   Keep each quiz's answer key in a separate file, and verify every key answer against the code
   before running.
4. **Blind grading.** Anonymise and shuffle the answer sets, then have a separate grader mark them
   against the keys (0 / 0.5 / 1 per question) without knowing which arm each came from.

## Caveats

- If the quiz author also wrote the material, the quiz checks the ideas that author thought were
  important, not every learning goal.
- Small samples: treat a point or two of difference as noise.
- Model learners hit a ceiling quickly on readable code. The informative test is still a human
  beginner working through the course while someone notes where they get stuck.

## Results

The method was first used on 2026-10-30 against an earlier, separate learning path (branch
`docs/learning-path-and-pedagogy`, not merged). There, learners who studied the improved material
scored 14/14 against 12.5/14 on recall, and 20/20 against 18/20 on transfer, with half the reading.
The differences were concentrated on understanding why MCTS behaves as it does and on knowing the
project's real state.

**Not yet run on this course.** Its quizzes need to be written and keyed against the course's
code (PUCT, self-play, the fixed encoding). That is the next step for this evaluation.
