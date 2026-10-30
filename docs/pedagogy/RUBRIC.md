# Pedagogy rubric

How well does this repository teach someone to build a chess engine, from a random mover to
AlphaZero-style self-play? Use this rubric to score the teaching material before and after
changes, so improvements are measured rather than asserted.

**Learner persona:** a competent Python programmer who knows the chess rules but has never written
a game engine, a search algorithm or a neural network. They arrive at the repository root and want
to work through the course.

**Scope:** what a learner reads or runs: `README.md`, `QUICKSTART.md`, `STATUS.md`, `course/`,
`tests/course/README.md`, `docs/*.md`, `docs/tutorials/`, and the docstrings of the modules the
course points to. `docs/history/` and `docs/plans/` are labelled project history and are out of
scope, except for criterion 9 (is that separation clear?).

## Criteria (each scored 0–4)

| # | Criterion | 0 | 2 | 4 |
|---|---|---|---|---|
| 1 | **Learning path:** one obvious entry point and a clear order through the material | No entry point; many competing files | An entry point exists, but the order is unclear or contradicted elsewhere | One start page; every module says what comes before and after |
| 2 | **Learning objectives:** each module states what the learner will be able to do | None | Some modules, vague | Every module, concrete and checkable |
| 3 | **Scaffolding:** intuition before formalism; prerequisites named | Jumps straight to code or jargon | Mixed | Each concept is motivated, built up and connected to the previous one |
| 4 | **Worked examples tied to the real code:** explanations point to the actual files and functions, and those references are correct | No links to code, or wrong ones | Some correct links | Most concepts anchored to correct file/function references |
| 5 | **Active learning:** exercises, predict-then-check prompts, retrieval questions | None | A few, without answers | Every module has practice, with answers or another way to self-check |
| 6 | **Feedback:** the learner can verify they got it right (tests, expected output) | None | Some | Every hands-on step has a verifiable result (e.g. a pytest checkpoint) |
| 7 | **Accuracy and honesty:** claims about strength and results match the code and the measurements; targets aren't presented as results | Many unsupported claims | Some | Claims are measured, or clearly labelled as estimates/targets |
| 8 | **Coverage:** every phase that exists in the code is taught | Most phases untaught | Some gaps | Every implemented phase is taught |
| 9 | **Signal-to-noise:** the learner isn't buried in duplicate, stale or process-log documents | Learning material lost among logs | Some clutter | Teaching material clearly separated from project history |
| 10 | **Currency:** material reflects the current state (nothing built is described as "coming next"; no broken commands) | Largely stale | Some stale parts | Current |

**Total: 40.** Graders must cite evidence (a file and what they saw) for every score.

## How to grade

Use a fresh, independent grader (a person, or an agent that hasn't seen the changes) with this
file and the read-only repository. It must not see earlier scores. Keep the grading prompt
identical between runs so scores are comparable. Expect about ±1 point between graders.

## Objective checks

`python docs/pedagogy/check_docs.py` runs every `python …`/`pytest …` command found in learner docs
inside a temporary copy of the repo, and checks every file the docs mention. Commands that read a
Lichess download are reported as "needs data", interactive ones as "interactive", and shell
one-liners (`&&`, `source`) as skipped. It also counts exercise and self-check prompts. It
complements criteria 4, 5, 6 and 10; it doesn't replace judgement.

## Score history

| Date | State | Rubric | check_docs.py (refs / commands broken) |
|---|---|---|---|
| 2026-10-30 | 9-module course (main at 39b207e) | 32/40 | 2 / 1 |
| 2026-10-30 | + tutorial reference fixes, README data note, UCI-score and split-by-game fixes, Module 8 case studies 6–7 | 34/40* | 0 / 0 |

\* Graded on a copy without `docs/pedagogy/` (it holds the answer keys), so the grader marked
STATUS.md's pointer to it as broken (currency 2/4). That folder exists in the repository. The
Module 8 "Next" link and the `docs/tutorials/` banners were added after this grade.

### What the graders said would raise the score next

1. **Objectives as checkable lists.** Each module opens with a one-line *Goal*. Rewriting it as 3–4
   "you'll be able to…" outcomes would let learners tick off what they've learned (criterion 2).
2. **Self-checks for Modules 8–9.** These are the only modules without "Check your understanding"
   questions (criterion 5).
3. **Expected ranges for measurement exercises.** For example Module 2's node counts and killer moves,
   and Module 5's c_puct sweep: without a plausible range, a learner can't tell a bug from noise
   (criterion 6).
4. **A small bundled dataset for Module 4.** The supervised-training part needs a large Lichess
   download and hours of CPU. A tiny shipped PGN would let laptop learners finish the loop
   (criterion 6).
5. **README's Elo ladder** still headlines historical guesses, footnoted as such. Module 7's measured
   numbers should replace them once someone runs it (criterion 7).
