# Pedagogy rubric

How well does this repository teach someone to build a chess engine, from random
moves up to a neural network? Use this rubric to score the project's teaching material
before and after changes, so improvements are measured rather than asserted.

**Learner persona:** a competent Python programmer who knows basic chess rules but has
never written a game engine, search algorithm or neural network. They arrive at the
repository root and want to understand and reproduce the project.

**Scope:** everything a learner would read or run. That means the README, the other root
`*.md` files, `docs/`, the `*.html` tutorials, `notebooks/`, and the docstrings in the modules those
point to.

*Scope change (2026-10-30):* `docs/journal/` holds the project's day-by-day logs, and its README
labels it as history, not teaching material. It is out of scope for criteria 1–8 and 10.
Criterion 9 still judges whether that separation is clear. The baseline was scored when those
files were at the root and in scope; `check_docs.py` still reports the journal's broken
references separately so they don't disappear from view.

## Criteria (each scored 0–4)

| # | Criterion | 0 | 2 | 4 |
|---|---|---|---|---|
| 1 | **Learning path:** one obvious entry point and a clear order through the material | No entry point; many competing files | An entry point exists but the order is unclear or contradicted elsewhere | One start page; every unit says what comes before and after |
| 2 | **Learning objectives:** each unit states what the learner will be able to do | None | Some units, vague | Every unit, concrete and checkable |
| 3 | **Scaffolding:** intuition before formalism; prerequisites named | Jumps straight to code or jargon | Mixed | Each concept is motivated, built up and connected to the previous one |
| 4 | **Worked examples tied to the real code:** explanations point to the actual files and functions, and those references are correct | No links to code, or wrong ones | Some correct links | Most concepts are anchored to correct file/function references |
| 5 | **Active learning:** exercises, predict-then-check prompts, retrieval questions | None | A few, without answers | Every unit has practice, with answers or a way to self-check |
| 6 | **Feedback:** the learner can verify they got it right (expected output, tests, checks) | None | Some expected outputs | Every hands-on step has a verifiable expected result |
| 7 | **Accuracy and honesty:** claims about strength or results match the code and the measurements; targets are not presented as results | Many unsupported or false claims | Some | Claims are measured, or clearly labelled as estimates or targets |
| 8 | **Coverage:** every phase that exists in the code is taught (random → minimax → MCTS → NN → NN-MCTS) | Most phases untaught | Some gaps | Every implemented phase has teaching material |
| 9 | **Signal-to-noise:** the learner isn't buried in duplicate, stale or process-log documents | Learning material lost among logs | Some clutter | Teaching material is clearly separated from project history |
| 10 | **Currency:** material reflects the current state (nothing built is still described as "coming next", and no commands are broken) | Largely stale | Some stale parts | Current |

**Total: 40.** Graders must cite evidence (a file and what they saw) for every score.

## Objective checks (automated)

`python docs/pedagogy/check_docs.py` reports:
- **Broken file references:** paths like `search/mcts.py` mentioned in learner docs that don't exist.
- **Broken commands:** `python …` commands from learner docs that crash. Each is run
  with a timeout and stdin closed. A timeout counts as "runs"; a traceback counts as "broken".
- **Exercise count:** the number of exercises or self-check questions found in learner material.

These checks complement criteria 4, 5, 6 and 10. They don't replace the human judgement.

## Score history (independent blind graders, identical prompt each time)

| Date | State | Score | check_docs.py (refs / commands broken) |
|---|---|---|---|
| 2026-10-30 | Original (commit a1908f8) | 11/40 | 24 / 14 |
| 2026-10-30 | + LEARN.md, 10 units, accurate STATUS/README, journal moved | 35/40 | 5 / 0 (false positives) |
| 2026-10-30 | + tutorial caveats, README trimmed, checker fixes | 34/40 | 0 / 0 |
| 2026-10-30 | + objectives and self-checks in HTML tutorials | 36/40 | 0 / 0 |

Graders differ by about ±1 point. The learner-outcome results are in `OUTCOME_TEST.md`.
