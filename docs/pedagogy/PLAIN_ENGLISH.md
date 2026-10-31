# Plain-English rubric

Does the learner-facing writing read like a well-edited professional tutorial? This rubric
measures the *language*, not the teaching design (that's [RUBRIC.md](RUBRIC.md)). Use it before
and after editing, so improvements are measured rather than asserted.

**Reader:** a competent Python programmer who knows the chess rules but has never written a game
engine, a search algorithm or a neural network. English may not be their first language.

**The standard:** technical ideas are taught in full, not avoided or watered down. What changes is
how they're written: every term is explained in plain words when it first appears, and the prose
says exactly what it means, with no figures of speech, slang or AI-style flourishes.

**Scope:** the files listed in `plain_check.py` (`README.md`, `QUICKSTART.md`, `STATUS.md`,
`course/`, `tests/course/README.md`, `docs/*.md`). Code blocks, commands and code comments are out
of scope.

## Criteria (each scored 0–4, per file)

| # | Criterion | 0 | 2 | 4 |
|---|---|---|---|---|
| 1 | **Technical ideas explained clearly.** Each technical term is explained in plain words when it first appears, or links to where it is. Intuition comes before formulas. Hard ideas are explained, not skipped. | Terms used cold; formulas with no explanation | Most terms explained, some used before they're defined | Every term explained at first use; a newcomer could follow each explanation |
| 2 | **No unnecessary jargon.** No unexplained abbreviations, insider shorthand or developer/ML slang where an everyday word would do. | Dense with shorthand | Some shorthand | Technical words only where the idea needs them |
| 3 | **Literal language.** No idioms, metaphors, jokes or figurative verbs ("demolish", "bolted on", "the rope is there"). An analogy is fine only if it helps understanding and is plainly an analogy. | Figurative language throughout | A few figures of speech | Says exactly what it means |
| 4 | **No AI-style writing.** No chains of dashes, "it's not X, it's Y" contrasts, dramatic one-line fragments, bold scattered for emphasis, hype, self-narration, rhetorical questions or filler adverbs ("honestly", "quietly", "genuinely"). | Reads as generated | Some tells | Reads as written by a careful human editor |
| 5 | **Clear structure and professional tone.** Short active sentences, one idea per sentence, steps numbered with the expected result, the same word used for the same thing, a calm and consistent tone. | Hard to follow | Mostly clear, uneven | Consistently clear and professional |

**Per file: 20.** A file passes at **17/20 or more with no criterion below 3.** The material
passes when every file does.

## How to grade

Use fresh, independent graders (people, or agents that haven't seen the edits). Give them this
file and the read-only repository, and keep the grading prompt identical between runs. Graders
must not see earlier scores. For every score below 4, a grader quotes the passages that cost the
points, so the result can be checked and acted on.

## Mechanical check

`python docs/pedagogy/plain_check.py` reports, per file: dashes per 1,000 words, flagged phrases
(figurative, casual and AI-style wording, with line numbers), average sentence length and the
share of sentences over 30 words. Targets: no more than 3 dashes per 1,000 words, no flagged
phrase left unless it is a technical term used literally (for example "killer move"), and
under 10% of sentences over 30 words. The script finds candidates; the graders judge them.

Editing must not change meaning. Every fact, number, command, file reference and code block stays
as it was, and `python docs/pedagogy/check_docs.py` must still report 0 broken references and 0
broken commands.

## Score history

| Date | State | Files passing | Mean /20 | plain_check (dash/1k, flags) |
|---|---|---|---|---|
| 2026-10-31 | Baseline (main at 45a778b), 19 files, 4 graders | 0/19 | 11.6 (range 8-15) | 6.9, 81 |
| 2026-10-31 | First editing pass, 4 fresh graders | 10/19 | 16.8 (range 15-20) | 0.0, 0 |
| 2026-10-31 | Second editing pass, 4 fresh graders | 18/19 (Module 5: 16) | 17.6 (range 16-19) | 0.0, 0 |
| 2026-10-31 | + Module 5 fixes, regraded by a fresh grader | 19/19 | 17.7 (range 17-19) | 0.0, 0 |

Lowest scores in the last full round (17/20, the pass mark): README, STATUS,
course/README, RL_CONCEPTS, Modules 2, 4, 5, 6 and 7. The remaining points are
mostly terms used a few lines before they are explained, and some names that
vary for the same thing. The graders' lists for that round are a good place to
start the next pass.
