# Module 9 — Capstone: The Experiment Report

**Goal:** produce the one artifact that proves you completed this course —
a short experiment report that could convince a skeptical engineer. Modules
0–8 each checked a piece; this is the summative assessment, and it cannot
be completed by running commands.

## The deliverable

One document (markdown is fine, ~2 pages), committed to your fork as
`REPORT.md`, containing four sections:

### 1. The measured ladder

A table of *your* engines against a calibrated opponent (Module 7's rig —
Stockfish at fixed strength via `cutechess-cli`, ≥100 games per row):

| Engine | Opponent | Games | Score | Elo diff (±95%) |
|---|---|---|---|---|
| minimax d3 | SF-1400 | 100 | … | … |
| MCTS 200 | SF-1400 | 100 | … | … |
| PUCT + your net | SF-1400 | 100 | … | … |

Every number measured by you, every Elo with an error bar, and one
sentence comparing the result to the repo's historical self-estimates
(~1200–1600). The gap between guess and measurement is a finding, not an
embarrassment.

### 2. The training story

For your supervised net (Module 4) and your self-play run (Module 6):
dataset size, wall-clock, final train/val losses, val accuracy, iterations,
buffer size, arena scores per iteration, promotions. Plus the honest
paragraph: what does the promotion pattern mean given arena noise (Module
6, Q1)? Would you trust this champion? Why?

### 3. One experiment you designed

Pick a question the course never answers directly — examples: does the
Module 1 rook bonus survive 200 games instead of 20? Does doubling
simulations beat doubling games-per-iteration at fixed compute (Module 7's
knob exercise)? Does mirror augmentation move val accuracy? State the
hypothesis, the setup, the result *with uncertainty*, and the conclusion
you'd defend. A clean negative result scores full marks.

### 4. The postmortem

Three short answers, in your own words:
- The bug you created (not the course's case studies — yours): what was it,
  what caught it, what habit would have caught it sooner?
- The concept that took longest to click, and the moment it did.
- If you kept going: the next 20 hours, spent where, expecting what.

## Rubric (self-assess, honestly)

| Criterion | Pass looks like |
|---|---|
| Measurement | ≥100-game matches, error bars computed, no naked point estimates |
| Reproducibility | Commands/configs included; a reader could rerun every row |
| Interpretation | Claims sized to evidence; noise acknowledged; at least one "we can't conclude X from this" |
| Experiment design | One variable isolated; hypothesis stated *before* the run |
| Honesty | Failures and surprises reported, not smoothed over |

## Why this can't be faked

Every earlier checkpoint verifies code. This one verifies *judgment*: the
numbers must be yours, the error bars must be computed, and section 4 has
no answer key. If you can write this report and defend it, you didn't just
follow a course — you did a small piece of real research engineering.

**When you're done:** the ultimate external validation is a Lichess BOT
account (Module 8's "where to go next") — a public rating is an error bar
the whole world can audit.
