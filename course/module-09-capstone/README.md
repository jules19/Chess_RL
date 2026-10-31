# Module 9: Capstone: The Experiment Report

> **Where you are:** the end. Modules 0-8 each checked a piece; this checks that you can put them together and defend the result.
> **Time:** 1-2 days, most of it waiting for 100-game matches.
> **You'll be able to:**
> - present a table of your engines' measured strengths, ranked, where every number has a 95% interval
> - explain your own training run in reinforcement-learning terms
> - design, run and report one experiment of your own
> - say what you'd conclude, and what you can't conclude, from your evidence

Write a short experiment report that could convince a sceptical engineer.
It is the final assessment of the whole course, and you cannot complete it
by running commands alone.

## The deliverable

One document (Markdown is fine, about 2 pages), committed to your fork as
`REPORT.md`, with four sections.

### 1. The measured strength table

A table of your engines against a calibrated opponent, from weakest to
strongest. Use Module 7's setup: Stockfish at a fixed strength through
`cutechess-cli`, with at least 100 games per row.

| Engine | Opponent | Games | Score | Elo difference (95% interval) |
|---|---|---|---|---|
| minimax, depth 3 | Stockfish at 1400 | 100 | … | … |
| MCTS, 200 simulations | Stockfish at 1400 | 100 | … | … |
| PUCT with your network | Stockfish at 1400 | 100 | … | … |

Here MCTS is Monte Carlo tree search (Module 3) and PUCT is the
network-guided search from Module 5. You measure every number yourself,
and every Elo figure has a 95% interval. Add one sentence comparing the
result with the repository's historical self-estimates (about 1200–1600).
A gap between the guess and the measurement is a valid finding; report it
as one.

### 2. The training story

For your supervised network (Module 4) and your self-play run (Module 6),
report these numbers:
- dataset size and running time
- final training and validation losses, and validation accuracy
- iterations and replay buffer size
- arena scores per iteration, and promotions

Then write a paragraph that answers three questions. What does the
promotion pattern mean, given arena noise (Module 6, Q1)? Would you trust
this champion? Why?

Then explain the run as reinforcement learning, in your own words, pointing
at your own numbers. Cover:
1. what the agent, the reward and the experience were
2. which step evaluated the policy and which improved it
3. what kind of value target you used, and what it cost you
4. where exploration came from
5. why the replay buffer and the gate were set the way they were

Use your tic-tac-toe runs (Module 6 Part A) wherever chess at your scale was
too noisy to show an effect.

### 3. One experiment you designed

Pick a question the course never answers directly. Examples:
- Does the Module 1 rook bonus survive 200 games instead of 20?
- At fixed computing time, does doubling the simulations beat doubling the
  games per iteration (Module 7's scaling exercise)?
- Does mirror augmentation (training on left-right mirrored positions,
  [Module 4](../module-04-neural-network/README.md) exercise 3) change
  validation accuracy?

State the hypothesis, the setup, the result with uncertainty, and the
conclusion you'd defend. A well-run experiment that finds no effect scores full marks.

### 4. Looking back

Three short answers, in your own words:
- The bug you created (one of yours, not one of the course's case studies):
  what was it, what caught it, what habit would have caught it sooner?
- The concept that took you longest to understand, and the moment you
  understood it.
- If you had 20 more hours: what would you spend them on, and what result
  would you expect?

## Rubric (assess yourself, and be strict)

| Criterion | Pass looks like |
|---|---|
| Measurement | Matches of at least 100 games, 95% intervals computed, no point estimate without an interval |
| Reproducibility | Commands and settings included; a reader could rerun every row |
| Interpretation | No claim goes further than the evidence; noise acknowledged; at least one "we can't conclude X from this" |
| Experiment design | One variable isolated; hypothesis stated before the run |
| Honesty | Failures and surprises reported in full |
| RL understanding | The run is explained as policy evaluation and improvement, with the reward, value target and exploration named and tied to your own results |

## What this report checks

The earlier checkpoints test your code. This report also tests your
judgement. The numbers must be ones you measured, the 95% intervals must be
computed, and section 4 has no answer key. A report you can defend shows
that you can design, run and report an experiment on your own.

**When you're done:** for an external check, create a Lichess BOT account.
BOT is Lichess's name for an account played by an engine instead of a
person; see Module 8's "Where to go next". Its public rating comes with an
uncertainty that anyone can check.

Previous: [Module 8: Engineering Extras](../module-08-engineering/README.md)
