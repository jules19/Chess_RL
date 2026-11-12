# Frozen assessment v1: grading key

Each question is worth 0, 0.5, or 1 point. Give 1 for the complete core idea,
0.5 for a correct but materially incomplete answer, and 0 for a wrong or absent
answer. Grade correctness against the assigned snapshot: safety changes mean
that a truthful answer about the baseline differs from a truthful answer about
the revised version. Do not reward matching terminology or prose style.
There are 6 recall and 6 transfer questions (12 points total).

1. Policy suggests moves; value estimates outcome for side to move. Human
   targets: played move and eventual outcome. Self-play: visit distribution
   and eventual outcome. Sources: training/dataset.py, selfplay/self_play.py.
2. selfplay/tictactoe_lab.py; vs_perfect is percentage of games not lost by
   the learned table alone against perfect play (draws; perfect cannot lose).
   Sources: course/module-06-selfplay-rl/README.md; selfplay/tictactoe_lab.py.
3. Prepared capture-biased fixture, trained on one escape-square board, not
   human games; lab fits policy component only, with no invented game result.
   Sources: selfplay/mind_change_lab.py, selfplay/assets/mind_change/README.md.
4. Zero/draw. Predicting zero can give tiny value error with no demonstrated
   chess improvement. Sources: selfplay/self_play.py; Module 6 Part B.
5. Remove the skip decorator FIRST, run pytest on
   tests/course/test_module02_search.py to observe failure, implement the
   deadline/abort and rerun. A still-skipped test cannot expose failure.
   Source: tests/course/test_module02_search.py. Baseline prose has this wrong.
6. Fetch if necessary, then git diff main origin/course-solutions --
   engine/evaluator.py (an equivalent git show of the remote file is fine).
   No checkout is necessary. Source: tests/course/README.md and repository
   branch evidence. Baseline prose names a local branch absent in fresh clones.
7. Search updates a tree/visits with fixed weights; training updates weights
   toward the visit distribution, affecting raw-network inference. Sources:
   search/puct.py; selfplay/train_loop.py; change-of-mind lab.
8. No. Include a control where the apparent tactic is harmless (escape square)
   and assess whether probability falls indiscriminately. Transfer on one board
   is not proof of learned discrimination or general strength. Source: lab.
9. Tiny sample/inconclusive; do not declare superiority. Check mechanism via
   a fixed position, nodes/visits or a tactical assertion; use diverse games,
   appropriate budgets, more samples and uncertainty for strength. Sources:
   course/module-03-mcts/README.md, course/module-07-measure-and-scale/README.md.
10. Separate named directories. Baseline silently rewrites log/champion/buffer;
    revised infrastructure should refuse occupied directories. Config describes
    a run but alone prevents no overwrite. Source: selfplay/train_loop.py.
11. Champion, replay buffer, iteration numbering and compatible settings; log
    preserved. All saved artifacts must belong to the same completed iteration
    (detect/reject inconsistency or use a committed snapshot). Restart incomplete
    work. Adam is recreated per iteration, so no cross-iteration Adam state is
    needed here. Same seed alone does not restore consumed RNG state. Sources:
    selfplay/train_loop.py; tests/course/test_module06_resume.py. Full credit
    requires consistency plus non-identical-continuation caveat; half credit for
    basic model/buffer restoration with one of these explanations missing.
12. Before: hypothesis/prediction, setup and planned measurement. After: result,
    uncertainty, interpretation, surprises. A named persistent journal/report
    location is enough; JOURNAL.md specifically is not required for credit.
    Sources: course/module-09-capstone/README.md and module prediction prompts.

Freeze date: 2026-11-11. Check source contents before running either arm.
Do not alter questions or criteria after viewing learner answers. Corrections
require a new version and new baseline. AI comprehension results do not measure
human enjoyment, installation friction, patience, or willingness to continue.
