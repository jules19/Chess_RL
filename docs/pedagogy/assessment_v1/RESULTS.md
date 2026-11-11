# Course experience changes: assessment and verification

`revised_snapshot.patch` is the reconstruction of the uncommitted assessed version, not current course text; it is retained because that study snapshot was not committed at assessment time.

Date: 2026-11-11.

## Frozen comprehension comparison

Questions and key were frozen before the course edits; their SHA-256 hashes
are in `freeze.json`. The baseline was captured from the working tree, including
the already shipped change-of-mind lab. Its tracked baseline is recoverable
from the commit recorded there; `baseline_manifest.json` verifies exact file
contents. `revised_manifest.json` records the revised learner snapshot.
Snapshots excluded pedagogy/answer keys, history and plans; the revised snapshot
also excludes the newly added local reference-solution patch. The baseline had
no such patch. No learner was allowed to consult reference solutions.

Two fresh AI learners per arm used the same inherited session model (no model
overrides), the same 12-call read-only budget and the same questions. The exact
runtime model identifier is not exposed in this assessment artifact. The only
prompt difference was the snapshot path. Both baseline responses were received
before implementation began. Learners saved their verbatim responses afterwards;
those administrative writes were not part of the study budget.

A fresh grader received shuffled participant IDs, the questions and the frozen
key, without the identity mapping or source snapshots. Answers themselves can
reveal version differences, so anonymisation does not guarantee complete
blinding. See `blind_grades.md` for per-question scores and
`grading_identity_map.json` for the mapping retained after grading.

| Participant | Recall / 6 | Transfer / 6 | Total / 12 |
|---|---:|---:|---:|
| Baseline A | 6 | 6 | 12 |
| Baseline B | 6 | 6 | 12 |
| Revised A | 6 | 6 | 12 |
| Revised B | 6 | 6 | 12 |

**No measured comprehension gain.** All four responses reached the ceiling.
The baseline learners reconstructed missing or contradictory instructions from
code and their Git knowledge, explicitly identifying those gaps. Revised
learners could cite the corrected instructions directly. That is a qualitative
observation, not an independent score or evidence of improved human learning.

These are strong AI readers with source access, not novice human participants.
The quiz author also made the course changes; the frozen quiz limits moving
scoring criteria but does not remove author bias. Two learners per arm and a
coarse score cannot establish small effects. Do not tune the quiz to make these
changes look successful. A new assessment needs a new version and baseline.

## Historical checks on the assessed version

Environment: Python 3.11, PyTorch 2.6.0 CPU, dependencies supplied through a
local temporary dependency directory. These are observed checks, not inferred
from the quiz scores:

- Full project suite: **103 passed, 16 skipped**, 26.40 seconds. Skips are the
  learner exercises, including the expanded resume exercise's parameter cases.
- Protection checks include occupied-directory refusal before model creation,
  preservation of every existing byte, actual warm-start architecture in config,
  matching snapshot state/log, and failure during buffer save leaving the prior
  completion marker and snapshot usable.
- The reference resume patch was tested in a disposable copy with the course
  skip marker removed: **12 passed** (7 resume cases plus 5 artifact checks),
  7.09 seconds. The patch was also applied to a fresh copy of the main file and
  compared with the tested solution. It remains unapplied in the learner source.
- Real menu interactions: quit, Play with `e4`, Continue, and the complete
  change-of-mind route all returned successfully. The lab still switches its
  raw preference to f4. Simulated missing torch produces installation guidance
  and a no-torch alternative rather than a traceback.
- Move parsing tests cover SAN and UCI, castling, promotion, ambiguity, illegal
  moves, null moves and preservation of the board before a move is accepted.
- Documentation static check: **0 broken learner file references, 0 malformed
  code fences**. This is not a claim that all 56 documented commands were run.
  Downloaded data and the learner's journal are recognised generated artifacts.
- The single-file remote solutions diff works with the existing remote branch.
  Fetching that branch still requires network access on a fresh clone.

## Exercise-instruction audit

The original checkout collected 10 course tests, not 11. All were inspected.
The Module 2 timeout exercise contained the concrete sequencing error: it told
learners to observe a failure while the test was still skipped. That order is
fixed. All exercises now share an explicit recovery guide that says to remove
the entire multiline decorator, test the initial failure, implement, and rerun.
The resume exercise uses a module-level marker because it now has several
contracts. Its location changes from Module 6 step 12 to step 9, before the
long run at step 10. Other exercise skips remain intentional.

## Historical implementation scope

Implemented: protected run folders, recorded settings, committed iteration
snapshots, the earlier resume exercise and tested reference, working recovery
instructions and hints, standard chess move input, a short entry menu, a light
journal template, an opening session and core/extended learning routes.

In that assessed version, resume remained a learner exercise and snapshot
history retained every buffer, costing additional disk space; recovery promises completed iterations, not power-loss
safety or exact random-stream replay. One writer per run is required.

No human usability sessions have been conducted. The opening session's
20-minute duration is a design target after installation, not a measured human
completion time. Use the observation checklist in `course_assessment_protocol.md`
(one directory above this report) to test setup, frustration, explanation on
new positions, enjoyment and willingness to continue.


## Subsequent review and simplification

The quiz predates the following changes. Its scores do **not** assess the final
run-files helper, reduced resume exercise or later prose corrections. The raw
answers, frozen questions, grades and original manifests remain unchanged.

The run-file machinery now lives in `selfplay/run_files.py`, with a supplied
validated loader. Learners reconnect champion, experience, iteration count and
log; they do not implement hashes or completion markers. Default storage keeps
only the latest complete save, with full history optional. The training loop
remains about its original length. See [run files](../../RUN_FILES.md) for
retention, recovery limits and the measured model/buffer storage cost.

Review of the course text corrected the instruction to remove a skip before
observing failure, clarified that the optional chess preview requires torch,
and removed an unconditional claim that random-initialised games never finish.
The opening time remains a target; the journal is limited to meaningful
predictions and interpretations. These are editorial judgments, not learner
outcomes.

Validation of the revised implementation:

- First PR in isolation: **92 passed, 12 skipped**, 25.43 seconds.
- Full suite: **111 passed, 12 skipped**, 31.63 seconds. All skips remain
  intentional learner checks (nine other checks and three resume checks).
- The supplied reference patch, applied in a disposable copy with the resume
  skip removed: **16 passed**, comprising three resume checks and 13 supplied
  run-files checks. The reference remains unapplied in the teaching source.
- A separate real CLI run with a 1×8 network completed, a second start was
  refused with every saved byte preserved, and the reference resumed it from
  88 to 176 positions while retaining the first log row. This is a functional
  smoke check, separate from the default-network storage measurement.
- Run-files tests cover preservation of earlier results, latest-only and full
  history retention, incompatible architecture, corrupt saves, interrupted
  writes/exports, and filesystems without hard links.

No fresh comprehension comparison or human usability session has been run
against this final revision. The frozen quiz is retained as evidence of its
ceiling, not a release gate or proof of improved learning.

## Reconstructing the assessed sources

`baseline_manifest.json` matches commit
`a765fe66d30f13ecd583e87b8220feeb13e5189e`. `revised_snapshot.patch` reconstructs
exactly the changed files in `revised_manifest.json` on that baseline; it is a
historical artifact, not a patch to apply to the current course. Both manifests
were checked against the original snapshots before publication.

In a separate disposable checkout:

```bash
git checkout --detach a765fe66d30f13ecd583e87b8220feeb13e5189e
git apply /path/to/current/course/docs/pedagogy/assessment_v1/revised_snapshot.patch
```

Consult `freeze.json` and the protocol for exclusions. Do not expose assessment
keys, this evidence directory or reference solutions to learners. The revised
manifest describes the assessed version, not the final PR stack.


## Next evaluation: observe human beginners

The version-neutral key accepts correct descriptions of either course, including
opposite overwrite behaviours and different journal filenames. Seven questions
(2, 3, 5, 6, 10, 11 and 12) concern behaviours or instructions touched by the
redesign. Combined with strong AI readers who could reconstruct procedures from
source, this comparison had little ability to distinguish the versions. In
particular, baseline learners could infer the correct skipped-test procedure
despite the faulty prose. The ceiling does not establish that the learner
experience is equivalent.

The next evaluation will be observation of 2–3 Python-capable chess beginners
doing the opening session, using the protocol's human checklist. Record where
they get stuck, requests for help, their explanations and whether they want to
continue. Do not run another AI quiz as a substitute. These sessions have not
yet taken place.
