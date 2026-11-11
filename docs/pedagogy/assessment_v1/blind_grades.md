# Blind assessment v1 grades

Graded only the frozen questions, frozen key, and `participant_1.md` through `participant_4.md` supplied in `/tmp/chess-course-assessment/grading/`. No repository sources, identity mapping, or other snapshots were inspected. Scores assess substance against the frozen key, using 0, 0.5, or 1 point per question. Either supported version of the program's behaviour is accepted as directed by the key.

| Question | participant_1 | participant_2 | participant_3 | participant_4 |
| --- | ---: | ---: | ---: | ---: |
| Q1 | 1 | 1 | 1 | 1 |
| Q2 | 1 | 1 | 1 | 1 |
| Q3 | 1 | 1 | 1 | 1 |
| Q4 | 1 | 1 | 1 | 1 |
| Q5 | 1 | 1 | 1 | 1 |
| Q6 | 1 | 1 | 1 | 1 |
| Q7 | 1 | 1 | 1 | 1 |
| Q8 | 1 | 1 | 1 | 1 |
| Q9 | 1 | 1 | 1 | 1 |
| Q10 | 1 | 1 | 1 | 1 |
| Q11 | 1 | 1 | 1 | 1 |
| Q12 | 1 | 1 | 1 | 1 |
| Recall subtotal | 6/6 | 6/6 | 6/6 | 6/6 |
| Transfer subtotal | 6/6 | 6/6 | 6/6 | 6/6 |
| **Total** | **12/12** | **12/12** | **12/12** | **12/12** |

No deductions for any participant: every answer supplies the complete core idea required by the frozen key.

Specific grading decisions:

- Q5: All four explicitly remove the complete skip decorator before the initial test run and explain that a skipped test cannot expose the failure. The question asks how to observe the initial failure; omitting an explicit post-implementation rerun does not materially omit that requested procedure.
- Q6: All four provide a remote-reference workflow without checkout. The `git show` alternatives in participants 2 and 3 are explicitly accepted by the key; distinguishing an inferred command from documented instructions does not reduce credit.
- Q10: All four recommend separate directories and explain why configuration alone cannot protect artifacts. Descriptions of either overwriting or refusing occupied directories receive credit under the frozen key.
- Q11: All four explain restoring the champion, buffer, iteration and associated configuration/architecture and log history; consistency across completed artifacts; fresh Adam state at iteration boundaries; and why the same seed does not restore consumed RNG state. Participants 2 and 3 explicitly identify the limitations of sequential saves rather than assume crash consistency, and describe restarting from a consistent completed boundary.
- Q12: All four identify a persistent location for a prediction and planned setup/measurement before collecting results, followed by results, uncertainty and interpretation while preserving the original prediction. Either `REPORT.md` or `JOURNAL.md` meets the key.

The assessment gives a ceiling result for these four submissions and does not distinguish their comprehension at this scoring resolution. No participant group is inferred, and these scores establish neither a statistical improvement nor human enjoyment, installation friction, patience, or willingness to continue.
