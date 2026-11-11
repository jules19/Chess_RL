# Course comprehension assessment v1

Freeze course_quiz_v1.md and course_quiz_v1_key.md before any course changes.
Archive each course snapshot and its per-file SHA-256 manifest. Include shipped
lab assets; exclude .git, docs/pedagogy (keys), outputs, caches and environments. Exclude course/solutions in both arms too
(the baseline has no such folder); learners must not inspect reference solutions.
The snapshot captures working files, not merely HEAD. Keep the frozen question
file alongside, outside the course tree. Record its hash in each run.

For each arm, use two fresh AI learners with the same model and prompt, no
conversation history, and at most 12 read-only tool calls. Start at README.md.
Learners may read source and course tests, but not execute code, access the
network, inspect other snapshots, reference solution branches or answer keys.
Supply the same quiz and require file citations. A missing instruction is
itself evidence; do not let learners ask the author for help.

Wait for baseline answers before modifying course material. Save verbatim
answers. After the changes, repeat with two fresh learners and the same budget.
Shuffle and anonymise all answers for a fresh grader; give the frozen key and
ask for per-question scores/reasons, without arm labels. Permit answers that
correctly describe either snapshot's actual behaviour, as specified in the key.
Record learner/grade artifacts and report individual scores, not just averages.
A handful of AI learners is a diagnostic, not statistical evidence of improved
human outcomes. A high baseline can leave little room to distinguish versions.

Human follow-up (not replaceable by AI): observe Python-capable chess beginners
starting unaided. Record setup completion, requests for help, time to the first
prediction and observed result, ability to explain search versus training,
response to a new position, and whether they want to continue. Invite comments
on frustration and enjoyment. Do not claim this study occurred until it has.
