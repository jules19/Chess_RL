# Observation sheet: the opening session

Use one copy of this sheet per tester. It puts the "Human follow-up" in
[course_assessment_protocol.md](course_assessment_protocol.md) into practice.
The question is simple: can a newcomer get from the README to the end of
Module 0's opening session on their own, understand what they saw, and want
to continue?

## Who to test

- Can program in Python (has used a terminal and installed a package).
- Knows the rules of chess. Has never built a chess engine or studied
  reinforcement learning.
- Has never seen this repository.
- Uses their own computer, with Python 3.9+ and git already installed. Do not
  install anything for them beforehand.

Two or three testers is enough to find the big problems. Record initials only.

## Your role

You are watching, not teaching. Ask them to think aloud.

**Say at the start:** "This is a test of the course, not of you. Start at the
README on GitHub and do what it says, up to the end of Module 0's opening
session. Please say what you're thinking as you go. I'll mostly stay quiet;
if you're stuck for a few minutes I'll step in, and that tells us where the
course needs fixing."

**When they get stuck,** go up this ladder one step at a time, and write down
the highest step you reached:

1. Wait 3 minutes.
2. Ask: "What are you trying to do right now?"
3. Point to the place in the README or module page that answers it.
4. Tell them the answer.

Don't explain concepts during the session, even if they ask. Write the
question down instead and come back to it in the debrief.

## Session record

Tester: ____  Date: ____  Computer and OS: ____  Connection: home / office / mobile

Write clock times. "Help" means the highest ladder step reached (0 = none).

| Step | Start | End | Help | What happened: hesitations, questions, errors, comments |
|---|---|---|---|---|
| README: clone, venv, `pip install -e ".[dev]"` | | | | |
| README: `pytest -q` | | | | |
| Finds the opening session in Module 0 | | | | |
| 1. Journal created | | | | |
| 2. PyTorch download started in a second terminal | | | | Download finished at: ____ |
| 3. Tic-tac-toe run | | | | |
| 4. No-exploration run | | | | |
| 5. Chess lab | | | | Waited for the download? Y / N |
| Stop and explain | | | | |

**Predictions.** Did they write each prediction in `JOURNAL.md` *before*
running? Step 3: Y / N. Step 4: Y / N. Step 5: Y / N.

**First result.** Minutes from the start of setup to the first result they
could describe (the tic-tac-toe table): ____

**What they should see** (to check against, not to show them):
- Step 3: `vs_perfect` rises from 0 to about 89.
- Step 4: `coverage` stays near 0.5% and `vs_perfect` stays at 50.
- Step 5: the network first prefers Rxa7 at 99.9%. With more simulations,
  search switches to f4, because Rxa7 runs into Re1#. After training,
  Rxa7 drops to about 17%.

## Debrief (after the session, about 10 minutes)

Ask these in order. Write their words, then tick what they covered.

**A. Search versus training.** "In the chess lab, two different things changed
the engine's choice. What did each one do? Which change would still be there
if you ran it again tomorrow?"

- [ ] Search looked ahead from this one position (and found the mate).
- [ ] Training changed the network itself, so its first instinct changed.
- [ ] Training's change is kept; search's is redone every time.

**B. A new case: tic-tac-toe.** "We tested the learned table *without* search.
Suppose it could also look ahead during the test, for as long as it likes.
Would it lose more often or less often? Why?"

- [ ] Less often.
- [ ] Because looking ahead can catch mistakes the table still makes.

Measured, from 400 games against a perfect player: the table alone did not
lose 90% of games. With a short search (25 simulations) it did not lose 86%,
slightly worse. With 100 simulations it did not lose 97%, and with 400, 100%.
If they say "a quick look might not help", that is correct: give credit, and
note it as a sign of real understanding.

**C. A new case: chess.** "Another engine wants to take a free bishop, but
taking it lets the opponent mate next move. It thinks for only 5
simulations. Will it see the mate? What would make it see it?"

- [ ] Probably not: the lab showed that 10 simulations still chose the bad capture.
- [ ] More search (more simulations), or training on positions like this one.

**D. Wanting to continue.** "From 1 to 5, how much do you want to do the
next session?" ____ "What would make it a 5?" ____

**E. Their view.**
- "What was the most confusing moment?" ____
- "What was the best moment?" ____
- "How would you describe this course to a friend?" ____

## Summary (one row per tester)

| Tester | OS | Finished? | Highest help | Minutes to first result | Total minutes | A (of 3) | B (of 2) | C (of 2) | Wants to continue (1–5) |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

"Finished" means reached "Stop and explain" with help no higher than step 3.

## Afterwards

Bring the sheets back. Problems that recur across testers come first. Anything
that needed step 4 help is a course bug, even if it happened only once.
Don't count this study as done in `OUTCOME_TEST.md` until it has actually been run.
