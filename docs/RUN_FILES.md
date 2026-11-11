# Self-play run files

Use a new folder for each experiment, for example
`--checkpoint-dir runs/m6_main_01`. Starting in any occupied folder is refused
before model creation or output writes. Another attempt should use a new suffix,
such as `_02`. There is no destructive override.

`config.json` records the settings actually used. A loaded checkpoint determines
the network architecture, even when command-line size arguments have defaults.
The original configuration stays with the experiment. One process may write a
run at a time.

## What is retained

By default, only the **last complete save** is retained: champion, replay buffer,
and the log of all completed iterations. Candidate weights are not archived.
A replacement is written before the previous save is removed, so allow space
for two saves during writing. If a process stops mid-write, its pending files
may remain until the next successful save; the last complete save is still usable.

`--keep-history` retains every completed iteration, including its candidate,
champion and buffer. This is an explicit disk-space tradeoff, not the default.
It must be chosen at the start of the run; resume uses the saved setting.

The familiar top-level `champion.pt`, `replay_buffer.npz` and `run_log.csv` names
remain available for evaluation and inspection. They are hard links where
supported, so they share data with the complete save rather than duplicate it.
Filesystems without hard links use copies and consume additional space. Treat
all these files as read-only; edit a copy for your own experiments.

## Resuming is the learner exercise

`selfplay/run_files.py` supplies `load_run(directory, device="cpu")`. It validates
and returns a `SavedRun` with:

- `champion`: the model, with its recorded architecture;
- `buffer`: the experience and capacity;
- `iteration`: the last completed iteration;
- `settings`: the original effective run settings;
- `files`: a writer carrying the committed log, ready to save the next result.

Use `saved.require_architecture(res_blocks, channels)` if sizes were explicitly
requested. Omitting sizes inherits the saved model. Module 6 asks learners to
connect this state to the learning loop and CLI, not implement the file format.
The supplied checks reject missing, corrupt and inconsistent saves.

Recovery follows `completed.json` rather than potentially stale top-level names.
The marker is replaced only when all files for a save exist. Hash checks detect
incomplete or changed files, including a changed configuration. These are
implementation details in the supporting module; they need not distract from
reading `train_loop.py`.

Continuation restarts an incomplete iteration. It does not reproduce the random
stream of an uninterrupted run, and does not claim power-loss durability.
An optimiser is created afresh each iteration, so no Adam state is needed at
that boundary. The former experimental snapshot format is not supported by this
new loader; keep those folders intact and use their original code to read them.

## Disk measurement

The earlier local measurement supplied during review was about **7.3 MB for one
4×128 model file**, or about **22 MB for three model files per iteration** in
the previous design. That measurement did not establish the size of a realistic
replay buffer. Buffer size depends on the positions and compression, and grows
until capacity is reached; model size alone is not a disk budget.
