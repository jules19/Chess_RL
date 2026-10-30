# Unit 4: Talking to chess GUIs with UCI

**You'll be able to:**
- explain what UCI is and why engines and GUIs are separate programs
- hold a UCI conversation with the engine by hand
- spot a protocol bug by checking the engine's output against the spec

**Before this:** [Unit 3](03_quiescence.md). **Next:** [Unit 5, MCTS](05_mcts.md). **Time:** ~30–45 min.
**Longer reading:** `minimax_tutorial.html`, section "Day 8: UCI Protocol"; GUI setup in
[`docs/guides/UCI_SETUP_MAC.md`](../guides/UCI_SETUP_MAC.md); logging in
[`docs/guides/UCI_LOGGING_README.md`](../guides/UCI_LOGGING_README.md).

## The idea

Chess engines don't draw boards. The **Universal Chess Interface (UCI)** is a plain-text
protocol: a GUI (Cute Chess, Arena, …) starts the engine as a separate program, writes
commands to its standard input and reads replies from its standard output. Any UCI engine
works with any UCI GUI.

The core conversation:

| GUI sends | Engine replies | Meaning |
|---|---|---|
| `uci` | `id …`, `option …`, `uciok` | "Who are you and what can I configure?" |
| `isready` | `readyok` | "Are you ready?" (used to synchronise) |
| `position startpos moves e2e4` | nothing | "Here is the current position" |
| `go depth 2` | `info …` then `bestmove e7e5` | "Think, then tell me your move" |
| `quit` | nothing | Exit |

## In the code

| Where | What to look for |
|---|---|
| `uci/engine.py`, `UCIEngine.run` | The read-a-line, dispatch-a-command loop |
| `uci/engine.py`, `handle_uci` | The `option` lines: engine type (random, material, minimax, mcts), depth, simulations |
| `uci/engine.py`, `handle_position` | Rebuilds the board from `startpos`/`fen` plus the move list |
| `uci/engine.py`, `get_best_move` | Chooses which engine from units 0–5 to call |
| `uci/engine.py`, `handle_go` | Runs the search and prints `info` and `bestmove` |
| `chess_rl_uci.py` | The launcher a GUI should point to |

## Try it

Have a whole conversation in one command:

```bash
printf 'uci\nisready\nposition startpos moves e2e4\ngo depth 2\nquit\n' | python chess_rl_uci.py
```

The end of the output should be:

```
uciok
readyok
info depth 2 score cp 10
bestmove e7e5
```

## Exercises

**1.** What does `position startpos moves e2e4` followed by `go` ask the engine to do, and which
side does the engine play?

<details><summary>Answer</summary>

Start from the initial position, play `e2e4`, then find a move for the side to move, which
is **Black**. The engine has no memory of "its colour". It always moves for whoever's turn it
is in the position it was given.
</details>

**2. Find the bug:** the UCI specification says `score cp` is "the score from the engine's
point of view", i.e. the side to move, based on its search. Look at the `info depth` line in
`handle_go`. Where does the number `10` really come from? Two things are wrong with it.

<details><summary>Answer</summary>

`score = evaluate(self.board)` gives:
1. a **static** evaluation of the current position, not the result of the search the engine
   just ran, and
2. a score from **White's** point of view (unit 1), even when the engine is playing Black.

So `cp 10` here means "White is slightly better before Black moves", and a GUI will show
it as Black's own assessment. A fix would return the minimax score from the search and
negate it when Black is to move. Try writing that fix as practice.
</details>

**3.** Switch the engine to MCTS for one move by adding a `setoption` line before `go`:

```bash
printf 'uci\nsetoption name Engine Type value mcts\nsetoption name MCTS Simulations value 50\nposition startpos\ngo\nquit\n' | python chess_rl_uci.py
```

Why might a GUI user notice this engine is slower than the minimax one?

<details><summary>Answer</summary>

Each MCTS simulation plays out a whole guided game (unit 5), roughly 20–25 simulations per second
here, so even 50 simulations take a couple of seconds. Minimax at depth 2 takes a fraction
of a second.
</details>

## Check yourself

1. Why is it useful that the engine and the GUI are separate programs?
2. What must an engine print after `uci`, and what after `isready`?

<details><summary>Answers</summary>

1. Any engine works with any GUI, and the engine can stay simple (text in, text out).
2. `uciok` (after its id and options); `readyok`.
</details>
