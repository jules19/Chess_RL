# Unit 0: The game loop

**You'll be able to:**
- explain the loop at the heart of every chess program
- say why a random player's games are long and usually drawn
- run and read the project's test suite

**Before this:** setup in [`LEARN.md`](../../LEARN.md). **Next:** [Unit 1, Judging a position](01_evaluation.md). **Time:** ~30 min.

## The idea

A chess program, stripped down, is a loop:

```
while the game isn't over:
    look at the board
    choose a move        ← everything interesting happens here
    play it
```

Everything in this project is a better answer to "choose a move". Unit 0 uses the
dumbest possible answer, **pick any legal move at random**, so you can see the loop on
its own.

We don't write the rules of chess ourselves. The [`python-chess`](https://python-chess.readthedocs.io/)
library generates legal moves and detects checkmate, stalemate and draws. That keeps every
unit focused on *choosing* moves, not on rules bugs.

## In the code

| Where | What to look for |
|---|---|
| `cli/play.py`, `random_move` | One line: `random.choice(list(board.legal_moves))` |
| `cli/play.py`, `play_random_vs_random` | The loop itself: `while not board.is_game_over()` |
| `cli/play.py`, `run_test_suite` | Plays 10 games and summarises the results |
| `cli/play.py`, `main` | The command-line modes (`random`, `human`, `minimax`, `mcts`, `test`, …) |

## Try it

```bash
python cli/play.py test
```

The ending looks like this (your numbers will differ, because the moves are random):

```
📊 Test Suite Results
White wins: 0
Black wins: 1
Draws: 9
Average game length: 380.2 moves
Min: 218, Max: 519
```

Most games are draws, and they're **long**: hundreds of half-moves. "Moves" here counts each
player's turn separately (chess programmers call one turn a **ply**).

To watch one game, or play against the random mover yourself:

```bash
python cli/play.py random     # watch random vs random
python cli/play.py human      # you play; enter moves like e2e4
```

## Exercises

**1. Predict before you run:** what fraction of random-vs-random games do you expect to end in
checkmate? Write down a guess, then run the test suite twice.

<details><summary>Answer</summary>

Usually only 0–3 of 10. Checkmate needs pieces to cooperate, and random play almost never
does that on purpose. Games mostly end by the 50-move rule, threefold repetition or
insufficient material once enough pieces have been traded off at random.
</details>

**2.** Why is it useful to build a random player at all, if it's so bad?

<details><summary>Answer</summary>

It tests the loop and the rules (every game ends, no illegal moves) with nothing else that
could go wrong. It also gives every later engine a **baseline**: any engine worth
having must beat random almost every time. Unit 8 relies on this.
</details>

**3.** Open `cli/play.py` and find where the game result is decided after the loop ends. How
does the program know *why* a game ended?

<details><summary>Answer</summary>

It asks the board: `board.is_checkmate()`, `board.is_stalemate()`,
`board.is_insufficient_material()` and so on (look just after the `while` loop in
`play_random_vs_random`). `python-chess` tracks repetitions and the 50-move counter for us.
</details>

## Check yourself

1. What does "ply" mean?
2. In the game loop, which single step do all the later units improve?

<details><summary>Answers</summary>

1. One move by one player (half of a "full move").
2. "Choose a move".
</details>
