# Quick start

This gets you playing chess against the computer in under 5 minutes. To learn how it works,
continue with [`LEARN.md`](LEARN.md) afterwards. Its unit 0 covers the same game loop in more depth.

## Setup (First Time)

```bash
# Install dependencies
pip install -r requirements.txt
```

## Run Your First Game

### Option 1: Watch Random vs Random
```bash
python cli/play.py random
```

This will play a complete game where both sides move randomly. You'll see:
- Board state after each move
- Move history
- Game termination (checkmate, stalemate, or draw)

### Option 2: Play Against the Computer
```bash
python cli/play.py human
```

- Choose your color (White or Black)
- Enter moves in UCI format (e.g., `e2e4`, `g1f3`)
- The computer will respond with random moves

### Option 3: Run Test Suite
```bash
python cli/play.py test
```

Plays 10 random vs random games and shows statistics:
- Win/loss/draw distribution
- Average game length
- Validates that all games terminate correctly

## What You Just Built

✅ **Move generation**: Using python-chess (battle-tested)
✅ **Game loop**: Complete games that terminate correctly
✅ **Board visualization**: See the position after each move
✅ **Legal move validation**: Prevents illegal moves
✅ **Draw detection**: 50-move rule, repetition, stalemate

## Stronger opponents

The same script plays every engine in the project:

```bash
python cli/play.py human-minimax   # you vs minimax (units 2-3)
python cli/play.py human-mcts      # you vs Monte Carlo Tree Search (unit 5)
python cli/play.py minimax         # watch minimax play the random mover
```

## Validation Checklist

Run the test suite and check:
- [ ] All 10 games finish (no hangs) and the summary `📊 Test Suite Results` is printed
- [ ] Every game ends by checkmate, stalemate or a draw rule
- [ ] Most games are **draws**, and they're **long**: an average of roughly 300–400 moves (plies) is
      normal for random play. One sample run: 0 White wins, 1 Black win, 9 draws, average 380 moves.

If a game never ends or crashes, something is wrong with the setup.

## FAQ

**Q: The games are taking too long!**
A: Random play leads to very long games. This is expected. Engines that evaluate positions (unit 1 onwards) finish games much faster.

**Q: Can I make it faster?**
A: Use `python cli/play.py test`. It plays 10 games without drawing the board, printing a short result per game and a summary.

**Q: The computer makes terrible moves!**
A: Correct: in `random` and `human` mode it plays randomly. Try `human-minimax` for a real opponent.

**Q: How do I enter moves in UCI format?**
A: Format is `[from][to]`, e.g., `e2e4` moves the pawn from e2 to e4. For pawn promotion, add the piece: `e7e8q` (promote to queen).

## Success Criteria

You have finished the quick start if:
1. You can run a complete random vs random game
2. You can play a game against the random player
3. The test suite passes (all games terminate)

**Time to complete**: 30 minutes to 2 hours (including setup)

**What's next**: [`LEARN.md`](LEARN.md), starting with unit 0.
