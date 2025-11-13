# Quick Start Guide - Baby Steps

This guide gets you playing chess against the computer in under 5 minutes.

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

## Next Steps

See `RISK_REDUCTION.md` for the complete roadmap. The immediate next steps are:

1. **Material Evaluation** (Tomorrow): Make the computer count piece values
2. **Minimax Search** (Day 3-4): Add look-ahead to find tactics
3. **Position Evaluation** (Day 5-6): Add center control, development, king safety
4. **UCI Interface** (Day 7): Play against the engine in a chess GUI

After 1 week, you'll have a ~1200-1400 strength chess engine that plays sensible chess.

## Validation Checklist

Run the test suite and verify:
- [ ] All 10 games terminate (no hangs)
- [ ] Mix of white wins, black wins, and draws
- [ ] Games end with checkmate OR stalemate OR draw
- [ ] Average game length is reasonable (50-200 moves)

If any test fails, there's a bug in the move generation or game rules.

## FAQ

**Q: The games are taking too long!**
A: Random play can lead to very long games. This is expected. Once we add evaluation (tomorrow), games will be much shorter.

**Q: Can I make it faster?**
A: Yes, edit `cli/play.py` and set `verbose=False` in the game loop.

**Q: The computer makes terrible moves!**
A: Correct! It's playing randomly. Tomorrow we'll add material counting so it at least captures pieces.

**Q: How do I enter moves in UCI format?**
A: Format is `[from][to]`, e.g., `e2e4` moves the pawn from e2 to e4. For pawn promotion, add the piece: `e7e8q` (promote to queen).

## Success Criteria

You've completed Baby Step #1 if:
1. You can run a complete random vs random game
2. You can play a game against the random player
3. The test suite passes (all games terminate)

**Time to complete**: 30 minutes to 2 hours (including setup)

**What's next**: Material evaluation - make the computer prefer to capture pieces and avoid losing pieces.
