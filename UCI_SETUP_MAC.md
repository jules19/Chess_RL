# UCI Setup Guide for Mac

This guide explains how to use the Chess_RL engine with chess GUIs on macOS.

## What is UCI?

**UCI (Universal Chess Interface)** is a standardized protocol that allows chess engines to communicate with chess GUIs (Graphical User Interfaces). Once configured, you can:

- Play against your Chess_RL engine with a nice graphical board
- Test different engine types (random, material, minimax)
- Run engine vs engine matches
- Analyze positions
- Track engine improvements over time

## Prerequisites

1. **Python 3** (macOS usually comes with Python 3)
2. **python-chess library** (already installed if you've been running the CLI)

Check your Python version:
```bash
python3 --version
# Should show Python 3.8 or higher
```

## Chess GUIs for Mac

Here are the best free chess GUIs for macOS:

### 1. **Cute Chess** (Recommended for Mac) ⭐
- **Download**: https://cutechess.com/
- **Pros**: Native Mac app, clean interface, tournament support
- **Cons**: None for our use case
- **Best for**: Testing and analyzing your engine

### 2. **PyChess**
- **Download**: http://pychess.org/
- **Pros**: Cross-platform, Python-based, easy to use
- **Cons**: Requires GTK+ on Mac (slightly complex setup)
- **Best for**: If you're already familiar with PyChess

### 3. **Tarrasch Chess GUI**
- **Download**: https://github.com/billforsternz/tarrasch-chess-gui
- **Pros**: Simple, lightweight
- **Cons**: Less feature-rich
- **Best for**: Quick testing

## Quick Start: Setting Up Cute Chess

### Step 1: Install Cute Chess

1. Download from https://cutechess.com/
2. Drag to your Applications folder
3. Open Cute Chess

### Step 2: Configure Chess_RL Engine

1. In Cute Chess, go to **Tools → Settings**
2. Click the **Engines** tab
3. Click the **+** button to add a new engine

4. Fill in the details:
   - **Name**: `Chess_RL Minimax`
   - **Command**: Click **Browse** and navigate to your project folder
   - Select: `/path/to/Chess_RL/chess_rl_uci.py`
   - **Working Directory**: `/path/to/Chess_RL/`
   - **Protocol**: UCI

   Example full command:
   ```
   /usr/local/bin/python3 /Users/yourname/Chess_RL/chess_rl_uci.py
   ```

5. Click **OK**

### Step 3: Configure Engine Options

After adding the engine, you can configure it:

1. Select your engine in the list
2. Click **Configure** (gear icon)
3. You'll see these options:

   - **Engine Type**:
     - `random` - Makes random legal moves
     - `material` - Greedy material counting (1-ply)
     - `minimax` - Alpha-beta search with positional eval (default)

   - **Search Depth**: 1-6 (default: 3)
     - Depth 2: Very fast, ~800 Elo
     - Depth 3: Good balance, ~1200-1400 Elo (recommended)
     - Depth 4: Slower but stronger, ~1500-1600 Elo
     - Depth 5+: Much slower, diminishing returns

   - **Debug**: Enable to see debug messages (optional)

4. Click **OK**

### Step 4: Play a Game!

**Option A: Human vs Engine**
1. Go to **Game → New**
2. Set:
   - **White**: `Human`
   - **Black**: `Chess_RL Minimax`
   - **Time Control**: Whatever you prefer (or none)
3. Click **OK**
4. Make moves by clicking pieces!

**Option B: Engine vs Engine**
1. Add a second engine (or use Chess_RL with different settings)
2. Go to **Game → New**
3. Set:
   - **White**: `Chess_RL Minimax depth 3`
   - **Black**: `Chess_RL Material`
4. Watch them play!

## Advanced: Engine Tournament

Test your engine's strength by running a tournament:

1. Go to **Tools → Tournament**
2. Add engines:
   - Chess_RL Minimax (depth 3)
   - Chess_RL Material
   - Chess_RL Random
3. Set:
   - **Games**: 10 or 20
   - **Rounds**: 1
   - **Time control**: 1 min per game (or more)
4. Click **Start**

Cute Chess will play all matches and show you:
- Win/loss/draw statistics
- Estimated Elo ratings
- Performance comparison

Example results you might see:
```
1. Chess_RL Minimax (depth 3)  - 8/10 points  (~1400 Elo)
2. Chess_RL Material           - 5/10 points  (~1100 Elo)
3. Chess_RL Random             - 2/10 points  (~600 Elo)
```

## Command Line Testing (Optional)

You can test the UCI engine directly from the terminal:

```bash
cd /path/to/Chess_RL

# Start the engine
python3 chess_rl_uci.py

# Type these commands (one per line):
uci                                    # Identify engine
isready                               # Check if ready
ucinewgame                            # Start new game
position startpos                     # Set starting position
go depth 3                           # Calculate best move
quit                                  # Exit

# Expected output:
# id name Chess_RL v0.1.0
# id author Your Name
# ...options...
# uciok
# readyok
# info depth 3 score cp 20
# bestmove e2e4
```

## Troubleshooting

### Issue: "Engine fails to start"

**Solution 1**: Check Python path
```bash
which python3
# Use the full path in engine configuration
```

**Solution 2**: Test engine manually
```bash
cd /path/to/Chess_RL
python3 chess_rl_uci.py
# Should wait for input (type 'uci' then Enter)
```

**Solution 3**: Check permissions
```bash
chmod +x chess_rl_uci.py
```

### Issue: "Engine moves very slowly"

**Solution**: Reduce search depth
- In engine settings, set **Search Depth** to 2 or 3
- Depth 4+ can take 10-30 seconds per move

### Issue: "Engine makes illegal moves"

This shouldn't happen, but if it does:
1. Check that you're using the latest version
2. Enable debug mode in engine settings
3. Check the debug log in Cute Chess (Tools → Settings → Logging)

## Using with Other GUIs

### Arena Chess (via Wine)

Arena is Windows-only but can run via Wine:
```bash
# Install Wine
brew install wine-stable

# Download Arena from http://www.playwitharena.de/
# Install and configure Chess_RL as above
```

### PyChess

1. Install PyChess: `brew install pychess`
2. Open PyChess
3. Go to **Edit → Engines**
4. Add engine similar to Cute Chess setup

### lichess-bot (Play Online!)

You can even put your engine on Lichess.org as a bot:

1. Install lichess-bot: https://github.com/lichess-bot-devs/lichess-bot
2. Configure it to use `chess_rl_uci.py`
3. Your engine will be available for anyone to challenge on Lichess!

(This is more advanced - see lichess-bot documentation)

## Comparing Engine Strengths

Here's the approximate strength of each engine type:

| Engine Type | Search Depth | Approx. Elo | Playing Style |
|-------------|--------------|-------------|---------------|
| Random      | N/A          | ~600        | Random moves |
| Material    | 1-ply        | ~1000-1100  | Greedy captures |
| Minimax     | 2            | ~1000-1200  | Basic tactics |
| Minimax     | 3            | ~1200-1400  | Positional + tactics |
| Minimax     | 4            | ~1400-1600  | Strong positional |
| Minimax     | 5+           | ~1500-1700  | Very strong (slow) |

As you improve the engine (MCTS, neural networks, etc.), you can track Elo progression!

## Next Steps

1. **Play against your engine** - See if you can beat it!
2. **Run tournaments** - Compare different depths and engine types
3. **Improve the engine** - Move on to Week 2: MCTS implementation
4. **Track progress** - As you add new features, measure Elo improvement

## Example: Full Tournament Setup

Let's set up a mini-tournament to test all your engines:

**In Cute Chess:**
1. Add these engine configurations:
   - Chess_RL Random (Engine Type: random)
   - Chess_RL Material (Engine Type: material)
   - Chess_RL Mini-2 (Engine Type: minimax, Depth: 2)
   - Chess_RL Mini-3 (Engine Type: minimax, Depth: 3)
   - Chess_RL Mini-4 (Engine Type: minimax, Depth: 4)

2. Run tournament: All vs All, 10 games each
3. Compare results!

Example expected results:
```
Tournament Results (50 games total):

Place  Engine              Score    Elo
1      Chess_RL Mini-4     45/50    1520
2      Chess_RL Mini-3     38/50    1380
3      Chess_RL Mini-2     28/50    1180
4      Chess_RL Material   15/50    1020
5      Chess_RL Random     4/50     650
```

This gives you a baseline to compare against as you improve the engine!

---

**Congratulations!** 🎉 You now have a fully functional UCI chess engine that you can play against, test, and improve. Have fun!
