# UCI setup guide for Mac

This guide explains how to use the Chess_RL engine with chess GUIs on macOS.

## What is UCI?

The Universal Chess Interface (UCI) is a standard set of text commands that a
chess engine and a chess GUI (graphical user interface: the program that shows
the board) use to communicate. After setup, you can:

- Play against your Chess_RL engine on a graphical board.
- Test the engine types (random, material, minimax, mcts, puct; see Step 3).
- Run matches between two engines.
- Analyse positions.
- Track how the engine improves over time.

## Prerequisites

1. **Python 3.** macOS usually includes it.
2. **The python-chess library.** It is already installed if you have run the
   command-line programs.

Check your Python version:
```bash
python3 --version
# Should show Python 3.8 or higher
```

## Chess GUIs for Mac

These free chess GUIs work on macOS. This guide uses Cute Chess.

### 1. Cute Chess (used in this guide)
- **Download**: https://cutechess.com/
- **Advantages**: native Mac application, clear interface, tournaments.

### 2. PyChess
- **Download**: http://pychess.org/
- **Advantages**: runs on several operating systems, written in Python.
- **Disadvantages**: on a Mac it needs GTK+, a graphical toolkit you must
  install separately.

### 3. Tarrasch Chess GUI
- **Download**: https://github.com/billforsternz/tarrasch-chess-gui
- **Advantages**: small and simple; suitable for quick tests.
- **Disadvantages**: fewer features.

## Quick start: setting up Cute Chess

### Step 1: Install Cute Chess

1. Download Cute Chess from https://cutechess.com/.
2. Drag it to your Applications folder.
3. Open Cute Chess. Expected: the Cute Chess window opens.

### Step 2: Add the Chess_RL engine

1. In Cute Chess, go to **Tools → Settings**.
2. Click the **Engines** tab.
3. Click **+** to add an engine. Expected: the engine details dialog opens.
4. Fill in the details:
   - **Name**: `Chess_RL Minimax`
   - **Command**: the full path to python3, a space, then the full path to
     `/path/to/Chess_RL/chess_rl_uci.py` (see the example below, and
     Troubleshooting Solution 1 to find the python3 path). If you use
     **Browse** to select only the script, it must be executable
     (Troubleshooting Solution 3).
   - **Working Directory**: `/path/to/Chess_RL/`
   - **Protocol**: UCI

   Example full command:
   ```
   /usr/local/bin/python3 /Users/yourname/Chess_RL/chess_rl_uci.py
   ```

5. Click **OK**. Expected: Chess_RL Minimax appears in the engine list.

Step 4 and the tournaments below also use two more engine entries. Add them now
by repeating this step with these names, then set their Engine Type in Step 3:
- Chess_RL Material (Engine Type: material)
- Chess_RL Random (Engine Type: random)

Each entry keeps its own options.

### Step 3: Configure the engine options

After you add the engine, you can configure it:

1. Select your engine in the list.
2. Click **Configure** (the gear icon).
3. Set these options:

   - **Engine Type**:
     - `random`: plays a random legal move.
     - `material`: plays the move that leaves the best material count (piece
       values). It looks one ply (one move by one side) ahead.
     - `minimax` (the default): searches moves and replies to a fixed depth,
       with alpha-beta pruning (skipping moves that cannot change the result)
       and a positional evaluation (a score from material, piece placement,
       king safety and pawn structure). See course Module 2.
     - `mcts`: Monte Carlo tree search (MCTS). It judges moves by playing many
       quick games (rollouts) forward from the position, with the evaluation
       function helping to choose the rollout moves. See course Module 3.
     - `puct`: the search used by AlphaZero (DeepMind's self-taught chess
       program), guided by a trained neural network. It needs torch (the
       PyTorch library) and a checkpoint (a saved network; see **Model File**).
       If either is missing, the engine plays minimax and gives the reason in
       an `info string` message (a text line sent to the GUI). See Module 5.

   - **Search Depth**: 1-6 (default: 3). Minimax only. Depth is the number of
     plies the search looks ahead. If the game has a time control (a limit on
     each side's thinking time), minimax ignores this setting and uses
     iterative deepening: it searches to depth 1, then 2, and so on, until
     the time for the move runs out.
     - Depth 2: very fast.
     - Depth 3: a good balance of speed and strength (recommended).
     - Depth 4: slower but stronger.
     - Depth 5 and above: much slower, with smaller gains for each extra ply.

   - **MCTS Simulations**: 50-1000 (default: 200). MCTS only. A simulation is
     one pass of the search: it follows the tree to a new position, evaluates
     it (with a rollout for MCTS, with the network for PUCT) and updates the
     statistics of the moves on that path.
   - **PUCT Simulations**: 16-3200 (default: 200). PUCT only.
   - **Model File**: the path to a trained checkpoint
     (default: `checkpoints/best_model.pt`). PUCT only.

   - **Debug**: turn this on to see debug messages (optional).

   For approximate strengths, see
   [Comparing engine strengths](#comparing-engine-strengths).

4. Click **OK**.

### Step 4: Play a game

**Option A: Human against engine**
1. Go to **Game → New**.
2. Set:
   - **White**: `Human`
   - **Black**: `Chess_RL Minimax`
   - **Time Control**: any setting you prefer, or none.
3. Click **OK**. Expected: the board appears and it is your move.
4. Click the pieces to move. Expected: the engine replies to each move.

**Option B: Engine against engine**
1. Go to **Game → New**.
2. Set **White** to `Chess_RL Minimax` (from Step 2) and **Black** to
   `Chess_RL Material`.
3. Set a time control, or none.
4. Click **OK**. Expected: the two engines play without input from you.

## Advanced: engine tournament

To compare engines, run a tournament:

1. Go to **Tools → Tournament**.
2. Add these engine entries, all created in Step 2:
   - Chess_RL Minimax
   - Chess_RL Material
   - Chess_RL Random
3. Set:
   - **Games**: 10 or 20
   - **Rounds**: 1
   - **Time control**: 1 minute per game or more
4. Click **Start**.

Expected: Cute Chess plays all the matches and shows the wins, losses and
draws, and estimated Elo ratings. Elo is the rating scale used in chess; a
higher number means a stronger player.

The example below is illustrative only. It writes each entry's depth after the
name. Its Elo labels are historical self-estimates, never measured; course
Module 7 measures them.
```
1. Chess_RL Minimax (depth 3)  - 8/10 points  (~1400 Elo)
2. Chess_RL Material           - 5/10 points  (~1100 Elo)
3. Chess_RL Random             - 2/10 points  (~600 Elo)
```

## Command-line testing (optional)

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
# id author Chess_RL contributors
# ...options...
# uciok
# readyok
# info depth 3 score cp -2
# bestmove g1f3
#
# The score is the search result in centipawns from the engine's (side to
# move's) point of view: -2 means "about level".
```

## Troubleshooting

### Issue: "Engine fails to start"

**Solution 1**: Check the Python path.
```bash
which python3
# Use the full path in engine configuration
```

**Solution 2**: Test the engine by hand.
```bash
cd /path/to/Chess_RL
python3 chess_rl_uci.py
# Should wait for input (type 'uci' then Enter)
```

**Solution 3**: Check the file permissions.
```bash
chmod +x chess_rl_uci.py
```

### Issue: "Engine moves very slowly"

**Solution**: Reduce the search depth.
- In the engine settings, set **Search Depth** to 2 or 3.
- At depth 4 and above, a move can take 10-30 seconds.

### Issue: "Engine makes illegal moves"

The engine is not expected to do this. If it does:
1. Check that you are using the latest version.
2. Turn on debug mode in the engine settings.
3. Check the debug log in Cute Chess (Tools → Settings → Logging).

## Using other GUIs

### Arena Chess (via Wine)

Arena runs only on Windows. On a Mac you can run it with Wine, a program that
runs Windows applications on macOS:
```bash
# Install Wine
brew install wine-stable

# Download Arena from http://www.playwitharena.de/
# Install and configure Chess_RL as above
```

### PyChess

1. Install PyChess: `brew install pychess`
2. Open PyChess.
3. Go to **Edit → Engines**.
4. Add the engine in the same way as in the Cute Chess setup.

### lichess-bot (play online)

lichess-bot connects a UCI engine to a bot account on Lichess.org, a free
chess website:

1. Install lichess-bot: https://github.com/lichess-bot-devs/lichess-bot
2. Configure it to use `chess_rl_uci.py`.
3. Expected: other Lichess users can challenge your engine to a game.

This setup is more advanced. See the lichess-bot documentation.

## Comparing engine strengths

Approximate strength of each engine type (historical self-estimates; see the
note under the table):

| Engine Type | Setting | Approx. Elo* | Playing Style |
|-------------|--------------|-------------|---------------|
| Random      | none          | ~600        | Random moves |
| Material    | 1 ply        | ~1000-1100  | Takes the largest capture |
| Minimax     | depth 2      | ~1000-1200  | Basic tactics |
| Minimax     | depth 3      | ~1200-1400  | Positional play and tactics |
| Minimax     | depth 4      | ~1400-1600  | Stronger positional play |
| Minimax     | depth 5+     | ~1500-1700  | Stronger again, but slow |
| MCTS        | 200 simulations     | ~1400-1600  | Spends more simulations on the moves with the best results so far |
| PUCT + network  | 200+ simulations    | depends on your training | The stage the course builds towards |

*Historical self-estimates, not measurements. For measured ratings with a 95%
interval (the range of ratings that fits the match results at 95%
confidence), see [course Module 7](../course/module-07-measure-and-scale/README.md)
and [STATUS.md](../STATUS.md). Every engine stage in the table can be selected
in this GUI, so you can use it for those measurements.

## Next steps

1. **Play against your engine.** Try to beat it.
2. **Run tournaments.** Compare different depths and engine types.
3. **Improve the engine.** Work through the [course modules](../course/README.md).
4. **Track progress.** When you add a feature, measure the change in Elo as
   course Module 7 describes.

## Example: full tournament setup

This example runs a small tournament between the random, material and minimax
stages.

**In Cute Chess:**
1. Use the Chess_RL Random and Chess_RL Material entries from Step 2, and add
   three minimax entries in the same way:
   - Chess_RL Mini-2 (Engine Type: minimax, Depth: 2)
   - Chess_RL Mini-3 (Engine Type: minimax, Depth: 3)
   - Chess_RL Mini-4 (Engine Type: minimax, Depth: 4)

2. Run the tournament with each pair of engines playing 10 games. With 5
   engines there are 10 pairs, so the tournament has 100 games and each
   engine plays 40.
3. Compare the results.

The block below shows only the layout of the results table. Its numbers are
illustrative, not measured, and its game counts (50 games in total, scores
out of 50) do not match the schedule above.
```
Tournament Results (50 games total):

Place  Engine              Score    Elo
1      Chess_RL Mini-4     45/50    1520
2      Chess_RL Mini-3     38/50    1380
3      Chess_RL Mini-2     28/50    1180
4      Chess_RL Material   15/50    1020
5      Chess_RL Random     4/50     650
```

Keep these results as a baseline to compare against as you improve the
engine.

---

You now have a working UCI chess engine that you can play against, test and
improve.
