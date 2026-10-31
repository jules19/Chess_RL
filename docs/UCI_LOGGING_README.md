# UCI transaction logging and PGN export

## Overview

The Chess_RL engine talks to chess GUIs (graphical user interfaces: programs
that show the board) through the Universal Chess Interface (UCI), a protocol
of text commands and responses. The engine can record two things:

1. **UCI transaction log.** Every UCI command and response, with a timestamp.
2. **PGN game export.** Each completed game, saved in PGN (Portable Game
   Notation), the standard text format for chess games.

The engine and its logging are in `uci/engine.py`. The setup guides use
[chess_rl_uci.py](../chess_rl_uci.py), a launcher that starts the same engine
and accepts the same arguments.

## Features

### UCI transaction logging

The log records every command the engine receives and every response it
sends:
- Commands: `uci`, `isready`, `position`, `go`, `setoption`, and others.
- Responses: `uciok`, `readyok`, `bestmove`, `info`, and others.
- Each line has a timestamp with millisecond precision.
- The file is opened in append mode (new entries go at the end), so earlier
  sessions are kept.

**Example log output:**
```
============================================================
UCI Log started: 2025-11-14 10:23:45
============================================================
[2025-11-14 10:23:45.123] IN : uci
[2025-11-14 10:23:45.124] OUT: id name Chess_RL v0.1.0
[2025-11-14 10:23:45.124] OUT: id author Chess_RL contributors
[2025-11-14 10:23:45.125] OUT: option name Engine Type type combo default minimax...
[2025-11-14 10:23:45.125] OUT: uciok
[2025-11-14 10:23:45.456] IN : position startpos moves e2e4
[2025-11-14 10:23:45.457] IN : go depth 4
[2025-11-14 10:23:45.789] OUT: info depth 4 score cp 25
[2025-11-14 10:23:45.790] OUT: bestmove e7e5
```

### PGN game export

Each game is saved in PGN format with:
- Standard PGN headers (Event, Site, Date, Round, White, Black, Result).
- The full list of moves in Standard Algebraic Notation (SAN), the usual
  written form of chess moves, such as Nf3.
- The game result (checkmate, stalemate or draw).
- For a game from a custom starting position, the position in the headers,
  in FEN (Forsyth-Edwards Notation, a one-line text description of a
  position).

Chess analysis tools (Lichess, Chess.com, ChessBase and others) can read the
files.

**Example PGN output:**
```pgn
[Event "Chess_RL Game"]
[Site "Local"]
[Date "2025.11.14"]
[Round "-"]
[White "Player"]
[Black "Chess_RL minimax"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6
8. c3 O-O 9. h3 Na5 10. Bc2 c5 11. d4 Qc7 1-0
```

## Usage

### Method 1: command-line arguments (recommended for testing)

Turn on logging when you start the engine:

```bash
# Enable UCI transaction logging
python3 uci/engine.py --uci-log uci_transactions.log

# Enable PGN game export
python3 uci/engine.py --pgn-log games.pgn

# Enable both
python3 uci/engine.py --uci-log debug.log --pgn-log mygames.pgn
```

### Method 2: UCI options (recommended for GUIs)

Configure logging in your chess GUI's engine settings:

1. Add the engine to your GUI (Arena, Cute Chess, PyChess or another UCI GUI).
2. Open the engine configuration dialog.
3. Set these options:

**Available UCI options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `UCI Log` | checkbox | false | Turn UCI transaction logging on or off |
| `UCI Log File` | string | `uci_transactions.log` | Path to the UCI log file |
| `PGN Export` | checkbox | false | Turn PGN game export on or off |
| `PGN Export File` | string | `games.pgn` | Path to the PGN export file |

**Example UCI commands:**
```
setoption name UCI Log value true
setoption name UCI Log File value /home/user/chess_logs/debug.log
setoption name PGN Export value true
setoption name PGN Export File value /home/user/chess_logs/games.pgn
```

## Use cases

### Game analysis
Open exported games in an analysis tool, such as:
- Lichess (upload the file and run the analysis)
- Chess.com (computer analysis)
- ChessBase or Fritz
- Stockfish
- your own analysis scripts

### Debugging
- Find problems in the UCI communication between the GUI and the engine.
- Check the order of the commands the GUI sends.
- Reproduce a game position from the position commands in the log.

## Implementation details

### UCI transaction logging
- Each line is written to the file at once (flushed after each write), so you
  can read the log while the engine runs.
- Each line is labelled IN (received by the engine) or OUT (sent by it).
- Debug messages (produced when the Debug option is on) are sent to the GUI as
  info string lines, but they are not written to the transaction log.

### PGN Export
- A game is saved when:
  - a new game starts (`ucinewgame`)
  - the engine quits (`quit`)
  - you turn the PGN Export option off (the game in progress is saved first)
  - a position the GUI sends is the end of the game (see the results below)
- The game result is detected automatically:
  - Checkmate: `1-0` or `0-1`
  - Stalemate, insufficient material: `1/2-1/2`
  - Fifty-move rule, repetition: `1/2-1/2`
  - Game still in progress: `*`
- Move lines are wrapped at 80 characters.

## Dependencies

No dependencies beyond the existing requirements:
- `python-chess` (already required)
- Standard library: `argparse`, `datetime`, `os`

## Testing

To test the logging features:

```bash
# Run a simple test game with logging
echo -e "uci\nisready\nucinewgame\nposition startpos\ngo depth 3\nquit" | \
  python3 uci/engine.py --uci-log test.log --pgn-log test.pgn

# Check the logs
cat test.log
cat test.pgn
```

Expected: test.log contains the commands and responses with timestamps, and
test.pgn contains one game with the engine's move.

## Possible future improvements

None of these is implemented yet:
- Log detail levels (minimal, normal, verbose).
- Rotation (starting a new file) for large log files.
- A JSON format for UCI logs, easier for programs to read.
- EPD export (Extended Position Description: a format for single positions).
- Statistics: nodes (positions searched), time, nps (nodes per second).
- Opening book recording (an opening book is a file of opening moves).
