# UCI Transaction Logging and PGN Export

## Overview

The Chess_RL UCI engine now supports comprehensive logging capabilities:

1. **UCI Transaction Logging**: Captures all UCI protocol commands and responses with timestamps
2. **PGN Game Export**: Automatically exports completed games in standard PGN format

## Features

### UCI Transaction Logging

Records every UCI command received and every response sent by the engine, including:
- Commands: `uci`, `isready`, `position`, `go`, `setoption`, etc.
- Responses: `uciok`, `readyok`, `bestmove`, `info`, etc.
- Timestamps with millisecond precision
- Appends to log file (supports multiple sessions)

**Example log output:**
```
============================================================
UCI Log started: 2025-11-14 10:23:45
============================================================
[2025-11-14 10:23:45.123] IN : uci
[2025-11-14 10:23:45.124] OUT: id name Chess_RL v0.1.0
[2025-11-14 10:23:45.124] OUT: id author Your Name
[2025-11-14 10:23:45.125] OUT: option name Engine Type type combo default minimax...
[2025-11-14 10:23:45.125] OUT: uciok
[2025-11-14 10:23:45.456] IN : position startpos moves e2e4
[2025-11-14 10:23:45.457] IN : go depth 4
[2025-11-14 10:23:45.789] OUT: info depth 4 score cp 25
[2025-11-14 10:23:45.790] OUT: bestmove e7e5
```

### PGN Game Export

Automatically tracks and exports games in PGN (Portable Game Notation) format:
- Standard PGN headers (Event, Site, Date, Round, Players, Result)
- Full move history in Standard Algebraic Notation (SAN)
- Proper handling of game outcomes (checkmate, stalemate, draws)
- Support for custom starting positions (FEN)
- Compatible with chess analysis tools (Lichess, Chess.com, ChessBase, etc.)

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

### Method 1: Command-Line Arguments (Recommended for Testing)

Enable logging when starting the engine:

```bash
# Enable UCI transaction logging
python3 uci/engine.py --uci-log uci_transactions.log

# Enable PGN game export
python3 uci/engine.py --pgn-log games.pgn

# Enable both
python3 uci/engine.py --uci-log debug.log --pgn-log mygames.pgn
```

### Method 2: UCI Options (Recommended for GUIs)

Configure logging through your chess GUI's engine settings:

1. Start the engine in your GUI (Arena, Cutechess, PyChess, etc.)
2. Open the engine configuration dialog
3. Configure these options:

**Available UCI Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `UCI Log` | checkbox | false | Enable/disable UCI transaction logging |
| `UCI Log File` | string | `uci_transactions.log` | Path to UCI log file |
| `PGN Export` | checkbox | false | Enable/disable PGN game export |
| `PGN Export File` | string | `games.pgn` | Path to PGN export file |

**Example UCI commands:**
```
setoption name UCI Log value true
setoption name UCI Log File value /home/user/chess_logs/debug.log
setoption name PGN Export value true
setoption name PGN Export File value /home/user/chess_logs/games.pgn
```

## Use Cases

### Game Analysis
Export games to PGN and analyze them in:
- Lichess (upload and analyze)
- Chess.com (computer analysis)
- ChessBase / Fritz
- Stockfish analysis
- Your own analysis tools

### Debugging
- Diagnose UCI protocol issues
- Verify command sequences
- Track engine decision-making
- Identify performance bottlenecks
- Reproduce specific game states

### Training Data Collection
- Collect games for machine learning
- Build opening repertoires
- Analyze engine weaknesses
- Compare different engine configurations

### Development
- Test UCI protocol compliance
- Debug GUI integration issues
- Verify move generation
- Track search statistics

## Implementation Details

### UCI Transaction Logging
- Logs written immediately (flushed after each write)
- Timestamps include milliseconds
- Input/output clearly labeled (IN/OUT)
- Debug messages excluded from transaction log (to avoid duplication)
- File opened in append mode (preserves previous sessions)

### PGN Export
- Moves tracked in Standard Algebraic Notation (SAN)
- Games saved when:
  - New game starts (`ucinewgame`)
  - Engine quits (`quit`)
  - PGN export disabled
- Automatic game result detection:
  - Checkmate: `1-0` or `0-1`
  - Stalemate, insufficient material: `1/2-1/2`
  - Fifty-move rule, repetition: `1/2-1/2`
  - In progress: `*`
- Support for non-standard starting positions (FEN recorded in headers)
- Proper PGN formatting (80-character line wrapping)

## Files Modified

- `uci/engine.py`: Main UCI engine with logging implementation

## Dependencies

No additional dependencies required beyond existing requirements:
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

## Future Enhancements

Potential improvements:
- Configurable log levels (minimal, normal, verbose)
- Rotation policy for large log files
- JSON format option for UCI logs (easier parsing)
- EPD export for position analysis
- Statistics tracking (nodes, time, nps)
- Opening book recording
