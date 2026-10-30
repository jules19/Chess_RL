#!/bin/bash
# Launch the UCI engine with the project's virtualenv, from wherever the repository lives.
# Point a chess GUI at this script (see docs/guides/UCI_SETUP_MAC.md).
DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$DIR/venv/bin/activate" ]; then
    source "$DIR/venv/bin/activate"
fi

exec python "$DIR/chess_rl_uci.py"
