#!/bin/bash
# UCI engine launcher for chess GUIs and cutechess-cli.
#
# Portable: derives the repo location from this script's own path (an
# earlier version hardcoded one developer's home directory — the same class
# of bug as the hardcoded sys.path that broke CI; see uci/engine.py).
# Uses the repo's .venv if present, otherwise system python3.

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -x "$REPO_DIR/.venv/bin/python" ]; then
    PYTHON="$REPO_DIR/.venv/bin/python"
else
    PYTHON="python3"
fi

exec "$PYTHON" "$REPO_DIR/chess_rl_uci.py" "$@"
