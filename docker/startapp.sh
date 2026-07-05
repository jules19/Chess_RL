#!/bin/sh
# Entry point displayed by the web-VNC layer (jlesage/baseimage-gui).
#
# Before launching Cute Chess, seed its engine list on first run so
# "Chess_RL" and "Stockfish" are already configured — no manual engine
# registration needed. $HOME is /config, a persistent volume, so any
# changes made in the GUI afterwards stick.

CONFIG_DIR="$HOME/.config/cutechess"

mkdir -p "$CONFIG_DIR"
if [ ! -f "$CONFIG_DIR/engines.json" ]; then
    cp /defaults/engines.json "$CONFIG_DIR/engines.json"
fi

exec /usr/games/cutechess
