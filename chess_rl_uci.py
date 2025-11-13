#!/usr/bin/env python3
"""
Chess_RL UCI Engine Launcher

Simple launcher for the Chess_RL UCI engine.
Use this script when configuring chess GUIs.

Usage:
    python3 chess_rl_uci.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run UCI engine
from uci.engine import main

if __name__ == "__main__":
    main()
