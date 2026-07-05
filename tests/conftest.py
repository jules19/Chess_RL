"""
Pytest configuration.

Adds the repository root to sys.path so tests import the project packages
(engine, net, search, selfplay, ...) without requiring `pip install -e .`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
