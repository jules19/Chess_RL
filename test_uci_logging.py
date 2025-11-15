#!/usr/bin/env python3
"""
Test script for UCI logging features.
"""

import subprocess
import time
import os

def test_uci_logging():
    """Test UCI transaction logging and PGN export."""

    print("Testing UCI logging features...")
    print("=" * 60)

    # Clean up any existing log files
    for f in ["test_uci.log", "test_games.pgn"]:
        if os.path.exists(f):
            os.remove(f)

    # Start the UCI engine with logging enabled
    proc = subprocess.Popen(
        ["python3", "uci/engine.py", "--uci-log", "test_uci.log", "--pgn-log", "test_games.pgn"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    def send_command(cmd):
        """Send a command and read response."""
        print(f">> {cmd}")
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()
        time.sleep(0.1)  # Give it time to process

    # Initialize engine
    send_command("uci")
    time.sleep(0.2)

    send_command("isready")
    time.sleep(0.1)

    # Start a new game
    send_command("ucinewgame")
    send_command("isready")

    # Play a few moves
    send_command("position startpos moves e2e4")
    send_command("go depth 3")
    time.sleep(0.5)

    send_command("position startpos moves e2e4 e7e5")
    send_command("go depth 3")
    time.sleep(0.5)

    # Quit
    send_command("quit")

    # Wait for process to exit
    proc.wait(timeout=2)

    print("\n" + "=" * 60)
    print("Checking log files...")
    print("=" * 60)

    # Check UCI log
    if os.path.exists("test_uci.log"):
        print("\n✓ UCI transaction log created!")
        with open("test_uci.log", "r") as f:
            content = f.read()
            print(f"  Size: {len(content)} bytes")
            print(f"  Lines: {content.count(chr(10))}")
            print("\n  First few lines:")
            for line in content.split("\n")[:10]:
                if line:
                    print(f"    {line}")
    else:
        print("\n✗ UCI transaction log NOT created")

    # Check PGN log
    if os.path.exists("test_games.pgn"):
        print("\n✓ PGN export file created!")
        with open("test_games.pgn", "r") as f:
            content = f.read()
            print(f"  Size: {len(content)} bytes")
            if content.strip():
                print("\n  Content:")
                for line in content.split("\n"):
                    if line:
                        print(f"    {line}")
            else:
                print("  (Empty - no complete game)")
    else:
        print("\n✗ PGN export file NOT created")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_uci_logging()
