#!/usr/bin/env python3
"""
Quick test script for Chess_RL UCI engine.
Shows that the UCI engine is working properly.
"""

import subprocess
import sys

def test_uci_engine():
    """Test the UCI engine with basic commands."""

    print("=" * 60)
    print("Testing Chess_RL UCI Engine")
    print("=" * 60)
    print()

    # Commands to send to the engine
    commands = [
        "uci",
        "isready",
        "ucinewgame",
        "position startpos",
        "go depth 3",
        "position startpos moves e2e4",
        "go depth 2",
        "quit"
    ]

    print("Sending commands to engine:")
    for cmd in commands:
        print(f"  > {cmd}")
    print()
    print("-" * 60)
    print()

    # Create the input string
    input_str = "\n".join(commands) + "\n"

    # Run the engine
    try:
        result = subprocess.run(
            [sys.executable, "chess_rl_uci.py"],
            input=input_str,
            capture_output=True,
            text=True,
            timeout=30
        )

        print("Engine Output:")
        print(result.stdout)

        if result.stderr:
            print("Errors:")
            print(result.stderr)

        # Check for expected responses
        output = result.stdout

        print()
        print("=" * 60)
        print("Validation:")
        print("=" * 60)

        checks = [
            ("uciok" in output, "✓ UCI identification works"),
            ("readyok" in output, "✓ Ready check works"),
            ("bestmove" in output, "✓ Move generation works"),
            ("e2e4" in output or "d2d4" in output or "g1f3" in output,
             "✓ Engine makes reasonable moves (e4, d4, or Nf3)"),
        ]

        all_passed = True
        for passed, message in checks:
            status = "✅" if passed else "❌"
            print(f"{status} {message}")
            if not passed:
                all_passed = False

        print()
        if all_passed:
            print("🎉 All tests passed! Engine is working correctly.")
            print()
            print("Next steps:")
            print("1. Install Banksia GUI: https://banksiagui.com/")
            print("2. Add this engine to Banksia GUI")
            print("3. Play a game!")
        else:
            print("❌ Some tests failed. Check the output above.")
            return 1

        return 0

    except subprocess.TimeoutExpired:
        print("❌ Error: Engine timed out (took more than 30 seconds)")
        return 1
    except FileNotFoundError:
        print("❌ Error: Could not find chess_rl_uci.py")
        print("Make sure you're running this from the Chess_RL directory")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(test_uci_engine())
