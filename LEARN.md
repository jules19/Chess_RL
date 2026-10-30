# Learn to build a chess engine: the learning path

This is the **one place to start**. Ten short units take you from a program that plays random
moves to a neural network guiding a tree search, using the real code in this repository.

**Who this is for:** you can write Python and you know how the chess pieces move. You don't
need to know anything about game engines, search algorithms or neural networks.

**How each unit works:**
- **You'll be able to:** what you can do once you've finished it
- **The idea:** intuition first, in plain language
- **In the code:** exactly where it lives (file and function), and what to look for
- **Try it:** a command to run, with the output you should see
- **Exercises:** practice, with answers hidden under "Answer"
- **Check yourself:** quick recall questions to see whether it stuck

Commit to an answer before you open each "Answer". Guessing first and then checking is what
makes it stick.

## Setup (once)

```bash
pip install -r requirements.txt
python cli/play.py test
```

The second command plays 10 random-vs-random games. If it prints `📊 Test Suite Results`,
you're ready. All commands are run from the repository root.

Optional: `python test/check_torch_device.py` shows whether the neural-network units will run on
the Apple GPU (MPS), CUDA or the CPU. Everything works on CPU; it's just slower.

## The path

| # | Unit | Core idea | Main code |
|---|---|---|---|
| 0 | [The game loop](docs/learn/00_game_loop.md) | A chess program is a loop: pick a move, play it, repeat | `cli/play.py` |
| 1 | [Judging a position](docs/learn/01_evaluation.md) | Turn a board into one number | `engine/evaluator.py` |
| 2 | [Looking ahead: minimax and alpha-beta](docs/learn/02_minimax.md) | Assume the opponent plays their best reply | `search/minimax.py` |
| 3 | [The horizon effect and quiescence](docs/learn/03_quiescence.md) | Don't stop searching in the middle of a fight | `search/minimax.py` |
| 4 | [Talking to chess GUIs: UCI](docs/learn/04_uci.md) | A text protocol any GUI understands | `uci/engine.py` |
| 5 | [Monte Carlo Tree Search](docs/learn/05_mcts.md) | Sample many games and spend effort where it pays off | `search/mcts.py` |
| 6 | [A neural network that sees the board](docs/learn/06_network.md) | Boards and moves as tensors; policy and value heads | `net/encoding.py`, `net/model.py` |
| 7 | [Training on human games](docs/learn/07_training.md) | Imitate the move, predict the result | `training/` |
| 8 | [Measuring honestly](docs/learn/08_measuring.md) | A target is not a result; how to measure strength | `notebooks/01_checkpoint_reality_check.py` |
| 9 | [Neural network + search, and the road to AlphaZero](docs/learn/09_nn_mcts.md) | Why plugging a network into rollouts is slow, and what AlphaZero does instead | `search/nn_mcts.py` |

Units 0–4 are classical chess programming, units 5–9 the modern machine-learning approach.
Each unit takes 30–90 minutes.

## Longer reading

The units link to sections of these tutorials when you want more depth:
- [`minimax_tutorial.html`](minimax_tutorial.html): units 1–4 in long form, with diagrams
- [`neural_network_tutorial.html`](neural_network_tutorial.html): background for units 6–7
- [`chess_rl_complete_tutorial.html`](chess_rl_complete_tutorial.html): roadmap, glossary and history of AlphaZero

## Where the project actually stands

See [`STATUS.md`](STATUS.md) for measured results, not targets. Spoiler: the neural network
has been built and trained, but it is still weak (unit 8 shows how we know). Self-play
reinforcement learning (Phase 4) hasn't been started. The project's day-by-day history is in
[`docs/journal/`](docs/journal/).
