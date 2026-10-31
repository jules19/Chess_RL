# Project Status

**Last updated:** 2026-10-31

## Where things stand

| Phase | What it is | Code | Validated |
|-------|------------|------|-----------|
| 0 | Random engine | ✅ | ✅ yes |
| 1 | Minimax search with a hand-written evaluation ([Modules 1-2](course/module-02-minimax-search/README.md)) | ✅ | ✅ yes |
| 2 | Monte Carlo tree search, MCTS ([Module 3](course/module-03-mcts/README.md)) | ✅ | ✅ yes (strength not yet measured) |
| 3a | Neural network trained on human games ([Module 4](course/module-04-neural-network/README.md)) | ✅ | ⚠️ partly |
| 3b | Search guided by the network ([Module 5](course/module-05-puct/README.md)) | ✅ | ⚠️ partly |
| 4 | Self-play reinforcement learning ([Module 6](course/module-06-selfplay-rl/README.md)) | ✅ | ⚠️ partly |

Terms used below: Elo is the chess rating scale, and a 95% interval is the
range the true result very likely lies in. Stockfish is a strong
open-source chess engine. [Module 7](course/module-07-measure-and-scale/README.md)
explains Elo and the interval.

- **Phase 0.** Games against the random engine complete correctly.
- **Phase 1.** The engine looks ahead through possible moves (minimax) and
  skips lines that cannot change the result (alpha-beta). It also has
  quiescence search, a transposition table, iterative deepening and UCI
  support; Module 2 explains each. It passes unit tests and tactical
  puzzles. Measured: at depth 2 (looking two plies ahead, where a ply is one
  move by one side), it scored 46.5/100 against Stockfish 15.1. Stockfish
  was limited to a rating of 1350 with its UCI_Elo setting and had 0.1 s
  per move. That gives about 1330 Elo (95% interval: 1260-1390).
- **Phase 2.** The engine plays many games out to the end from the current
  position (rollouts) and picks moves by their results. The rollouts are
  guided rather than fully random, and a filter removes moves that leave a
  piece that the opponent can capture without losing anything (blunders).
  Moves in the search are chosen with UCT, a rule that balances moves that
  have done well against moves tried less often. It passes unit tests. Its
  strength against minimax has not yet been measured with a 95% interval.
  Module 3 shows that 6-game matches can't tell which engine is stronger.
- **Phase 3a.** The network suggests moves (its policy) and scores
  positions (its value). It was trained on one month of Lichess games:
  4,794 games, for 3 epochs (3 passes over the data). It was then tested on
  positions from games kept out of training. Its first-choice move matched
  the move played in 27.7% of those positions. Always picking the most
  common move matches 9.3%. The value part only just beats always
  predicting an even position ("always even"), and only after epoch 1.
  After that it overfits: it memorises the training games. It needs several
  months of data.
- **Phase 3b.** Two designs: `nn_mcts.py` (Monte Carlo tree search that
  uses the network to score positions) and PUCT (`search/puct.py`), the
  search AlphaZero uses, which combines search results with the network's
  move suggestions. Both pass unit tests with a stand-in network that has
  no chess knowledge. With the one-month network, PUCT scored 4/6 against
  Monte Carlo tree search with rollouts (95% interval 30-90%, so not yet
  conclusive).
- **Phase 4.** The loop stores self-play games for training, trains the
  network, and plays a match to decide whether the new network replaces
  the old one. It runs from start to finish at small scale (verified). No
  long training run has been done yet.

**Test suite:** `pytest --collect-only -q` lists every test. On 2026-10-31
there were 87, and 10 of them are the course exercises in `tests/course/`,
which are skipped until you do them. GitHub runs the whole suite
automatically each time new code is uploaded to the repository
(`.github/workflows/ci.yml`). All tests pass.

## Changes

The change history is in [docs/history/CHANGELOG.md](docs/history/CHANGELOG.md).
The course itself starts at [course/README.md](course/README.md).

## What's next (in priority order)

The remaining work is experiments to run. No new code is needed, and the
course takes you through each one:

1. **Train on more real data** (course Module 4). The quick run and a
   one-month run are done (results in Module 4, exercise 7). The next step
   is several months of Lichess games, until the value part beats "always
   even" by a clear margin (`training/eval_supervised.py`).
2. **Test PUCT with the trained network** (Module 5). Play PUCT with the
   network against Monte Carlo tree search with rollouts, with the same
   number of simulations (search steps) per move for each.
3. **Run a real self-play loop** (Module 6). Start from the network trained
   on human games, run it overnight, and record how often a new network
   replaces the old one.
4. **Measure strength reliably** (Module 7). Use cutechess-cli (a
   command-line program that plays matches between engines) to play at
   least 100 games against Stockfish limited to a set strength. Report Elo
   with its 95% interval. These measurements replace all the historical
   estimates.

## Environment

- Development machine: an Apple M1 Mac. Experiments run on the CPU only,
  inside a Docker container, because the Mac's graphics processor (GPU) is
  not available there. `auto_device()` picks the device for the network:
  an NVIDIA GPU if there is one, otherwise an Apple GPU, otherwise the CPU. The times quoted in the course are from
  that 8-core CPU.
- Install: `pip install -e ".[train,dev]"`.
- No money has been spent on cloud computing so far. None is needed until
  self-play is run at a much larger scale.
