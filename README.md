# Chess_RL
Incremental development of a chess program, from random moves towards AlphaZero-style reinforcement learning.

---

## **🚀 Start here**

**To learn how it works, follow [`LEARN.md`](LEARN.md).** It's a 10-unit path through the real code, with
exercises and answers. Each unit builds on the previous one.

**To play right now:**
```bash
pip install -r requirements.txt
python cli/play.py minimax        # watch minimax play a random mover (also: random, mcts, human-minimax)
python chess_rl_uci.py            # UCI engine for chess GUIs; see docs/guides/UCI_SETUP_MAC.md
```

## **What exists today**

| Phase | What | Code | Status |
|---|---|---|---|
| 0 | Random mover | `cli/play.py` | ✅ Built |
| 1 | Hand-written evaluation + minimax with alpha-beta and quiescence; UCI interface | `engine/evaluator.py`, `search/minimax.py`, `uci/engine.py` | ✅ Built |
| 2 | Monte Carlo Tree Search with evaluator-guided rollouts | `search/mcts.py` | ✅ Built |
| 3a | Policy-value network trained on human games | `net/`, `training/` | ✅ Built; ⚠️ the trained model is weak (see below) |
| 3b | Network plugged into MCTS rollouts | `search/nn_mcts.py` | ✅ Built; very slow (~465 network calls per simulation) |
| 4 | Self-play reinforcement learning | none | ❌ Not started |

**Measured, not estimated:** the trained network (5 epochs, 411 games) picks the human's move 13% of the time
(random: 6%), scored 0 wins in 40 games against random, material-only and minimax opponents, and lost every
game against minimax. Full numbers and method: [`STATUS.md`](STATUS.md) and
[unit 8](docs/learn/08_measuring.md). Earlier documents quoted Elo ratings for each phase; **those were never
measured** and have been removed.

**Repository map:** teaching material is in `LEARN.md`, `docs/learn/`, the three HTML tutorials and
`notebooks/`. How-to guides are in `docs/guides/`. Day-by-day project history is in `docs/journal/`.
The pedagogy rubric and doc checker are in `docs/pedagogy/`.

---

## **Long-term vision**

The end goal is an engine that **teaches itself**: it plays millions of games against itself, using MCTS guided
by a neural network, trains that network on the results, and keeps the new version only if it beats the old one
(the AlphaZero loop, [unit 9](docs/learn/09_nn_mcts.md)). That's Phase 4, which isn't built yet. The full vision
and the hardware each scale would need are in [`docs/VISION.md`](docs/VISION.md).
