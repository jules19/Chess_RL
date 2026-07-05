# Module 7 — Measure & Scale

**Goal:** replace every "~1400-1600 Elo" guess in this repo's history with a
number you can defend, then learn which knobs convert compute into strength.

## Part 1: Honest measurement

**The uncomfortable truth:** every Elo figure in `docs/history/` is a
self-estimate. Self-play ladders systematically overestimate — beating your
own previous engine says little about absolute strength, and two
deterministic engines replay the same game forever (which is why
`training/evaluate.py` supports temperature).

**The fix: calibrated opponents.**

1. Install a reference engine and a match runner:
   `cutechess-cli` + Stockfish (its `UCI_LimitStrength`/`UCI_Elo` options go
   down to ~1320). *Locked-down machine? Both are pre-installed in the
   Docker GUI container — see [docs/DOCKER_GUI.md](../../docs/DOCKER_GUI.md),
   including the exact `docker exec … cutechess-cli` incantation.*
2. Your engine already speaks UCI (`uci/engine.py`) — this is the moment
   Phase 1's UCI work pays off.
3. Run a real match:
   ```bash
   cutechess-cli \
     -engine cmd=./run_engine.sh name=chess_rl \
     -engine cmd=stockfish name=sf1400 option.UCI_LimitStrength=true option.UCI_Elo=1400 \
     -each proto=uci tc=40/60 -games 100 -pgnout match.pgn
   ```
4. Convert score → Elo difference: `ΔElo = -400·log10(1/score - 1)`.
   **Report the error bar**: with 100 games, ±~50 Elo. With 10 games you
   know almost nothing — compute the interval before believing any match
   result in this course.

**Do:** measure your minimax (depth 3), MCTS (200 sims), and best network
against at least two Stockfish levels. Make a table. Compare with the
historical guesses and enjoy the discrepancy.

## Part 2: Scaling knobs, in order of leverage

When you get real compute (a GPU, a weekend), spend it in this order:

1. **Simulations per move in self-play** (quality of training targets
   compounds; AlphaZero used 800 vs our toy 16-64)
2. **Games per iteration** (more data per generation)
3. **Batched NN inference** (Module 5, exercise 4 — this is what makes 1-2
   affordable; single-position inference wastes ~90% of a GPU)
4. **Parallel self-play workers** (games are embarrassingly parallel;
   `multiprocessing` + one shared inference server is the classic design)
5. **Network size** (last! 4×128 is nowhere near saturated at our data
   scale; bigger nets on too little data just overfit)

**Do (exercise):** pick knob 1. Double simulations (32→64) at fixed
wall-clock by halving games, run both configurations for 5 iterations, and
arena the two champions. Which conversion of compute won? Data beats
intuition here, which is the point of the module.

## Check your understanding

<details>
<summary><b>Q1.</b> Your engine scores 6/10 against Stockfish-1400. Compute (roughly) the Elo difference this implies — and the honest range.</summary>

Point estimate: ΔElo = −400·log10(1/0.6 − 1) ≈ **+70**. But the 95% band on
a 10-game score of 60% spans roughly 30–85%, i.e. anywhere from about
−150 to +300 Elo. Ten games told you almost nothing — which is the whole
argument for the 100-game minimum in this module.
</details>

<details>
<summary><b>Q2.</b> Why do two deterministic engines make a "10-game match" meaningless in a different way than noise does?</summary>

Same position + same engines + no randomness → the identical game replays
ten times. You have a sample size of one wearing a costume. Fixes: vary
openings (cutechess's `-openings` flag), or a little temperature
(`training/evaluate.py` supports it). Noise you can average away; zero
variance you cannot.
</details>

## Checkpoint

- [ ] A measured rating with an error bar, from ≥100 games against a
      calibrated opponent
- [ ] One scaling experiment with a conclusion you'd defend

Next: [Module 8 — Engineering Extras](../module-08-engineering/README.md)
