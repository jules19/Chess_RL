# Module 7 — Measure & Scale

> **Where you are:** after the loop works. Now: how strong is it really, and where should more compute go?
> **Time:** 3-4 hours of your time, plus matches that run for hours in the background.
> **You'll be able to:**
> - turn a match score into an Elo difference *with* a 95% interval, by hand and with `cli/match.py`
> - work out how many games a claim needs before you run the match
> - measure an engine against a calibrated opponent instead of its own past self
> - design a scaling experiment that changes one knob and holds compute fixed

Replace every "~1400-1600 Elo" guess in this repo's history with a number you
can defend, then learn which knobs convert compute into strength.

## Part 0: how much does a match tell you?

A game between two engines is a noisy coin flip whose bias you're trying to
measure. Count a win as 1, a draw as ½ and a loss as 0. After *n* games with
average score *p*, the 95% interval for the true score is roughly

```
p ± 1.96 · √( p(1−p) / n )
```

- 20 games at 50%: ±22 percentage points, i.e. anywhere from 28% to 72%.
  That's why 11.5/20 in Module 1 proved nothing.
- 100 games at 55%: about ±10 points (45% to 65%), roughly ±70 Elo.
- To tell 55% from 50% reliably you need several hundred games.

The rough formula breaks down for small *n* and for scores near 0% or 100%
(4/4 is not proof of 100%). `cli/match.py` uses the *Wilson* interval, which
stays honest there: instead of centring the interval on the observed score,
it asks which true scores would make your result unsurprising, and that
pulls the interval in from 0% and 100% when the sample is small. It then
turns both ends into Elo with
`ΔElo = −400 · log10(1/score − 1)`. Draws make the true interval a little
narrower than this, so these numbers err on the cautious side.

**RL lens.** Measuring an RL agent is measuring a noisy process, twice
over: the games are noisy, and so is training (Module 6's seeds). An agent
that beats its own previous version tells you it changed, not how strong it
is. Calibrated opponents and intervals are how you avoid fooling yourself.

## Part 1: Honest measurement

**The uncomfortable truth:** every Elo figure in `docs/history/` is a
self-estimate. Self-play ladders systematically overestimate — beating your
own previous engine says little about absolute strength, and two
deterministic engines replay the same game forever (which is why
`training/evaluate.py` supports temperature).

**The fix: calibrated opponents.** Stockfish can limit its own strength to
a stated rating (`UCI_LimitStrength` and `UCI_Elo`, down to about 1320).
Beat Stockfish-1320 55% of the time and you're roughly 1350, on a scale
other people use. You need only one thing installed: Stockfish.

**Getting Stockfish.** Pick one:
- **The Docker GUI container** has it already, with this repo mounted at
  `/app` ([docs/DOCKER_GUI.md](../../docs/DOCKER_GUI.md)). Start it with
  `cd docker && docker compose up -d --build`. Nothing is installed on your
  machine.
- **Install it** with your package manager (`apt install stockfish` on
  Debian/Ubuntu, `brew install stockfish` on a Mac). On a work machine with
  application control (ThreatLocker and the like), use the container instead.

If you're unsure, use the container: it's also where cutechess-cli lives.
This course has no measured Stockfish result to compare yours with yet. The
only numbers on record are the historical self-estimates (about 1200-1600
for minimax and MCTS), and testing those is the point of this module.

**Route A — `cli/match.py`, no extra tools (start here).** The `uci:` engine
spec plays any UCI engine. Smoke test, 10 games, a few minutes:
```bash
# inside the container (Stockfish is at /usr/games/stockfish):
docker exec -it chess_rl_gui sh -c "cd /app && /opt/venv/bin/python cli/match.py \
    minimax:depth=2 \
    uci:cmd=/usr/games/stockfish,movetime=0.1,UCI_LimitStrength=true,UCI_Elo=1320 \
    --games 10"
# or, with Stockfish installed locally:
python3 cli/match.py minimax:depth=2 \
    uci:cmd=stockfish,movetime=0.1,UCI_LimitStrength=true,UCI_Elo=1320 --games 10
```
**What to expect:** whatever the score, the interval will be about ±30
points at 10 games: a smoke test that the rig works, not a rating. Then
raise `--games` to 100 and leave it overnight. One caution: Stockfish's
`UCI_Elo` is calibrated at longer time controls than 0.1 s a move, so treat
short-time numbers as rough and say so in your report.

**Route B — `cutechess-cli`, the standard tool** (installed in the
container). It handles clocks and adjudication itself. Your engine already
speaks UCI (`uci/engine.py`) — this is the moment Phase 1's UCI work pays
off. `run_engine.sh`, in the repository root, starts it for any GUI or
match runner:
```bash
cutechess-cli \
  -engine cmd=./run_engine.sh name=chess_rl \
  -engine cmd=stockfish name=sf1400 option.UCI_LimitStrength=true option.UCI_Elo=1400 \
  -each proto=uci tc=40/60 -games 100 -pgnout match.pgn
```
[docs/DOCKER_GUI.md](../../docs/DOCKER_GUI.md) has the exact
`docker exec … cutechess-cli` version.

**Reading the result.** Convert score → Elo difference with an interval
(Part 0). `cli/match.py` prints both; cutechess-cli prints its own Elo
estimate and error, so check it against the formula. With 100 games expect
about ±70 Elo. With 10 games you know almost nothing.

**Do:** measure your minimax (depth 3), MCTS (200 sims), and best network
against at least two Stockfish levels (Route A works for all of them:
`puct:...,net=checkpoints/best_model.pt` is a spec too). Make a table. Compare with the
historical guesses and enjoy the discrepancy.

## Part 2: Scaling knobs, in order of leverage

When you get real compute (a GPU, a weekend), spend it in this order:

1. **Simulations per move in self-play** (quality of training targets
   compounds; AlphaZero used 800 vs our toy 16-64)
2. **Games per iteration** (more data per generation)
3. **Batched NN inference** (Module 5, exercise 5 — this is what makes 1-2
   affordable; single-position inference wastes ~90% of a GPU)
4. **Parallel self-play workers** (games are embarrassingly parallel;
   `multiprocessing` + one shared inference server is the classic design)
5. **Network size** (last! 4×128 is nowhere near saturated at our data
   scale; bigger nets on too little data just overfit)

**Do (exercise):** pick knob 1. Double simulations (32→64) at fixed
wall-clock by halving games, and run both configurations for 5 iterations,
with every other flag the same:
```bash
python3 selfplay/train_loop.py --init-checkpoint checkpoints/best_model.pt --iterations 5 \
    --simulations 32 --games-per-iter 20 --train-steps 300 --checkpoint-dir run_a
python3 selfplay/train_loop.py --init-checkpoint checkpoints/best_model.pt --iterations 5 \
    --simulations 64 --games-per-iter 10 --train-steps 300 --checkpoint-dir run_b
```
Time the first iteration of each: if one is much slower, the comparison
isn't at fixed compute, so adjust `--games-per-iter` until they match. Each
run is a few hours on a laptop CPU. Then play the two champions against each
other:
```bash
python3 cli/match.py puct:sims=64,net=run_a/champion.pt puct:sims=64,net=run_b/champion.pt --games 40
```
Which conversion of compute won? Before you run it, use Part 0 to decide how
big a difference 40 games can detect. (It's about ±15 points around 50%,
roughly ±110 Elo: only a large difference will show. Say so in your
conclusion if the interval contains 50%.) Data beats intuition here, which is
the point of the module.

## Check your understanding

<details>
<summary><b>Q1.</b> Your engine scores 6/10 against Stockfish-1400. Compute (roughly) the Elo difference this implies — and the honest range.</summary>

Point estimate: ΔElo = −400·log10(1/0.6 − 1) ≈ **+70**. But the 95% band on
a 10-game score of 60% spans roughly 31–83% (Wilson), i.e. anywhere from
about −140 to +280 Elo. Ten games told you almost nothing — which is the whole
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

<details>
<summary><b>Q3.</b> You want to know whether a change is worth +30 Elo. Roughly how many games do you need, and what does that tell you about Module 6's 6-game arena?</summary>

+30 Elo is a score of about 54%. To separate 54% from 50% you need the
interval's half-width under 4 points: 1.96 · √(0.25 / n) < 0.04 gives n of
about 600 games. A 6-game arena can only detect huge differences (its
half-width is about ±40 points), so it is a sanity check against disasters,
not a measurement.
</details>

## Checkpoint

- [ ] A measured rating with an error bar, from ≥100 games against a
      calibrated opponent
- [ ] One scaling experiment with a conclusion you'd defend

Previous: [Module 6 — Self-Play Reinforcement Learning](../module-06-selfplay-rl/README.md) · Next: [Module 8 — Engineering Extras](../module-08-engineering/README.md)
