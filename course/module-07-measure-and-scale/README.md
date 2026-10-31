# Module 7: Measure and Scale

> **Where you are:** the training loop works. Now you find out how strong the engine is, and which settings give the most strength for extra computing time.
> **Time:** 3-4 hours of your time, plus matches that run for hours in the background.
> **You'll be able to:**
> - turn a match score into an Elo rating difference with a 95% interval (both explained below), by hand and with `cli/match.py`
> - work out how many games a claim needs before you run the match
> - measure an engine against a calibrated opponent instead of its own past self
> - design a scaling experiment that changes one setting and keeps the total computing time fixed

Replace every "~1400-1600 Elo" guess in this repository's history with a
number you can defend. Then learn which settings turn extra computing time
into playing strength.

*Elo* is the rating scale chess uses. Only differences matter: a player 100
Elo points stronger than another is expected to score about 64%, and a
400-point gap means an expected score of about 91%.

## Part 0: What a match result tells you

As an analogy, a game between two engines is like a coin flip with an
unknown bias, and the match measures that bias. Count a win as 1, a draw as
½ and a loss as 0. The score you observe after a few games is not the true
score, the one you would get after unlimited games. A *95% interval* is a
range built so that, 95% of the time, it contains the true score. Its
*half-width* is the distance from the centre to either end (the ± figure).

After *n* games with average score *p*, the 95% interval for the true score
is roughly the formula below. It is wider when there are fewer games (small
*n*) and when the score is near 50%:

```
p ± 1.96 · √( p(1−p) / n )
```

- 20 games at 50%: ±22 percentage points, that is, anywhere from 28% to 72%.
  That's why 11.5/20 in Module 1 proved nothing.
- 100 games at 55%: about ±10 percentage points (45% to 65%), roughly ±70 Elo.
- To tell 55% from 50% reliably you need several hundred games.

The rough formula is inaccurate for small *n* and for scores near 0% or 100%
(4/4 is not proof of 100%). `cli/match.py` uses the *Wilson* interval, which
stays accurate in those cases. Instead of centring the interval on the
observed score, it asks which true scores would make your result
unsurprising. When the sample is small, this keeps the interval away from 0%
and 100%. The script then converts both ends of the interval into Elo.

The conversion below turns a score (0 to 1) into a rating difference. A
score of 0.5 gives 0, and the difference grows quickly as the score nears 0
or 1: `ΔElo = −400 · log10(1/score − 1)`.
Draws make the true interval a little
narrower than this, so these numbers are slightly cautious.

**RL connection.** A reinforcement learning (RL) agent is noisy in two ways: the
games are random, and so is training (Module 6's seeds). If an agent beats its
own previous version, you learn that it changed. You do not learn how strong
it is. Calibrated opponents and intervals give you a measurement of strength
that does not depend on the agent's own history.

## Part 1: Measurement against a calibrated opponent

**The problem.** Every Elo figure in `docs/history/` is a self-estimate,
mostly from matches against the project's own earlier engines. Beating your
previous engine says little about absolute strength, so these figures tend
to be too high.

A second problem affects any match: two deterministic engines (with no
randomness) replay the same game every time. That is why
`training/evaluate.py` supports temperature (randomness in move choice).

**The fix: calibrated opponents.** The Stockfish engine can limit its
strength to a stated rating with two Universal Chess Interface (UCI)
options, `UCI_LimitStrength` and `UCI_Elo`. UCI is the standard text
protocol between chess engines and the programs that run them. In this
course, "Stockfish at 1350" means Stockfish limited to a rating of 1350.

The lowest rating you can set is 1350 in Stockfish 15 (Debian 12's
package) and 1320 in Stockfish 16 and later. To check yours, send `uci` and look at `option name UCI_Elo`. If you score 55%
against Stockfish at 1350, your rating is roughly 1385, on a scale other
people use. Stockfish is the only extra program you need.

**Getting Stockfish.** Pick one:
- **The Docker GUI container** has it already, with this repository mounted at
  `/app` ([docs/DOCKER_GUI.md](../../docs/DOCKER_GUI.md)). Start it with
  `cd docker && docker compose up -d --build`. Nothing is installed on your
  machine.
- **Install it** with your package manager (`apt install stockfish` on
  Debian/Ubuntu, `brew install stockfish` on a Mac). On a work machine with
  application-control software that blocks unapproved programs (for
  example ThreatLocker), use the container instead.

If you're unsure, use the container, which also has cutechess-cli (Route B
below). The historical self-estimates put minimax and Monte Carlo
tree search (MCTS) at about 1200-1600. The run below is the project's first
measured rating. Your result will differ a little with hardware, Stockfish
version and chance, and the interval accounts for that.

**Route A: `cli/match.py`, no extra tools (start here).** An engine
specification starting with `uci:` plays any UCI engine. Start with a quick check of 10 games, which takes
a few minutes:
```bash
# inside the container (Stockfish is at /usr/games/stockfish):
docker exec -it chess_rl_gui sh -c "cd /app && /opt/venv/bin/python cli/match.py \
    minimax:depth=2 \
    uci:cmd=/usr/games/stockfish,movetime=0.1,UCI_LimitStrength=true,UCI_Elo=1350 \
    --games 10"
# or, with Stockfish installed locally:
python3 cli/match.py minimax:depth=2 \
    uci:cmd=stockfish,movetime=0.1,UCI_LimitStrength=true,UCI_Elo=1350 --games 10
```
**Expected (measured with Stockfish 15.1, 8-core laptop CPU):** at 10-20
games the 95% interval is about ±20-30 percentage points. That only checks that the
setup works; it is not a rating. Our 20-game run scored 8/20, with an
interval of 22% to 61%, in about 4 minutes.

Then raise `--games` to 100, which takes about 20 minutes. In our 100-game
reference run, minimax depth 2 scored 46.5/100 against Stockfish at 1350,
with 37 wins, 19 draws and 44 losses (written +37 =19 −44). The 95% interval is 37% to 56%, which is −24 Elo (−92 to +43).
That puts the engine at about 1330, somewhere between 1260 and 1390.

This depth 2 result is already inside the project's historical estimate
for depth 3 (about 1200-1400, docs/history/DAY_7_UCI.md). At 100 games, it
still cannot be told apart from 1350.

One caution: Stockfish's `UCI_Elo` is calibrated at longer time controls
than 0.1 s a move. Treat results from games at 0.1 s a move as approximate,
and say so in your report.

**Route B: `cutechess-cli`, the standard tool** (installed in the
container). It runs matches between UCI engines and handles clocks and
adjudication (ending a game early with a declared result) itself. Your
engine already implements UCI (`uci/engine.py`), so cutechess-cli can run it
with no extra code.
`run_engine.sh`, in the repository root, starts it for any graphical
interface or match runner:
```bash
cutechess-cli \
  -engine cmd=./run_engine.sh name=chess_rl \
  -engine cmd=stockfish name=sf1400 option.UCI_LimitStrength=true option.UCI_Elo=1400 \
  -each proto=uci tc=40/60 -games 100 -pgnout match.pgn
```
[docs/DOCKER_GUI.md](../../docs/DOCKER_GUI.md) has the exact
`docker exec … cutechess-cli` version.

**Reading the result.** Convert the score into an Elo difference with a
95% interval (Part 0). `cli/match.py` prints both. cutechess-cli prints its
own Elo estimate and error, so check it against the formula. With 100 games,
expect about ±70 Elo. With 10 games, the interval is too wide to be
useful.

**Do:** measure your minimax (depth 3), MCTS (200 simulations) and best
network against at least two Stockfish levels. Route A works for all of
them, because `puct:...,net=checkpoints/best_model.pt` is also an engine
specification. Make a
table, and compare it with the historical guesses.

## Part 2: Scaling settings in order of effect

When you have more computing time (a graphics processing unit (GPU), or a
whole weekend), spend it in this order:

1. **Simulations per move in self-play.** Better search gives better
   training targets, and the gain builds up over iterations. AlphaZero used
   800 simulations; this course uses 16-64.
2. **Games per iteration.** This gives more data per iteration.
3. **Batched neural network inference** (Module 5, exercise 5): running the
   network on many positions in one call. This makes settings 1 and 2
   affordable. With one position at a time, about 90% of a GPU's capacity
   is unused.
4. **Parallel self-play workers.** Games are independent, so they can run in
   parallel with no coordination. The usual design is `multiprocessing` with
   one shared inference server (one process that runs the network for all
   workers).
5. **Network size.** Change this last. The current network (4 residual
   blocks of 128 channels, Module 4) has not yet learned as much as its
   size allows from the amount of data this course generates. Bigger networks trained on too little data overfit: they memorise
   the training positions and do worse on new ones.

**Do (exercise):** test setting 1. Double the simulations (from 32 to 64)
and halve the games, so the running time stays the same. Run both
configurations for 5 iterations, with every other flag the same:
```bash
python3 selfplay/train_loop.py --init-checkpoint checkpoints/best_model.pt --iterations 5 \
    --simulations 32 --games-per-iter 20 --train-steps 300 --checkpoint-dir run_a
python3 selfplay/train_loop.py --init-checkpoint checkpoints/best_model.pt --iterations 5 \
    --simulations 64 --games-per-iter 10 --train-steps 300 --checkpoint-dir run_b
```
Time the first iteration of each. If one is much slower, the comparison is
not at fixed computing time, so adjust `--games-per-iter` until they match.
Each run takes a few hours on a laptop CPU.

After the runs, play the two champions against each other in a 40-game
match:
```bash
python3 cli/match.py puct:sims=64,net=run_a/champion.pt puct:sims=64,net=run_b/champion.pt --games 40
```
Which use of the computing time won? Before you run the match, use Part 0
to work out how big a difference 40 games can detect, and write it down.
If the interval contains 50%, say so in your conclusion.

<details>
<summary>How big a difference can 40 games detect?</summary>

About ±15 percentage points around 50%, which is roughly ±110 Elo, so the
match can show only a large difference.
</details>

**Laptop version (under half an hour in total):** run the same pair with
`--iterations 2 --games-per-iter 4` and `--train-steps 50` (use `2` games
per iteration for run B). Then play a 10-game match instead of the 40-game
one. This covers the whole procedure, but the interval will contain 50%, so
say so in your conclusion. To answer the question, you need the full runs.

## Check your understanding

<details>
<summary><b>Q1.</b> Your engine scores 6/10 against Stockfish at 1400. Compute (roughly) the Elo difference this implies, and its 95% interval.</summary>

Point estimate: ΔElo = −400·log10(1/0.6 − 1) ≈ +70. But the 95% interval on
a 10-game score of 60% spans roughly 31-83% (Wilson), that is, anywhere from
about −140 to +280 Elo. That range is too wide to be useful, which is why
this module asks for at least 100 games.
</details>

<details>
<summary><b>Q2.</b> Why do two deterministic engines make a "10-game match" meaningless in a different way than noise does?</summary>

With the same position, the same engines and no randomness, the identical
game replays ten times. You have ten copies of one game, so your sample size
is one. To fix it, vary the openings (cutechess's `-openings` flag) or add a
little temperature (`training/evaluate.py` supports it). Playing more
games reduces noise, but more copies of one game add no information.
</details>

<details>
<summary><b>Q3.</b> You want to know whether a change is worth +30 Elo. Roughly how many games do you need, and what does that tell you about Module 6's 6-game arena?</summary>

+30 Elo is a score of about 54%. To separate 54% from 50% you need the
interval's half-width under 4 percentage points: 1.96 · √(0.25 / n) < 0.04
gives n of about 600 games. A 6-game arena can only detect huge differences
(its half-width is about ±40 percentage points). It can catch a much worse candidate, but
it cannot measure a small improvement.
</details>

## Checkpoint

- [ ] A measured rating with a 95% interval, from at least 100 games against a
      calibrated opponent
- [ ] One scaling experiment with a conclusion you'd defend

Previous: [Module 6: Self-Play Reinforcement Learning](../module-06-selfplay-rl/README.md) · Next: [Module 8: Engineering Extras](../module-08-engineering/README.md)
