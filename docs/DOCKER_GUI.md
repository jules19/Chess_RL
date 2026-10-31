# Chess GUI in Docker (browser-based, nothing installed on the host)

Play against every engine stage in a chess GUI (graphical user interface: a
program that shows the board). You need only Docker and a browser. The setup
suits locked-down computers where allowlisting software, such as ThreatLocker,
blocks unapproved programs. Cute Chess (the GUI), Python, torch (the PyTorch
neural network library) and Stockfish (a strong chess engine) are all inside
the container. The GUI appears in a browser tab through web-VNC (VNC sends a
remote screen to you, and your mouse and keyboard back).

## Architecture

```
your machine                          container
┌───────────────────┐   bind mount   ┌─────────────────────────────┐
│ this repo (files) │ ─────────────► │ /app  (the same files)      │
│ your editor       │                │ Cute Chess GUI ──launches──►│
│ browser ◄─────────┼── :5800 ───────│ web-VNC (KasmVNC)           │
│ Docker            │                │ /opt/venv python + torch    │
└───────────────────┘                │ Stockfish (sparring/rating) │
                                     └─────────────────────────────┘
```

The repository stays on your machine. A bind mount makes the same folder
visible inside the container at `/app`, without copying it. The engine runs
inside the container and reads your current files, so if you edit code on the
host and start a new game, the engine runs your changes. Checkpoints (saved
trained networks) are written to the mounted folder, so they stay on the host
after the container stops.

## Usage

Start the container:

```bash
cd docker
docker compose up --build     # first build downloads torch; takes a while
```

Expected: the build finishes and the container keeps running.

Open http://localhost:5800 in your browser. Expected: the Cute Chess window
appears in the browser tab.

To play a game:

1. Choose `Game → New…`. Expected: the new game dialog opens.
2. Choose *Human* for one side and Chess_RL (already registered) for the
   other.
3. To choose the engine stage, select Chess_RL, then *Configure*, and set
   Engine Type to one of these values:
   - `random`: plays a random legal move.
   - `material`: plays the move that leaves the best material count (piece
     values), looking one move ahead.
   - `minimax`: searches several moves ahead with alpha-beta search, which
     skips moves that cannot change the result (Module 2).
   - `mcts`: Monte Carlo tree search, which plays many quick games from the
     position to judge each move (Module 3).
   - `puct`: the search used by AlphaZero (DeepMind's self-taught chess
     program), guided by a trained neural network (Module 5).

   Expected: the next game uses the stage you chose.

Stockfish is also registered. Play it yourself, or start Chess_RL against
Stockfish and watch. Course Module 7 uses this setup to measure strength. The
container also includes `cutechess-cli`, which runs 100-game matches without
the GUI.

## Using the neural network (`puct`) stage

Give the engine paths as the container sees them. If your trained checkpoint is at `checkpoints/best_model.pt` in the
repository, the container sees it at `/app/checkpoints/best_model.pt`. This
is already the default.

To use another file, set the engine's Model File option. If the model or
torch is missing, the engine plays minimax instead of stopping with an error.
It gives the reason in an `info string` message (a text line the engine sends
to the GUI). To see it, open the engine debug window (`View → Debug`), then
start a game.

## Notes and troubleshooting

- **Port bound to the local machine only** (`127.0.0.1:5800`). The web GUI has
  no login. Do not make it reachable from other machines unless you add one.
- **Native window instead of a browser tab.** macOS has a built-in VNC client.
  Map port 5900 as well (`"127.0.0.1:5900:5900"`). Then, in Finder, press ⌘K
  and connect to `vnc://localhost:5900`.
- **Smaller image.** `docker compose build --build-arg INSTALL_TORCH=false`
  leaves out torch (saves about 900 MB). The random, material, minimax
  and MCTS stages still work. The repository calls these Phases 0-2; the
  neural network stages are Phases 3 and 4.
- **Engine list is empty.** The list is copied into the `cutechess-config`
  volume (Docker storage that keeps the GUI settings) only on the first start.
  To reset it, run `docker compose down -v`. This deletes the GUI settings,
  but not your repository.
- **Matches without the GUI.** This command runs the command-line version of
  Cute Chess inside the container:
  ```bash
  docker exec -it chess_rl_gui /usr/games/cutechess-cli \
    -engine name=chess_rl cmd=/opt/venv/bin/python3 arg=/app/chess_rl_uci.py dir=/app \
    -engine name=stockfish cmd=/usr/games/stockfish \
    -each proto=uci tc=40/60 -games 10 -pgnout /app/match.pgn
  ```
- **Run the test suite in the container**, for example if the host has no
  Python:
  ```bash
  docker exec -it chess_rl_gui /opt/venv/bin/pip install pytest
  docker exec -it chess_rl_gui sh -c "cd /app && /opt/venv/bin/python -m pytest -q"
  ```
