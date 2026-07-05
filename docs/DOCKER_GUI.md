# Chess GUI in Docker (browser-based, nothing installed on the host)

Play against every engine stage in a real chess GUI without installing
anything on your machine beyond Docker and a browser. Designed for locked-down
hosts (application allowlisting like ThreatLocker): Cute Chess, Python,
torch, and Stockfish all live inside the container; the GUI renders in a
browser tab via web-VNC.

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

**Nothing moves into the container.** The repo stays on your machine and is
bind-mounted at `/app`. The engine process *executes* inside the container,
reading the live working tree — edit code on the host, start a new game in
the GUI, and it runs your changes. Checkpoints you train land in the mounted
folder, so they survive the container and are visible on the host.

## Usage

```bash
cd docker
docker compose up --build     # first build downloads torch; takes a while
```

Open **http://localhost:5800**. You'll see the Cute Chess window.

To play:
1. `Game → New…`
2. Choose *Human* for one side, **Chess_RL** (pre-registered) for the other
3. To pick the engine stage: select Chess_RL → *Configure* → set
   **Engine Type** to `random` / `material` / `minimax` / `mcts` / `puct`

**Stockfish is also pre-registered** — play it yourself, or set up
Chess_RL vs Stockfish and watch (that's course Module 7's measurement rig;
`cutechess-cli` is in the container too for headless 100-game matches).

## Using the neural network (`puct`) stage

Container paths, not host paths: if your trained checkpoint is at
`checkpoints/best_model.pt` in the repo, the container sees it at
`/app/checkpoints/best_model.pt` (that's already the default). Configure via
the engine's **Model File** option. Missing model / missing torch degrades
gracefully to minimax — watch for the `info string` explanation in the
engine debug window (`View → Debug`, then start a game).

## Notes & troubleshooting

- **Port bound to localhost only** (`127.0.0.1:5800`) — the web GUI has no
  auth; don't expose it beyond the machine without adding some.
- **Prefer a native window?** macOS has a built-in VNC client: map port
  5900 too (`"127.0.0.1:5900:5900"`), then Finder → ⌘K →
  `vnc://localhost:5900`.
- **Smaller image:** `docker compose build --build-arg INSTALL_TORCH=false`
  skips torch (~900 MB saved) — Phases 0-2 still fully work.
- **Engine list didn't appear?** It's seeded only on first start of the
  `cutechess-config` volume. Reset with
  `docker compose down -v` (destroys GUI settings, not your repo).
- **Headless matches** (no GUI needed), from a shell in the container:
  ```bash
  docker exec -it chess_rl_gui /usr/games/cutechess-cli \
    -engine name=chess_rl cmd=/opt/venv/bin/python3 arg=/app/chess_rl_uci.py dir=/app \
    -engine name=stockfish cmd=/usr/games/stockfish \
    -each proto=uci tc=40/60 -games 10 -pgnout /app/match.pgn
  ```
- **Run the test suite in the container** (e.g. if the host has no Python):
  ```bash
  docker exec -it chess_rl_gui /opt/venv/bin/pip install pytest
  docker exec -it chess_rl_gui sh -c "cd /app && /opt/venv/bin/python -m pytest -q"
  ```
