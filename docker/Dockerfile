# Cute Chess GUI + this repo's engine + Stockfish, rendered in your browser.
#
# Built on jlesage/baseimage-gui, which wraps a single Linux GUI app with a
# web-based VNC layer (KasmVNC): open http://localhost:5800 and the Cute
# Chess window is there. Nothing is installed on the host — the point of
# this image is that Docker + a browser is all the host ever needs.
#
# The repo itself is NOT baked into the image. docker-compose bind-mounts it
# at /app, so the engine the GUI launches is the live working tree you're
# editing on the host.
#
# Build (from the docker/ directory):
#   docker compose build                      # includes CPU torch (~1.5 GB image)
#   docker compose build --build-arg INSTALL_TORCH=false   # Phases 0-2 only (~600 MB)

FROM jlesage/baseimage-gui:debian-12-v4

# cutechess  - the GUI (also provides cutechess-cli for headless matches)
# stockfish  - calibrated opponent for strength measurement (course Module 7)
# python3    - runs the engine; venv keeps it isolated from system python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cutechess \
        stockfish \
        python3 \
        python3-venv \
        fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# Engine dependencies live in the image (the code lives in the bind mount).
# Torch is optional: Phases 0-2 (random/material/minimax/mcts) don't need it;
# the 'puct' engine type does.
ARG INSTALL_TORCH=true
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir chess numpy && \
    if [ "$INSTALL_TORCH" = "true" ]; then \
        /opt/venv/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Pre-registered engine list (Chess_RL + Stockfish), seeded on first start
COPY engines.json /defaults/engines.json

# baseimage-gui convention: /startapp.sh is what gets displayed
COPY startapp.sh /startapp.sh
RUN chmod +x /startapp.sh

ENV APP_NAME="Cute Chess"

# 5800 = web UI (browser), 5900 = raw VNC (macOS Screen Sharing, optional)
EXPOSE 5800 5900
