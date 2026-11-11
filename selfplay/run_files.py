"""Supplied experiment-file handling; Module 6 learners study train_loop.py.

One writer per run. A completion marker selects a fully written snapshot.
Only that snapshot is retained by default; --keep-history keeps older ones.
This handles interrupted processes, not power-loss durability or exact RNG replay.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile

import torch

from net.model import load_model
from selfplay.replay_buffer import ReplayBuffer

LOG_HEADER = "iteration,positions_in_buffer,policy_loss,value_loss,arena_score,promoted\n"
_STATE_FILES = ("champion.pt", "replay_buffer.npz", "run_log.csv")


def save_checkpoint(model, path, iteration):
    """Write the same self-describing model format used elsewhere in the course."""
    torch.save({"model_state_dict": model.state_dict(),
                "input_channels": model.input_channels,
                "num_res_blocks": model.num_res_blocks,
                "num_channels": model.num_channels,
                "selfplay_iteration": iteration}, path)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export(source, destination):
    """Keep the familiar top-level filename without duplicating large files."""
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        # Filesystems without hard links can still run the course (extra space).
        shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


class RunFiles:
    """Store results; the loop only creates, configures and saves this object."""

    def __init__(self, directory, *, log_text=LOG_HEADER, keep_history=False):
        self.directory = Path(directory)
        self.log_text = log_text
        self.keep_history = keep_history

    @classmethod
    def create(cls, directory, *, keep_history=False):
        path = Path(directory)
        message = (f"Run folder '{path}' already contains files. Nothing was overwritten. "
                   "Choose a NEW folder, for example --checkpoint-dir runs/m6_smoke_02. "
                   "To continue a run, complete Module 6's resume exercise first.")
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(message)
        path.mkdir(parents=True, exist_ok=True)
        try:
            # Exclusive reservation prevents simultaneous new runs sharing a folder.
            with (path / ".run-started").open("x") as reservation:
                reservation.write("Reserved for one training run.\n")
        except FileExistsError:
            raise FileExistsError(message) from None
        (path / "run_log.csv").write_text(LOG_HEADER)
        return cls(path, keep_history=keep_history)

    def write_config(self, settings, model):
        settings = dict(settings)
        settings.update(res_blocks=model.num_res_blocks, channels=model.num_channels,
                        input_channels=model.input_channels,
                        device=str(next(model.parameters()).device),
                        checkpoint_dir=str(self.directory), keep_history=self.keep_history,
                        init_checkpoint=str(settings["init_checkpoint"]) if settings["init_checkpoint"] else None,
                        arena_simulations=settings["arena_simulations"] or settings["simulations"])
        config = {"settings": settings, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "python_version": platform.python_version(), "torch_version": str(torch.__version__)}
        with (self.directory / "config.json").open("x") as output:
            json.dump(config, output, indent=2)
            output.write("\n")

    def save(self, champion, buffer, iteration, row, *, candidate=None):
        """Commit state and a log row together, then discard older saves by default.

        Pending writes never replace the last complete save. Candidate weights
        are retained only with full history, since resume needs the champion.
        """
        snapshots = self.directory / "iterations"
        snapshots.mkdir(exist_ok=True)
        pending = Path(tempfile.mkdtemp(prefix="pending-", dir=snapshots))
        snapshot = pending.with_name(f"iteration-{iteration:06d}-{pending.name[8:]}")
        next_log = self.log_text + row
        save_checkpoint(champion, pending / "champion.pt", iteration)
        buffer.save(str(pending / "replay_buffer.npz"))
        (pending / "run_log.csv").write_text(next_log)
        names = list(_STATE_FILES)
        if self.keep_history and candidate is not None:
            save_checkpoint(candidate, pending / "candidate.pt", iteration)
            names.append("candidate.pt")
        os.replace(pending, snapshot)
        marker = {"iteration": iteration,
                  "snapshot": snapshot.relative_to(self.directory).as_posix(),
                  "config_sha256": _sha256(self.directory / "config.json"),
                  "sha256": {name: _sha256(snapshot / name) for name in names}}
        marker_path = self.directory / "completed.json"
        marker_path.with_suffix(".tmp").write_text(json.dumps(marker, indent=2) + "\n")
        os.replace(marker_path.with_suffix(".tmp"), marker_path)
        self.log_text = next_log
        for name in _STATE_FILES:
            _export(snapshot / name, self.directory / name)
        if not self.keep_history:
            # Only our generated directories are eligible for cleanup. Unknown
            # entries are left alone. Keep the new snapshot throughout cleanup.
            for old in snapshots.iterdir():
                if (old != snapshot and not old.is_symlink() and old.is_dir()
                        and old.name.startswith(("pending-", "iteration-"))):
                    shutil.rmtree(old)


@dataclass
class SavedRun:
    champion: object
    buffer: ReplayBuffer
    iteration: int
    settings: dict
    files: RunFiles

    def require_architecture(self, res_blocks=None, channels=None):
        """Omitted sizes inherit the model; explicit incompatible sizes fail."""
        if ((res_blocks is not None and res_blocks != self.champion.num_res_blocks)
                or (channels is not None and channels != self.champion.num_channels)):
            raise ValueError("Requested architecture (res_blocks/channels) differs from saved model")


def load_run(directory, device="cpu"):
    """Supplied, validated loader. Resume exercises need no file-format knowledge.

    Returns a champion, buffer, iteration, settings and writer carrying the
    committed log. Ignores incomplete saves; rejects missing or damaged state.
    """
    directory = Path(directory).resolve()
    try:
        marker = json.loads((directory / "completed.json").read_text())
        if _sha256(directory / "config.json") != marker["config_sha256"]:
            raise ValueError("Run configuration changed since the completed save")
        settings = json.loads((directory / "config.json").read_text())["settings"]
        snapshot = (directory / marker["snapshot"]).resolve()
        if snapshot.parent != directory / "iterations":
            raise ValueError("Invalid snapshot location")
        names = marker["sha256"]
        if not set(_STATE_FILES) <= set(names) or set(names) - set((*_STATE_FILES, "candidate.pt")):
            raise ValueError("Invalid snapshot file list")
        for name, expected in names.items():
            if _sha256(snapshot / name) != expected:
                raise ValueError(f"Incomplete or damaged save: {name}")
        champion, checkpoint = load_model(snapshot / "champion.pt", device=device)
        if (champion.num_res_blocks, champion.num_channels, champion.input_channels) != (
                settings["res_blocks"], settings["channels"], settings["input_channels"]):
            raise ValueError("Saved architecture disagrees with configuration")
        buffer = ReplayBuffer.load(snapshot / "replay_buffer.npz")
        iteration = marker["iteration"]
        log_text = (snapshot / "run_log.csv").read_text()
        rows = log_text.splitlines()
        if (not isinstance(iteration, int) or iteration < 1
                or checkpoint["selfplay_iteration"] != iteration
                or buffer.buffer.maxlen != settings["buffer_capacity"]
                or not log_text.startswith(LOG_HEADER)
                or [int(row.split(",")[0]) for row in rows[1:]] != list(range(1, iteration + 1))
                or int(rows[-1].split(",")[1]) != len(buffer)):
            raise ValueError("Saved state and log do not describe the same completed iteration")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("Invalid completed save or configuration") from error
    return SavedRun(champion, buffer, iteration, settings,
                    RunFiles(directory, log_text=log_text, keep_history=settings["keep_history"]))
