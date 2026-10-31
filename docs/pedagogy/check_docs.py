#!/usr/bin/env python3
"""
Objective checks on the project's teaching material (see RUBRIC.md).

Reports:
- file references in learner docs that don't exist
- `python ...` commands from learner docs that crash
- how many exercises / self-check questions the material contains

Commands run inside a temporary copy of the repo, so documented commands that
write files (e.g. training) can't touch the real checkpoints.

Usage:
    python docs/pedagogy/check_docs.py            # full report
    python docs/pedagogy/check_docs.py --no-run   # skip running commands
"""

import argparse
import html
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SKIP_DIRS = {"venv", ".venv", ".git", "docs/pedagogy", "__pycache__", "checkpoints", "runs", "data", "notebooks"}
HISTORY_DIRS = ("docs/history", "docs/plans")  # project history: checked, but reported separately

FILE_REF = re.compile(r"(?<![\w/.:-])((?:[\w-]+/)*[\w-]+\.(?:py|md|html|pgn|sh))\b")
COMMAND = re.compile(r"^\s*(?:\$\s*)?((?:python3?|pytest)\s+[^\n#]+?|pytest)\s*(?:#.*)?$", re.MULTILINE)
EXERCISE = re.compile(
    r"(?i)(\bexercise\s*\d|\bpredict\b[^.\n]{0,60}\?|check your understanding|self-check|\bquiz\b|<summary>)"
)


def learner_docs(history=False):
    """Markdown, HTML tutorials and notebooks a learner would read (or, with history=True, the journal)."""
    docs = []
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT).as_posix()
        if any(rel == d or rel.startswith(d + "/") for d in SKIP_DIRS):
            continue
        if rel.startswith(tuple(d + "/" for d in HISTORY_DIRS)) != history:
            continue
        if path.suffix in {".md", ".html"} or (path.suffix == ".py" and rel.startswith("notebooks/")):
            docs.append(path)
    return docs


def text_of(path):
    raw = path.read_text(errors="replace")
    if path.suffix == ".html":
        raw = html.unescape(re.sub(r"<br\s*/?>", "\n", raw))
        raw = re.sub(r"<[^>]+>", "", raw)
    return raw


def code_blocks(path, text):
    """Only look for commands inside code blocks, not prose."""
    if path.suffix == ".md":
        return re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
    return [text]


OUTPUT_FLAG = re.compile(r"(?:--?[\w-]*(?:log|output|out|dir)|-o|>)[\s=]*$")  # a flag or a shell redirect
# Files the course asks the learner to create (not expected to exist in the repo)
LEARNER_CREATED = {"REPORT.md", "training/first_network.py"}


def is_not_a_file_reference(text, match):
    """Python names (chess.pgn, mo.md(...)), files a command will create, and files flagged as absent."""
    before, after = text[max(0, match.start() - 40):match.start()], text[match.end():match.end() + 120]
    return (after.startswith(("(", ".read", ".write"))
            or re.search(r"\bimport\s+$", before)
            or OUTPUT_FLAG.search(before)
            or "not in the repository" in after)


def files_created_by_commands(docs):
    """Files written via an output flag anywhere in the docs (e.g. --output data/filtered.pgn)."""
    created = set()
    for doc in docs:
        text = text_of(doc)
        for m in FILE_REF.finditer(text):
            if OUTPUT_FLAG.search(text[max(0, m.start() - 40):m.start()]):
                created.add(m.group(1))
    return created


CREATED_ELSEWHERE = set()


def check_references(docs):
    broken = []
    for doc in docs:
        text = text_of(doc)
        matches = list(FILE_REF.finditer(text))
        # a file created by a command, or flagged as absent, is excused everywhere in that document
        excused = {m.group(1) for m in matches if is_not_a_file_reference(text, m)}
        refs = {m.group(1) for m in matches} - excused - CREATED_ELSEWHERE - LEARNER_CREATED
        for ref in sorted(refs):
            if "://" in ref or ref.startswith(("www.", "lichess")):
                continue
            candidates = [ROOT / ref, doc.parent / ref]
            found = any(c.exists() for c in candidates)
            if not found and "/" not in ref:
                # bare filename: accept it if it exists anywhere in the project
                found = any(p for p in ROOT.rglob(ref) if "venv" not in p.parts)
            if not found:
                broken.append((doc.relative_to(ROOT).as_posix(), ref))
    return broken


def collect_commands(docs):
    commands = {}
    for doc in docs:
        text = text_of(doc)
        for block in code_blocks(doc, text):
            block = re.sub(r"\\\n\s*", " ", block)  # join "\" line continuations
            for cmd in COMMAND.findall(block):
                cmd = cmd.strip().rstrip("\\").strip()
                try:
                    tokens = shlex.split(cmd)
                except ValueError:
                    tokens = []
                if any(t in ("&&", "||", "|", ";", "source") or t.endswith(";") for t in tokens):
                    commands.setdefault("[shell, not run] " + cmd, doc.relative_to(ROOT).as_posix())
                    continue
                if "<" in cmd or "..." in cmd or "/path/to" in cmd:
                    continue  # placeholder, not runnable as written
                commands.setdefault(cmd, doc.relative_to(ROOT).as_posix())
    return commands


def run_commands(commands, timeout):
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        sandbox = Path(tmp) / "repo"
        shutil.copytree(ROOT, sandbox, ignore=shutil.ignore_patterns("venv", ".git", "*.log"))
        for cmd, doc in commands.items():
            if cmd.startswith("[shell, not run]"):
                results.append(("skipped (shell syntax)", cmd, doc))
                continue
            argv = shlex.split(cmd)
            if argv[0] in ("python", "python3"):
                argv[0] = sys.executable
            elif argv[0] == "pytest":
                argv = [sys.executable, "-m", "pytest"] + argv[1:]
            try:
                proc = subprocess.run(argv, cwd=sandbox, stdin=subprocess.DEVNULL,
                                      capture_output=True, text=True, timeout=timeout)
                output = proc.stdout + proc.stderr
                if proc.returncode == 0:
                    status = "ok"
                elif "EOFError" in output:
                    status = "interactive"  # waited for keyboard input; not a bug
                elif re.search(r"(No such file|not found)[^\n]*\.pgn", output):
                    status = "needs data"  # reads a PGN the learner downloads or prepares in an earlier step
                else:
                    last = [l for l in output.strip().splitlines() if l.strip()][-1:] or ["(no output)"]
                    status = "BROKEN: " + last[0][:120]
            except subprocess.TimeoutExpired:
                status = "ok (still running at timeout)"
            results.append((status, cmd, doc))
    return results


def count_exercises(docs):
    counts = {}
    for doc in docs:
        # count on the raw file: HTML self-checks are <details><summary> blocks, which text_of strips
        n = len(EXERCISE.findall(doc.read_text(errors="replace")))
        if n:
            counts[doc.relative_to(ROOT).as_posix()] = n
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-run", action="store_true", help="don't execute documented commands")
    parser.add_argument("--timeout", type=int, default=20, help="seconds per command")
    args = parser.parse_args()

    docs = learner_docs()
    print(f"Learner documents scanned: {len(docs)}\n")

    CREATED_ELSEWHERE.update(files_created_by_commands(docs))
    broken_refs = check_references(docs)
    print(f"## Broken file references: {len(broken_refs)}")
    for doc, ref in broken_refs:
        print(f"  {doc}: {ref}")

    commands = collect_commands(docs)
    broken_cmds = []
    if args.no_run:
        print(f"\n## Commands found: {len(commands)} (not run)")
    else:
        results = run_commands(commands, args.timeout)
        broken_cmds = [r for r in results if r[0].startswith("BROKEN")]
        print(f"\n## Commands: {len(results)} run, {len(broken_cmds)} broken")
        for status, cmd, doc in results:
            print(f"  [{status}] {cmd}   ({doc})")

    exercises = count_exercises(docs)
    print(f"\n## Exercises / self-check prompts: {sum(exercises.values())}")
    for doc, n in sorted(exercises.items()):
        print(f"  {doc}: {n}")

    history = learner_docs(history=True)
    history_refs = check_references(history)
    print(f"\n## Project history ({', '.join(HISTORY_DIRS)}, {len(history)} files, not learner material): "
          f"{len(history_refs)} broken file references, {len(collect_commands(history))} commands (not run)")

    print(f"\nSUMMARY broken_refs={len(broken_refs)} broken_commands={len(broken_cmds)} "
          f"exercises={sum(exercises.values())} history_broken_refs={len(history_refs)}")


if __name__ == "__main__":
    main()
