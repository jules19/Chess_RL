#!/usr/bin/env python3
"""
Mechanical plain-English checks on the learner docs (see PLAIN_ENGLISH.md).

For each file it reports, outside code blocks and inline code:
- dashes used as punctuation (em dashes), per 1,000 words
- flagged phrases: idioms and figures of speech, casual or hype words, and
  phrases typical of AI-generated text, with line numbers
- average sentence length, and the share of sentences over 30 words
- exclamation marks

It can't judge whether a technical term is explained well; the graders do
that with the rubric.

Usage:
    python docs/pedagogy/plain_check.py            # summary table and every flag
    python docs/pedagogy/plain_check.py --summary  # table only
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

FILES = [
    "README.md",
    "QUICKSTART.md",
    "STATUS.md",
    "course/README.md",
    "course/RL_CONCEPTS.md",
    *sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "course").glob("module-*/README.md")),
    "tests/course/README.md",
    "docs/DOCKER_GUI.md",
    "docs/UCI_SETUP_MAC.md",
    "docs/UCI_LOGGING_README.md",
]

# Each entry: (category, regex). Matched case-insensitively against prose only.
FLAGS = [
    # Idioms, metaphors and other figurative language
    ("figurative", r"\bhilarious(ly)?\b"),
    ("figurative", r"\bdemolish\w*"),
    ("figurative", r"\bvibes?\b"),
    ("figurative", r"\bdark tunnels?\b"),
    ("figurative", r"\bpile of\b"),
    ("figurative", r"\bbolted\b"),
    ("figurative", r"\bknobs?\b"),
    ("figurative", r"\brungs?\b"),
    ("figurative", r"\bpay-?off\b"),
    ("figurative", r"\bpunchline\b"),
    ("figurative", r"\bmagic(al)?\b"),
    ("figurative", r"\bunder the hood\b"),
    ("figurative", r"\bthe rope\b"),
    ("figurative", r"\bslippery\b"),
    ("figurative", r"\bpins? (it|this|that|them) down\b"),
    ("figurative", r"\bsecret sauce\b"),
    ("figurative", r"\bboils? down\b"),
    ("figurative", r"\bin a nutshell\b"),
    ("figurative", r"\bsweet spot\b"),
    ("figurative", r"\bbaked in\b"),
    ("figurative", r"\bfor free\b"),
    ("figurative", r"\bsuperpower\w*"),
    ("figurative", r"\bheavy lifting\b"),
    ("figurative", r"\bmoving parts\b"),
    ("figurative", r"\bbells and whistles\b"),
    ("figurative", r"\bgold-?plat\w*"),
    ("figurative", r"\bscrutiny\b"),
    ("figurative", r"\bholes?\b(?! in)"),
    ("figurative", r"\bthread\b"),
    ("figurative", r"\bblind spots?\b"),
    ("figurative", r"\bnorth star\b"),
    ("figurative", r"\bsilver bullet\b"),
    ("figurative", r"\bout of the box\b"),
    ("figurative", r"\bon steroids\b"),
    ("figurative", r"\blights? up\b"),
    ("figurative", r"\bwhack-a-mole\b"),
    ("figurative", r"\brabbit holes?\b"),
    ("figurative", r"\bsmoke test\b"),
    ("figurative", r"\bfoot-?guns?\b"),
    ("figurative", r"\bbites?\b(?! of)"),
    ("figurative", r"\bflavou?rs?\b"),
    ("figurative", r"\bsanity\b"),
    # Casual, hype or slangy wording
    ("casual", r"\btoy\b"),
    ("casual", r"\bdumb\b"),
    ("casual", r"\bcrazy\b"),
    ("casual", r"\bawesome\b"),
    ("casual", r"\bkiller\b(?! move| heuristic| table| slot)"),
    ("casual", r"\bnail(ed|s)?\b"),
    ("casual", r"\btrick(s|y)?\b"),
    ("casual", r"\bspoiler\b"),
    ("casual", r"\bgame-?changer\b"),
    ("casual", r"\bgotcha\w*"),
    ("casual", r"#1\b"),
    ("casual", r"\bstuff\b"),
    ("casual", r"\bbasically\b"),
    ("casual", r"\bsuper\b(?!vis)"),
    # Phrases typical of AI-generated text
    ("ai-style", r"\bdelve\w*"),
    ("ai-style", r"\bit'?s worth noting\b"),
    ("ai-style", r"\bcrucially\b"),
    ("ai-style", r"\bthe key (insight|idea|takeaway) (is|here)\b"),
    ("ai-style", r"\bhere'?s the thing\b"),
    ("ai-style", r"\bat its (heart|core)\b"),
    ("ai-style", r"\bjourney\b"),
    ("ai-style", r"\bunlock\w*"),
    ("ai-style", r"\bleverag\w*"),
    ("ai-style", r"\blandscape\b"),
    ("ai-style", r"\bseamless\w*"),
    ("ai-style", r"\brobust\w*"),
    ("ai-style", r"\bempower\w*"),
    ("ai-style", r"\bhonest(ly)?\b"),
    ("ai-style", r"\bgenuine(ly)?\b"),
    ("ai-style", r"\bquietly\b"),
    ("ai-style", r"\bsilently\b"),
    ("ai-style", r"\bdeliberately\b"),
    ("ai-style", r"\bnot (just|merely|only) [^.\n]{0,60}[,;]\s*(but|it'?s)\b"),
    ("ai-style", r"\bisn'?t [^.\n]{2,40}[;,] it'?s\b"),
    ("ai-style", r"\bthat'?s the (point|whole point|lesson)\b"),
]
FLAG_RE = [(cat, re.compile(rx, re.IGNORECASE)) for cat, rx in FLAGS]


def prose_lines(text):
    """Yield (line number, prose) with fenced code blocks, inline code, links' targets and HTML removed."""
    in_fence = False
    for n, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        line = re.sub(r"`[^`]*`", " CODE ", line)
        line = re.sub(r"\]\([^)]*\)", "]", line)
        line = re.sub(r"https?://\S+", " URL ", line)
        line = re.sub(r"<[^>]+>", " ", line)
        yield n, line


def sentences(prose):
    """Split prose into sentences. Table rows, headings and list items count as their own units."""
    units = []
    para = []
    for line in prose.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("|") or re.match(r"^([-*>]|\d+\.)\s", s):
            if para:
                units.append(" ".join(para))
                para = []
            if s and not s.startswith("|") and not s.startswith("#"):
                para = [re.sub(r"^([-*>]|\d+\.)\s+", "", s)]
            continue
        para.append(s)
    if para:
        units.append(" ".join(para))
    out = []
    for u in units:
        u = re.sub(r"\b(e\.g|i\.e|etc|vs|approx|cf)\.", r"\1", u)
        out.extend(p for p in re.split(r"(?<=[.?!:])\s+(?=[A-Z*(])", u) if len(p.split()) >= 3)
    return out


def check(path):
    text = (ROOT / path).read_text()
    lines = list(prose_lines(text))
    prose = "\n".join(l for _, l in lines)
    words = len(re.findall(r"\b\w+\b", prose))
    flags = []
    for n, line in lines:
        for cat, rx in FLAG_RE:
            for m in rx.finditer(line):
                flags.append((n, cat, m.group(0)))
    sents = sentences(prose)
    lengths = [len(s.split()) for s in sents]
    return {
        "path": path,
        "words": words,
        "dashes": prose.count("—") + len(re.findall(r"\s--\s", prose)),
        "exclaim": len(re.findall(r"\w!(\s|$)", prose)),
        "avg_sentence": sum(lengths) / len(lengths) if lengths else 0,
        "long_share": sum(l > 30 for l in lengths) / len(lengths) if lengths else 0,
        "flags": flags,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", action="store_true", help="print the table only")
    args = parser.parse_args()

    results = [check(p) for p in FILES if (ROOT / p).exists()]
    print(f"{'file':44} {'words':>6} {'dash/1k':>8} {'flags':>6} {'avg sent':>9} {'>30w':>6} {'!':>3}")
    tot_w = tot_d = tot_f = 0
    for r in results:
        per_k = 1000 * r["dashes"] / r["words"] if r["words"] else 0
        print(
            f"{r['path']:44} {r['words']:6} {per_k:8.1f} {len(r['flags']):6} "
            f"{r['avg_sentence']:9.1f} {100 * r['long_share']:5.0f}% {r['exclaim']:3}"
        )
        tot_w += r["words"]
        tot_d += r["dashes"]
        tot_f += len(r["flags"])
    print(f"{'TOTAL':44} {tot_w:6} {1000 * tot_d / tot_w:8.1f} {tot_f:6}")

    if not args.summary:
        for r in results:
            for n, cat, word in r["flags"]:
                print(f"{r['path']}:{n}: {cat}: {word}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
