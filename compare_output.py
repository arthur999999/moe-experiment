#!/usr/bin/env python3
"""
Runs the model with the fixed Phase 1 baseline config, cleans the output
the same way reference_output.txt was cleaned, and diffs against it.

Usage:
    python compare_output.py                  # uses default binary/model
    python compare_output.py --extra-args --some-flag  # pass extra flags to llama-cli
"""

import argparse
import re
import subprocess
import sys
import difflib

LLAMA_CLI = "./llama.cpp/build/bin/llama-cli"
MODEL_PATH = "./models/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"
PROMPT = "Explain how mixture of experts routing works."
REFERENCE_FILE = "reference_output.txt"

BASE_ARGS = [
    "-m", MODEL_PATH,
    "-p", PROMPT,
    "-n", "128",
    "-t", "4",
    "--temp", "0",
    "--top-k", "1",
    "--seed", "42",
    "-no-cnv",
    "-st",
    "--no-repack",
    "--simple-io",
    "--log-disable",
]

ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m')


def run_model(extra_args=None):
    cmd = [LLAMA_CLI] + BASE_ARGS
    if extra_args:
        cmd += extra_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"WARNING: llama-cli exited with code {result.returncode}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
    return result.stdout


def clean_output(raw: str) -> str:
    """Mirrors the sed/awk pipeline used to produce reference_output.txt:
    strip ANSI codes, keep only lines between the echoed prompt and the
    stats line, collapse repeated blank lines.
    """
    text = ANSI_ESCAPE.sub('', raw)
    lines = text.splitlines()

    collecting = False
    kept = []
    for line in lines:
        if line.startswith(f"> {PROMPT}"):
            collecting = True
            continue
        if line.startswith("[ Prompt:"):
            collecting = False
            continue
        if collecting:
            kept.append(line)

    # collapse consecutive blank lines
    cleaned = []
    prev_blank = False
    for line in kept:
        is_blank = (line.strip() == "")
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank

    return "\n".join(cleaned).strip() + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    try:
        with open(REFERENCE_FILE) as f:
            reference = f.read()
    except FileNotFoundError:
        print(f"ERROR: {REFERENCE_FILE} not found. Run Phase 1 setup first.", file=sys.stderr)
        sys.exit(1)

    raw_output = run_model(args.extra_args)
    current = clean_output(raw_output)

    if current == reference:
        print("MATCH — output identical to reference")
        sys.exit(0)
    else:
        print("MISMATCH — diff (reference vs current):")
        diff = difflib.unified_diff(
            reference.splitlines(keepends=True),
            current.splitlines(keepends=True),
            fromfile="reference",
            tofile="current",
        )
        sys.stdout.writelines(diff)
        sys.exit(1)


if __name__ == "__main__":
    main()
