#!/usr/bin/env python3
"""Parse a macOS `powermetrics --samplers cpu_power,gpu_power` text log and
combine it with a qe-bench JSON result into one file.

This has not been validated against a real powermetrics log (no sudo
access in the environment that wrote it) — if `power.window_matched` comes
back false in the output, or the watt values look wrong, inspect the raw
log at the path in `power.raw_log` and adjust HEADER_RE / the power-line
regexes below to match what your macOS version actually prints. Run
`sudo powermetrics -i 500 --samplers cpu_power,gpu_power -n 3` by hand to
see the real format if unsure.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime

# Header line powermetrics prints before each sample block. Real format
# (confirmed 2026-09-06, macOS 26A5425a) carries a SECOND parenthetical
# group after the timestamp that the original non-greedy `(.+?)\) \*\*\*`
# pattern didn't account for:
#   *** Sampled system activity (Sun Sep  6 07:09:46 2026 +1000) (208.10ms elapsed) ***
# Matching the timestamp shape directly, rather than "whatever's between
# the first ( and the last ) ***", avoids depending on that second group
# being present/absent/formatted consistently across macOS versions.
HEADER_RE = re.compile(
    r"\*\*\* Sampled system activity \(([A-Za-z]{3} [A-Za-z]{3}\s+\d{1,2} \d{2}:\d{2}:\d{2} \d{4} [+-]\d{4})\)"
)
GPU_POWER_RE = re.compile(r"^GPU Power:\s*([\d.]+)\s*mW", re.MULTILINE)
CPU_POWER_RE = re.compile(r"^CPU Power:\s*([\d.]+)\s*mW", re.MULTILINE)
COMBINED_POWER_RE = re.compile(r"^Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW", re.MULTILINE)


def parse_header_timestamp(raw: str) -> float | None:
    # powermetrics header format varies by macOS version/locale; try a few.
    formats = [
        "%a %b %d %H:%M:%S %Y %z",
        "%a %b  %d %H:%M:%S %Y %z",
        "%Y-%m-%d %H:%M:%S %z",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(raw.strip(), fmt).timestamp()
        except ValueError:
            continue
    return None


def parse_blocks(log_text: str) -> list[dict]:
    """Split the log into per-sample blocks and extract power + timestamp."""
    headers = list(HEADER_RE.finditer(log_text))
    blocks = []
    for i, h in enumerate(headers):
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(log_text)
        block_text = log_text[start:end]
        ts = parse_header_timestamp(h.group(1))
        gpu = GPU_POWER_RE.search(block_text)
        cpu = CPU_POWER_RE.search(block_text)
        combined = COMBINED_POWER_RE.search(block_text)
        blocks.append({
            "timestamp": ts,
            "gpu_mw": float(gpu.group(1)) if gpu else None,
            "cpu_mw": float(cpu.group(1)) if cpu else None,
            "combined_mw": float(combined.group(1)) if combined else None,
        })
    return blocks


def summarize(values: list[float]) -> dict:
    if not values:
        return {"mean_mw": None, "max_mw": None, "min_mw": None, "n_samples": 0}
    return {
        "mean_mw": statistics.mean(values),
        "max_mw": max(values),
        "min_mw": min(values),
        "n_samples": len(values),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power-log", required=True)
    ap.add_argument("--bench-log", required=True)
    ap.add_argument("--start-epoch", type=float, required=True)
    ap.add_argument("--end-epoch", type=float, required=True)
    ap.add_argument("--platform", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.power_log) as f:
        log_text = f.read()
    with open(args.bench_log) as f:
        bench_result = json.load(f)

    blocks = parse_blocks(log_text)
    timestamped = [b for b in blocks if b["timestamp"] is not None]
    window_matched = len(timestamped) > 0

    if window_matched:
        # Small padding since block timestamps mark sample start, not the
        # exact moment load began/ended.
        in_window = [b for b in timestamped if args.start_epoch - 1 <= b["timestamp"] <= args.end_epoch + 1]
        used = in_window if in_window else timestamped
        gpu_values = [b["gpu_mw"] for b in used if b["gpu_mw"] is not None]
        cpu_values = [b["cpu_mw"] for b in used if b["cpu_mw"] is not None]
        combined_values = [b["combined_mw"] for b in used if b["combined_mw"] is not None]
    else:
        # Fallback: no parseable header/timestamp structure at all (covers
        # both "headers present but timestamp format didn't match" and "no
        # header lines in this log at all") — pull every power value out
        # of the raw text directly, independent of block structure. Only
        # correct if the log genuinely covers just the benchmark window
        # plus its padding; not a precise load-only window either way.
        gpu_values = [float(v) for v in GPU_POWER_RE.findall(log_text)]
        cpu_values = [float(v) for v in CPU_POWER_RE.findall(log_text)]
        combined_values = [float(v) for v in COMBINED_POWER_RE.findall(log_text)]

    power_summary = {
        "window_matched": window_matched,
        "n_blocks_total": len(blocks),
        "n_blocks_used": len(used) if window_matched else max(len(gpu_values), len(cpu_values), len(combined_values)),
        "gpu_power": summarize(gpu_values),
        "cpu_power": summarize(cpu_values),
        "combined_power": summarize(combined_values),
        "raw_log": args.power_log,
        "caveat": (
            "window_matched=false means header timestamps could not be parsed; "
            "stats fall back to every sample in the log (includes ~2s idle "
            "padding before/after the benchmark, not a precise load-only window)."
            if not window_matched else
            "Stats computed only from samples whose timestamp fell within the "
            "benchmark's start/end epoch (+/-1s padding)."
        ),
    }

    combined = {
        "platform": args.platform,
        "benchmark": bench_result,
        "power": power_summary,
    }

    with open(args.out, "w") as f:
        json.dump(combined, f, indent=2)

    print(json.dumps(power_summary, indent=2))


if __name__ == "__main__":
    main()
