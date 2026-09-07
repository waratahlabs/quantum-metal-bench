#!/usr/bin/env python3
"""Parse the CSV produced by benchmark_with_power_cuda.sh and combine it
with the CLI's JSON benchmark output into one file."""

from __future__ import annotations

import argparse
import csv
import json
import statistics


def summarize(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "max": None, "min": None, "n_samples": 0}
    return {
        "mean": statistics.mean(values),
        "max": max(values),
        "min": min(values),
        "n_samples": len(values),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power-csv", required=True)
    ap.add_argument("--bench-log", required=True)
    ap.add_argument("--start-epoch", type=float, required=True)
    ap.add_argument("--end-epoch", type=float, required=True)
    ap.add_argument("--platform", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    with open(args.power_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "timestamp": float(row["timestamp_epoch"]),
                    "power_w": float(row["power_draw_w"]),
                    "util_pct": float(row["utilization_gpu_pct"]),
                    "clock_mhz": float(row["clocks_sm_mhz"]),
                })
            except (ValueError, KeyError):
                continue  # skip malformed rows (e.g. a failed nvidia-smi call)

    in_window = [r for r in rows if args.start_epoch - 0.5 <= r["timestamp"] <= args.end_epoch + 0.5]
    used = in_window if in_window else rows

    with open(args.bench_log) as f:
        bench_result = json.load(f)

    # Physically-impossible-clock sanity check: flag rather than silently
    # trust, since this is exactly what F12 found on the WSL2 rig.
    IMPLAUSIBLE_CLOCK_MHZ = 3500  # generous ceiling above any current consumer GPU boost clock
    implausible_clocks = [r["clock_mhz"] for r in used if r["clock_mhz"] > IMPLAUSIBLE_CLOCK_MHZ]

    power_summary = {
        "n_samples_total": len(rows),
        "n_samples_used": len(used),
        "window_matched": len(in_window) > 0,
        "power_w": summarize([r["power_w"] for r in used]),
        "utilization_pct": summarize([r["util_pct"] for r in used]),
        "clock_mhz": summarize([r["clock_mhz"] for r in used]),
        "implausible_clock_readings": len(implausible_clocks),
        "telemetry_suspect": len(implausible_clocks) > 0,
        "raw_csv": args.power_csv,
    }
    if power_summary["telemetry_suspect"]:
        power_summary["warning"] = (
            f"{len(implausible_clocks)} sample(s) reported clocks.sm above "
            f"{IMPLAUSIBLE_CLOCK_MHZ} MHz, which is not physically possible for "
            "any current consumer GPU -- do not trust the power numbers in this file."
        )

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
