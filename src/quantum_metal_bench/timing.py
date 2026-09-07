"""N-repetition timing harness (PRD §3.B). Never reports a single-shot as a mean."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

HIGH_VARIANCE_THRESHOLD = 0.10  # stddev/mean ratio


@dataclass
class TimingResult:
    n_reps: int
    execution_time_sec_mean: float
    execution_time_sec_stddev: float
    execution_time_sec_min: float
    execution_time_sec_max: float
    high_variance: bool
    samples: list[float] = field(default_factory=list)


def time_repeated(fn: Callable[[], None], n_reps: int = 5, warmup: bool = True) -> TimingResult:
    """Run fn() n_reps times (after one discarded warmup call) using perf_counter_ns.

    n_reps must be >= 1; the caller is responsible for enforcing PRD's N>=5
    floor for anything that will be published — this function will not
    silently pad a smaller n_reps to look like a proper sample.
    """
    if n_reps < 1:
        raise ValueError("n_reps must be >= 1")

    if warmup:
        fn()  # discarded

    samples: list[float] = []
    for _ in range(n_reps):
        start = time.perf_counter_ns()
        fn()
        end = time.perf_counter_ns()
        samples.append((end - start) / 1e9)

    mean = statistics.mean(samples)
    stddev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    high_variance = (stddev / mean) > HIGH_VARIANCE_THRESHOLD if mean > 0 else False

    return TimingResult(
        n_reps=n_reps,
        execution_time_sec_mean=mean,
        execution_time_sec_stddev=stddev,
        execution_time_sec_min=min(samples),
        execution_time_sec_max=max(samples),
        high_variance=high_variance,
        samples=samples,
    )
