import time

import pytest

from quantum_metal_bench.timing import time_repeated


def test_reports_mean_stddev_over_multiple_reps_not_a_single_shot():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1

    result = time_repeated(fn, n_reps=5)
    assert result.n_reps == 5
    assert len(result.samples) == 5
    # +1 for the discarded warmup call
    assert calls["n"] == 6


def test_high_variance_flagged_when_stddev_exceeds_threshold():
    call_count = {"n": 0}

    def variable_latency():
        call_count["n"] += 1
        # Alternate fast/slow to force stddev/mean > 10%
        time.sleep(0.001 if call_count["n"] % 2 == 0 else 0.02)

    result = time_repeated(variable_latency, n_reps=6)
    assert result.high_variance is True


def test_rejects_zero_reps():
    with pytest.raises(ValueError):
        time_repeated(lambda: None, n_reps=0)
