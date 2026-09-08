import json
import subprocess
from unittest.mock import MagicMock, patch

import numpy as np

from quantum_metal_bench.backends import run_metal_backend


def _fake_completed_process(payload: dict, returncode: int = 0) -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = json.dumps(payload)
    proc.stderr = ""
    return proc


def test_unsupported_circuit_type_is_visibly_skipped_not_crashed():
    ground_truth = np.array([1.0, 0.0], dtype=np.complex128)
    result = run_metal_backend("clifford_t", 4, 2, 5, 0, ground_truth)
    assert result.status == "SKIPPED_CIRCUIT_TYPE"
    assert result.skip_reason is not None


@patch("quantum_metal_bench.backends.metal_available", return_value=(False, "not on macOS"))
def test_metal_unavailable_reports_visible_skip_not_silent_omission(_mock_avail):
    ground_truth = np.array([1.0, 0.0], dtype=np.complex128)
    result = run_metal_backend("qaoa", 4, 2, 5, 0, ground_truth)
    assert result.status == "METAL_LEG_SKIPPED"
    assert result.skip_reason == "not on macOS"


@patch("quantum_metal_bench.backends.metal_available", return_value=(True, None))
@patch("quantum_metal_bench.backends.subprocess.run")
def test_successful_run_computes_fidelity_against_ground_truth(mock_run, _mock_avail):
    # 2-qubit ground truth: uniform superposition
    ground_truth = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.complex128)
    payload = {
        "n_reps": 5,
        "execution_time_sec_mean": 0.01,
        "execution_time_sec_stddev": 0.0005,
        "gate_count": 20,
        "statevector_real": [0.5, 0.5, 0.5, 0.5],
        "statevector_imag": [0.0, 0.0, 0.0, 0.0],
    }
    mock_run.return_value = _fake_completed_process(payload)

    result = run_metal_backend("qaoa", 2, 2, 5, 0, ground_truth)
    assert result.status == "SUCCESS"
    assert result.fidelity is not None
    assert result.fidelity > 0.999
    assert result.n_reps == 5


@patch("quantum_metal_bench.backends.metal_available", return_value=(True, None))
@patch("quantum_metal_bench.backends.subprocess.run")
def test_nonzero_exit_reports_failed_other_with_stderr_not_silent(mock_run, _mock_avail):
    mock_run.return_value = _fake_completed_process({}, returncode=1)
    mock_run.return_value.stderr = "qe-bench bench: circuit 'x' not supported\n"

    ground_truth = np.array([1.0, 0.0], dtype=np.complex128)
    result = run_metal_backend("qaoa", 2, 2, 5, 0, ground_truth)
    assert result.status == "FAILED_OTHER"
    assert "not supported" in result.skip_reason


@patch("quantum_metal_bench.backends.metal_available", return_value=(True, None))
@patch("quantum_metal_bench.backends.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="qe-bench", timeout=300))
def test_timeout_reports_unconfirmed_not_silent_success(_mock_run, _mock_avail):
    ground_truth = np.array([1.0, 0.0], dtype=np.complex128)
    result = run_metal_backend("qaoa", 2, 2, 5, 0, ground_truth)
    assert result.status == "TIMEOUT_UNCONFIRMED_OOM"


@patch("quantum_metal_bench.backends.metal_available", return_value=(True, None))
@patch("quantum_metal_bench.backends.subprocess.run")
def test_large_n_skips_fidelity_and_never_touches_statevector_fields(mock_run, _mock_avail):
    # Regression test for the real n=28 OOM: qe-bench signals
    # statevector_omitted at large n instead of emitting the full arrays.
    # The Python side must accept this without requiring (or touching)
    # statevector_real/statevector_imag, and must work with ground_truth=None
    # since the caller has no reason to compute one at this scale either.
    payload = {
        "n_reps": 5,
        "execution_time_sec_mean": 9.38,
        "execution_time_sec_stddev": 0.31,
        "gate_count": 476,
        "statevector_omitted": True,
        "statevector_omit_reason": "n_qubits > 16: full statevector JSON dump is O(2**n) memory and OOMs at scale",
    }
    mock_run.return_value = _fake_completed_process(payload)

    result = run_metal_backend("qaoa", 28, 4, 5, 0, None)
    assert result.status == "SUCCESS"
    assert result.accuracy_status == "SKIPPED_FIDELITY_LARGE_N"
    assert result.fidelity is None
    assert result.fidelity_threshold_used is None
    assert result.execution_time_sec_mean == 9.38
    assert result.n_reps == 5
