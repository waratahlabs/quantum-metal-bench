from unittest.mock import MagicMock, patch

import numpy as np

from quantum_metal_bench.backends import run_cuda_backend


def _dummy_circuit_fn():
    return None  # never called when device construction is mocked to raise/return


def test_unsupported_circuit_type_is_visibly_skipped():
    ground_truth = np.array([1.0, 0.0], dtype=np.complex128)
    result = run_cuda_backend("something_else", 4, 2, 5, 0, _dummy_circuit_fn, gate_count=10, ground_truth=ground_truth)
    assert result.status == "SKIPPED_CIRCUIT_TYPE"


@patch("quantum_metal_bench.backends.cuda_available", return_value=(False, "not installed"))
def test_plugin_unavailable_reports_visible_skip_not_silent_omission(_mock_avail):
    ground_truth = np.array([1.0, 0.0], dtype=np.complex128)
    result = run_cuda_backend("qaoa", 4, 2, 5, 0, _dummy_circuit_fn, gate_count=10, ground_truth=ground_truth)
    assert result.status == "NVIDIA_LEG_SKIPPED"
    assert result.skip_reason == "not installed"


@patch("quantum_metal_bench.backends.cuda_available", return_value=(True, None))
def test_device_construction_failure_degrades_to_failed_other_not_a_crash(_mock_avail):
    # Simulate a driver mismatch / CUDA init failure inside qml.device() —
    # must not propagate and kill the whole sweep.
    import quantum_metal_bench.backends as backends_mod

    def raising_circuit_fn():
        raise RuntimeError("CUDA driver version mismatch")

    ground_truth = np.array([1.0, 0.0], dtype=np.complex128)
    with patch.dict("sys.modules", {"pennylane": MagicMock()}):
        import pennylane as qml  # the mocked module

        qml.device.side_effect = RuntimeError("CUDA driver version mismatch")
        result = run_cuda_backend("qaoa", 2, 2, 5, 0, raising_circuit_fn, gate_count=10, ground_truth=ground_truth)

    assert result.status == "FAILED_OTHER"
    assert "CUDA driver version mismatch" in result.skip_reason


@patch("quantum_metal_bench.backends.cuda_available", return_value=(True, None))
def test_successful_run_computes_fidelity_against_ground_truth(_mock_avail):
    ground_truth = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.complex128)

    fake_qnode = MagicMock(return_value=np.array([0.5, 0.5, 0.5, 0.5], dtype=np.complex128))

    with patch.dict("sys.modules", {"pennylane": MagicMock()}):
        import pennylane as qml  # the mocked module

        qml.device.return_value = MagicMock()
        qml.QNode.return_value = fake_qnode

        result = run_cuda_backend(
            "qaoa", 2, 2, 5, 0, lambda: None, gate_count=20, ground_truth=ground_truth
        )

    assert result.status == "SUCCESS"
    assert result.fidelity is not None
    assert result.fidelity > 0.999
    assert result.n_reps == 5


@patch("quantum_metal_bench.backends.cuda_available", return_value=(True, None))
def test_large_n_skips_fidelity_without_ground_truth(_mock_avail):
    # Symmetric with the Metal regression test: above
    # MAX_QUBITS_FOR_FIDELITY_CHECK, the caller has no reason to compute a
    # CPU ground truth either, so this must work with ground_truth=None.
    fake_state = np.zeros(2**20, dtype=np.complex128)
    fake_qnode = MagicMock(return_value=fake_state)

    with patch.dict("sys.modules", {"pennylane": MagicMock()}):
        import pennylane as qml  # the mocked module

        qml.device.return_value = MagicMock()
        qml.QNode.return_value = fake_qnode

        result = run_cuda_backend(
            "qaoa", 20, 4, 5, 0, lambda: None, gate_count=200, ground_truth=None
        )

    assert result.status == "SUCCESS"
    assert result.accuracy_status == "SKIPPED_FIDELITY_LARGE_N"
    assert result.fidelity is None
    assert result.fidelity_threshold_used is None
