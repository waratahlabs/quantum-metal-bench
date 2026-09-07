import numpy as np

from quantum_metal_bench.fidelity import (
    classify_accuracy,
    fidelity_threshold,
    max_amplitude_error,
    state_fidelity,
)


def test_threshold_is_looser_at_high_gate_count_than_naive_constant():
    naive_constant = 0.999999
    low_gate_threshold = fidelity_threshold(n_qubits=10, gate_count=100)
    high_gate_threshold = fidelity_threshold(n_qubits=28, gate_count=100_000)

    # The whole point: a fixed constant is wrong at scale. The analytic
    # threshold at high gate_count must be measurably looser.
    assert high_gate_threshold < low_gate_threshold
    assert high_gate_threshold < naive_constant


def test_threshold_does_not_collapse_to_zero_at_realistic_scale():
    # F11 regression: the old bound multiplied by 2**n_qubits, so a
    # realistic QAOA run (n=28, gate_count=476, matching the repo's
    # locked benchmark protocol) collapsed the threshold to exactly 0.0,
    # making the accuracy gate unable to reject anything. The corrected
    # bound (gate_count * eps only — unitary evolution doesn't accumulate
    # error proportional to statevector dimension) must stay well above
    # zero and comfortably below 1 at this scale.
    threshold = fidelity_threshold(n_qubits=28, gate_count=476)
    assert threshold > 0.999, "threshold collapsed — accuracy gate can no longer reject anything"
    assert threshold < 1.0


def test_threshold_independent_of_qubit_count_for_fixed_gate_count():
    # Gate application is unitary/norm-preserving, so the analytic bound
    # should depend on gate_count alone, not on n_qubits — pinning this so
    # a future change can't silently reintroduce the dimension-scaling bug.
    assert fidelity_threshold(n_qubits=4, gate_count=476) == fidelity_threshold(n_qubits=28, gate_count=476)


def test_state_fidelity_of_identical_states_is_one():
    psi = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.complex64)
    # Tolerance matches float32 precision, not float64 — a complex64 input
    # cannot round-trip through vdot tighter than ~1.2e-7 (machine epsilon).
    assert abs(state_fidelity(psi, psi) - 1.0) < 1e-6


def test_classify_accuracy_distinguishes_real_bug_from_expected_drift():
    # Passes both -> SUCCESS
    assert classify_accuracy(fidelity=0.9999999, n_qubits=10, gate_count=50) == "SUCCESS"

    # Fails naive constant but within analytic bound at huge scale -> EXPECTED_FLOAT32_DRIFT
    n, gates = 28, 100_000
    threshold = fidelity_threshold(n, gates)
    borderline_fidelity = (threshold + 0.999999) / 2  # between analytic threshold and naive constant
    assert threshold < borderline_fidelity < 0.999999
    assert classify_accuracy(borderline_fidelity, n, gates) == "EXPECTED_FLOAT32_DRIFT"

    # Fails even the analytic (loose) bound -> real bug
    assert classify_accuracy(fidelity=0.5, n_qubits=10, gate_count=50) == "FAILED_ACCURACY"


def test_state_fidelity_clamped_to_one_despite_float_rounding():
    # Cross-precision comparison (float32-rounded vs itself) can overshoot
    # 1.0 by a hair due to floating point — must clamp, physically impossible otherwise.
    psi = np.array([0.6, 0.8], dtype=np.complex128) * (1 + 1e-8)
    assert state_fidelity(psi, psi) <= 1.0


def test_max_amplitude_error():
    a = np.array([1.0, 0.0], dtype=np.complex64)
    b = np.array([0.9, 0.1], dtype=np.complex64)
    assert abs(max_amplitude_error(a, b) - 0.1) < 1e-6
