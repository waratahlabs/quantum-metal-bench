"""Analytic float32 fidelity threshold (PRD §3.C).

A fixed constant like 0.999999 is wrong at scale: float32 rounding error
accumulates with gate count. This derives a threshold from machine epsilon
and gate count instead of reusing one number everywhere.

F11 fix (2026-09-07): the original bound multiplied by the statevector
dimension (2**n_qubits) on the theory that error scales with the cost of
a matrix-vector product. That's the wrong model for *this* quantity. Gate
application in this circuit is unitary (Hadamard/CNOT/RZ/RX all preserve
vector norm) — the standard numerical-analysis result for a chain of
unitary transformations (e.g. Higham, "Accuracy and Stability of Numerical
Algorithms", ch. 19) bounds accumulated error by gate_count * eps,
independent of dimension, because a unitary matrix has condition number 1.
Multiplying by dim conflated "how much arithmetic a gate costs" with "how
much error a gate accumulates" — those are different quantities, and only
the second one belongs in a fidelity bound. Confirmed against real hardware
data (n=28, gate_count=476, measured fidelity 0.999999999999964): the old
formula gave a threshold of 0.0 (useless); the corrected formula gives
~0.99943, a real discriminating bound the measured fidelity comfortably
clears without being trivially satisfiable.
"""

from __future__ import annotations

import numpy as np

FLOAT32_EPS = np.finfo(np.float32).eps  # ~1.19e-7
SAFETY_MARGIN = 10.0  # multiplicative slack on the analytic bound


def analytic_error_bound(n_qubits: int, gate_count: int) -> float:
    """Expected accumulated float32 error bound for a chained gate application.

    Gate application here is unitary and norm-preserving, so accumulated
    error scales with the number of chained operations (gate_count), not
    with the statevector dimension. `n_qubits` is accepted for API
    stability and potential future use (e.g. per-amplitude noise models)
    but does not currently affect the bound.
    """
    del n_qubits  # unitary evolution: error doesn't scale with dimension
    return gate_count * FLOAT32_EPS


def fidelity_threshold(n_qubits: int, gate_count: int) -> float:
    """Pass threshold for fidelity, loosened at high (n, gate_count) vs a naive constant."""
    bound = analytic_error_bound(n_qubits, gate_count) * SAFETY_MARGIN
    # Fidelity threshold = 1 - bound, floored so it never goes negative/absurd
    # at extreme qubit counts (bound can exceed 1 well past hardware limits).
    # np.float32 inputs (FLOAT32_EPS) propagate through arithmetic — cast
    # back to a native float so downstream json.dumps doesn't choke on it.
    return float(max(0.0, 1.0 - min(bound, 1.0)))


def state_fidelity(psi_test: np.ndarray, psi_true: np.ndarray) -> float:
    """|<psi_test|psi_true>|^2, both assumed normalized.

    Clamped to [0, 1]: cross-precision comparisons (float32 vs float64) can
    round to e.g. 1.0000007 despite fidelity being physically bounded by 1 —
    an unclamped value would misleadingly read as a bug in published data.
    """
    overlap = np.vdot(psi_test, psi_true)
    return float(np.clip(np.abs(overlap) ** 2, 0.0, 1.0))


def max_amplitude_error(psi_test: np.ndarray, psi_true: np.ndarray) -> float:
    return float(np.max(np.abs(psi_test - psi_true)))


def classify_accuracy(
    fidelity: float,
    n_qubits: int,
    gate_count: int,
    naive_constant: float = 0.999999,
) -> str:
    """Return SUCCESS / FAILED_ACCURACY / EXPECTED_FLOAT32_DRIFT per PRD §3.C.

    - Passes the analytic threshold -> SUCCESS
    - Fails the analytic threshold -> FAILED_ACCURACY (real bug)
    - Passes analytic but fails the naive fixed constant -> EXPECTED_FLOAT32_DRIFT
      (so this doesn't get miscounted as a bug by a maintainer skimming a
      fixed-threshold comparison)
    """
    threshold = fidelity_threshold(n_qubits, gate_count)
    if fidelity < threshold:
        return "FAILED_ACCURACY"
    if fidelity < naive_constant:
        return "EXPECTED_FLOAT32_DRIFT"
    return "SUCCESS"
