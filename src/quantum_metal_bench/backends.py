"""Backend adapters. Metal shells out to the vendored qe-bench binary
(metal/) rather than reimplementing the kernel in Python (ISA Decisions).
CUDA runs in-process via PennyLane's lightning.gpu device (no separate
binary to shell out to — it's a PennyLane plugin, not a standalone kernel
like QuantumEdgeKit)."""

from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from .fidelity import classify_accuracy, fidelity_threshold, state_fidelity
from .schema import BackendResult
from .timing import time_repeated

# metal/.build/release/qe-bench relative to the repo root (this file lives at
# src/quantum_metal_bench/backends.py -> repo root is two parents up).
_REPO_ROOT = Path(__file__).resolve().parents[2]
QE_BENCH_PATH = _REPO_ROOT / "metal" / ".build" / "release" / "qe-bench"

METAL_SUPPORTED_CIRCUITS = {"qaoa"}  # clifford_t has no cross-language RNG parity yet

# Cross-language fidelity parity (Metal vs Python ground truth) is already
# established at small n (~1e-7 agreement, verified once). Re-proving it via
# a full statevector comparison at every benchmark run is O(2**n) memory and
# doesn't scale -- a real n=28 run reached ~18GB and got OOM-killed. Above
# this threshold, both backends skip the fidelity check and report timing
# only. Shared contract with Swift's `maxQubitsForFidelityCheck` in
# metal/Sources/QuantumEdgeCLI/main.swift -- keep in sync.
MAX_QUBITS_FOR_FIDELITY_CHECK = 16


def metal_available() -> tuple[bool, str | None]:
    """Returns (available, reason_if_not). Never raises — callers must be
    able to report a visible skip rather than crash the whole sweep."""
    if platform.system() != "Darwin":
        return False, "Metal backend requires macOS"
    if not QE_BENCH_PATH.exists():
        return False, f"qe-bench binary not found at {QE_BENCH_PATH} — run `swift build -c release` in metal/"
    return True, None


def run_metal_backend(
    circuit: str,
    n_qubits: int,
    depth: int,
    reps: int,
    seed: int,
    ground_truth: np.ndarray | None,
) -> BackendResult:
    """Runs the given circuit on the Metal backend via qe-bench and compares
    its resulting statevector against `ground_truth` (the CPU float64 result
    for the SAME circuit params — caller's responsibility to pass the right
    one, since a mismatched circuit would produce a meaningless fidelity).

    Above MAX_QUBITS_FOR_FIDELITY_CHECK, `ground_truth` is ignored (may be
    None) — qe-bench itself skips emitting the statevector at this scale, so
    there's nothing to compare. Returns SUCCESS with accuracy_status
    SKIPPED_FIDELITY_LARGE_N and no fidelity fields, not a failure."""
    if circuit not in METAL_SUPPORTED_CIRCUITS:
        return BackendResult(
            status="SKIPPED_CIRCUIT_TYPE",
            skip_reason=f"Metal has no cross-language circuit mirror for '{circuit}' yet (only qaoa)",
        )

    available, reason = metal_available()
    if not available:
        return BackendResult(status="METAL_LEG_SKIPPED", skip_reason=reason)

    try:
        proc = subprocess.run(
            [
                str(QE_BENCH_PATH), "bench",
                "--circuit", circuit,
                "--qubits", str(n_qubits),
                "--depth", str(depth),
                "--reps", str(reps),
                "--seed", str(seed),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return BackendResult(status="TIMEOUT_UNCONFIRMED_OOM")

    if proc.returncode != 0:
        return BackendResult(
            status="FAILED_OTHER",
            skip_reason=(proc.stderr or "qe-bench exited nonzero with no stderr").strip()[:500],
        )

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return BackendResult(status="FAILED_OTHER", skip_reason="qe-bench produced non-JSON stdout")

    n_reps = payload["n_reps"]
    mean = payload["execution_time_sec_mean"]
    stddev = payload["execution_time_sec_stddev"]
    high_variance = (stddev / mean) > 0.10 if mean > 0 else False

    if payload.get("statevector_omitted"):
        return BackendResult(
            status="SUCCESS",
            execution_time_sec_mean=mean,
            execution_time_sec_stddev=stddev,
            n_reps=n_reps,
            high_variance=high_variance,
            accuracy_status="SKIPPED_FIDELITY_LARGE_N",
        )

    real = np.array(payload["statevector_real"], dtype=np.float64)
    imag = np.array(payload["statevector_imag"], dtype=np.float64)
    metal_state = real + 1j * imag

    fidelity = state_fidelity(metal_state, ground_truth)
    gate_count = payload.get("gate_count", 0)
    threshold = fidelity_threshold(n_qubits, gate_count)
    accuracy_status = classify_accuracy(fidelity, n_qubits, gate_count)

    return BackendResult(
        status="SUCCESS",
        execution_time_sec_mean=mean,
        execution_time_sec_stddev=stddev,
        n_reps=n_reps,
        high_variance=high_variance,
        fidelity=fidelity,
        fidelity_threshold_used=threshold,
        accuracy_status=accuracy_status,
    )


def cuda_available() -> tuple[bool, str | None]:
    """Returns (available, reason_if_not). Checks for the plugin's importable
    module rather than trying to construct a device — the latter can hang or
    hard-crash the interpreter on a misconfigured CUDA install rather than
    raising a catchable exception (an OOM/driver-mismatch mid-init that
    lands the process instead of returning an error), which would put this
    check outside the same visible-skip discipline as everything else here."""
    if importlib.util.find_spec("pennylane_lightning.lightning_gpu") is None:
        return False, "pennylane-lightning-gpu not installed (CUDA-only wheel — this install has no NVIDIA GPU)"
    return True, None


CUDA_SUPPORTED_CIRCUITS = {"qaoa", "clifford_t"}  # runs the actual Python circuit fn, no cross-language mirror needed


def run_cuda_backend(
    circuit: str,
    n_qubits: int,
    depth: int,
    reps: int,
    seed: int,
    circuit_fn,
    gate_count: int,
    ground_truth: np.ndarray | None,
) -> BackendResult:
    """Runs `circuit_fn` on PennyLane's lightning.gpu device in-process.

    Unlike Metal, this has no subprocess/OOM sandboxing (PRD §4 Data Point 3)
    yet — lightning.gpu's CUDA malloc failures typically raise a catchable
    Python exception rather than SIGKILL-ing the process the way an iOS/
    unified-memory OOM does, so a bare try/except covers the common case,
    but a driver-level OOM that takes the whole process down would currently
    escape this adapter same as it would any other in-process backend.
    Verified against real RTX 5070 Ti hardware on bare-metal Linux.

    Above MAX_QUBITS_FOR_FIDELITY_CHECK, `ground_truth` is ignored (may be
    None) and the fidelity comparison is skipped — see run_metal_backend's
    docstring for why full-statevector comparison doesn't scale.
    """
    if circuit not in CUDA_SUPPORTED_CIRCUITS:
        return BackendResult(
            status="SKIPPED_CIRCUIT_TYPE",
            skip_reason=f"CUDA adapter has no handling for circuit '{circuit}'",
        )

    available, reason = cuda_available()
    if not available:
        return BackendResult(status="NVIDIA_LEG_SKIPPED", skip_reason=reason)

    import pennylane as qml

    try:
        dev = qml.device("lightning.gpu", wires=n_qubits)
        qnode = qml.QNode(circuit_fn, dev)

        holder: dict = {}

        def run_once():
            holder["state"] = qnode()

        timing = time_repeated(run_once, n_reps=reps)
    except Exception as exc:  # noqa: BLE001 — deliberately broad: any CUDA
        # failure (OOM, driver mismatch, context init) must degrade to a
        # visible status, never propagate and kill the whole sweep.
        return BackendResult(status="FAILED_OTHER", skip_reason=str(exc)[:500])

    if n_qubits > MAX_QUBITS_FOR_FIDELITY_CHECK or ground_truth is None:
        return BackendResult(
            status="SUCCESS",
            execution_time_sec_mean=timing.execution_time_sec_mean,
            execution_time_sec_stddev=timing.execution_time_sec_stddev,
            n_reps=timing.n_reps,
            high_variance=timing.high_variance,
            accuracy_status="SKIPPED_FIDELITY_LARGE_N",
        )

    cuda_state = np.asarray(holder["state"])
    fidelity = state_fidelity(cuda_state, ground_truth)
    threshold = fidelity_threshold(n_qubits, gate_count)
    accuracy_status = classify_accuracy(fidelity, n_qubits, gate_count)

    return BackendResult(
        status="SUCCESS",
        execution_time_sec_mean=timing.execution_time_sec_mean,
        execution_time_sec_stddev=timing.execution_time_sec_stddev,
        n_reps=timing.n_reps,
        high_variance=timing.high_variance,
        fidelity=fidelity,
        fidelity_threshold_used=threshold,
        accuracy_status=accuracy_status,
    )
