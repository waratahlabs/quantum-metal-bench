"""qmb — quantum-metal-bench CLI (PRD §6)."""

from __future__ import annotations

import json

import click
import numpy as np
import pennylane as qml

from . import METHODOLOGY_VERSION
from .backends import run_cuda_backend, run_metal_backend
from .circuits import CIRCUIT_BUILDERS
from .fidelity import classify_accuracy, fidelity_threshold, max_amplitude_error, state_fidelity
from .metadata import capture_metadata
from .schema import BackendResult, BenchmarkRun, QubitResult, statevector_size_mb
from .timing import time_repeated


@click.group()
@click.version_option(METHODOLOGY_VERSION)
def main():
    """quantum-metal-bench: cross-platform quantum statevector benchmark."""


@main.command()
@click.option("--circuit", type=click.Choice(list(CIRCUIT_BUILDERS)), default="qaoa")
@click.option("--qubits", type=int, default=10)
@click.option("--depth", type=int, default=4)
@click.option("--reps", type=int, default=5, help="Repetitions per backend (PRD floor: 5)")
@click.option("--backend", type=click.Choice(["cpu", "metal", "cuda"]), default="cpu",
              help="cpu (lightning.qubit), metal (shells out to qe-bench, qaoa only), or cuda (lightning.gpu, in-process)")
@click.option("--seed", type=int, default=0)
def single(circuit: str, qubits: int, depth: int, reps: int, backend: str, seed: int):
    """Run one (circuit, qubits, depth) cell on one backend and print its JSON result."""
    if reps < 5:
        click.echo(f"warning: reps={reps} is below the PRD floor of 5 — result is not publishable", err=True)

    builder = CIRCUIT_BUILDERS[circuit]
    circuit_fn, gate_count = builder(qubits, depth, seed)

    dev = qml.device("lightning.qubit", wires=qubits)
    qnode = qml.QNode(circuit_fn, dev)

    holder: dict = {}

    def run_once():
        holder["state"] = qnode()

    # When benchmarking a GPU backend, the CPU run only exists to produce a
    # ground-truth state for fidelity comparison — its own timing is
    # discarded, so there's no reason to pay for N reps of it (at n=28 a
    # single CPU rep already costs minutes; 5x that for a number nobody
    # reads would be pure waste).
    cpu_reps = reps if backend == "cpu" else 1
    timing = time_repeated(run_once, n_reps=cpu_reps, warmup=(backend == "cpu"))
    ground_truth = holder["state"]

    if backend == "metal":
        result = run_metal_backend(circuit, qubits, depth, reps, seed, np.asarray(ground_truth))
    elif backend == "cuda":
        result = run_cuda_backend(circuit, qubits, depth, reps, seed, circuit_fn, gate_count, np.asarray(ground_truth))
    else:
        fidelity = state_fidelity(ground_truth, ground_truth)  # CPU compared against itself
        threshold = fidelity_threshold(qubits, gate_count)
        accuracy_status = classify_accuracy(fidelity, qubits, gate_count)
        result = BackendResult(
            status="SUCCESS",
            execution_time_sec_mean=timing.execution_time_sec_mean,
            execution_time_sec_stddev=timing.execution_time_sec_stddev,
            n_reps=timing.n_reps,
            high_variance=timing.high_variance,
            fidelity=fidelity,
            fidelity_threshold_used=threshold,
            accuracy_status=accuracy_status,
        )
    click.echo(json.dumps(result.to_dict(), indent=2))


@main.command()
@click.option("--circuit", type=click.Choice(list(CIRCUIT_BUILDERS)), default="qaoa")
@click.option("--qubit-min", type=int, default=4)
@click.option("--qubit-max", type=int, default=16)
@click.option("--depth", type=int, default=4)
@click.option("--reps", type=int, default=5)
@click.option("--seed", type=int, default=0)
@click.option("--skip-metal", is_flag=True, default=False, help="Skip the Metal leg even on macOS (faster iteration)")
@click.option("--skip-cuda", is_flag=True, default=False, help="Skip the CUDA leg even if lightning.gpu is installed")
def sweep(circuit: str, qubit_min: int, qubit_max: int, depth: int, reps: int, seed: int, skip_metal: bool, skip_cuda: bool):
    """Sweep qubit counts across CPU + Metal + CUDA (each visibly skipped when unavailable) and emit a full BenchmarkRun JSON (PRD §5)."""
    metadata = capture_metadata()
    run = BenchmarkRun(
        benchmark_id=f"sweep-{circuit}",
        metadata=metadata,
        circuit={"type": circuit, "depth": depth},
    )

    builder = CIRCUIT_BUILDERS[circuit]
    for n in range(qubit_min, qubit_max + 1):
        circuit_fn, gate_count = builder(n, depth, seed)
        dev = qml.device("lightning.qubit", wires=n)
        qnode = qml.QNode(circuit_fn, dev)

        holder: dict = {}

        def run_once():
            holder["state"] = qnode()

        timing = time_repeated(run_once, n_reps=reps)
        state = holder["state"]
        fidelity = state_fidelity(state, state)
        threshold = fidelity_threshold(n, gate_count)
        accuracy_status = classify_accuracy(fidelity, n, gate_count)

        qr = QubitResult(qubits=n, statevector_size_mb=statevector_size_mb(n))
        qr.backends["pennylane_cpu_float64"] = BackendResult(
            status="SUCCESS",
            execution_time_sec_mean=timing.execution_time_sec_mean,
            execution_time_sec_stddev=timing.execution_time_sec_stddev,
            n_reps=timing.n_reps,
            high_variance=timing.high_variance,
            fidelity=fidelity,
            fidelity_threshold_used=threshold,
            accuracy_status=accuracy_status,
        )
        if skip_metal:
            qr.backends["metal_gpu_float32"] = BackendResult(status="METAL_LEG_SKIPPED", skip_reason="--skip-metal passed")
        else:
            qr.backends["metal_gpu_float32"] = run_metal_backend(circuit, n, depth, reps, seed, np.asarray(state))

        if skip_cuda:
            qr.backends["cuquantum_gpu_float32"] = BackendResult(status="NVIDIA_LEG_SKIPPED", skip_reason="--skip-cuda passed")
        else:
            qr.backends["cuquantum_gpu_float32"] = run_cuda_backend(
                circuit, n, depth, reps, seed, circuit_fn, gate_count, np.asarray(state)
            )
        run.results.append(qr)

    click.echo(json.dumps(run.to_dict(), indent=2))


if __name__ == "__main__":
    main()
