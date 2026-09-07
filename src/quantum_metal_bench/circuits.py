"""Circuit generators for the two benchmark topologies (PRD §3.A)."""

from __future__ import annotations

import math
import random

import pennylane as qml


def qaoa_angles(depth: int, seed: int) -> tuple[list[float], list[float]]:
    """Deterministic, cross-language-reproducible QAOA angle schedule.

    Uses plain modular arithmetic (not `random.Random`) specifically so the
    Swift `qe-bench` Metal adapter can build the byte-identical circuit from
    the same (depth, seed) — a real fidelity comparison between backends
    requires them to run the *same* circuit, and Python's `random` module
    has no portable equivalent in Swift. This formula is the shared contract;
    see metal/Sources/QuantumEdgeCLI/QAOACircuit.swift for the Swift mirror
    — changing one without the other silently breaks cross-backend fidelity.
    """
    gammas = [((seed * 37 + layer * 17 + 1) % 100) / 100.0 * (math.pi / 2) for layer in range(depth)]
    betas = [((seed * 53 + layer * 11 + 1) % 100) / 100.0 * (math.pi / 4) for layer in range(depth)]
    return gammas, betas


def qaoa_circuit(n_qubits: int, depth: int, seed: int = 0):
    """QAOA circuit: layered cost/mixer operators over a ring graph.

    Returns a callable with no arguments, suitable for qml.QNode wrapping.
    Gate count is tracked so fidelity.py can derive an analytic error bound.
    """
    edges = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    gammas, betas = qaoa_angles(depth, seed)

    def circuit():
        for q in range(n_qubits):
            qml.Hadamard(wires=q)
        for layer in range(depth):
            for i, j in edges:
                qml.CNOT(wires=[i, j])
                qml.RZ(gammas[layer], wires=j)
                qml.CNOT(wires=[i, j])
            for q in range(n_qubits):
                qml.RX(2 * betas[layer], wires=q)
        return qml.state()

    gate_count = n_qubits + depth * (len(edges) * 3 + n_qubits)
    return circuit, gate_count


def random_clifford_t_circuit(n_qubits: int, depth: int, seed: int = 0):
    """Random Clifford+T circuit: standardized high-depth gate application stress test."""
    rng = random.Random(seed)
    single_qubit_gates = [qml.Hadamard, qml.S, qml.T]

    def circuit():
        for _ in range(depth):
            for q in range(n_qubits):
                gate = rng.choice(single_qubit_gates)
                gate(wires=q)
            if n_qubits > 1:
                a, b = rng.sample(range(n_qubits), 2)
                qml.CNOT(wires=[a, b])
        return qml.state()

    gate_count = depth * (n_qubits + (1 if n_qubits > 1 else 0))
    return circuit, gate_count


CIRCUIT_BUILDERS = {
    "qaoa": qaoa_circuit,
    "clifford_t": random_clifford_t_circuit,
}
