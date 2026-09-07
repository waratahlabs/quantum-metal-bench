"""
PennyLane device v2 correctness test — Metal vs default.qubit.
All gates pass PL wire indices directly; convention is now handled in bridge.swift.
"""
import sys
import numpy as np
import pennylane as qml
sys.path.insert(0, ".")
from pennylane_metal.device import MetalQubit

TOL = 1e-4
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def run(name, circuit_fn, n):
    ref = qml.device("default.qubit", wires=n)
    metal = MetalQubit(wires=n)

    @qml.qnode(ref)
    def ref_circuit():
        circuit_fn()
        return qml.state()

    tape = qml.tape.QuantumScript(
        ops=[op for op in qml.tape.make_qscript(circuit_fn)().operations],
        measurements=[qml.state()]
    )
    metal_sv = metal.execute(tape)

    ref_sv = ref_circuit()
    delta = np.max(np.abs(metal_sv - ref_sv))
    status = PASS if delta < TOL else FAIL
    print(f"[{status}] {name:30s}  max|Δ|={delta:.2e}")
    return delta < TOL

results = []

# 1q gates
results.append(run("H on wire 0", lambda: qml.Hadamard(wires=0), 2))
results.append(run("H on wire 1", lambda: qml.Hadamard(wires=1), 2))
results.append(run("X on wire 0", lambda: qml.PauliX(wires=0), 2))
results.append(run("RZ(π/3) on wire 1", lambda: qml.RZ(np.pi/3, wires=1), 3))
results.append(run("RX(π/4) on wire 2", lambda: qml.RX(np.pi/4, wires=2), 3))
results.append(run("Y on wire 0", lambda: qml.PauliY(wires=0), 2))

# Bell state: H(0) then CNOT(0→1)
def bell():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
results.append(run("Bell state H(0)+CNOT(0,1)", bell, 2))

# Bell state reversed wire ordering CNOT(1→0)
def bell_rev():
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[1, 0])
results.append(run("Bell state H(1)+CNOT(1,0)", bell_rev, 2))

# GHZ 3-qubit
def ghz():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
results.append(run("GHZ 3-qubit", ghz, 3))

# CZ gate
def cz_circuit():
    qml.Hadamard(wires=0)
    qml.Hadamard(wires=1)
    qml.CZ(wires=[0, 1])
results.append(run("CZ(0,1) after H⊗H", cz_circuit, 2))

# Mixed 1q+2q on 4 qubits
def mixed4():
    qml.Hadamard(wires=0)
    qml.RZ(np.pi/5, wires=1)
    qml.CNOT(wires=[0, 2])
    qml.RX(np.pi/7, wires=3)
    qml.CZ(wires=[1, 3])
results.append(run("Mixed 4-qubit circuit", mixed4, 4))

print()
print(f"Results: {sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
