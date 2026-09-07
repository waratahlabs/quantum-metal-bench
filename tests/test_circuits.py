import pennylane as qml

from quantum_metal_bench.circuits import qaoa_circuit, random_clifford_t_circuit


def test_qaoa_circuit_runs_and_returns_correct_qubit_count():
    circuit_fn, gate_count = qaoa_circuit(n_qubits=4, depth=2, seed=1)
    dev = qml.device("lightning.qubit", wires=4)
    qnode = qml.QNode(circuit_fn, dev)
    state = qnode()
    assert state.shape == (2**4,)
    assert gate_count > 0


def test_clifford_t_circuit_runs_and_returns_correct_qubit_count():
    circuit_fn, gate_count = random_clifford_t_circuit(n_qubits=5, depth=3, seed=1)
    dev = qml.device("lightning.qubit", wires=5)
    qnode = qml.QNode(circuit_fn, dev)
    state = qnode()
    assert state.shape == (2**5,)
    assert gate_count > 0


def test_circuits_are_deterministic_given_seed():
    fn_a, _ = qaoa_circuit(n_qubits=3, depth=2, seed=42)
    fn_b, _ = qaoa_circuit(n_qubits=3, depth=2, seed=42)
    dev = qml.device("lightning.qubit", wires=3)
    state_a = qml.QNode(fn_a, dev)()
    state_b = qml.QNode(fn_b, dev)()
    assert (state_a == state_b).all()
