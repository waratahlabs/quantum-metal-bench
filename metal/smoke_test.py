#!/usr/bin/env python3
"""Smoke test: Metal statevector vs PennyLane default.qubit on Bell state."""
import ctypes
import os
import sys
import numpy as np

DYLIB = os.path.join(os.path.dirname(__file__), ".build", "release", "libQuantumEdgeBridge.dylib")

lib = ctypes.CDLL(DYLIB)
lib.qe_init.argtypes = [ctypes.c_int32]
lib.qe_init.restype = ctypes.c_int32
lib.qe_reset.argtypes = []
lib.qe_reset.restype = None
lib.qe_statevector_pointer.argtypes = []
lib.qe_statevector_pointer.restype = ctypes.c_void_p
lib.qe_apply_1q.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
lib.qe_apply_1q.restype = None
lib.qe_apply_2q.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32]
lib.qe_apply_2q.restype = None

# --- ISC-30: qe_init(2) returns 0 ---
ret = lib.qe_init(2)
assert ret == 0, f"qe_init returned {ret}, expected 0"
print("ISC-30 PASS: qe_init(2) == 0")

# --- ISC-31: initial state is |00⟩ ---
addr = lib.qe_statevector_pointer()
arr_t = (ctypes.c_float * 8).from_address(addr)
sv = np.frombuffer(arr_t, dtype=np.float32).view(np.complex64)

expected_00 = np.array([1+0j, 0+0j, 0+0j, 0+0j], dtype=np.complex64)
assert np.allclose(sv, expected_00, atol=1e-6), f"Initial state wrong: {sv}"
print("ISC-31 PASS: initial state |00⟩ =", sv)

# --- ISC-32: after H on qubit 0, state = [1/√2, 0, 1/√2, 0] ---
inv2 = float(1 / np.sqrt(2))
H = np.array([[inv2, inv2], [inv2, -inv2]], dtype=np.complex64)
H_flat = H.view(np.float32).flatten()
H_ptr = H_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
lib.qe_apply_1q(H_ptr, 0)

expected_h = np.array([inv2, inv2, 0, 0], dtype=np.complex64)
assert np.allclose(sv, expected_h, atol=1e-5), f"After H state wrong: {sv}"
print("ISC-32 PASS: after H(0), state =", np.round(sv, 5))

# --- ISC-33: after CNOT (control=0, target=1), state = Bell [1/√2,0,0,1/√2] ---
# CNOT(control=q_lo=0, target=q_hi=1):
#   |00>->|00>, |01>->|11>, |10>->|10>, |11>->|01>
CNOT = np.array([
    [1,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,1,0,0],
], dtype=np.complex64)
CNOT_flat = CNOT.view(np.float32).flatten()
CNOT_ptr = CNOT_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
lib.qe_apply_2q(CNOT_ptr, 0, 1)

expected_bell = np.array([inv2, 0, 0, inv2], dtype=np.complex64)
assert np.allclose(sv, expected_bell, atol=1e-5), f"After CNOT state wrong: {sv}"
print("ISC-33 PASS: after CNOT(0,1), Bell state =", np.round(sv, 5))

# --- Compare with PennyLane default.qubit ---
try:
    import pennylane as qml
    dev_ref = qml.device("default.qubit", wires=2)

    @qml.qnode(dev_ref)
    def bell_ref():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    ref_state = bell_ref()
    print(f"\nPennyLane default.qubit Bell state: {np.round(ref_state, 5)}")
    print(f"Metal kernel Bell state:            {np.round(sv, 5)}")
    assert np.allclose(np.abs(sv)**2, np.abs(ref_state)**2, atol=1e-5), \
        "Probabilities mismatch vs default.qubit"
    print("MATCH: Metal probabilities agree with default.qubit to 1e-5")
except ImportError:
    print("PennyLane not installed — skipping reference comparison")

print("\nAll smoke tests PASSED")
sys.exit(0)
