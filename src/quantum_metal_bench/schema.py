"""Output JSON schema matching PRD §5. Dataclasses -> dict for json.dumps."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from . import METHODOLOGY_VERSION


@dataclass
class BackendResult:
    status: str  # SUCCESS | FAILED_ACCURACY | OOM_SIGKILL | OOM_UNCONFIRMED | NVIDIA_LEG_SKIPPED | ...
    execution_time_sec_mean: float | None = None
    execution_time_sec_stddev: float | None = None
    n_reps: int | None = None
    high_variance: bool | None = None
    peak_memory_mb: float | None = None
    fidelity: float | None = None
    fidelity_threshold_used: float | None = None
    accuracy_status: str | None = None
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class QubitResult:
    qubits: int
    statevector_size_mb: float
    backends: dict[str, BackendResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qubits": self.qubits,
            "statevector_size_mb": self.statevector_size_mb,
            "backends": {k: v.to_dict() for k, v in self.backends.items()},
        }


@dataclass
class BenchmarkRun:
    benchmark_id: str
    metadata: dict[str, Any]
    circuit: dict[str, Any]
    results: list[QubitResult] = field(default_factory=list)
    methodology_version: str = METHODOLOGY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "methodology_version": self.methodology_version,
            "metadata": self.metadata,
            "circuit": self.circuit,
            "results": [r.to_dict() for r in self.results],
        }


def statevector_size_mb(n_qubits: int, bytes_per_amplitude: int = 8) -> float:
    """2**n complex amplitudes * bytes_per_amplitude (8 for complex64/float32 pairs)."""
    return (2**n_qubits) * bytes_per_amplitude / (1024 * 1024)
