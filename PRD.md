# Product Requirements Document (PRD): `quantum-metal-bench`

## 1. Executive Summary

**`quantum-metal-bench`** is an open-source Python CLI and benchmarking harness that measures quantum statevector simulation performance across three hardware architectures — Apple Silicon (Metal), NVIDIA GPUs (CUDA/cuQuantum), and multi-core CPUs (x86/ARM) — under a methodology designed to survive scrutiny from the maintainers of the frameworks it benchmarks.

The open research question: **how does unified-memory GPU compute (Apple Silicon Metal) scale relative to discrete-GPU (NVIDIA, via cuQuantum-backed simulators) and CPU-bound simulators (Qiskit Aer, PennyLane `lightning.qubit`) as statevector size approaches each device's memory ceiling?** The harness reports what it measures; it does not presuppose which architecture wins. Any performance claim made from this data (including material used to promote Quantum Edge / Waratah Labs) must cite the published raw JSON and methodology version, and Quantum Edge's own conflict of interest as the harness author is disclosed up front (§2).

*Prior version note: an earlier draft of this PRD stated as its thesis that Metal "can drastically outperform" CPU simulators. That framing is the single biggest credibility risk to an "unassailable" benchmark and has been removed — see §8 Positioning & Conflict of Interest.*

## 2. Target Audience & Conflict of Interest Disclosure

* **Quantum Framework Maintainers** (Xanadu, IBM) evaluating backend architectures.
* **Systems / Hardware Engineers** interested in Apple Silicon unified memory vs. traditional PCIe GPU bandwidth.
* **Quantum AI Researchers** running local simulations (e.g., Quantum Prompt Optimisation / QAOA).

**Disclosure:** `quantum-metal-bench` is built and published by Waratah Labs, which also builds Quantum Edge (a Metal-accelerated quantum tooling product). This is a direct conflict of interest for any claim favoring Metal. Mitigations, in order of strength:

1. All raw JSON output is published alongside any derived claim — no number appears in marketing copy without a link to the run that produced it.
2. Methodology (this document) is versioned (semver) and any pre-publication change to the timing/fidelity rules after results are seen is logged, not silently edited.
3. Before the first public run is published, the methodology and code are reviewed by at least one person with no stake in Quantum Edge's outcome (§9).
4. The harness ships fully open-source, including the losing-scenario code paths — if Metal loses a comparison, that result is not gated behind a flag that excludes it from the default report.

## 3. Benchmark Methodology (Ensuring Reproducibility)

### A. Circuit Topologies

1. **QAOA (Quantum Approximate Optimization Algorithm):** High entanglement, layered cost/mixer operators. Represents real-world optimization workloads.
2. **Random Clifford+T:** Standardized, high circuit depth. Tests raw gate application speed and cache-miss penalties.

### B. Execution Isolation (The Timing Standard)

* **Warmup Phase:** Run the circuit once at $n=4$ to force Python JIT compiling, Metal pipeline state object (PSO) compilation, and CUDA context initialization. *This time is discarded.*
* **Start Trigger:** The timer starts *exactly* at the handoff to the backend C++/Swift/CUDA API (e.g., Qiskit's `backend.run()` or PennyLane's `device.execute()`).
* **Stop Trigger:** The timer stops only when the final statevector array is fully populated in host-accessible memory.
* **Repetition (new):** Every (circuit, depth, qubit-count, backend) cell is run **N≥5** times after warmup. The harness reports mean, stddev, and min/max, not a single point estimate. A cell with stddev >10% of mean is flagged `HIGH_VARIANCE` in the output rather than silently averaged — this is usually thermal throttling or background load, and hiding it would misrepresent steady-state performance.
* **Thermal control (new):** On laptop-class Apple Silicon, the harness polls `powermetrics` (or equivalent) for CPU/GPU die temperature and P-core clock speed alongside each run. A sweep that shows throttling (sustained clock drop >15% from the first rep) is marked in the output; long sweeps insert a cooldown pause between qubit-count steps to reduce (not eliminate) this confound.

### C. The Fidelity Anchor

* **The Ground Truth:** PennyLane `lightning.qubit` (float64) output is designated as the mathematical ground truth.
* **Validation, up to n=16:** the backend's resulting statevector is compared against the ground truth using maximum amplitude error and state fidelity ($\langle \psi_{\text{test}} \vert{} \psi_{\text{true}} \rangle$).
* **Validation, above n=16 (methodology v0.2.0):** skipped. A full-statevector comparison is O(2^n) memory — transferring, re-parsing, and holding the ground truth plus the backend's own output simultaneously. Real hardware testing found this reached ~18GB and triggered an OOM kill on the Metal leg at n=28, well before any qubit-count-driven memory ceiling that would be a genuine, reportable result. Cross-backend fidelity parity is established once, at small n (verified to ~1e-7 agreement between the Metal Swift kernel and the Python ground truth) — re-proving it on every large-n timing run doesn't scale and isn't necessary. Runs above this threshold report `accuracy_status: "SKIPPED_FIDELITY_LARGE_N"` with `fidelity`/`fidelity_threshold_used` omitted, and timing is otherwise reported normally. This is a deliberate scope narrowing of what's checked at scale, not a silent gap — logged here per this document's own versioning rule (§2, item 2).
* **Threshold is analytically derived, not fixed (changed from original 0.999999 constant):** float32 rounding error accumulates with gate count and qubit count. Before sweeping, the harness computes an expected float32 error bound analytically (accumulated machine epsilon × gate count, a standard result for chained floating-point matrix-vector products) for each (circuit, depth, n) triple, and sets the pass threshold as that bound with a documented safety margin — not a single constant reused across all scales. A result failing the analytic-bound-adjusted threshold is `FAILED_ACCURACY` (real bug); a result that only fails a naive fixed `0.999999` cutoff at high qubit count due to expected float32 accumulation is reported separately as `EXPECTED_FLOAT32_DRIFT`, so framework maintainers can't dismiss the whole benchmark as "doesn't understand its own numerics."

## 4. Data Capture Specifications

### Data Point 1: Hardware & Environment Metadata

*Captured once at harness initialization.*

* **CPU:** Architecture (x86_64 / arm64), Cores (P-cores / E-cores), Clock speed.
* **GPU:** Metal Device Name (e.g., "Apple M1 Pro") or CUDA Device Name (e.g., "NVIDIA RTX 5070 Ti").
* **Memory:** Total System RAM, Unified Memory architecture flag (Boolean).
* **Software:** OS version, Python version, Qiskit version, PennyLane version, `lightning.gpu`/cuQuantum version, Metal/CUDA toolkit versions.
* **Thermal baseline (new):** idle die temperature and clock speed immediately before the sweep starts, for later throttle comparison.

### Data Point 2: Runtime Telemetry (Per Circuit, Per Depth, Per Repetition)

*Captured continuously during execution.*

* **Wall-Clock Execution Time:** Measured via `time.perf_counter_ns()` strictly around the execution handoff, per repetition (N≥5), reported as mean/stddev/min/max.
* **Peak Memory (RSS):** Monitored via a background thread polling `psutil` or `rusage`.
* **VRAM / Unified Memory Allocation:** For Metal/CUDA, capture the size of the `MTLBuffer` or CUDA malloc dedicated to the statevector (exactly $2^{n+1} \times 4$ bytes for float32).
* **Thermal/clock sample (new):** die temperature and clock speed at time of measurement, for throttle-flagging.

### Data Point 3: The Memory Wall (OOM Handling)

*Captured at failure.*

* When a device hits its memory limit, the OS forcefully terminates the process. The harness sandboxes each $n$-qubit step via `multiprocessing`/`subprocess`.
* **Platform-specific detection (changed — original spec assumed macOS `SIGKILL`-only):**
  * **macOS:** subprocess exit code `-9` → record `OOM_SIGKILL`.
  * **Linux (relevant for the CUDA leg):** the kernel/cgroup OOM killer does not always deliver a clean `-9` to the harness's direct child — it can kill a deeper CUDA driver process, hang, or return a nonzero exit unrelated to `-9`. The harness additionally: (a) sets an explicit wall-clock timeout per subprocess step as a hang backstop, (b) checks `dmesg`/cgroup `memory.events` `oom_kill` counter after an abnormal exit to positively confirm OOM vs. a crash, and (c) records `OOM_UNCONFIRMED` (rather than silently calling it `OOM_SIGKILL`) when the signal doesn't match the macOS pattern but memory pressure was present.
* On confirmed OOM, the harness records the failure state and stops sweeping higher for that backend.

## 5. Output Schema Definition

```json
{
  "benchmark_id": "sweep-qaoa-mac-m1-pro-17180292",
  "methodology_version": "0.1.0",
  "metadata": {
    "os": "macOS 14.5",
    "cpu": "Apple M1 Pro (10 cores)",
    "gpu": "Apple M1 Pro (16 cores)",
    "ram_gb": 32,
    "unified_memory": true,
    "idle_temp_c": 38.2,
    "idle_clock_mhz": 3200
  },
  "circuit": {
    "type": "QAOA",
    "depth": 4
  },
  "results": [
    {
      "qubits": 20,
      "statevector_size_mb": 16.7,
      "backends": {
        "pennylane_cpu_float64": {
          "status": "SUCCESS",
          "execution_time_sec_mean": 0.145,
          "execution_time_sec_stddev": 0.006,
          "n_reps": 5,
          "high_variance": false,
          "peak_memory_mb": 34.2,
          "fidelity": 1.0
        },
        "metal_gpu_float32": {
          "status": "SUCCESS",
          "execution_time_sec_mean": 0.021,
          "execution_time_sec_stddev": 0.002,
          "n_reps": 5,
          "high_variance": false,
          "peak_memory_mb": 17.5,
          "fidelity": 0.99999999,
          "fidelity_threshold_used": 0.99999990,
          "accuracy_status": "SUCCESS"
        },
        "cuquantum_gpu_float32": {
          "status": "SUCCESS",
          "execution_time_sec_mean": 0.018,
          "execution_time_sec_stddev": 0.003,
          "n_reps": 5,
          "high_variance": false,
          "peak_memory_mb": 19.1,
          "fidelity": 0.99999995,
          "fidelity_threshold_used": 0.99999990,
          "accuracy_status": "SUCCESS"
        }
      }
    },
    {
      "qubits": 29,
      "statevector_size_mb": 8589.9,
      "backends": {
        "metal_gpu_float32": {
          "status": "FAILED_OOM_SIGKILL",
          "execution_time_sec_mean": null,
          "peak_memory_mb": 8100.0,
          "fidelity": null
        }
      }
    }
  ]
}
```

## 6. Architecture & Implementation Stack

* **CLI Framework:** `Click`.
* **Benchmarking Engine:** custom `perf_counter_ns` loops with explicit N-repetition and stddev/variance reporting (not a single-shot `pytest-benchmark` default).
* **Process Isolation:** `multiprocessing`, with a wall-clock timeout backstop per step (Linux OOM hang mitigation, §4).
* **Metal Integration:**
  * Option A: Package the existing Swift `QuantumEdgeKit` as a macOS CLI binary, invoked by the Python harness via `subprocess`.
  * Option B: Wrap the Metal compute kernels using `metal-cpp` and expose them to Python via `pybind11` (preferred for eventual PennyLane PR).
* **CUDA/NVIDIA integration (new — was missing from the original stack despite NVIDIA being a named comparison target):** `lightning.gpu` (cuQuantum-backed) as the NVIDIA leg, installed and benchmarked with the same harness, not left as a gap filled implicitly by generic `lightning.qubit` CPU numbers. If cuQuantum-backed `lightning.gpu` is unavailable in a given environment, the harness explicitly records `NVIDIA_LEG_SKIPPED: <reason>` in the output rather than silently omitting the NVIDIA row — a missing comparison must be visible, not invisible.
* **Thermal telemetry:** `powermetrics` (macOS) / `nvidia-smi` (NVIDIA) polling thread.

## 7. Statistical & Reporting Standards (new section)

* Every headline number in any published chart or PR must trace to a `benchmark_id` in the raw JSON.
* Any chart comparing backends must show error bars (stddev across N≥5 reps), not bare point values.
* `HIGH_VARIANCE`, `EXPECTED_FLOAT32_DRIFT`, `OOM_UNCONFIRMED`, and `NVIDIA_LEG_SKIPPED` statuses must render visibly in any generated markdown/chart output — no silent success-only filtering.

## 8. Positioning & Conflict of Interest

* Public materials referencing benchmark results must include, near the claim, a one-line disclosure that Waratah Labs (Quantum Edge's maker) authored the benchmark, plus a link to the raw JSON and methodology version used.
* Marketing copy may summarize results but must not restate them without the qualifying conditions attached in the raw data (e.g., a Metal win at n=20 must not be generalized to "Metal wins" without noting the OOM ceiling and the qubit range tested).

## 9. Independent Review Gate (new)

Before the first public run's results are published anywhere (blog, README, social), the methodology (this document) and the harness code are reviewed by at least one person outside Waratah Labs / not incentivized by Quantum Edge's outcome. Their sign-off (or documented disagreement) is recorded alongside the first published results. This does not need to repeat per run — it is a one-time methodology gate, re-triggered only if the timing/fidelity/repetition rules materially change.
