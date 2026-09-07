"""Hardware & environment metadata capture (PRD §4, Data Point 1)."""

from __future__ import annotations

import platform
import subprocess
from typing import Any

import psutil


def capture_metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "python_version": platform.python_version(),
        "unified_memory": platform.system() == "Darwin" and platform.machine() == "arm64",
    }
    meta["idle_temp_c"], meta["idle_clock_mhz"] = _thermal_baseline()
    return meta


def _thermal_baseline() -> tuple[float | None, float | None]:
    """Best-effort thermal baseline. Returns (None, None) when unavailable
    rather than fabricating a value — an unmeasured baseline must not look
    like a measured one downstream (PRD §3.B thermal control)."""
    if platform.system() != "Darwin":
        return None, None
    try:
        subprocess.run(["which", "powermetrics"], capture_output=True, check=True, timeout=2)
    except Exception:
        return None, None
    # powermetrics requires sudo; left as a documented manual step rather
    # than silently skipped or fabricated. Real capture wired in when the
    # harness is run with elevated privileges (see README).
    return None, None
