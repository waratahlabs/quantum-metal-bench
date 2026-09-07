"""Platform-specific OOM detection (PRD §4, Data Point 3).

macOS delivers a clean SIGKILL (-9) to the child on memory pressure. Linux's
cgroup/kernel OOM killer does not reliably do the same for a process using a
CUDA context — it can kill a driver process, hang, or exit with something
other than -9. This module distinguishes a confirmed OOM from an
unconfirmed one instead of assuming the macOS pattern everywhere.
"""

from __future__ import annotations

import platform
import subprocess


def classify_subprocess_exit(returncode: int, timed_out: bool, dmesg_oom_hit: bool = False) -> str:
    """Classify a subprocess's exit into a PRD-schema status string.

    returncode: value from subprocess.Popen/run (negative == killed by signal on POSIX)
    timed_out: True if the harness's own wall-clock timeout backstop fired
    dmesg_oom_hit: True if a post-hoc dmesg/cgroup memory.events check found
                   an oom_kill event attributable to this run (Linux only)
    """
    system = platform.system()

    if timed_out:
        # A hang is exactly the failure mode PRD flags for Linux cgroup OOM —
        # never silently report SUCCESS or misattribute to SIGKILL.
        return "OOM_UNCONFIRMED" if dmesg_oom_hit else "TIMEOUT_UNCONFIRMED_OOM"

    if system == "Darwin":
        if returncode == -9:
            return "OOM_SIGKILL"
        return "SUCCESS" if returncode == 0 else "FAILED_OTHER"

    if system == "Linux":
        if returncode == -9 and dmesg_oom_hit:
            return "OOM_SIGKILL"
        if dmesg_oom_hit:
            # Killed by cgroup OOM but not via a clean -9 to our direct child.
            return "OOM_UNCONFIRMED"
        return "SUCCESS" if returncode == 0 else "FAILED_OTHER"

    return "SUCCESS" if returncode == 0 else "FAILED_OTHER"


def check_dmesg_oom() -> bool:
    """Best-effort check of dmesg / cgroup memory.events for a recent oom_kill.

    Returns False (not True) on any failure to read — an inability to check
    is not evidence of OOM, and must never be silently upgraded to one.
    """
    try:
        out = subprocess.run(
            ["dmesg", "-T"], capture_output=True, text=True, timeout=5
        ).stdout
        return "Out of memory" in out or "oom_kill" in out.lower()
    except Exception:
        return False
