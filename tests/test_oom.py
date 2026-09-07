from unittest.mock import patch

from quantum_metal_bench.oom import classify_subprocess_exit


def test_macos_clean_sigkill_is_oom_sigkill():
    with patch("quantum_metal_bench.oom.platform.system", return_value="Darwin"):
        assert classify_subprocess_exit(returncode=-9, timed_out=False) == "OOM_SIGKILL"


def test_macos_normal_success():
    with patch("quantum_metal_bench.oom.platform.system", return_value="Darwin"):
        assert classify_subprocess_exit(returncode=0, timed_out=False) == "SUCCESS"


def test_linux_sigkill_without_dmesg_confirmation_is_unconfirmed_not_sigkill():
    # This is the exact gap flagged in the PRD review: Linux cgroup OOM does
    # not reliably deliver a clean -9 to our direct child, so a bare -9
    # without dmesg corroboration must NOT be auto-labeled OOM_SIGKILL.
    with patch("quantum_metal_bench.oom.platform.system", return_value="Linux"):
        result = classify_subprocess_exit(returncode=-9, timed_out=False, dmesg_oom_hit=False)
        assert result == "FAILED_OTHER"


def test_linux_sigkill_with_dmesg_confirmation_is_oom_sigkill():
    with patch("quantum_metal_bench.oom.platform.system", return_value="Linux"):
        result = classify_subprocess_exit(returncode=-9, timed_out=False, dmesg_oom_hit=True)
        assert result == "OOM_SIGKILL"


def test_linux_hang_with_dmesg_hit_is_oom_unconfirmed():
    with patch("quantum_metal_bench.oom.platform.system", return_value="Linux"):
        result = classify_subprocess_exit(returncode=-1, timed_out=True, dmesg_oom_hit=True)
        assert result == "OOM_UNCONFIRMED"


def test_hang_without_any_oom_evidence_is_timeout_unconfirmed_not_silently_success():
    with patch("quantum_metal_bench.oom.platform.system", return_value="Linux"):
        result = classify_subprocess_exit(returncode=-1, timed_out=True, dmesg_oom_hit=False)
        assert result == "TIMEOUT_UNCONFIRMED_OOM"
