"""Tests for the rune_bench.attestation module.

Covers:
- AttestationResult dataclass
- NoOpDriver: always passes, any target
- TPM2Driver: tool-presence check, verify success/failure/timeout/FileNotFoundError
- get_driver factory: noop default, tpm2 driver, env-var fallback, unknown name
- api_contracts: RunBenchmarkRequest.attestation_required field
- api_backend: _verify_attestation helper and run_benchmark integration
- config: attestation section injected into env vars
"""

from __future__ import annotations

import subprocess  # nosec  # tests require subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rune_bench.attestation.factory import get_driver
from rune_bench.attestation.interface import AttestationDriver, AttestationResult
from rune_bench.attestation.noop import NoOpDriver
from rune_bench.attestation.tpm2 import TPM2Driver


# ---------------------------------------------------------------------------
# AttestationResult
# ---------------------------------------------------------------------------


def test_attestation_result_is_frozen():
    result = AttestationResult(passed=True, pcr_digest="abc", message="ok")
    with pytest.raises((FrozenInstanceError, AttributeError)):
        result.passed = False  # type: ignore[misc]


def test_attestation_result_fields():
    r = AttestationResult(passed=False, pcr_digest=None, message="fail")
    assert r.passed is False
    assert r.pcr_digest is None
    assert r.message == "fail"


# ---------------------------------------------------------------------------
# NoOpDriver
# ---------------------------------------------------------------------------


def test_noop_driver_passes_any_target():
    driver = NoOpDriver()
    for target in ("", "~/.kube/config", "/path/to/kubeconfig", "some-cluster"):
        result = driver.verify(target)
        assert result.passed is True
        assert result.pcr_digest is None
        assert "NoOp" in result.message


def test_noop_driver_is_attestation_driver_subclass():
    assert isinstance(NoOpDriver(), AttestationDriver)


# ---------------------------------------------------------------------------
# TPM2Driver — initialisation (tool presence)
# ---------------------------------------------------------------------------


@patch("rune_bench.attestation.tpm2.shutil.which", return_value=None)
def test_tpm2_driver_raises_when_tools_missing(mock_which):
    with pytest.raises(RuntimeError, match="tpm2-tools"):
        TPM2Driver()


def test_tpm2_driver_no_policy_only_requires_tpm2_quote():
    """Driver without pcr_policy_path must NOT require tpm2_checkquote."""
    with patch(
        "rune_bench.attestation.tpm2.shutil.which",
        side_effect=lambda cmd: "/usr/bin/tpm2_quote" if cmd == "tpm2_quote" else None,
    ):
        # Should not raise — tpm2_checkquote is absent but not required
        driver = TPM2Driver(pcr_policy_path=None)
    assert driver.pcr_policy_path is None


def test_tpm2_driver_with_policy_requires_both_tools():
    """Driver with pcr_policy_path MUST require tpm2_checkquote."""
    with patch(
        "rune_bench.attestation.tpm2.shutil.which",
        side_effect=lambda cmd: "/usr/bin/tpm2_quote" if cmd == "tpm2_quote" else None,
    ):
        with pytest.raises(RuntimeError, match="tpm2_checkquote"):
            TPM2Driver(pcr_policy_path="/etc/pcr.policy")


# ---------------------------------------------------------------------------
# TPM2Driver — verify
# ---------------------------------------------------------------------------


def _make_tpm2_driver(pcr_policy_path: str | None = None) -> TPM2Driver:
    """Construct a TPM2Driver bypassing the tool-presence check."""
    with patch("rune_bench.attestation.tpm2.shutil.which", return_value="/usr/bin/tpm2_quote"):
        return TPM2Driver(pcr_policy_path=pcr_policy_path)


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> MagicMock:
    proc = MagicMock(spec=subprocess.CompletedProcess)
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


def test_tpm2_driver_verify_success_no_policy():
    driver = _make_tpm2_driver()
    with patch("rune_bench.attestation.tpm2.subprocess.run", return_value=_completed(0, "pcr-digest-hex")) as mock_run:
        result = driver.verify("~/.kube/config")

    assert result.passed is True
    assert result.pcr_digest == "pcr-digest-hex"
    assert "passed" in result.message
    mock_run.assert_called_once()


def test_tpm2_driver_verify_empty_stdout_gives_none_digest():
    driver = _make_tpm2_driver()
    with patch("rune_bench.attestation.tpm2.subprocess.run", return_value=_completed(0, "")):
        result = driver.verify("target")

    assert result.passed is True
    assert result.pcr_digest is None


def test_tpm2_driver_verify_quote_failure():
    driver = _make_tpm2_driver()
    with patch("rune_bench.attestation.tpm2.subprocess.run", return_value=_completed(1, "", "ERROR: device not found")):
        result = driver.verify("target")

    assert result.passed is False
    assert "tpm2_quote failed" in result.message
    assert "device not found" in result.message


def test_tpm2_driver_verify_quote_timeout():
    driver = _make_tpm2_driver()
    with patch("rune_bench.attestation.tpm2.subprocess.run", side_effect=subprocess.TimeoutExpired("tpm2_quote", 30)):
        result = driver.verify("target")

    assert result.passed is False
    assert "timed out" in result.message


def test_tpm2_driver_verify_quote_file_not_found():
    driver = _make_tpm2_driver()
    with patch("rune_bench.attestation.tpm2.subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(RuntimeError, match="tpm2_quote binary disappeared"):
            driver.verify("target")


def test_tpm2_driver_verify_success_with_policy():
    driver = _make_tpm2_driver(pcr_policy_path="/etc/rune/pcr.policy")
    quote_ok = _completed(0, "abc123")
    check_ok = _completed(0)
    with patch("rune_bench.attestation.tpm2.subprocess.run", side_effect=[quote_ok, check_ok]):
        result = driver.verify("target")

    assert result.passed is True
    assert result.pcr_digest == "abc123"


def test_tpm2_driver_verify_checkquote_failure():
    driver = _make_tpm2_driver(pcr_policy_path="/etc/rune/pcr.policy")
    quote_ok = _completed(0, "abc123")
    check_fail = _completed(2, "", "policy mismatch")
    with patch("rune_bench.attestation.tpm2.subprocess.run", side_effect=[quote_ok, check_fail]):
        result = driver.verify("target")

    assert result.passed is False
    assert "policy mismatch" in result.message
    assert result.pcr_digest == "abc123"


def test_tpm2_driver_verify_checkquote_timeout():
    driver = _make_tpm2_driver(pcr_policy_path="/etc/rune/pcr.policy")
    quote_ok = _completed(0, "abc123")
    with patch(
        "rune_bench.attestation.tpm2.subprocess.run",
        side_effect=[quote_ok, subprocess.TimeoutExpired("tpm2_checkquote", 30)],
    ):
        result = driver.verify("target")

    assert result.passed is False
    assert "tpm2_checkquote timed out" in result.message


def test_tpm2_driver_verify_checkquote_file_not_found():
    driver = _make_tpm2_driver(pcr_policy_path="/etc/rune/pcr.policy")
    quote_ok = _completed(0, "abc123")
    with patch(
        "rune_bench.attestation.tpm2.subprocess.run",
        side_effect=[quote_ok, FileNotFoundError],
    ):
        with pytest.raises(RuntimeError, match="tpm2_checkquote binary disappeared"):
            driver.verify("target")


# ---------------------------------------------------------------------------
# get_driver factory
# ---------------------------------------------------------------------------


def test_get_driver_default_is_noop():
    driver = get_driver()
    assert isinstance(driver, NoOpDriver)


def test_get_driver_explicit_noop():
    driver = get_driver({"driver": "noop"})
    assert isinstance(driver, NoOpDriver)


def test_get_driver_tpm2():
    with patch("rune_bench.attestation.tpm2.shutil.which", return_value="/usr/bin/tpm2_quote"):
        driver = get_driver({"driver": "tpm2"})
    assert isinstance(driver, TPM2Driver)


def test_get_driver_tpm2_with_pcr_policy():
    with patch("rune_bench.attestation.tpm2.shutil.which", return_value="/usr/bin/tpm2_quote"):
        driver = get_driver({"driver": "tpm2", "pcr_policy_path": "/etc/rune/pcr.policy"})
    assert isinstance(driver, TPM2Driver)
    assert driver.pcr_policy_path == "/etc/rune/pcr.policy"


def test_get_driver_unknown_raises():
    with pytest.raises(ValueError, match="Unknown attestation driver"):
        get_driver({"driver": "hsm"})


def test_get_driver_reads_env_var(monkeypatch):
    monkeypatch.setenv("RUNE_ATTESTATION_DRIVER", "noop")
    driver = get_driver()
    assert isinstance(driver, NoOpDriver)


def test_get_driver_env_var_tpm2(monkeypatch):
    monkeypatch.setenv("RUNE_ATTESTATION_DRIVER", "tpm2")
    monkeypatch.delenv("RUNE_ATTESTATION_PCR_POLICY_PATH", raising=False)
    with patch("rune_bench.attestation.tpm2.shutil.which", return_value="/usr/bin/tpm2_quote"):
        driver = get_driver()
    assert isinstance(driver, TPM2Driver)
    assert driver.pcr_policy_path is None


def test_get_driver_pcr_policy_from_env(monkeypatch):
    monkeypatch.setenv("RUNE_ATTESTATION_DRIVER", "tpm2")
    monkeypatch.setenv("RUNE_ATTESTATION_PCR_POLICY_PATH", "/env/pcr.policy")
    with patch("rune_bench.attestation.tpm2.shutil.which", return_value="/usr/bin/tpm2_quote"):
        driver = get_driver()
    assert driver.pcr_policy_path == "/env/pcr.policy"


def test_get_driver_config_overrides_env(monkeypatch):
    monkeypatch.setenv("RUNE_ATTESTATION_DRIVER", "tpm2")
    # explicit config takes precedence over env var
    driver = get_driver({"driver": "noop"})
    assert isinstance(driver, NoOpDriver)


# ---------------------------------------------------------------------------
# api_contracts — RunBenchmarkRequest.attestation_required
# ---------------------------------------------------------------------------


def test_run_benchmark_request_attestation_defaults_false():
    from rune_bench.api_contracts import RunBenchmarkRequest

    req = RunBenchmarkRequest(
        vastai=False,
        template_hash="abc",
        min_dph=1.0,
        max_dph=2.0,
        reliability=0.99,
        backend_url="http://localhost:11434",
        question="q",
        model="llama3.1:8b",
        backend_warmup=False,
        backend_warmup_timeout=60,
        kubeconfig="~/.kube/config",
        vastai_stop_instance=False,
    )
    assert req.attestation_required is False


def test_run_benchmark_request_attestation_true():
    from rune_bench.api_contracts import RunBenchmarkRequest

    req = RunBenchmarkRequest(
        vastai=False,
        template_hash="abc",
        min_dph=1.0,
        max_dph=2.0,
        reliability=0.99,
        backend_url=None,
        question="q",
        model="llama3.1:8b",
        backend_warmup=False,
        backend_warmup_timeout=60,
        kubeconfig="/k",
        vastai_stop_instance=False,
        attestation_required=True,
    )
    assert req.attestation_required is True


def test_run_benchmark_request_from_cli_attestation_default():
    from rune_bench.api_contracts import RunBenchmarkRequest

    req = RunBenchmarkRequest.from_cli(
        vastai=False,
        template_hash="abc",
        min_dph=0.0,
        max_dph=0.0,
        reliability=0.99,
        backend_url=None,
        question="q",
        model="m",
        backend_warmup=False,
        backend_warmup_timeout=60,
        kubeconfig=Path("/k"),
        vastai_stop_instance=False,
    )
    assert req.attestation_required is False


def test_run_benchmark_request_from_cli_attestation_true():
    from rune_bench.api_contracts import RunBenchmarkRequest

    req = RunBenchmarkRequest.from_cli(
        vastai=False,
        template_hash="abc",
        min_dph=0.0,
        max_dph=0.0,
        reliability=0.99,
        backend_url=None,
        question="q",
        model="m",
        backend_warmup=False,
        backend_warmup_timeout=60,
        kubeconfig=Path("/k"),
        vastai_stop_instance=False,
        attestation_required=True,
    )
    assert req.attestation_required is True


def test_run_benchmark_request_to_dict_includes_attestation():
    from rune_bench.api_contracts import RunBenchmarkRequest

    req = RunBenchmarkRequest(
        vastai=False,
        template_hash="x",
        min_dph=0.0,
        max_dph=0.0,
        reliability=0.9,
        backend_url=None,
        question="q",
        model="m",
        backend_warmup=False,
        backend_warmup_timeout=60,
        kubeconfig="/k",
        vastai_stop_instance=False,
        attestation_required=True,
    )
    d = req.to_dict()
    assert d["attestation_required"] is True


# ---------------------------------------------------------------------------
# api_backend — _verify_attestation and run_benchmark integration
# ---------------------------------------------------------------------------


def _make_benchmark_request(attestation_required: bool = False):
    from rune_bench.api_contracts import RunBenchmarkRequest

    return RunBenchmarkRequest(
        vastai=False,
        template_hash="x",
        min_dph=0.0,
        max_dph=0.0,
        reliability=0.9,
        backend_url="http://localhost:11434",
        question="q",
        model="llama3.1:8b",
        backend_warmup=False,
        backend_warmup_timeout=60,
        kubeconfig="/k",
        vastai_stop_instance=False,
        attestation_required=attestation_required,
    )


def test_verify_attestation_passes_with_noop(monkeypatch):
    monkeypatch.delenv("RUNE_ATTESTATION_DRIVER", raising=False)
    from rune_bench.api_backend import _verify_attestation

    _verify_attestation("/k")  # should not raise


def test_verify_attestation_raises_on_failure(monkeypatch):
    monkeypatch.setenv("RUNE_ATTESTATION_DRIVER", "noop")
    failing_result = AttestationResult(passed=False, pcr_digest=None, message="PCR mismatch")
    with patch("rune_bench.attestation.noop.NoOpDriver.verify", return_value=failing_result):
        from rune_bench.api_backend import _verify_attestation

        with pytest.raises(RuntimeError, match="Attestation failed"):
            _verify_attestation("/k")


def test_run_benchmark_skips_attestation_when_not_required(monkeypatch):
    monkeypatch.delenv("RUNE_ATTESTATION_DRIVER", raising=False)
    request = _make_benchmark_request(attestation_required=False)

    mock_provider = MagicMock()
    mock_provider.provision.return_value = MagicMock(
        backend_url="http://localhost:11434",
        model="llama3.1:8b",
        provider_handle="cid-123",
    )

    mock_runner = MagicMock()
    mock_runner.ask.return_value = "answer"

    with (
        patch("rune_bench.api_backend._make_resource_provider_for_benchmark", return_value=mock_provider),
        patch("rune_bench.api_backend._make_agent_runner", return_value=mock_runner),
        patch("rune_bench.api_backend._verify_attestation") as mock_verify,
    ):
        from rune_bench.api_backend import run_benchmark

        result = run_benchmark(request)

    mock_verify.assert_not_called()
    assert result["answer"] == "answer"


def test_run_benchmark_calls_attestation_when_required(monkeypatch):
    monkeypatch.delenv("RUNE_ATTESTATION_DRIVER", raising=False)
    request = _make_benchmark_request(attestation_required=True)

    mock_provider = MagicMock()
    mock_provider.provision.return_value = MagicMock(
        backend_url="http://localhost:11434",
        model="llama3.1:8b",
        provider_handle="cid-456",
    )

    mock_runner = MagicMock()
    mock_runner.ask.return_value = "ans"

    with (
        patch("rune_bench.api_backend._make_resource_provider_for_benchmark", return_value=mock_provider),
        patch("rune_bench.api_backend._make_agent_runner", return_value=mock_runner),
        patch("rune_bench.api_backend._verify_attestation") as mock_verify,
    ):
        from rune_bench.api_backend import run_benchmark

        result = run_benchmark(request)

    mock_verify.assert_called_once_with("/k")
    assert result["answer"] == "ans"


def test_run_benchmark_aborts_when_attestation_fails(monkeypatch):
    monkeypatch.delenv("RUNE_ATTESTATION_DRIVER", raising=False)
    request = _make_benchmark_request(attestation_required=True)

    with patch(
        "rune_bench.api_backend._verify_attestation",
        side_effect=RuntimeError("Attestation failed"),
    ):
        from rune_bench.api_backend import run_benchmark

        with pytest.raises(RuntimeError, match="Attestation failed"):
            run_benchmark(request)


# ---------------------------------------------------------------------------
# config — attestation section is injected into env vars
# ---------------------------------------------------------------------------


def test_load_config_injects_attestation_driver(tmp_path, monkeypatch):
    cfg_file = tmp_path / "rune.yaml"
    cfg_file.write_text(
        "version: '1'\n"
        "defaults:\n"
        "  model: llama3.1:8b\n"
        "  attestation:\n"
        "    driver: tpm2\n"
        "    pcr_policy_path: /etc/rune/pcr.policy\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RUNE_ATTESTATION_DRIVER", raising=False)
    monkeypatch.delenv("RUNE_ATTESTATION_PCR_POLICY_PATH", raising=False)

    import importlib
    import rune_bench.common.config as config_mod
    importlib.reload(config_mod)

    effective = config_mod.load_config()
    assert effective["attestation"]["driver"] == "tpm2"
    import os
    assert os.environ.get("RUNE_ATTESTATION_DRIVER") == "tpm2"
    assert os.environ.get("RUNE_ATTESTATION_PCR_POLICY_PATH") == "/etc/rune/pcr.policy"


def test_load_config_injects_top_level_attestation(tmp_path, monkeypatch):
    """Top-level attestation: section (outside defaults) must be injected."""
    cfg_file = tmp_path / "rune.yaml"
    cfg_file.write_text(
        "version: '1'\n"
        "defaults:\n"
        "  model: llama3.1:8b\n"
        "attestation:\n"
        "  driver: tpm2\n"
        "  pcr_policy_path: /top/level/pcr.policy\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RUNE_ATTESTATION_DRIVER", raising=False)
    monkeypatch.delenv("RUNE_ATTESTATION_PCR_POLICY_PATH", raising=False)

    import importlib
    import rune_bench.common.config as config_mod
    importlib.reload(config_mod)

    config_mod.load_config()
    import os
    assert os.environ.get("RUNE_ATTESTATION_DRIVER") == "tpm2"
    assert os.environ.get("RUNE_ATTESTATION_PCR_POLICY_PATH") == "/top/level/pcr.policy"


def test_load_config_defaults_attestation_overrides_top_level(tmp_path, monkeypatch):
    """attestation under defaults takes precedence over top-level attestation."""
    cfg_file = tmp_path / "rune.yaml"
    cfg_file.write_text(
        "version: '1'\n"
        "defaults:\n"
        "  attestation:\n"
        "    driver: noop\n"
        "attestation:\n"
        "  driver: tpm2\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RUNE_ATTESTATION_DRIVER", raising=False)

    import importlib
    import rune_bench.common.config as config_mod
    importlib.reload(config_mod)

    config_mod.load_config()
    import os
    assert os.environ.get("RUNE_ATTESTATION_DRIVER") == "noop"


def test_load_config_attestation_env_not_overridden_when_set(tmp_path, monkeypatch):
    cfg_file = tmp_path / "rune.yaml"
    cfg_file.write_text(
        "version: '1'\n"
        "defaults:\n"
        "  attestation:\n"
        "    driver: tpm2\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUNE_ATTESTATION_DRIVER", "noop")  # already set

    import importlib
    import rune_bench.common.config as config_mod
    importlib.reload(config_mod)

    config_mod.load_config()
    import os
    # existing env var must not be overwritten by YAML
    assert os.environ.get("RUNE_ATTESTATION_DRIVER") == "noop"
