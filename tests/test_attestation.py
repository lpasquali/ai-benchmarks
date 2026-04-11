# SPDX-License-Identifier: Apache-2.0
"""Tests for the rune_bench.attestation module."""

from __future__ import annotations

import subprocess  # nosec  # tests require subprocess
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from rune_bench.agents.base import AgentResult
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
# TPM2Driver
# ---------------------------------------------------------------------------

def _completed(returncode: int, stdout: str = "", stderr: str = "") -> MagicMock:
    proc = MagicMock(spec=subprocess.CompletedProcess)
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc

def test_tpm2_driver_verify_success_no_policy():
    with patch("rune_bench.attestation.tpm2.shutil.which", return_value="/usr/bin/tpm2_quote"):
        driver = TPM2Driver()
    with patch("rune_bench.attestation.tpm2.subprocess.run", return_value=_completed(0, "pcr-digest-hex")) as mock_run:
        result = driver.verify("~/.kube/config")

    assert result.passed is True
    assert result.pcr_digest == "pcr-digest-hex"
    assert "passed" in result.message
    mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# api_backend integration
# ---------------------------------------------------------------------------

def _make_benchmark_request(attestation_required: bool = False):
    from rune_bench.api_contracts import RunBenchmarkRequest

    return RunBenchmarkRequest(
        provisioning=None,
        backend_url="http://localhost:11434",
        question="q",
        model="llama3.1:8b",
        backend_warmup=False,
        backend_warmup_timeout=60,
        kubeconfig="/k",
        attestation_required=attestation_required,
    )

@pytest.mark.asyncio
async def test_run_benchmark_skips_attestation_when_not_required(monkeypatch):
    monkeypatch.delenv("RUNE_ATTESTATION_DRIVER", raising=False)
    request = _make_benchmark_request(attestation_required=False)

    from rune_bench.resources.base import ProvisioningResult
    mock_provider = AsyncMock()
    mock_provider.provision.return_value = ProvisioningResult(
        backend_url="http://localhost:11434",
        model="llama3.1:8b",
        provider_handle="cid-123",
    )

    mock_runner = AsyncMock()
    mock_runner.ask_structured.return_value = AgentResult(answer="answer")

    async def mock_cost(*a, **k): return 0.0

    with (
        patch("rune_bench.api_backend._make_resource_provider_for_benchmark", return_value=mock_provider),
        patch("rune_bench.api_backend._make_agent_runner", return_value=mock_runner),
        patch("rune_bench.api_backend._verify_attestation") as mock_verify,
        patch("rune_bench.api_backend.calculate_run_cost", side_effect=mock_cost),
    ):
        from rune_bench.api_backend import run_benchmark
        result = await run_benchmark(request)

    mock_verify.assert_not_called()
    assert result["answer"] == "answer"

@pytest.mark.asyncio
async def test_run_benchmark_calls_attestation_when_required(monkeypatch):
    monkeypatch.delenv("RUNE_ATTESTATION_DRIVER", raising=False)
    request = _make_benchmark_request(attestation_required=True)

    from rune_bench.resources.base import ProvisioningResult
    mock_provider = AsyncMock()
    mock_provider.provision.return_value = ProvisioningResult(
        backend_url="http://localhost:11434",
        model="llama3.1:8b",
        provider_handle="cid-456",
    )

    mock_runner = AsyncMock()
    mock_runner.ask_structured.return_value = AgentResult(answer="ans")
    
    async def mock_cost(*a, **k): return 0.0

    with (
        patch("rune_bench.api_backend._make_resource_provider_for_benchmark", return_value=mock_provider),
        patch("rune_bench.api_backend._make_agent_runner", return_value=mock_runner),
        patch("rune_bench.api_backend._verify_attestation") as mock_verify,
        patch("rune_bench.api_backend.calculate_run_cost", side_effect=mock_cost),
    ):
        from rune_bench.api_backend import run_benchmark
        result = await run_benchmark(request)

    mock_verify.assert_called_once_with("/k")
    assert result["answer"] == "ans"
