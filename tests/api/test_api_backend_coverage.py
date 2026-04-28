# SPDX-License-Identifier: Apache-2.0
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from rune_bench.api_backend import run_benchmark, _make_resource_provider_for_benchmark
from rune_bench.api_contracts import RunBenchmarkRequest, RunTelemetry


@pytest.mark.asyncio
async def test_run_benchmark_telemetry_no_cost():
    # Hit line 257 in api_backend.py
    req = RunBenchmarkRequest(
        model="m",
        question="q",
        backend_url="u",
        provisioning=None,
        backend_warmup=True,
        backend_warmup_timeout=10,
        kubeconfig="k",
    )

    mock_runner = MagicMock()
    mock_result = MagicMock()
    mock_result.metadata = None
    mock_result.telemetry = RunTelemetry(
        tokens=MagicMock(), latency=[], cost_estimate_usd=None
    )
    mock_result.artifacts = []
    mock_result.result_type = "text"

    mock_runner.ask_structured = AsyncMock(return_value=mock_result)

    mock_provider = MagicMock()
    # provision() must be an AsyncMock
    mock_provision_result = MagicMock()
    mock_provision_result.backend_url = "http://u"
    mock_provision_result.model = "m"
    mock_provider.provision = AsyncMock(return_value=mock_provision_result)
    mock_provider.teardown = AsyncMock()

    with patch(
        "rune_bench.api_backend._make_resource_provider_for_benchmark",
        return_value=mock_provider,
    ):
        with patch(
            "rune_bench.api_backend._make_agent_runner", return_value=mock_runner
        ):
            with patch("rune_bench.api_backend.calculate_run_cost", return_value=0.5):
                res = await run_benchmark(req)
                assert res["telemetry"]["cost_estimate_usd"] == 0.5


@pytest.mark.asyncio
async def test_run_benchmark_telemetry_missing():
    # Hit line 263 in api_backend.py
    req = RunBenchmarkRequest(
        model="m",
        question="q",
        backend_url="u",
        provisioning=None,
        backend_warmup=True,
        backend_warmup_timeout=10,
        kubeconfig="k",
    )

    mock_runner = MagicMock()
    mock_result = MagicMock()
    mock_result.metadata = None
    mock_result.telemetry = None
    mock_result.artifacts = []
    mock_result.result_type = "text"

    mock_runner.ask_structured = AsyncMock(return_value=mock_result)

    mock_provider = MagicMock()
    mock_provision_result = MagicMock()
    mock_provision_result.backend_url = "http://u"
    mock_provision_result.model = "m"
    mock_provider.provision = AsyncMock(return_value=mock_provision_result)
    mock_provider.teardown = AsyncMock()

    with patch(
        "rune_bench.api_backend._make_resource_provider_for_benchmark",
        return_value=mock_provider,
    ):
        with patch(
            "rune_bench.api_backend._make_agent_runner", return_value=mock_runner
        ):
            with patch("rune_bench.api_backend.calculate_run_cost", return_value=0.5):
                res = await run_benchmark(req)
                assert res["telemetry"]["cost_estimate_usd"] == 0.5


def test_make_resource_provider_no_backend_type():
    req = RunBenchmarkRequest(
        model="m",
        question="q",
        backend_url="http://u",
        provisioning=None,
        backend_warmup=True,
        backend_warmup_timeout=10,
        kubeconfig="k",
    )
    provider = _make_resource_provider_for_benchmark(req)
    assert provider._backend_type == "ollama"
