# SPDX-License-Identifier: Apache-2.0
"""Tests for preflight cost-check paths: CLI behaviour and api_client validation."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

import rune as rune_cli
from rune_bench.api_client import RuneApiClient
from rune_bench.common.costs import CostEstimator, FailClosedError
from rune_bench.workflows import run_preflight_cost_check, SpendGateAction
from rune_bench.api_contracts import CostEstimationResponse


# ─── run_preflight_cost_check (workflows) ─────────────────────────────────────

@pytest.mark.asyncio
async def test_preflight_returns_empty_when_not_vastai():
    result = await run_preflight_cost_check(vastai=False, max_dph=3.0, min_dph=2.0)
    assert result == {}

@pytest.mark.asyncio
async def test_preflight_local_backend_returns_estimate():
    mock_resp = MagicMock(spec=CostEstimationResponse)
    mock_resp.to_dict.return_value = {"projected_cost_usd": 1.0, "cost_driver": "vastai"}
    
    with patch("rune_bench.common.costs.CostEstimator.estimate", new_callable=AsyncMock) as mock_est:
        mock_est.return_value = mock_resp
        result = await run_preflight_cost_check(
            vastai=True,
            max_dph=3.0,
            min_dph=2.0,
            estimated_duration_seconds=3600,
            backend_mode="local",
        )
        assert result["cost_driver"] == "vastai"
        assert result["projected_cost_usd"] == 1.0

@pytest.mark.asyncio
async def test_preflight_http_backend_calls_client(monkeypatch):
    mock_client = MagicMock()
    mock_client.get_cost_estimate.return_value = {
        "projected_cost_usd": 4.5,
        "cost_driver": "vastai",
        "resource_impact": "medium",
        "warning": None,
    }
    result = await run_preflight_cost_check(
        vastai=True,
        max_dph=3.0,
        min_dph=2.0,
        backend_mode="http",
        http_client=mock_client,
    )
    assert result["projected_cost_usd"] == 4.5
    mock_client.get_cost_estimate.assert_called_once()

@pytest.mark.asyncio
async def test_preflight_raises_runtime_error_missing_http_client():
    with pytest.raises(RuntimeError, match="http_client is required"):
        await run_preflight_cost_check(
            vastai=True,
            max_dph=3.0,
            min_dph=2.0,
            backend_mode="http",
            http_client=None,
        )

@pytest.mark.asyncio
async def test_preflight_re_raises_fail_closed_error():
    with patch("rune_bench.common.costs.CostEstimator.estimate", side_effect=FailClosedError("closed")):
        with pytest.raises(FailClosedError, match="closed"):
            await run_preflight_cost_check(
                vastai=True,
                max_dph=3.0,
                min_dph=2.0,
                backend_mode="local",
            )


# ─── CLI _run_preflight_cost_check behaviour ──────────────────────────────────

@pytest.mark.asyncio
async def test_cli_preflight_spend_gate_aborted(monkeypatch):
    async def mock_preflight(**_):
        return {"projected_cost_usd": 100.0, "cost_driver": "vastai", "warning": None}
    
    with patch("rune.run_preflight_cost_check", side_effect=mock_preflight):
        # Patch evaluate_spend_gate to return BLOCK
        monkeypatch.setattr("rune.evaluate_spend_gate", lambda *a, **k: SpendGateAction.BLOCK)
        with pytest.raises(rune_cli.typer.Exit):
            await rune_cli._run_preflight_cost_check(vastai=True, max_dph=3.0, min_dph=2.0, yes=False)

@pytest.mark.asyncio
async def test_cli_preflight_spend_gate_accepted(monkeypatch):
    async def mock_preflight(**_):
        return {"projected_cost_usd": 100.0, "cost_driver": "vastai", "warning": None}

    with patch("rune.run_preflight_cost_check", side_effect=mock_preflight):
        monkeypatch.setattr("rune.evaluate_spend_gate", lambda *a, **k: SpendGateAction.ALLOW)
        await rune_cli._run_preflight_cost_check(vastai=True, max_dph=3.0, min_dph=2.0, yes=True)

@pytest.mark.asyncio
async def test_cli_preflight_unavailable_warning(monkeypatch):
    async def mock_err(*a, **k): raise RuntimeError("API error")
    with patch("rune.run_preflight_cost_check", side_effect=mock_err):
        # If yes=True, should log and return
        await rune_cli._run_preflight_cost_check(vastai=True, max_dph=3.0, min_dph=2.0, yes=True)
        
        # If yes=False, should raise Exit(1)
        with pytest.raises(rune_cli.typer.Exit):
            await rune_cli._run_preflight_cost_check(vastai=True, max_dph=3.0, min_dph=2.0, yes=False)

@pytest.mark.asyncio
async def test_cli_preflight_fail_closed_error(monkeypatch):
    async def mock_fail(*a, **k): raise FailClosedError("No driver")
    with patch("rune.run_preflight_cost_check", side_effect=mock_fail):
        with pytest.raises(rune_cli.typer.Exit):
            await rune_cli._run_preflight_cost_check(vastai=True, max_dph=3.0, min_dph=2.0, yes=False)

@pytest.mark.asyncio
async def test_cli_preflight_not_vastai_noop():
    # If vastai=False, it should return early without calling run_preflight_cost_check
    with patch("rune.run_preflight_cost_check", side_effect=AsyncMock(return_value={})) as mock_run:
        await rune_cli._run_preflight_cost_check(vastai=False, max_dph=3.0, min_dph=2.0, yes=True)
        mock_run.assert_not_called()
