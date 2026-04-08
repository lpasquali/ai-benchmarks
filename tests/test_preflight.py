# SPDX-License-Identifier: Apache-2.0
"""Tests for preflight cost-check paths: CLI behaviour and api_client validation."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from rune_bench.api_client import RuneApiClient
from rune_bench.common.costs import CostEstimator, FailClosedError
from rune_bench.workflows import run_preflight_cost_check


# ─── run_preflight_cost_check (workflows) ─────────────────────────────────────


def test_preflight_returns_empty_when_not_vastai():
    result = run_preflight_cost_check(vastai=False, max_dph=3.0, min_dph=2.0)
    assert result == {}


def test_preflight_local_backend_returns_estimate():
    result = run_preflight_cost_check(
        vastai=True,
        max_dph=3.0,
        min_dph=2.0,
        estimated_duration_seconds=3600,
        backend_mode="local",
    )
    assert result["cost_driver"] == "vastai"
    assert "projected_cost_usd" in result


def test_preflight_http_backend_calls_client(monkeypatch):
    mock_client = MagicMock()
    mock_client.get_cost_estimate.return_value = {
        "projected_cost_usd": 4.5,
        "cost_driver": "vastai",
        "resource_impact": "medium",
        "warning": None,
    }
    result = run_preflight_cost_check(
        vastai=True,
        max_dph=3.0,
        min_dph=2.0,
        backend_mode="http",
        http_client=mock_client,
    )
    assert result["projected_cost_usd"] == 4.5
    mock_client.get_cost_estimate.assert_called_once()


def test_preflight_http_backend_requires_client():
    with pytest.raises(RuntimeError, match="http_client is required"):
        run_preflight_cost_check(
            vastai=True,
            max_dph=3.0,
            min_dph=2.0,
            backend_mode="http",
            http_client=None,
        )


def test_preflight_raises_fail_closed_error():
    """FailClosedError propagates when no cost driver is configured in CostEstimator."""

    # Patch CostEstimator.estimate to raise FailClosedError
    with patch("rune_bench.common.costs.CostEstimator.estimate", side_effect=FailClosedError("no driver")):
        with pytest.raises(FailClosedError):
            run_preflight_cost_check(
                vastai=True,
                max_dph=3.0,
                min_dph=2.0,
                backend_mode="local",
            )


# ─── CLI _run_preflight_cost_check behaviour ──────────────────────────────────


def test_cli_preflight_decline_exits_1(monkeypatch, capsys):
    """User declining the confirmation prompt exits with code 1 (not 0)."""
    import typer
    import rune as rune_cli

    # Patch run_preflight_cost_check to return a cost above default threshold
    monkeypatch.setattr(
        rune_cli,
        "run_preflight_cost_check",
        lambda **_kw: {
            "projected_cost_usd": 99.0,
            "cost_driver": "vastai",
            "resource_impact": "high",
            "warning": None,
        },
    )
    # User types 'n'
    monkeypatch.setattr(rune_cli.console, "input", lambda _prompt: "n")
    # Make sure we're not in CI
    monkeypatch.delenv("CI", raising=False)

    with pytest.raises(typer.Exit) as exc_info:
        rune_cli._run_preflight_cost_check(vastai=True, max_dph=3.0, min_dph=2.0, yes=False)

    assert exc_info.value.exit_code == 1


def test_cli_preflight_estimation_failure_with_yes_continues(monkeypatch):
    """With --yes, a RuntimeError during estimation warns but does not exit."""
    import rune as rune_cli

    monkeypatch.setattr(
        rune_cli,
        "run_preflight_cost_check",
        lambda **_kw: (_ for _ in ()).throw(RuntimeError("server offline")),
    )

    # Should not raise
    rune_cli._run_preflight_cost_check(vastai=True, max_dph=3.0, min_dph=2.0, yes=True)


def test_cli_preflight_estimation_failure_without_yes_exits_1(monkeypatch):
    """Without --yes, a RuntimeError during estimation exits with code 1."""
    import typer
    import rune as rune_cli

    monkeypatch.setattr(
        rune_cli,
        "run_preflight_cost_check",
        lambda **_kw: (_ for _ in ()).throw(RuntimeError("estimation unavailable")),
    )

    with pytest.raises(typer.Exit) as exc_info:
        rune_cli._run_preflight_cost_check(vastai=True, max_dph=3.0, min_dph=2.0, yes=False)

    assert exc_info.value.exit_code == 1


# ─── api_client.get_cost_estimate cost_driver validation ──────────────────────


def test_get_cost_estimate_raises_for_none_cost_driver(monkeypatch):
    """get_cost_estimate raises FailClosedError when cost_driver is 'none'."""
    from rune_bench.common.costs import FailClosedError
    client = RuneApiClient("http://api:8080")
    monkeypatch.setattr(
        client,
        "_request",
        lambda *_a, **_kw: {"projected_cost_usd": 4.5, "cost_driver": "none"},
    )
    with pytest.raises(FailClosedError, match="cost_driver="):
        client.get_cost_estimate({})


def test_get_cost_estimate_raises_for_unknown_cost_driver(monkeypatch):
    """get_cost_estimate raises FailClosedError when cost_driver is 'unknown'."""
    from rune_bench.common.costs import FailClosedError
    client = RuneApiClient("http://api:8080")
    monkeypatch.setattr(
        client,
        "_request",
        lambda *_a, **_kw: {"projected_cost_usd": 4.5, "cost_driver": "unknown"},
    )
    with pytest.raises(FailClosedError, match="cost_driver="):
        client.get_cost_estimate({})


def test_get_cost_estimate_raises_for_missing_cost_driver(monkeypatch):
    """get_cost_estimate raises FailClosedError when cost_driver is absent."""
    from rune_bench.common.costs import FailClosedError
    client = RuneApiClient("http://api:8080")
    monkeypatch.setattr(
        client,
        "_request",
        lambda *_a, **_kw: {"projected_cost_usd": 4.5},
    )
    with pytest.raises(FailClosedError, match="cost_driver="):
        client.get_cost_estimate({})


def test_get_cost_estimate_accepts_valid_cost_driver(monkeypatch):
    """get_cost_estimate succeeds when cost_driver is a real driver name."""
    client = RuneApiClient("http://api:8080")
    monkeypatch.setattr(
        client,
        "_request",
        lambda *_a, **_kw: {"projected_cost_usd": 3.0, "cost_driver": "vastai"},
    )
    result = client.get_cost_estimate({})
    assert result["cost_driver"] == "vastai"


# ─── FailClosedError message ──────────────────────────────────────────────────


def test_fail_closed_error_message_uses_local_hardware():
    """FailClosedError message references --local-hardware, not --local."""
    from rune_bench.api_contracts import CostEstimationRequest

    req = CostEstimationRequest(
        vastai=False, aws=False, gcp=False, azure=False, local_hardware=False,
        min_dph=0.0, max_dph=0.0,
        local_tdp_watts=0.0, local_energy_rate_kwh=0.0,
        local_hardware_purchase_price=0.0, local_hardware_lifespan_years=0.0,
        model="", estimated_duration_seconds=3600,
    )
    estimator = CostEstimator()
    with pytest.raises(FailClosedError, match="local_hardware"):
        asyncio.run(estimator.estimate(req))


# ─── evaluate_spend_gate CI blocking behavior ─────────────────────────────────


def test_evaluate_spend_gate_blocks_in_ci(monkeypatch):
    """With CI=1 and cost above threshold, evaluate_spend_gate returns BLOCK."""
    from rune_bench.workflows import SpendGateAction, evaluate_spend_gate

    monkeypatch.setenv("CI", "1")
    action = evaluate_spend_gate(99.0, threshold=5.0, yes=False)
    assert action is SpendGateAction.BLOCK


def test_evaluate_spend_gate_allows_with_yes_in_ci(monkeypatch):
    """With --yes, evaluate_spend_gate returns ALLOW even in CI above threshold."""
    from rune_bench.workflows import SpendGateAction, evaluate_spend_gate

    monkeypatch.setenv("CI", "1")
    action = evaluate_spend_gate(99.0, threshold=5.0, yes=True)
    assert action is SpendGateAction.ALLOW


def test_cli_preflight_ci_block_exits_1(monkeypatch):
    """In CI mode with spend above threshold, CLI exits with code 1."""
    import typer
    import rune as rune_cli

    monkeypatch.setattr(
        rune_cli,
        "run_preflight_cost_check",
        lambda **_kw: {
            "projected_cost_usd": 99.0,
            "cost_driver": "vastai",
            "resource_impact": "high",
            "warning": None,
        },
    )
    monkeypatch.setenv("CI", "1")
    monkeypatch.delenv("RUNE_SPEND_WARNING_THRESHOLD", raising=False)

    with pytest.raises(typer.Exit) as exc_info:
        rune_cli._run_preflight_cost_check(vastai=True, max_dph=3.0, min_dph=2.0, yes=False)

    assert exc_info.value.exit_code == 1


def test_cli_preflight_ci_yes_bypasses_block(monkeypatch):
    """In CI with --yes, spend block is bypassed and function returns normally."""
    import rune as rune_cli

    monkeypatch.setattr(
        rune_cli,
        "run_preflight_cost_check",
        lambda **_kw: {
            "projected_cost_usd": 99.0,
            "cost_driver": "vastai",
            "resource_impact": "high",
            "warning": None,
        },
    )
    monkeypatch.setenv("CI", "1")
    # Should not raise
    rune_cli._run_preflight_cost_check(vastai=True, max_dph=3.0, min_dph=2.0, yes=True)
