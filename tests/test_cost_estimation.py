"""Tests for cost estimation: api_backend.get_cost_estimate and common.costs.CostEstimator."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rune_bench.api_backend import get_cost_estimate
from rune_bench.api_contracts import CostEstimationRequest, CostEstimationResponse
from rune_bench.common.costs import CostEstimator


# ─── get_cost_estimate (api_backend) ─────────────────────────────────────────


def _req(**kwargs) -> CostEstimationRequest:
    defaults = dict(
        vastai=False, aws=False, gcp=False, azure=False, local_hardware=False,
        min_dph=0.0, max_dph=0.0,
        local_tdp_watts=0.0, local_energy_rate_kwh=0.0,
        local_hardware_purchase_price=0.0, local_hardware_lifespan_years=0.0,
        model="", estimated_duration_seconds=3600,
    )
    defaults.update(kwargs)
    return CostEstimationRequest(**defaults)


def test_get_cost_estimate_vastai():
    r = _req(vastai=True, min_dph=2.0, max_dph=4.0)
    out = get_cost_estimate(r)
    assert out["cost_driver"] == "vastai"
    assert out["projected_cost_usd"] == pytest.approx(3.0, rel=1e-3)
    assert out["resource_impact"] == "medium"
    assert out["local_energy_kwh"] == 0.0
    assert out["confidence_score"] == 1.0
    assert out["warning"] is None


def test_get_cost_estimate_vastai_zero_max_dph():
    r = _req(vastai=True, min_dph=2.0, max_dph=0.0)
    out = get_cost_estimate(r)
    assert out["cost_driver"] == "vastai"
    assert out["projected_cost_usd"] == pytest.approx(2.0, rel=1e-3)


def test_get_cost_estimate_local_hardware():
    r = _req(
        local_hardware=True,
        local_tdp_watts=200.0,
        local_energy_rate_kwh=0.1,
        local_hardware_purchase_price=5000.0,
        local_hardware_lifespan_years=5.0,
        estimated_duration_seconds=7200,
    )
    out = get_cost_estimate(r)
    assert out["cost_driver"] == "local"
    assert out["local_energy_kwh"] == pytest.approx(0.4, rel=1e-3)
    assert out["projected_cost_usd"] > 0.0
    assert out["resource_impact"] == "low"


def test_get_cost_estimate_local_hardware_no_lifespan():
    r = _req(
        local_hardware=True,
        local_tdp_watts=100.0,
        local_energy_rate_kwh=0.15,
        local_hardware_lifespan_years=0.0,
    )
    out = get_cost_estimate(r)
    assert out["cost_driver"] == "local"


def test_get_cost_estimate_aws():
    r = _req(aws=True, min_dph=2.0, max_dph=3.0)
    out = get_cost_estimate(r)
    assert out["cost_driver"] == "aws"
    assert out["projected_cost_usd"] == pytest.approx(2.5, rel=1e-3)


def test_get_cost_estimate_gcp():
    r = _req(gcp=True, min_dph=1.5, max_dph=2.5)
    out = get_cost_estimate(r)
    assert out["cost_driver"] == "gcp"
    assert out["projected_cost_usd"] == pytest.approx(2.0, rel=1e-3)


def test_get_cost_estimate_azure():
    r = _req(azure=True, min_dph=3.0, max_dph=5.0)
    out = get_cost_estimate(r)
    assert out["cost_driver"] == "azure"
    assert out["projected_cost_usd"] == pytest.approx(4.0, rel=1e-3)


def test_get_cost_estimate_unknown_driver():
    r = _req()
    out = get_cost_estimate(r)
    assert out["cost_driver"] == "unknown"
    assert out["projected_cost_usd"] == 0.0
    assert out["resource_impact"] == "low"


def test_get_cost_estimate_high_impact():
    r = _req(vastai=True, min_dph=15.0, max_dph=15.0)
    out = get_cost_estimate(r)
    assert out["resource_impact"] == "high"


def test_get_cost_estimate_medium_impact():
    r = _req(vastai=True, min_dph=5.0, max_dph=5.0)
    out = get_cost_estimate(r)
    assert out["resource_impact"] == "medium"


# ─── CostEstimator (common.costs) ────────────────────────────────────────────


def _estimator_req(**kwargs) -> CostEstimationRequest:
    return _req(**kwargs)


def test_cost_estimator_local():
    estimator = CostEstimator()
    r = _estimator_req(
        local_hardware=True,
        local_tdp_watts=300.0,
        local_energy_rate_kwh=0.12,
        local_hardware_purchase_price=10000.0,
        local_hardware_lifespan_years=4.0,
        estimated_duration_seconds=3600,
    )
    result = asyncio.run(estimator.estimate(r))
    assert isinstance(result, CostEstimationResponse)
    assert result.cost_driver == "local"
    assert result.local_energy_kwh > 0
    assert result.projected_cost_usd > 0


def test_cost_estimator_vastai():
    estimator = CostEstimator()
    r = _estimator_req(vastai=True, max_dph=3.0, estimated_duration_seconds=3600)
    result = asyncio.run(estimator.estimate(r))
    assert result.cost_driver == "vastai"
    assert result.projected_cost_usd == pytest.approx(3.0, rel=1e-3)


def test_cost_estimator_vastai_no_max_dph():
    estimator = CostEstimator()
    r = _estimator_req(vastai=True, max_dph=0.0, estimated_duration_seconds=3600)
    result = asyncio.run(estimator.estimate(r))
    assert result.cost_driver == "vastai"
    assert result.projected_cost_usd == pytest.approx(2.5, rel=1e-3)


def test_cost_estimator_aws():
    estimator = CostEstimator()
    r = _estimator_req(aws=True, estimated_duration_seconds=3600)
    result = asyncio.run(estimator.estimate(r))
    assert result.cost_driver == "aws"
    assert result.projected_cost_usd > 0


def test_cost_estimator_gcp():
    estimator = CostEstimator()
    r = _estimator_req(gcp=True, estimated_duration_seconds=3600)
    result = asyncio.run(estimator.estimate(r))
    assert result.cost_driver == "gcp"
    assert result.projected_cost_usd > 0


def test_cost_estimator_azure_live_api(monkeypatch):
    """Test Azure estimation with mocked HTTP response."""
    import sys
    import types

    estimator = CostEstimator()

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"Items": [{"retailPrice": 3.06}]}

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)

    mock_httpx = types.ModuleType("httpx")
    mock_httpx.AsyncClient = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", mock_httpx)

    r = _estimator_req(azure=True, estimated_duration_seconds=3600)
    result = asyncio.run(estimator.estimate(r))
    assert result.cost_driver == "azure"
    assert result.projected_cost_usd == pytest.approx(3.06, rel=1e-2)


def test_cost_estimator_azure_api_failure(monkeypatch):
    """Test Azure estimation falls back to stub when API fails."""
    import sys
    import types

    estimator = CostEstimator()

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=Exception("connection refused"))

    mock_httpx = types.ModuleType("httpx")
    mock_httpx.AsyncClient = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", mock_httpx)

    r = _estimator_req(azure=True, estimated_duration_seconds=3600)
    result = asyncio.run(estimator.estimate(r))
    assert result.cost_driver == "azure"
    assert result.warning is not None


def test_cost_estimator_no_driver():
    """CostEstimator raises FailClosedError when no cost driver is configured."""
    from rune_bench.common.costs import FailClosedError

    estimator = CostEstimator()
    r = _estimator_req()
    with pytest.raises(FailClosedError, match="No cost driver selected"):
        asyncio.run(estimator.estimate(r))
