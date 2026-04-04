"""UAT tests for cost estimation drivers.

These tests validate the accuracy and plausibility of the cost estimation logic
implemented in ``rune_bench.common.costs.CostEstimator``.  They exercise
``CostEstimator.estimate()`` directly; the HTTP wiring of ``POST /v1/estimates``
(``api_backend.get_cost_estimate``) is covered by ``tests/test_cost_estimation.py``.

They are skipped by default in CI unless ``-m uat`` is passed explicitly.
To exclude them in CI pipelines that do not set ``-m uat``, add ``-m "not uat"``
to your ``pytest`` invocation:

    # Skip UAT (default CI):
    pytest -m "not uat"

    # Run UAT explicitly:
    pytest -m uat tests/uat/test_cost_estimation_uat.py

UAT requirements (rune#39):
1. Vast.ai estimate returns a plausible total projected cost for a short run.
2. Azure returns a value from the retail API or a documented stub.
3. AWS/GCP stub rates are within 20% of official on-demand reference prices.
4. Shadow calculation for local hardware TDP vs energy rate confirms amortization logic.
5. Any test run where projected_cost_usd > $1.00 raises CostLimitExceededError
   (enforced via the uat_cost_guard fixture below).
"""

from __future__ import annotations

import asyncio
from typing import Generator
from unittest.mock import AsyncMock, MagicMock

import pytest

from rune_bench.api_contracts import CostEstimationRequest, CostEstimationResponse
from rune_bench.common.costs import CostEstimator


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class CostLimitExceededError(RuntimeError):
    """Raised when a UAT estimate exceeds the $1.00 safety ceiling."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _req(**kwargs: object) -> CostEstimationRequest:
    defaults: dict[str, object] = dict(
        vastai=False,
        aws=False,
        gcp=False,
        azure=False,
        local_hardware=False,
        min_dph=0.0,
        max_dph=0.0,
        local_tdp_watts=0.0,
        local_energy_rate_kwh=0.0,
        local_hardware_purchase_price=0.0,
        local_hardware_lifespan_years=0.0,
        model="",
        estimated_duration_seconds=3600,
    )
    defaults.update(kwargs)
    return CostEstimationRequest(**defaults)  # type: ignore[arg-type]


def _run(coro):  # type: ignore[no-untyped-def]
    """Execute a coroutine in a new event loop."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# UAT cost guard fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def uat_cost_guard(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Intercept every UAT test result and raise if projected cost > $1.00.

    Works by monkeypatching CostEstimator.estimate so the real estimate is
    captured, then checked before being returned to the test.
    """
    if "uat" not in request.keywords:
        yield
        return

    original_estimate = CostEstimator.estimate

    async def guarded_estimate(
        self: CostEstimator, req: CostEstimationRequest
    ) -> CostEstimationResponse:
        result: CostEstimationResponse = await original_estimate(self, req)
        if result.projected_cost_usd > 1.00:
            raise CostLimitExceededError(
                f"UAT safety block: projected cost ${result.projected_cost_usd:.2f} "
                f"exceeds the $1.00 ceiling (driver={result.cost_driver}). "
                "Reduce estimated_duration_seconds or dph before running UAT."
            )
        return result

    CostEstimator.estimate = guarded_estimate  # type: ignore[method-assign]
    yield
    CostEstimator.estimate = original_estimate  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# UAT: Vast.ai driver
# ---------------------------------------------------------------------------


@pytest.mark.uat
def test_uat_vastai_estimate_plausible_value() -> None:
    """Vast.ai projected cost for a 10-minute run should be between $0.01 and $1.00.

    A 10-minute benchmark at max $3.00/hr incurs a total cost of $0.50 —
    well within the plausible range for a short GPU job.
    """
    estimator = CostEstimator()
    req = _req(vastai=True, min_dph=2.0, max_dph=3.0, estimated_duration_seconds=600)
    result = _run(estimator.estimate(req))

    assert result.cost_driver == "vastai", "Driver should be 'vastai'"
    assert 0.01 <= result.projected_cost_usd <= 1.00, (
        f"Expected plausible $/hr range, got ${result.projected_cost_usd:.4f}"
    )
    assert result.confidence_score > 0.0, "Confidence score must be positive"


@pytest.mark.uat
def test_uat_vastai_estimate_uses_midpoint_rate() -> None:
    """Projected total cost uses the midpoint of min_dph and max_dph.

    15 minutes at the midpoint of $1.00–$2.00/hr ($1.50/hr) = $0.375 total.
    """
    estimator = CostEstimator()
    req = _req(vastai=True, min_dph=1.0, max_dph=2.0, estimated_duration_seconds=900)
    result = _run(estimator.estimate(req))

    # 15 minutes at midpoint $1.50/hr = $0.375
    expected = (1.0 + 2.0) / 2 * (900 / 3600)
    assert result.projected_cost_usd == pytest.approx(expected, rel=0.01)


@pytest.mark.uat
def test_uat_vastai_estimate_zero_max_dph_uses_default() -> None:
    """When max_dph is 0, the estimator should fall back to an internal default."""
    estimator = CostEstimator()
    req = _req(vastai=True, min_dph=0.0, max_dph=0.0, estimated_duration_seconds=600)
    result = _run(estimator.estimate(req))

    assert result.cost_driver == "vastai"
    assert result.projected_cost_usd > 0, "Zero max_dph must use a sensible default"


# ---------------------------------------------------------------------------
# UAT: Azure driver (retail API or documented stub)
# ---------------------------------------------------------------------------


@pytest.mark.uat
def test_uat_azure_estimate_via_mocked_retail_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Azure returns a value sourced from the retail API (mocked for UAT isolation).

    The Azure retail API at prices.azure.com requires no authentication.
    This test validates that the estimator correctly parses the 'retailPrice'
    field from the API response.
    """
    import sys
    import types

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"Items": [{"retailPrice": 3.06}]}

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)

    mock_httpx = types.ModuleType("httpx")
    mock_httpx.AsyncClient = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", mock_httpx)

    estimator = CostEstimator()
    # 10 minutes at $3.06/hr = $0.51 — under the $1.00 ceiling
    req = _req(azure=True, estimated_duration_seconds=600)
    result = _run(estimator.estimate(req))

    assert result.cost_driver == "azure"
    assert result.projected_cost_usd == pytest.approx(3.06 * (600 / 3600), rel=0.01), (
        "Estimate should match the retail API price"
    )
    assert result.confidence_score >= 0.9, "Live-API confidence must be high (≥0.9)"


@pytest.mark.uat
def test_uat_azure_estimate_falls_back_to_stub_on_api_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the Azure retail API is unreachable, a documented stub rate is used.

    The stub rate of $3.06/hr (Standard_NC6s_v3, eastus) is the documented
    fallback.  The warning field must be set to indicate the fallback.
    """
    import sys
    import types

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=Exception("connection refused"))

    mock_httpx = types.ModuleType("httpx")
    mock_httpx.AsyncClient = MagicMock(return_value=mock_client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "httpx", mock_httpx)

    estimator = CostEstimator()
    req = _req(azure=True, estimated_duration_seconds=600)
    result = _run(estimator.estimate(req))

    assert result.cost_driver == "azure"
    assert result.projected_cost_usd > 0, "Fallback stub must still return a positive value"
    assert result.warning is not None, "Warning must be set when falling back to stub"
    assert "offline" in (result.warning or "").lower() or "azure" in (result.warning or "").lower(), (
        "Warning should mention the Azure API failure"
    )


# ---------------------------------------------------------------------------
# UAT: AWS stub — within 20% of official on-demand reference price
# ---------------------------------------------------------------------------

# Official AWS p3.2xlarge (1× V100, 16 GB) on-demand price in us-east-1: ~$3.06/hr.
# RUNE stub uses $2.50/hr as a conservative mid-point estimate.
# Deviation from on-demand reference: |2.50 - 3.06| / 3.06 ≈ 18.3% — within 20%.
_AWS_REFERENCE_RATE_USD_PER_HR = 3.06   # AWS p3.2xlarge on-demand (us-east-1)
_TOLERANCE = 0.20                       # 20%


@pytest.mark.uat
def test_uat_aws_stub_within_20_percent_of_official_price() -> None:
    """AWS stub rate should be within 20% of the published on-demand price.

    Reference: AWS p3.2xlarge (1× V100, 16 GB) on-demand in us-east-1 = $3.06/hr.
    RUNE stub rate: $2.50/hr.  Deviation: ~18.3% — within the 20% tolerance.

    The test verifies the implied per-hour rate embedded in the 10-minute
    projected_cost_usd value, not a spot price (spot prices fluctuate and
    cannot be reliably validated in UAT without live credentials).
    """
    estimator = CostEstimator()
    # 10 minutes → at most ~$0.42 — well under $1.00 ceiling
    req = _req(aws=True, estimated_duration_seconds=600)
    result = _run(estimator.estimate(req))

    assert result.cost_driver == "aws"

    implied_hourly_rate = result.projected_cost_usd / (600 / 3600)
    deviation = abs(implied_hourly_rate - _AWS_REFERENCE_RATE_USD_PER_HR) / _AWS_REFERENCE_RATE_USD_PER_HR
    assert deviation <= _TOLERANCE, (
        f"AWS stub rate ${implied_hourly_rate:.2f}/hr deviates {deviation:.1%} "
        f"from on-demand reference ${_AWS_REFERENCE_RATE_USD_PER_HR:.2f}/hr (tolerance {_TOLERANCE:.0%})"
    )


# ---------------------------------------------------------------------------
# UAT: GCP stub — within 20% of official on-demand reference price
# ---------------------------------------------------------------------------

# Reference: GCP n1-standard-8 + Tesla V100 (us-central1) on-demand ≈ $2.48/hr.
# This matches the tier the $2.20/hr stub approximates.
# Deviation: |2.20 - 2.48| / 2.48 ≈ 11.3% — within 20%.
_GCP_REFERENCE_RATE_USD_PER_HR = 2.48   # GCP n1 + V100 on-demand (us-central1)
_GCP_TOLERANCE = 0.20


@pytest.mark.uat
def test_uat_gcp_stub_within_20_percent_of_official_price() -> None:
    """GCP stub rate should be within 20% of the on-demand reference price.

    Reference: GCP n1-standard-8 + Tesla V100 (us-central1) on-demand ≈ $2.48/hr.
    RUNE stub rate: $2.20/hr.  Deviation: ~11.3% — within the 20% tolerance.

    The test verifies the implied per-hour rate embedded in the 10-minute
    projected_cost_usd value against the on-demand (not spot) reference price.
    """
    estimator = CostEstimator()
    req = _req(gcp=True, estimated_duration_seconds=600)
    result = _run(estimator.estimate(req))

    assert result.cost_driver == "gcp"

    implied_hourly_rate = result.projected_cost_usd / (600 / 3600)
    deviation = abs(implied_hourly_rate - _GCP_REFERENCE_RATE_USD_PER_HR) / _GCP_REFERENCE_RATE_USD_PER_HR
    assert deviation <= _GCP_TOLERANCE, (
        f"GCP stub rate ${implied_hourly_rate:.2f}/hr deviates {deviation:.1%} "
        f"from on-demand reference ${_GCP_REFERENCE_RATE_USD_PER_HR:.2f}/hr (tolerance {_GCP_TOLERANCE:.0%})"
    )


# ---------------------------------------------------------------------------
# UAT: Local hardware — shadow calculation (TDP × energy rate + amortization)
# ---------------------------------------------------------------------------


@pytest.mark.uat
def test_uat_local_hardware_shadow_calculation_energy_only() -> None:
    """Shadow calculation: energy cost = (TDP_W / 1000) × hours × rate_per_kwh.

    Example: 300 W GPU, $0.12/kWh, 10-minute run.
      energy_kwh = (300/1000) × (600/3600) = 0.05 kWh
      energy_cost = 0.05 × 0.12 = $0.006
      projected_cost_usd = round($0.006, 2) = $0.01
    (CostEstimator rounds projected_cost_usd to 2 decimal places.)
    """
    estimator = CostEstimator()
    req = _req(
        local_hardware=True,
        local_tdp_watts=300.0,
        local_energy_rate_kwh=0.12,
        local_hardware_purchase_price=0.0,
        local_hardware_lifespan_years=0.0,
        estimated_duration_seconds=600,
    )
    result = _run(estimator.estimate(req))

    assert result.cost_driver == "local"

    expected_kwh = (300.0 / 1000) * (600 / 3600)
    assert result.local_energy_kwh == pytest.approx(expected_kwh, rel=0.01), (
        f"Energy calculation mismatch: got {result.local_energy_kwh} kWh, "
        f"expected {expected_kwh:.4f} kWh"
    )

    expected_energy_cost = round(expected_kwh * 0.12, 2)
    assert result.projected_cost_usd == pytest.approx(expected_energy_cost, abs=0.005), (
        f"Energy cost mismatch: got ${result.projected_cost_usd:.4f}, "
        f"expected ${expected_energy_cost:.4f} (rounded to 2 dp)"
    )


@pytest.mark.uat
def test_uat_local_hardware_shadow_calculation_with_amortization() -> None:
    """Shadow calculation including hardware amortization.

    Example: 300 W GPU, $0.12/kWh, $5000 GPU, 5-year lifespan, 10-minute run.
      energy_kwh = (300/1000) × (600/3600) = 0.05 kWh
      energy_cost = 0.05 × 0.12 = $0.006
      total_lifetime_hours = 5 × 365 × 24 = 43,800 hr
      amort_cost = (5000 / 43800) × (600/3600) ≈ $0.01902
      total = energy_cost + amort_cost ≈ $0.025
    """
    estimator = CostEstimator()
    req = _req(
        local_hardware=True,
        local_tdp_watts=300.0,
        local_energy_rate_kwh=0.12,
        local_hardware_purchase_price=5000.0,
        local_hardware_lifespan_years=5.0,
        estimated_duration_seconds=600,
    )
    result = _run(estimator.estimate(req))

    assert result.cost_driver == "local"

    duration_hours = 600 / 3600
    energy_kwh = (300.0 / 1000) * duration_hours
    energy_cost = energy_kwh * 0.12
    total_lifetime_hours = 5.0 * 365 * 24
    amort_cost = (5000.0 / total_lifetime_hours) * duration_hours
    expected_total = energy_cost + amort_cost

    expected_total_raw = energy_cost + amort_cost
    expected_total = round(expected_total_raw, 2)

    assert result.projected_cost_usd == pytest.approx(expected_total, abs=0.005), (
        f"Total cost with amortization mismatch: got ${result.projected_cost_usd:.4f}, "
        f"expected ${expected_total:.4f} (rounded to 2 dp)"
    )
    assert result.local_energy_kwh == pytest.approx(energy_kwh, rel=0.01)


# ---------------------------------------------------------------------------
# UAT: $1.00 ceiling enforcement
# ---------------------------------------------------------------------------


@pytest.mark.uat
def test_uat_cost_limit_exceeded_raises_error() -> None:
    """Any estimate where projected_cost_usd > $1.00 must raise CostLimitExceededError.

    This test deliberately triggers a Vast.ai estimate that exceeds the ceiling.
    The uat_cost_guard fixture (autouse) enforces this invariant.
    """
    estimator = CostEstimator()
    # 1 hour at $3.00/hr = $3.00 → will exceed the $1.00 UAT ceiling
    req = _req(vastai=True, min_dph=2.0, max_dph=3.0, estimated_duration_seconds=3600)

    with pytest.raises(CostLimitExceededError, match="exceeds the \\$1.00 ceiling"):
        _run(estimator.estimate(req))


@pytest.mark.uat
def test_uat_cost_limit_not_triggered_for_cheap_runs() -> None:
    """Estimates at or below $1.00 must pass through without raising."""
    estimator = CostEstimator()
    # 10 minutes at $2.00/hr = $0.333 → under the ceiling
    req = _req(vastai=True, min_dph=1.0, max_dph=2.0, estimated_duration_seconds=600)
    result = _run(estimator.estimate(req))

    assert result.projected_cost_usd <= 1.00
    assert result.cost_driver == "vastai"
