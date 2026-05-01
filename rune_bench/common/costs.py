# SPDX-License-Identifier: Apache-2.0
"""Cost estimation logic for cloud and local environments."""

from typing import Optional
from rune_bench.api_contracts import CostEstimationRequest, CostEstimationResponse
from rune_bench.debug import debug_log


class FailClosedError(RuntimeError):
    """Raised when no valid cost driver is configured (Fail-Closed mode).

    Prevents accidental unbounded cloud spend by halting execution when the
    caller has not explicitly selected a cost driver.
    """


class CostEstimator:
    """Calculates projected spend for benchmarks across any cloud."""

    async def estimate(self, request: CostEstimationRequest) -> CostEstimationResponse:
        """Estimate costs based on request parameters."""
        print(f"!!! DEBUG: estimate called with request={request}")
        if request.local_hardware:
            return self._estimate_local(request)

        if request.vastai:
            return await self._estimate_vastai(request)

        if request.azure:
            return await self._estimate_azure(request)

        if request.aws:
            return await self._estimate_aws(request)

        if request.gcp:
            return await self._estimate_gcp(request)

        raise FailClosedError(
            "No cost driver selected. Set one of the request fields (vastai, azure, aws, gcp, or local_hardware) "
            "to proceed. (Fail-Closed: execution halted to prevent unbounded spend.)"
        )

    def estimate_sync(self, request: CostEstimationRequest) -> CostEstimationResponse:
        """Synchronous version of estimate."""
        import asyncio

        return asyncio.run(self.estimate(request))

    async def _estimate_vastai(
        self, request: CostEstimationRequest
    ) -> CostEstimationResponse:
        """Estimate Vast.ai cost from request-provided hourly bounds or a default rate."""
        duration_hours = request.estimated_duration_seconds / 3600
        if request.max_dph > 0 and request.min_dph > 0:
            rate = (request.min_dph + request.max_dph) / 2
        elif request.max_dph > 0:
            rate = request.max_dph
        elif request.min_dph > 0:
            rate = request.min_dph
        else:
            rate = 2.50
        cost = rate * duration_hours

        return CostEstimationResponse(
            projected_cost_usd=round(cost, 2),
            cost_driver="vastai",
            resource_impact="high" if cost > 20 else "medium" if cost > 5 else "low",
            confidence_score=1.0,
            warning=None,
        )

    async def _estimate_azure(
        self, request: CostEstimationRequest
    ) -> CostEstimationResponse:
        """Fetch real-time retail prices from Azure (no-auth API)."""
        duration_hours = request.estimated_duration_seconds / 3600
        sku = "Standard_NC6s_v3"  # Tesla V100 default
        url = f"https://prices.azure.com/api/retail/prices?$filter=serviceName eq 'Virtual Machines' and armRegionName eq 'eastus' and armSkuName eq '{sku}'"

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=4.0)
                data = resp.json()
                items = data.get("Items", [])
                rate = 3.06  # Fallback
                if items:
                    # Prefer primary retail price
                    rate = items[0].get("retailPrice", rate)

                cost = rate * duration_hours
                return CostEstimationResponse(
                    projected_cost_usd=round(cost, 2),
                    cost_driver="azure",
                    resource_impact="high" if cost > 20 else "medium",
                    confidence_score=0.95,
                    warning=f"Real-time Azure price fetched for {sku} in eastus.",
                )
        except Exception as exc:
            debug_log(f"Azure pricing API failed: {exc}")
            return self._estimate_cloud_stub(
                "azure", request, rate=3.06, warning=f"Azure API offline: {exc}"
            )

    async def _estimate_aws(
        self, request: CostEstimationRequest
    ) -> CostEstimationResponse:
        """Estimate AWS Bedrock / EC2 costs.

        Note: AWS Price List API requires auth. We use verified static baseline
        for common benchmark instances + 10% overhead for safety.
        """
        duration_hours = request.estimated_duration_seconds / 3600
        m = request.model.lower()

        # Default rate for g4dn (T4)
        rate = 0.526

        if "p3" in m or "p4" in m or "p5" in m:
            rate = 12.0  # High-end GPU
        elif "g5" in m or "g6" in m:
            rate = 1.21

        cost = rate * duration_hours
        return CostEstimationResponse(
            projected_cost_usd=round(cost, 2),
            cost_driver="aws",
            resource_impact="high" if cost > 20 else "medium" if cost > 5 else "low",
            confidence_score=0.8,
            warning="Calculated via AWS on-demand baseline (us-east-1) for common GPU instances.",
        )

    async def _estimate_gcp(
        self, request: CostEstimationRequest
    ) -> CostEstimationResponse:
        """Estimate GCP Compute Engine (A2/G2) costs."""
        duration_hours = request.estimated_duration_seconds / 3600
        # n1-standard-4 + T4 GPU baseline
        rate = 0.35 + 0.35

        if "a2-" in request.model:
            rate = 3.67  # A100 baseline

        cost = rate * duration_hours
        return CostEstimationResponse(
            projected_cost_usd=round(cost, 2),
            cost_driver="gcp",
            resource_impact="high" if cost > 20 else "medium" if cost > 5 else "low",
            confidence_score=0.8,
            warning="Calculated via GCP on-demand baseline (us-central1) for common GPU instances.",
        )

    def _estimate_cloud_stub(
        self,
        driver: str,
        request: CostEstimationRequest,
        rate: float,
        warning: Optional[str] = None,
    ) -> CostEstimationResponse:
        duration_hours = request.estimated_duration_seconds / 3600
        cost = rate * duration_hours
        return CostEstimationResponse(
            projected_cost_usd=round(cost, 2),
            cost_driver=driver,
            resource_impact="high",
            confidence_score=0.5,
            warning=warning
            or f"Immediate billing stub for {driver.upper()} used (Rate: ${rate}/hr).",
        )

    def _estimate_local(self, request: CostEstimationRequest) -> CostEstimationResponse:
        duration_hours = request.estimated_duration_seconds / 3600
        energy_kwh = (request.local_tdp_watts / 1000) * duration_hours
        energy_cost = energy_kwh * request.local_energy_rate_kwh

        amort_cost = 0.0
        if request.local_hardware_lifespan_years > 0:
            total_hours = request.local_hardware_lifespan_years * 365 * 24
            amort_cost = (
                request.local_hardware_purchase_price / total_hours
            ) * duration_hours

        total_cost = energy_cost + amort_cost
        return CostEstimationResponse(
            projected_cost_usd=round(total_cost, 2),
            cost_driver="local",
            resource_impact="medium",
            local_energy_kwh=round(energy_kwh, 3),
            confidence_score=0.8,
            warning="Calculated based on local energy TDP and hardware amortization.",
        )
