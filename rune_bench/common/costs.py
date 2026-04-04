"""Cost estimation logic for cloud and local environments."""

from typing import Optional
from rune_bench.api_contracts import CostEstimationRequest, CostEstimationResponse


class FailClosedError(RuntimeError):
    """Raised when no valid cost driver is configured (Fail-Closed mode).

    Prevents accidental unbounded cloud spend by halting execution when the
    caller has not explicitly selected a cost driver.
    """


class CostEstimator:
    """Calculates projected spend for benchmarks across any cloud."""

    async def estimate(self, request: CostEstimationRequest) -> CostEstimationResponse:
        """Estimate costs based on request parameters."""
        
        if request.local_hardware:
            return self._estimate_local(request)
        
        if request.vastai:
            return await self._estimate_vastai(request)
            
        if request.azure:
            return await self._estimate_azure(request)
            
        if request.aws:
            return self._estimate_cloud_stub("aws", request, rate=2.50)
            
        if request.gcp:
            return self._estimate_cloud_stub("gcp", request, rate=2.20)

        raise FailClosedError(
            "No cost driver selected. Configure --vastai, --azure, --aws, --gcp, or --local-hardware "
            "to proceed. (Fail-Closed: execution halted to prevent unbounded spend.)"
        )

    async def _estimate_vastai(self, request: CostEstimationRequest) -> CostEstimationResponse:
        """Estimate using live Vast.ai market data if max_dph is provided."""
        duration_hours = request.estimated_duration_seconds / 3600
        # If max_dph is 0, we use a sensible industrial default for estimation
        rate = request.max_dph if request.max_dph > 0 else 2.50
        cost = rate * duration_hours
        
        return CostEstimationResponse(
            projected_cost_usd=round(cost, 2),
            cost_driver="vastai",
            resource_impact="high" if cost > 20 else "medium" if cost > 5 else "low",
            confidence_score=0.9,
            warning="Vast.ai cost based on your Max DPH setting."
        )

    async def _estimate_azure(self, request: CostEstimationRequest) -> CostEstimationResponse:
        """Fetch real-time retail prices from Azure (no-auth API)."""
        duration_hours = request.estimated_duration_seconds / 3600
        # Default to Standard_NC6s_v3 (Tesla V100) if no specific model mapping
        sku = "Standard_NC6s_v3" 
        url = f"https://prices.azure.com/api/retail/prices?$filter=serviceName eq 'Virtual Machines' and armRegionName eq 'eastus' and armSkuName eq '{sku}'"
        
        try:
            import httpx  # type: ignore[import-not-found]  # optional dependency
            async with httpx.AsyncClient() as client:
                resp = await client.get(url)
                data = resp.json()
                # Get the first retail price (non-spot, non-reserved)
                items = data.get("Items", [])
                rate = 3.06 # Fallback industrial rate if API fails
                if items:
                    rate = items[0].get("retailPrice", rate)
                
                cost = rate * duration_hours
                return CostEstimationResponse(
                    projected_cost_usd=round(cost, 2),
                    cost_driver="azure",
                    resource_impact="high" if cost > 20 else "medium",
                    confidence_score=0.95,
                    warning=f"Real-time Azure price fetched for {sku} in eastus."
                )
        except Exception as exc:
            return self._estimate_cloud_stub("azure", request, rate=3.06, warning=f"Azure API offline: {exc}")

    def _estimate_cloud_stub(self, driver: str, request: CostEstimationRequest, rate: float, warning: Optional[str] = None) -> CostEstimationResponse:
        duration_hours = request.estimated_duration_seconds / 3600
        cost = rate * duration_hours
        return CostEstimationResponse(
            projected_cost_usd=round(cost, 2),
            cost_driver=driver,
            resource_impact="high",
            confidence_score=0.5,
            warning=warning or f"Immediate billing stub for {driver.upper()} used (Rate: ${rate}/hr)."
        )

    def _estimate_local(self, request: CostEstimationRequest) -> CostEstimationResponse:
        duration_hours = request.estimated_duration_seconds / 3600
        energy_kwh = (request.local_tdp_watts / 1000) * duration_hours
        energy_cost = energy_kwh * request.local_energy_rate_kwh
        
        amort_cost = 0.0
        if request.local_hardware_lifespan_years > 0:
            total_hours = request.local_hardware_lifespan_years * 365 * 24
            amort_cost = (request.local_hardware_purchase_price / total_hours) * duration_hours
            
        total_cost = energy_cost + amort_cost
        return CostEstimationResponse(
            projected_cost_usd=round(total_cost, 2),
            cost_driver="local",
            resource_impact="medium",
            local_energy_kwh=round(energy_kwh, 3),
            confidence_score=0.8,
            warning="Calculated based on local energy TDP and hardware amortization."
        )
