# SPDX-License-Identifier: Apache-2.0
"""Post-run cost calculation and finalized metrics."""

from typing import Optional
from rune_bench.api_contracts import CostEstimationRequest
from rune_bench.common.costs import CostEstimator

async def calculate_run_cost(backend: str, model: str, duration_s: int, estimator: Optional[CostEstimator] = None) -> float:
    """Calculate the estimated cost of a completed run in USD.
    
    Args:
        backend: The backend type (e.g., "vastai", "local", "azure").
        model: The model name used.
        duration_s: Total execution time in seconds.
        estimator: Optional CostEstimator instance.
        
    Returns:
        Estimated cost in USD.
    """
    if estimator is None:
        estimator = CostEstimator()
    
    # Map high-level backend strings to CostEstimationRequest flags
    request_kwargs = {
        "estimated_duration_seconds": duration_s,
        "model": model
    }
    
    if backend == "vastai":
        request_kwargs["vastai"] = True
    elif backend in ("azure", "aws", "gcp"):
        request_kwargs[backend] = True
    else:
        # Default to local hardware for cost calculation if not a known cloud
        request_kwargs["local_hardware"] = True
        request_kwargs["local_tdp_watts"] = 300.0  # Default assumption
        request_kwargs["local_energy_rate_kwh"] = 0.15
        
    try:
        response = await estimator.estimate(CostEstimationRequest(**request_kwargs))
        return response.projected_cost_usd
    except Exception:
        # Fallback to static pricing ($2.50/hr) if lookup fails
        return round((2.50 / 3600) * duration_s, 4)
