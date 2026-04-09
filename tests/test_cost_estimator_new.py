import pytest
from rune_bench.api_contracts import CostEstimationRequest
from rune_bench.common.costs import CostEstimator, FailClosedError
import asyncio

def test_cost_estimator_branches():
    estimator = CostEstimator()
    
    # Empty request fails closed
    req = CostEstimationRequest()
    with pytest.raises(FailClosedError):
        asyncio.run(estimator.estimate(req))
        
    # AWS stub
    req = CostEstimationRequest(aws={"instance_type": "g5.xlarge"}, estimated_duration_seconds=3600)
    resp = asyncio.run(estimator.estimate(req))
    assert resp.cost_driver == "aws"
    assert resp.projected_cost_usd == 2.50
        
    # GCP stub
    req = CostEstimationRequest(gcp={"machine_type": "a2-highgpu-1g"}, estimated_duration_seconds=3600)
    resp = asyncio.run(estimator.estimate(req))
    assert resp.cost_driver == "gcp"
    assert resp.projected_cost_usd == 2.20

    # Azure fallback
    req = CostEstimationRequest(azure={"vm_size": "Standard_NC6s_v3"}, estimated_duration_seconds=3600)
    resp = asyncio.run(estimator.estimate(req))
    assert resp.cost_driver == "azure"

    # Local hardware
    req = CostEstimationRequest(local_hardware=True, local_tdp_watts=300, local_energy_rate_kwh=0.15, estimated_duration_seconds=3600)
    resp = asyncio.run(estimator.estimate(req))
    assert resp.cost_driver == "local"

    # Vastai variants
    req = CostEstimationRequest(vastai=True, max_dph=0.5, estimated_duration_seconds=3600)
    resp = asyncio.run(estimator.estimate(req))
    assert resp.cost_driver == "vastai"
    assert resp.projected_cost_usd == 0.5

    req = CostEstimationRequest(vastai=True, min_dph=0.2, estimated_duration_seconds=3600)
    resp = asyncio.run(estimator.estimate(req))
    assert resp.cost_driver == "vastai"
    assert resp.projected_cost_usd == 0.2

    req = CostEstimationRequest(vastai=True, min_dph=0.2, max_dph=0.4, estimated_duration_seconds=3600)
    resp = asyncio.run(estimator.estimate(req))
    assert resp.cost_driver == "vastai"
    assert resp.projected_cost_usd == 0.3
