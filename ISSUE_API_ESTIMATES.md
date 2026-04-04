# ISSUE: Extend `rune-api` with `/v1/estimates` (Cost Projection)

## 1. Overview
As part of the `rune-ui` initiative, the `rune` core must be extended to provide cost estimation logic for all benchmarks *before* they are executed. This ensures that both the CLI and UI can provide consistent "Pre-Flight Spend Alerts."

## 2. Core Requirements
- **Unified Logic:** Centralize all cost calculation logic in the `rune_bench` core to avoid duplication in the UI/BFF.
- **Endpoint:** Implement `POST /v1/estimates` in `rune_bench/api_server.py`.
- **Request Payload:** The endpoint should accept the same payload as `RunBenchmarkRequest`.

## 3. Estimation Drivers
### A. Cloud (Immediate Billing)
- **Vast.ai:** Query the current market rates (`min_dph`, `max_dph`) and model VRAM impact to project spend.
- **Generic Stubs:** Implement stubs for **AWS, GCP, and Azure** for immediate billing calculations in future drivers.

### B. Local Hardware (Airgapped)
- **Energy Consumption:** Model cost based on GPU/CPU TDP, local electricity rates ($/kWh), and estimated run duration.
- **Amortization:** Optional "Hardware Lifetime" modeling to project capital depreciation.

## 4. Response Payload
The endpoint must return a structured estimation:
```json
{
  "projected_cost_usd": 12.50,
  "cost_driver": "vastai",
  "resource_impact": "high",
  "local_energy_kwh": 4.2,
  "confidence_score": 0.95,
  "warning": "Selected max_dph is higher than average for this template."
}
```

## 5. Tasks
- [ ] Implement the `CostEstimator` in `rune_bench/common/costs.py`.
- [ ] Add the `POST /v1/estimates` endpoint to the API server.
- [ ] Update `api_contracts.py` with the new Request/Response schemas.
- [ ] Add unit tests with 100% coverage for the estimation logic.
