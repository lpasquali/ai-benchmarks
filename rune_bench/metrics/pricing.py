# SPDX-License-Identifier: Apache-2.0
"""Pricing oracle for projected run costs (FinOps simulation)."""

from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class PricingProjection:
    total_cost_usd: float
    gpu_cost_usd: float
    token_cost_usd: float
    confidence: float
    historical_match: bool

class PricingSoothSayer:
    """Calculates projected costs based on historical averages and live market prices."""

    # Hardcoded commercial LLM costs (Price per 1M tokens) - April 2026 est.
    TOKEN_PRICING = {
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
        "deepseek-r1:32b": {"input": 0.0, "output": 0.0},  # Usually local/Vast
        "llama3.1:8b": {"input": 0.0, "output": 0.0},
    }

    def __init__(self, sdk: Optional[Any] = None):
        self._sdk = sdk

    async def get_live_dph(self, gpu_name: str) -> float:
        """Fetch live dph (Dollars Per Hour) from Vast.ai for a specific GPU."""
        if not self._sdk:
            # Fallback static pricing if no SDK (e.g., RTX 4090 ~ $0.40/hr)
            static_map = {"rtx4090": 0.40, "a100": 1.50, "h100": 3.50}
            return static_map.get(gpu_name.lower().replace(" ", ""), 0.50)
        
        try:
            # Simple heuristic: search for the GPU name and take the 5th percentile price
            query = f"gpu_name = {gpu_name} verified = true"
            offers = self._sdk.search_offers(query=query)
            if not offers:
                return 0.50
            
            prices = sorted([o.get("dph_total", 100.0) for o in offers])
            # Use a low-end but realistic price (5th percentile)
            idx = max(0, int(len(prices) * 0.05))
            return prices[idx]
        except Exception:
            return 0.50

    def get_token_rates(self, model: str) -> Dict[str, float]:
        """Return input/output rates per 1M tokens."""
        # Find closest match
        for key, val in self.TOKEN_PRICING.items():
            if key in model.lower():
                return val
        return {"input": 0.0, "output": 0.0}

    async def simulate(
        self, 
        agent: str, 
        model: str, 
        gpu: Optional[str] = None,
        suite: str = "standard"
    ) -> PricingProjection:
        """Simulate a run and return projected cost."""
        
        # Historical averages (Stubs - in production these would come from a DB)
        # Avg Duration (s), Avg Input (tokens), Avg Output (tokens)
        history = {
            "holmes": (180, 5000, 1200),
            "k8sgpt": (45, 2000, 800),
            "perplexity": (30, 1000, 500),
        }
        
        avg_dur, avg_input, avg_output = history.get(agent.lower(), (120, 1000, 500))
        historical_match = agent.lower() in history
        
        # GPU Cost
        dph = await self.get_live_dph(gpu or "rtx4090")
        gpu_cost = (avg_dur / 3600) * dph
        
        # Token Cost
        rates = self.get_token_rates(model)
        token_cost = ((avg_input / 1_000_000) * rates["input"]) + ((avg_output / 1_000_000) * rates["output"])
        
        total = gpu_cost + token_cost
        
        return PricingProjection(
            total_cost_usd=round(total, 4),
            gpu_cost_usd=round(gpu_cost, 4),
            token_cost_usd=round(token_cost, 4),
            confidence=0.9 if historical_match else 0.4,
            historical_match=historical_match
        )
