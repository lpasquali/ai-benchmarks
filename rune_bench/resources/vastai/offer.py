# SPDX-License-Identifier: Apache-2.0
"""Block 1 — Search Offers.

Finds the best available GPU offer on Vast.ai given price and
reliability constraints, ordered by descending VRAM / DL perf.
"""

from dataclasses import dataclass

from rune_bench.resources.vastai.sdk import VastAI

from rune_bench.debug import debug_log


@dataclass
class Offer:
    offer_id: int
    total_vram_mb: int
    raw: dict


class OfferFinder:
    """Query the Vast.ai marketplace and return the single best matching offer."""

    DEFAULT_ORDER = "gpu_total_ram-,dlperf-,total_flops-"

    def __init__(self, sdk: VastAI) -> None:
        self._sdk = sdk

    def find_best(
        self,
        min_dph: float,
        max_dph: float,
        reliability: float,
    ) -> Offer:
        """Return the best offer matching constraints.

        Raises:
            RuntimeError: if the API call fails or no offers are found.
        """
        query = {
            "reliability": {"gt": str(reliability)},
            "verified": {"eq": "True"},
            "dph": {"gte": str(min_dph), "lte": str(max_dph)}
        }
        try:
            debug_log(
                f"Vast.ai API call: search_offers query={query!r} order={self.DEFAULT_ORDER!r} disable_bundling=True"
            )
            results = self._sdk.search_offers(
                query=query,
                order=self.DEFAULT_ORDER,
                disable_bundling=True,
                raw=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Vast.ai offer search failed: {exc}") from exc

        debug_log(
            f"Vast.ai API result: search_offers count={len(results) if isinstance(results, list) else '<non-list>'}"
        )

        if not isinstance(results, list) or not results:
            raise RuntimeError("No matching offers found for the given constraints.")

        best = results[0]
        offer_id = best.get("id")
        total_vram = int(best.get("gpu_total_ram", 0))

        if not offer_id or total_vram <= 0:
            raise RuntimeError("Offer response is missing id or gpu_total_ram.")

        return Offer(offer_id=offer_id, total_vram_mb=total_vram, raw=best)
