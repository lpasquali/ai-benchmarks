# SPDX-License-Identifier: Apache-2.0
"""FinOps cost projection from historical jobs, LLM list prices, and live GPU offers."""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass
from typing import Any, Callable

from rune_bench.storage import StoragePort

# USD per 1M tokens (input, output) — indicative list prices; refresh periodically.
_LLM_USD_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-haiku": (0.25, 1.25),
}

_DEFAULT_LLM_PER_MILLION = (0.50, 1.50)

# Fallback $/hour when Vast.ai is unavailable or no matching offer (by substring).
_FALLBACK_GPU_DPH_USD: dict[str, float] = {
    "4090": 0.40,
    "a100": 1.40,
    "h100": 3.50,
    "3090": 0.25,
    "l40": 0.80,
}

_DEFAULT_ASSUMED_DURATION_S = 120.0
_DEFAULT_INPUT_TOKENS = 2048.0
_DEFAULT_OUTPUT_TOKENS = 512.0


@dataclass(frozen=True)
class PricingProjection:
    total_cost_usd: float
    gpu_cost_usd: float
    token_cost_usd: float
    confidence: float
    historical_match: bool


def _model_llm_rates(model: str) -> tuple[float, float]:
    m = model.strip().lower()
    if not m:
        return _DEFAULT_LLM_PER_MILLION
    for key, rates in _LLM_USD_PER_MILLION.items():
        if key in m or m in key:
            return rates
    return _DEFAULT_LLM_PER_MILLION


def _fallback_dph(gpu: str) -> float:
    g = gpu.strip().lower()
    for frag, dph in _FALLBACK_GPU_DPH_USD.items():
        if frag in g:
            return dph
    return 0.50


def _extract_tokens_from_result(result: dict[str, Any] | None) -> tuple[float | None, float | None]:
    """Best-effort prompt/output token counts from nested result payloads."""

    if not result:
        return None, None

    def walk(obj: Any) -> list[tuple[int, int]]:
        found: list[tuple[int, int]] = []
        if isinstance(obj, dict):
            pe = obj.get("prompt_eval_count")
            ec = obj.get("eval_count")
            if pe is not None and ec is not None:
                try:
                    found.append((int(pe), int(ec)))
                except (TypeError, ValueError):
                    pass
            pt = obj.get("prompt_tokens")
            ct = obj.get("completion_tokens")
            if pt is not None and ct is not None:
                try:
                    found.append((int(pt), int(ct)))
                except (TypeError, ValueError):
                    pass
            for v in obj.values():
                found.extend(walk(v))
        elif isinstance(obj, list):
            for v in obj:
                found.extend(walk(v))
        return found

    pairs = walk(result)
    if not pairs:
        return None, None
    # Use the last pair (often the innermost / final usage block).
    inp, out = pairs[-1]
    return float(inp), float(out)


def _job_matches_filters(
    kind: str,
    request_payload: dict[str, Any],
    *,
    agent: str,
    suite: str,
    model: str,
) -> bool:
    if kind not in ("benchmark", "agentic-agent"):
        print(f"DEBUG: Filter mismatch kind: {kind}")
        return False
    if agent:
        if kind != "agentic-agent":
            print(f"DEBUG: Filter mismatch agent-kind: {kind}")
            return False
        req_agent = request_payload.get("agent", "holmes")
        if req_agent != agent:
            print(f"DEBUG: Filter mismatch agent: req={req_agent} vs goal={agent}")
            return False
    if suite:
        if kind != "benchmark":
            print(f"DEBUG: Filter mismatch suite-kind: {kind}")
            return False
        req_suite = request_payload.get("template_hash", "")
        if req_suite != suite:
            print(f"DEBUG: Filter mismatch suite: req={req_suite} vs goal={suite}")
            return False
    if model:
        req_model = request_payload.get("model", "")
        if req_model != model:
            print(f"DEBUG: Filter mismatch model: req={req_model} vs goal={model}")
            return False
    return True


@dataclass(frozen=True)
class _HistoricalAgg:
    n: int
    avg_duration_s: float
    min_duration_s: float
    max_duration_s: float
    avg_input_tokens: float
    avg_output_tokens: float
    token_samples: int


def _aggregate(rows: list[dict[str, Any]]) -> _HistoricalAgg:
    if not rows:
        return _HistoricalAgg(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    durs: list[float] = []
    in_toks: list[float] = []
    out_toks: list[float] = []
    for row in rows:
        durs.append(float(row.get("duration_seconds", 0)))
        pi, po = _extract_tokens_from_result(row.get("result_payload"))
        if pi is not None and po is not None:
            in_toks.append(pi)
            out_toks.append(po)
    avg_d = statistics.mean(durs) if durs else 0.0
    min_d = min(durs) if durs else 0.0
    max_d = max(durs) if durs else 0.0
    if in_toks:
        return _HistoricalAgg(
            n=len(rows),
            avg_duration_s=avg_d,
            min_duration_s=min_d,
            max_duration_s=max_d,
            avg_input_tokens=statistics.mean(in_toks),
            avg_output_tokens=statistics.mean(out_toks),
            token_samples=len(in_toks),
        )
    return _HistoricalAgg(
        n=len(rows),
        avg_duration_s=avg_d,
        min_duration_s=min_d,
        max_duration_s=max_d,
        avg_input_tokens=_DEFAULT_INPUT_TOKENS,
        avg_output_tokens=_DEFAULT_OUTPUT_TOKENS,
        token_samples=0,
    )


def _vast_dph_stats(
    search_offers: Callable[..., list[dict[str, Any]]],
    gpu_substring: str,
    *,
    max_offers: int = 80,
) -> tuple[float, float, float]:
    """Return (median_dph, min_dph, max_dph) from live offers filtered by GPU name."""
    query = {"verified": {"eq": "True"}, "reliability": {"gt": "0.85"}}
    try:
        offers = search_offers(query=query, order="dph+", disable_bundling=True, raw=True)
    except Exception:
        offers = []
    if not isinstance(offers, list):
        offers = []
    gq = gpu_substring.strip().lower()
    dphs: list[float] = []
    for o in offers[:500]:
        name = str(o.get("gpu_name", "")).lower()
        if gq and gq not in name:
            continue
        raw = o.get("dph_total", o.get("dph"))
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if v > 0:
            dphs.append(v)
        if len(dphs) >= max_offers:
            break
    if not dphs:
        fb = _fallback_dph(gpu_substring)
        return fb, fb * 0.7, fb * 1.5
    return (
        statistics.median(dphs),
        min(dphs),
        max(dphs),
    )


class PricingSoothSayer:
    """Project run cost: GPU time from Vast.ai (or fallback) plus LLM token list pricing."""

    def __init__(
        self,
        store: StoragePort | None = None,
        *,
        vast_search_offers: Callable[..., list[dict[str, Any]]] | None = None,
    ) -> None:
        self._store = store
        self._vast_search_offers = vast_search_offers

    async def get_live_dph(self, gpu: str) -> float:
        """Helper: return live median DPH for a given GPU."""
        if not self._vast_search_offers:
            return _fallback_dph(gpu)
        mid, _, _ = _vast_dph_stats(self._vast_search_offers, gpu)
        return mid

    async def simulate(
        self,
        *,
        tenant_id: str = "default",
        agent: str = "",
        suite: str = "",
        gpu: str = "RTX 4090",
        model: str = "",
    ) -> dict[str, Any]:
        agent = agent.strip()
        suite = suite.strip()
        gpu = gpu.strip() or "RTX 4090"
        model = model.strip()

        raw_rows = []
        if self._store:
            raw_rows = self._store.list_jobs_for_finops(tenant_id=tenant_id, limit=2000)
        
        print(f"DEBUG: simulate tenant={tenant_id} raw_rows={len(raw_rows)}")
        matched: list[dict[str, Any]] = []
        for row in raw_rows:
            if _job_matches_filters(
                row["kind"],
                row["request_payload"],
                agent=agent,
                suite=suite,
                model=model,
            ):
                matched.append(row)

        print(f"DEBUG: matched size={len(matched)}")
        hist = _aggregate(matched)
        if hist.n == 0:
            dur_mid = _DEFAULT_ASSUMED_DURATION_S
            dur_low = _DEFAULT_ASSUMED_DURATION_S * 0.25
            dur_high = _DEFAULT_ASSUMED_DURATION_S * 3.0
            avg_in = _DEFAULT_INPUT_TOKENS
            avg_out = _DEFAULT_OUTPUT_TOKENS
            hist_note = "no_matching_history"
            confidence_str = "low"
            confidence_val = 0.4
            historical_match = False
        else:
            dur_mid = hist.avg_duration_s
            dur_low = hist.min_duration_s
            dur_high = hist.max_duration_s
            avg_in = hist.avg_input_tokens
            avg_out = hist.avg_output_tokens
            hist_note = "from_history"
            confidence_str = "high" if hist.n >= 5 and hist.token_samples >= 3 else "medium"
            confidence_val = 0.9 if confidence_str == "high" else 0.7
            historical_match = True

        in_per_m, out_per_m = _model_llm_rates(model or "local")

        def _llm_cost(inp: float, out: float) -> float:
            return (inp / 1_000_000.0) * in_per_m + (out / 1_000_000.0) * out_per_m

        # Live or fallback GPU pricing
        if self._vast_search_offers:
            dph_mid, dph_low, dph_high = _vast_dph_stats(self._vast_search_offers, gpu)
            vast_source = "live"
        else:
            fb = _fallback_dph(gpu)
            dph_mid, dph_low, dph_high = fb, fb * 0.7, fb * 1.5
            vast_source = "fallback"

        gpu_cost = (dur_mid / 3600.0) * dph_mid
        token_cost = _llm_cost(avg_in, avg_out)
        total = gpu_cost + token_cost

        return {
            "projected_cost_usd": round(total, 4),
            "cost_low_usd": round((dur_low / 3600.0) * dph_low + token_cost * 0.5, 4),
            "cost_high_usd": round((dur_high / 3600.0) * dph_high + token_cost * 2.0, 4),
            "confidence": confidence_str,
            "confidence_score": confidence_val,
            "currency": "USD",
            "historical_basis": hist_note,
            "historical_sample_count": hist.n,
            "token_samples_from_history": hist.token_samples,
            "avg_duration_seconds": round(dur_mid, 1),
            "llm_input_tokens_assumed": round(hist.avg_input_tokens, 0),
            "llm_output_tokens_assumed": round(hist.avg_output_tokens, 0),
            "vast_pricing_source": vast_source,
            "components_usd": {
                "gpu_compute": round(gpu_cost, 4),
                "llm_tokens": round(token_cost, 4),
            },
            # Compatibility with UI expectations
            "total_cost_usd": round(total, 4),
            "gpu_cost_usd": round(gpu_cost, 4),
            "token_cost_usd": round(token_cost, 4),
            "historical_match": historical_match,
        }


def make_pricing_sooth_sayer(store: StoragePort) -> PricingSoothSayer:
    """Factory to create a PricingSoothSayer with live VastAI search if API key set."""
    vast_key = os.environ.get("VAST_API_KEY", "").strip()
    if not vast_key:
        return PricingSoothSayer(store)
    try:
        from rune_bench.resources.vastai.sdk import VastAI
        sdk = VastAI(api_key=vast_key)
        return PricingSoothSayer(store, vast_search_offers=sdk.search_offers)
    except (ImportError, RuntimeError):
        return PricingSoothSayer(store)
