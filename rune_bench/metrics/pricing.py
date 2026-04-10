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
    "4090": 0.35,
    "a100": 1.40,
    "h100": 3.50,
    "3090": 0.25,
    "l40": 0.80,
}

_DEFAULT_ASSUMED_DURATION_S = 120.0
_DEFAULT_INPUT_TOKENS = 2048.0
_DEFAULT_OUTPUT_TOKENS = 512.0


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
        return False
    if agent:
        if kind != "agentic-agent":
            return False
        if request_payload.get("agent", "holmes") != agent:
            return False
    if suite:
        if kind != "benchmark":
            return False
        if request_payload.get("template_hash", "") != suite:
            return False
    if model:
        req_model = request_payload.get("model", "")
        if req_model != model:
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
        durs.append(float(row["duration_seconds"]))
        pi, po = _extract_tokens_from_result(row.get("result_payload"))
        if pi is not None and po is not None:
            in_toks.append(pi)
            out_toks.append(po)
    avg_d = statistics.mean(durs)
    min_d = min(durs)
    max_d = max(durs)
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
        store: StoragePort,
        *,
        vast_search_offers: Callable[..., list[dict[str, Any]]] | None = None,
    ) -> None:
        self._store = store
        self._vast_search_offers = vast_search_offers

    def simulate(
        self,
        *,
        tenant_id: str,
        agent: str = "",
        suite: str = "",
        gpu: str = "RTX 4090",
        model: str = "",
    ) -> dict[str, Any]:
        agent = agent.strip()
        suite = suite.strip()
        gpu = gpu.strip() or "RTX 4090"
        model = model.strip()

        raw_rows = self._store.list_jobs_for_finops(tenant_id=tenant_id, limit=2000)
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

        hist = _aggregate(matched)
        if hist.n == 0:
            dur_mid = _DEFAULT_ASSUMED_DURATION_S
            dur_low = _DEFAULT_ASSUMED_DURATION_S * 0.25
            dur_high = _DEFAULT_ASSUMED_DURATION_S * 3.0
            hist_note = "no_matching_history"
            confidence = "low"
        else:
            dur_mid = hist.avg_duration_s
            dur_low = hist.min_duration_s
            dur_high = hist.max_duration_s
            hist_note = "from_history"
            confidence = "high" if hist.n >= 5 and hist.token_samples >= 3 else "medium"

        in_per_m, out_per_m = _model_llm_rates(model or "local")

        def _llm_cost(inp: float, out: float) -> float:
            return (inp / 1_000_000.0) * in_per_m + (out / 1_000_000.0) * out_per_m

        in_mid, out_mid = hist.avg_input_tokens, hist.avg_output_tokens
        if hist.token_samples == 0:
            in_low, out_low = in_mid * 0.25, out_mid * 0.25
            in_high, out_high = in_mid * 2.5, out_mid * 2.5
        else:
            in_low, out_low = in_mid * 0.8, out_mid * 0.8
            in_high, out_high = in_mid * 1.2, out_mid * 1.2

        llm_mid = _llm_cost(in_mid, out_mid)
        llm_low = _llm_cost(in_low, out_low)
        llm_high = _llm_cost(in_high, out_high)

        if self._vast_search_offers is not None:
            dph_mid, dph_lo, dph_hi = _vast_dph_stats(self._vast_search_offers, gpu)
            vast_source = "live"
        else:
            fb = _fallback_dph(gpu)
            dph_mid, dph_lo, dph_hi = fb, fb * 0.7, fb * 1.5
            vast_source = "fallback"

        # GPU $/min from $/hour.
        dpm_mid = dph_mid / 60.0
        dpm_lo = dph_lo / 60.0
        dpm_hi = dph_hi / 60.0

        # Issue #214: (Avg Duration [minutes] * GPU $/min) + token costs.
        dur_min_mid = dur_mid / 60.0
        dur_min_lo = dur_low / 60.0
        dur_min_hi = dur_high / 60.0

        gpu_mid = dur_min_mid * dpm_mid
        gpu_low = dur_min_lo * dpm_lo
        gpu_high = dur_min_hi * dpm_hi

        total_mid = gpu_mid + llm_mid
        total_low = gpu_low + llm_low
        total_high = gpu_high + llm_high

        return {
            "currency": "USD",
            "projected_cost_usd": round(total_mid, 4),
            "cost_low_usd": round(total_low, 4),
            "cost_high_usd": round(total_high, 4),
            "confidence": confidence,
            "historical_sample_count": hist.n,
            "historical_basis": hist_note,
            "avg_duration_seconds": round(dur_mid, 3),
            "duration_low_seconds": round(dur_low, 3),
            "duration_high_seconds": round(dur_high, 3),
            "gpu": gpu,
            "gpu_dph_usd": round(dph_mid, 4),
            "gpu_dph_low_usd": round(dph_lo, 4),
            "gpu_dph_high_usd": round(dph_hi, 4),
            "vast_pricing_source": vast_source,
            "llm_model_effective": model or "default",
            "llm_input_tokens_assumed": round(in_mid, 1),
            "llm_output_tokens_assumed": round(out_mid, 1),
            "token_samples_from_history": hist.token_samples,
            "components_usd": {
                "gpu_compute": round(gpu_mid, 4),
                "llm_tokens": round(llm_mid, 4),
            },
        }


def make_pricing_sooth_sayer(store: StoragePort) -> PricingSoothSayer:
    """Factory: use Vast REST client when ``VAST_API_KEY`` is set."""
    key = os.environ.get("VAST_API_KEY", "").strip()
    if key:
        try:
            from rune_bench.resources.vastai.sdk import VastAI

            sdk = VastAI(api_key=key, raw=True)
            return PricingSoothSayer(store, vast_search_offers=sdk.search_offers)
        except Exception:
            pass
    return PricingSoothSayer(store, vast_search_offers=None)
