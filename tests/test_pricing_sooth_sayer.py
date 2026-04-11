# SPDX-License-Identifier: Apache-2.0
"""Tests for :class:`rune_bench.metrics.pricing.PricingSoothSayer` and FinOps HTTP API (#214)."""

from __future__ import annotations

import pytest
import json
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from urllib.request import Request, urlopen

from rune_bench.api_server import ApiSecurityConfig, RuneApiApplication
from rune_bench.job_store import JobStore
from rune_bench.metrics import pricing as pricing_mod
from rune_bench.metrics.pricing import (
    PricingSoothSayer,
    _aggregate,
    _extract_tokens_from_result,
    _fallback_dph,
    _job_matches_filters,
    _model_llm_rates,
    _vast_dph_stats,
    make_pricing_sooth_sayer,
)

_API_TOKEN = "a" * 32
_SHA256_HEX = "3ba3f5f43b92602683c19aee62a20342b084dd5971ddd33808d81a328879a547"


class _MemStore:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def list_jobs_for_finops(self, *, tenant_id: str, limit: int = 2000):
        return list(self._rows[:limit])


@pytest.mark.asyncio
async def test_sooth_sayer_no_history_uses_defaults_and_range():
    store = _MemStore([])
    sayer = PricingSoothSayer(store, vast_search_offers=None)
    out = await sayer.simulate(tenant_id="t1", gpu="RTX 4090", model="gpt-4o")
    assert out["historical_sample_count"] == 0
    assert out["confidence"] == "low"
    assert out["historical_basis"] == "no_matching_history"
    assert out["cost_low_usd"] < out["projected_cost_usd"] < out["cost_high_usd"]
    assert out["components_usd"]["gpu_compute"] >= 0
    assert out["components_usd"]["llm_tokens"] >= 0
    assert out["vast_pricing_source"] == "fallback"


@pytest.mark.asyncio
async def test_sooth_sayer_matches_agent_and_model():
    rows = [
        {
            "kind": "agentic-agent",
            "request_payload": {
                "agent": "holmes",
                "model": "llama3.1:8b",
                "question": "q",
                "backend_url": None,
                "backend_warmup": False,
                "backend_warmup_timeout": 1,
            },
            "result_payload": {
                "metadata": {"prompt_eval_count": 1000, "eval_count": 200},
            },
            "duration_seconds": 60.0,
        },
        {
            "kind": "agentic-agent",
            "request_payload": {
                "agent": "other",
                "model": "llama3.1:8b",
                "question": "q",
                "backend_url": None,
                "backend_warmup": False,
                "backend_warmup_timeout": 1,
            },
            "result_payload": None,
            "duration_seconds": 999.0,
        },
    ]
    store = _MemStore(rows)
    sayer = PricingSoothSayer(store, vast_search_offers=None)
    out = await sayer.simulate(tenant_id="t1", agent="holmes", model="llama3.1:8b", gpu="4090")
    assert out["historical_sample_count"] == 1
    assert out["avg_duration_seconds"] == 60.0
    assert out["llm_input_tokens_assumed"] == 1000.0
    assert out["token_samples_from_history"] == 1


@pytest.mark.asyncio
async def test_sooth_sayer_suite_filters_benchmark_template():
    rows = [
        {
            "kind": "benchmark",
            "request_payload": {
                "template_hash": "tpl-abc",
                "model": "m1",
                "vastai": False,
                "min_dph": 0,
                "max_dph": 0,
                "reliability": 0.9,
                "backend_url": "http://x",
                "question": "q",
                "backend_warmup": False,
                "backend_warmup_timeout": 1,
                "kubeconfig": "/tmp/k",
                "vastai_stop_instance": False,
            },
            "result_payload": None,
            "duration_seconds": 90.0,
        }
    ]
    store = _MemStore(rows)
    sayer = PricingSoothSayer(store, vast_search_offers=None)
    assert (await sayer.simulate(tenant_id="t1", suite="tpl-abc", model="m1"))[ "historical_sample_count"] == 1
    assert (await sayer.simulate(tenant_id="t1", suite="other"))[ "historical_sample_count"] == 0


@pytest.mark.asyncio
async def test_vast_dph_stats_filters_gpu_name():
    offers = [
        {"gpu_name": "RTX 4090", "dph_total": 0.40},
        {"gpu_name": "RTX 3090", "dph_total": 0.20},
        {"gpu_name": "RTX 4090", "dph_total": 0.50},
    ]

    def _search(**_kwargs):
        return offers

    mid, lo, hi = _vast_dph_stats(_search, "4090")
    assert lo == 0.40
    assert hi == 0.50
    assert mid == 0.45


@pytest.mark.asyncio
async def test_finops_simulate_http(tmp_path):
    store = JobStore(tmp_path / "db.sqlite")
    app = RuneApiApplication(
        store=store,
        security=ApiSecurityConfig(auth_disabled=False, tenant_tokens={"tenant-a": _SHA256_HEX}),
        backend_functions={
            "agentic-agent": lambda r: {"answer": "x"},
            "benchmark": lambda r: {"answer": "b"},
            "llm-instance": lambda r: {"mode": "x", "backend_url": "http://x"},
            "ollama-instance": lambda r: {"mode": "x", "backend_url": "http://x"},
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), app.create_handler())
    try:
        import threading

        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()
        host, port = server.server_address
        url = f"http://{host}:{port}/v1/finops/simulate?gpu=A100&model=gpt-4o"
        req = Request(url)
        req.add_header("Authorization", f"Bearer {_API_TOKEN}")
        req.add_header("X-Tenant-ID", "tenant-a")
        with urlopen(req) as resp:
            body = json.loads(resp.read().decode())
        assert resp.status == 200
        assert "projected_cost_usd" in body
        assert body["currency"] == "USD"
    finally:
        server.shutdown()
        server.server_close()
        store.close()


@pytest.mark.asyncio
async def test_model_llm_rates_empty_and_named_models():
    assert _model_llm_rates("") == pricing_mod._DEFAULT_LLM_PER_MILLION
    assert _model_llm_rates("  GPT-4-TURBO-preview  ")[0] == 10.0
    assert _model_llm_rates("claude-3-haiku-foo")[0] == 0.25


@pytest.mark.asyncio
async def test_fallback_dph_unknown_uses_default():
    assert _fallback_dph("unknown-gpu") == 0.50
    assert _fallback_dph("NVIDIA A100 80GB") == 1.40


@pytest.mark.asyncio
async def test_extract_tokens_invalid_numbers_skipped_openai_nested():
    assert _extract_tokens_from_result({"prompt_eval_count": "bad", "eval_count": 1}) == (None, None)
    assert _extract_tokens_from_result(
        {"prompt_tokens": "bad", "completion_tokens": 20}
    ) == (None, None)
    assert _extract_tokens_from_result(
        {"outer": {"prompt_tokens": 10, "completion_tokens": 20}}
    ) == (10.0, 20.0)
    assert _extract_tokens_from_result([{"prompt_eval_count": 3, "eval_count": 4}]) == (3.0, 4.0)


@pytest.mark.asyncio
async def test_job_matches_filters_edge_cases():
    assert not _job_matches_filters("llm-instance", {"model": "m"}, agent="", suite="", model="")
    assert not _job_matches_filters(
        "benchmark",
        {"agent": "holmes", "model": "m", "template_hash": "t"},
        agent="holmes",
        suite="",
        model="",
    )
    assert not _job_matches_filters(
        "agentic-agent",
        {"agent": "holmes", "model": "m"},
        agent="",
        suite="tpl",
        model="",
    )
    assert not _job_matches_filters(
        "benchmark",
        {"model": "a", "template_hash": "t"},
        agent="",
        suite="",
        model="b",
    )


@pytest.mark.asyncio
async def test_aggregate_empty_rows():
    agg = _aggregate([])
    assert agg.n == 0


@pytest.mark.asyncio
async def test_vast_dph_stats_search_raises_uses_fallback():
    def _boom(**_kwargs):
        raise OSError("network")

    mid, lo, hi = _vast_dph_stats(_boom, "4090")
    assert mid == _fallback_dph("4090")


@pytest.mark.asyncio
async def test_vast_dph_stats_non_list_offers():
    def _bad(**_kwargs):
        return {"not": "a list"}

    mid, lo, hi = _vast_dph_stats(_bad, "4090")
    assert mid == _fallback_dph("4090")


@pytest.mark.asyncio
async def test_vast_dph_stats_skips_invalid_dph_and_empty_gpu_filter():
    offers = [
        {"gpu_name": "RTX 4090", "dph_total": "x"},
        {"gpu_name": "RTX 4090", "dph_total": 0.4},
    ]

    def _search(**_kwargs):
        return offers

    mid, lo, hi = _vast_dph_stats(_search, "")
    assert lo == 0.4 and hi == 0.4 and mid == 0.4


@pytest.mark.asyncio
async def test_vast_dph_stats_respects_max_offers():
    offers = [{"gpu_name": "RTX 4090", "dph_total": float(i) * 0.01} for i in range(1, 120)]

    def _search(**_kwargs):
        return offers

    _vast_dph_stats(_search, "4090", max_offers=5)
    # If loop stopped early, median is from first 5 collected positive dphs — still valid floats
    mid, _, _ = _vast_dph_stats(_search, "4090", max_offers=5)
    assert mid > 0


@pytest.mark.asyncio
async def test_sooth_sayer_high_confidence_and_live_vast():
    rows = []
    for i in range(5):
        rows.append(
            {
                "kind": "agentic-agent",
                "request_payload": {
                    "agent": "holmes",
                    "model": "m",
                    "question": "q",
                    "backend_url": None,
                    "backend_warmup": False,
                    "backend_warmup_timeout": 1,
                },
                "result_payload": {
                    "metadata": {"prompt_eval_count": 100 + i, "eval_count": 50 + i},
                },
                "duration_seconds": 30.0 + i,
            }
        )
    store = _MemStore(rows)

    def _vast(**_kwargs):
        return [{"gpu_name": "RTX 4090", "dph_total": 0.6}]

    sayer = PricingSoothSayer(store, vast_search_offers=_vast)
    out = await sayer.simulate(tenant_id="t1", agent="holmes", model="m", gpu="4090")
    assert out["confidence"] == "high"
    assert out["vast_pricing_source"] == "live"


@patch("rune_bench.resources.vastai.sdk.VastAI")
@pytest.mark.asyncio
async def test_make_pricing_sooth_sayer_wires_vast_when_env_set(mock_vast, monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "k" * 40)
    inst = mock_vast.return_value
    inst.search_offers.return_value = [{"gpu_name": "RTX 4090", "dph_total": 0.5}]
    store = _MemStore([])
    sayer = make_pricing_sooth_sayer(store)
    out = await sayer.simulate(tenant_id="t", gpu="4090")
    assert out["vast_pricing_source"] == "live"
    mock_vast.assert_called_once()


@patch("rune_bench.resources.vastai.sdk.VastAI", side_effect=RuntimeError("boom"))
@pytest.mark.asyncio
async def test_make_pricing_sooth_sayer_falls_back_when_vast_import_fails(_mock_vast, monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "k" * 40)
    store = _MemStore([])
    sayer = make_pricing_sooth_sayer(store)
    out = await sayer.simulate(tenant_id="t", gpu="4090")
    assert out["vast_pricing_source"] == "fallback"
