# SPDX-License-Identifier: Apache-2.0
"""Tests for :class:`rune_bench.metrics.pricing.PricingSoothSayer` and FinOps HTTP API (#214)."""

from __future__ import annotations

import json
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

from rune_bench.api_server import ApiSecurityConfig, RuneApiApplication
from rune_bench.job_store import JobStore
from rune_bench.metrics.pricing import PricingSoothSayer, _vast_dph_stats

_API_TOKEN = "a" * 32
_SHA256_HEX = "3ba3f5f43b92602683c19aee62a20342b084dd5971ddd33808d81a328879a547"


class _MemStore:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def list_jobs_for_finops(self, *, tenant_id: str, limit: int = 2000):
        return list(self._rows[:limit])


def test_sooth_sayer_no_history_uses_defaults_and_range():
    store = _MemStore([])
    sayer = PricingSoothSayer(store, vast_search_offers=None)
    out = sayer.simulate(tenant_id="t1", gpu="RTX 4090", model="gpt-4o")
    assert out["historical_sample_count"] == 0
    assert out["confidence"] == "low"
    assert out["historical_basis"] == "no_matching_history"
    assert out["cost_low_usd"] < out["projected_cost_usd"] < out["cost_high_usd"]
    assert out["components_usd"]["gpu_compute"] >= 0
    assert out["components_usd"]["llm_tokens"] >= 0
    assert out["vast_pricing_source"] == "fallback"


def test_sooth_sayer_matches_agent_and_model():
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
    out = sayer.simulate(tenant_id="t1", agent="holmes", model="llama3.1:8b", gpu="4090")
    assert out["historical_sample_count"] == 1
    assert out["avg_duration_seconds"] == 60.0
    assert out["llm_input_tokens_assumed"] == 1000.0
    assert out["token_samples_from_history"] == 1


def test_sooth_sayer_suite_filters_benchmark_template():
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
    assert sayer.simulate(tenant_id="t1", suite="tpl-abc", model="m1")["historical_sample_count"] == 1
    assert sayer.simulate(tenant_id="t1", suite="other")["historical_sample_count"] == 0


def test_vast_dph_stats_filters_gpu_name():
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


def test_finops_simulate_http(tmp_path):
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
