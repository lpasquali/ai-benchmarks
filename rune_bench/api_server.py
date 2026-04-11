# SPDX-License-Identifier: Apache-2.0
"""HTTP server for RUNE API mode."""

from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import time
import asyncio
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, TypeAlias
from urllib.parse import parse_qs, urlparse

from rune_bench.api_backend import (
    get_cost_estimate,
    list_backend_models,
    run_agentic_agent,
    run_benchmark,
    run_llm_instance,
)
from rune_bench.api_contracts import (
    CostEstimationRequest,
    RunAgenticAgentRequest,
    RunBenchmarkRequest,
    RunLLMInstanceRequest,
    UpdateSettingsRequest,
)
from rune_bench.common import (
    get_raw_config,
    load_config,
    update_settings,
)
from rune_bench.metrics.pricing import PricingSoothSayer
from rune_bench.storage import StoragePort
from rune_bench.storage.sqlite import SQLiteStorageAdapter

_MIN_API_TOKEN_LEN = 32
_HTTP_REQUEST_SOCKET_TIMEOUT_S = 30.0

class RequestRateLimited(Exception):
    """Raised when a tenant exceeds their request rate limit."""
    pass

# Back-compat alias: legacy tests and callers still reference
# ``rune_bench.api_server.JobStore``. The class is now
# ``SQLiteStorageAdapter`` under ``rune_bench.storage.sqlite``.
JobStore = SQLiteStorageAdapter

BackendRequest: TypeAlias = (
    RunAgenticAgentRequest | RunBenchmarkRequest | RunLLMInstanceRequest | CostEstimationRequest
)
BackendHandler: TypeAlias = Callable[[BackendRequest], dict]


def _run_agentic_backend(request: BackendRequest) -> dict:
    if not isinstance(request, RunAgenticAgentRequest):
        raise RuntimeError("invalid request type for agentic-agent backend")
    return run_agentic_agent(request)


def _run_benchmark_backend(request: BackendRequest) -> dict:
    if not isinstance(request, RunBenchmarkRequest):
        raise RuntimeError("invalid request type for benchmark backend")
    return run_benchmark(request)


def _run_llm_instance_backend(request: BackendRequest) -> dict:
    if not isinstance(request, RunLLMInstanceRequest):
        raise RuntimeError("invalid request type for ollama-instance backend")
    return run_llm_instance(request)


def _get_cost_estimate_backend(request: BackendRequest) -> dict:
    if not isinstance(request, CostEstimationRequest):
        raise RuntimeError("invalid request type for cost-estimate backend")
    return get_cost_estimate(request)


@dataclass(frozen=True)
class ApiSecurityConfig:
    auth_disabled: bool
    tenant_tokens: dict[str, str]

    @classmethod
    def from_env(cls) -> "ApiSecurityConfig":
        auth_disabled = os.environ.get("RUNE_API_AUTH_DISABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
        tokens_raw = os.environ.get("RUNE_API_TOKENS", "").strip()
        
        tenant_tokens = {}
        if tokens_raw:
            for pair in tokens_raw.split(","):
                if ":" in pair:
                    tenant, token = pair.split(":", 1)
                    tenant_tokens[tenant.strip()] = token.strip()
        
        if not auth_disabled and not tenant_tokens:
            raise RuntimeError(
                "RUNE API auth is enabled but no tenants are configured. "
                f"Set RUNE_API_TOKENS='tenant-a:<{_MIN_API_TOKEN_LEN}+-char-secret>' "
                "or RUNE_API_AUTH_DISABLED=1 for development."
            )
        
        return cls(auth_disabled=auth_disabled, tenant_tokens=tenant_tokens)


class RuneApiApplication:
    """The main RUNE API application logic."""

    def __init__(
        self, 
        store: StoragePort,
        security: ApiSecurityConfig,
        backend_functions: dict[str, BackendHandler] | None = None,
    ) -> None:
        self.store = store
        self.security = security
        self.backend_functions = backend_functions or {
            "agentic-agent": _run_agentic_backend,
            "benchmark": _run_benchmark_backend,
            "ollama-instance": _run_llm_instance_backend,
            "llm-instance": _run_llm_instance_backend,
            "cost-estimate": _get_cost_estimate_backend,
        }
        self._rate_limits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "RuneApiApplication":
        config = load_config()
        security = ApiSecurityConfig.from_env()
        
        # Determine job store path
        db_url = os.environ.get("RUNE_DATABASE_URL", config.get("database_url", "sqlite:///home/ubuntu/.rune/jobs.db"))
        if db_url.startswith("sqlite:///"):
            db_path = Path(db_url[10:]).expanduser()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            store = SQLiteStorageAdapter(db_path)
        elif db_url.startswith("postgresql://") or db_url.startswith("postgres://"):
            from rune_bench.storage.postgres import PostgresStorageAdapter
            store = PostgresStorageAdapter(db_url)
        else:
            raise ValueError(f"Unsupported database URL scheme: {db_url}")
            
        return cls(store=store, security=security)

    def _enforce_request_rate_limit(self, tenant_id: str) -> None:
        now = time.time()
        with self._lock:
            history = self._rate_limits.get(tenant_id, [])
            # Keep only last 60 seconds
            history = [t for t in history if now - t < 60]
            if len(history) >= 100:  # 100 requests per minute limit
                raise RequestRateLimited(f"Tenant {tenant_id} exceeded rate limit (100 RPM)")
            history.append(now)
            self._rate_limits[tenant_id] = history

    def create_handler(self) -> type[BaseHTTPRequestHandler]:
        app = self

        class RuneApiHandler(BaseHTTPRequestHandler):
            def setup(self) -> None:
                super().setup()
                try:
                    self.request.settimeout(_HTTP_REQUEST_SOCKET_TIMEOUT_S)
                except (AttributeError, OSError):
                    pass

            def _write_json(self, code: int, data: dict | list) -> None:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(data).encode())

            def _authenticate(self) -> str | None:
                if app.security.auth_disabled:
                    return "default"
                
                auth_header = self.headers.get("Authorization", "")
                if not auth_header.startswith("Bearer "):
                    return None
                
                token = auth_header[7:].strip()
                tenant_id = self.headers.get("X-Tenant-ID", "").strip()
                
                if not tenant_id:
                    return None
                
                expected_token = app.security.tenant_tokens.get(tenant_id)
                if expected_token and hmac.compare_digest(token, expected_token):
                    return tenant_id
                
                return None

            def do_GET(self) -> None:
                tenant_id = self._authenticate()
                if not tenant_id:
                    self._write_json(401, {"error": "unauthorized"})
                    return

                parsed = urlparse(self.path)
                path = parsed.path
                
                try:
                    self._enforce_request_rate_limit(tenant_id)
                except RequestRateLimited as exc:
                    self._write_json(429, {"error": str(exc)})
                    return

                if path == "/v1/healthz":
                    self._write_json(200, {"status": "ok", "active_threads": threading.active_count()})
                    return

                if path == "/v1/catalog/models":
                    query = parse_qs(parsed.query)
                    backend_url = query.get("backend_url", [""])[0]
                    backend_type = query.get("backend_type", ["ollama"])[0]
                    if not backend_url:
                        self._write_json(400, {"error": "missing required query parameter: backend_url"})
                        return
                    try:
                        payload = list_backend_models(backend_url, backend_type=backend_type)
                    except (RuntimeError, ValueError) as exc:
                        self._write_json(400, {"error": str(exc)})
                        return
                    self._write_json(200, payload)
                    return

                if path == "/v1/metrics/summary":
                    query = parse_qs(parsed.query)
                    filter_job_id = query.get("job_id", [None])[0]
                    summary = app.store.get_events_summary(job_id=filter_job_id)
                    self._write_json(200, {"events": summary})
                    return

                if path == "/v1/finops/simulate":
                    query = parse_qs(parsed.query)
                    agent = query.get("agent", ["holmes"])[0]
                    model = query.get("model", ["llama3.1:8b"])[0]
                    gpu = query.get("gpu", ["rtx4090"])[0]
                    suite = query.get("suite", ["standard"])[0]
                    
                    try:
                        # Use a dedicated loop for the async simulation call
                        soothsayer = PricingSoothSayer(store=app.store)
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        projection = loop.run_until_complete(soothsayer.simulate(
                            tenant_id=tenant_id, agent=agent, model=model, gpu=gpu, suite=suite
                        ))
                        loop.close()
                        self._write_json(200, asdict(projection) if not isinstance(projection, dict) else projection)
                    except Exception as exc:
                        logging.exception("FinOps simulation failed")
                        self._write_json(400, {"error": str(exc)})
                    return

                if path == "/v1/settings":
                    config = get_raw_config()
                    # Redact sensitive fields
                    if "profiles" in config:
                        for p in config["profiles"].values():
                            if "api_token" in p:
                                p["api_token"] = "[REDACTED]"
                    self._write_json(200, config)
                    return

                if path.startswith("/v1/runs/") and path.endswith("/trace"):
                    run_id = path.split("/")[3]
                    job = app.store.get_job(run_id)
                    if not job:
                        self._write_json(404, {"error": "job not found"})
                        return
                    
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    
                    last_event_id = 0
                    while True:
                        events = app.store.get_events_for_job(run_id, after_id=last_event_id)
                        for e in events:
                            data = json.dumps(e)
                            try:
                                self.wfile.write(f"data: {data}\n\n".encode())
                                self.wfile.flush()
                                last_event_id = max(last_event_id, e.get("id", 0))
                            except (ConnectionResetError, BrokenPipeError):
                                return
                        
                        if job.status in ("succeeded", "failed", "cancelled"):
                            self.wfile.write(b"event: end\ndata: {}\n\n")
                            return
                        
                        time.sleep(1)
                    return

                if path.startswith("/v1/jobs/"):
                    job_id = path.split("/")[-1]
                    job = app.store.get_job(job_id)
                    if not job:
                        self._write_json(404, {"error": "job not found"})
                        return
                    self._write_json(200, asdict(job))
                    return

                self._write_json(404, {"error": "not found"})

            def do_POST(self) -> None:
                tenant_id = self._authenticate()
                if not tenant_id:
                    self._write_json(401, {"error": "unauthorized"})
                    return

                parsed = urlparse(self.path)
                path = parsed.path
                
                try:
                    self._enforce_request_rate_limit(tenant_id)
                except RequestRateLimited as exc:
                    self._write_json(429, {"error": str(exc)})
                    return

                length = int(self.headers.get("Content-Length", 0))
                if length > 10 * 1024 * 1024:  # 10MB limit
                    self._write_json(413, {"error": "request too large"})
                    return
                
                body = self.wfile.read(length) if length > 0 else b"{}"
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    self._write_json(400, {"error": "invalid JSON"})
                    return

                if path == "/v1/settings":
                    req = UpdateSettingsRequest(**data)
                    update_settings(req)
                    self._write_json(200, {"status": "updated"})
                    return

                if path.startswith("/v1/jobs/"):
                    kind = path.split("/")[-1]
                    handler = app.backend_functions.get(kind)
                    if not handler:
                        self._write_json(404, {"error": f"unknown job kind: {kind}"})
                        return
                    
                    # Create job in store
                    job_id, _ = app.store.create_job(tenant_id=tenant_id, kind=kind, request_payload=data)
                    
                    # Run in background
                    threading.Thread(
                        target=app._execute_job,
                        args=(job_id, handler, kind, data),
                        daemon=True
                    ).start()
                    
                    self._write_json(202, {"job_id": job_id, "status": "accepted"})
                    return

                self._write_json(404, {"error": "not found"})

        return RuneApiHandler

    def _execute_job(self, job_id: str, handler: BackendHandler, kind: str, payload: dict) -> None:
        self.store.update_job_status(job_id, "running")
        try:
            # Map payload to contract object
            if kind == "agentic-agent":
                req = RunAgenticAgentRequest.from_dict(payload)
            elif kind == "benchmark":
                req = RunBenchmarkRequest.from_dict(payload)
            elif kind in ("ollama-instance", "llm-instance"):
                req = RunLLMInstanceRequest.from_dict(payload)
            else:
                raise ValueError(f"unsupported kind: {kind}")

            # Execute
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(handler(req))
            loop.close()
            
            self.store.complete_job(job_id, "succeeded", result_payload=result)
        except Exception as exc:
            logging.exception("Job %s failed", job_id)
            self.store.complete_job(job_id, "failed", error_message=str(exc))

    def serve(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        server = ThreadingHTTPServer((host, port), self.create_handler())
        server.timeout = _HTTP_REQUEST_SOCKET_TIMEOUT_S
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
