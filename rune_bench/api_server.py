"""HTTP server for RUNE API mode."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, TypeAlias
from urllib.parse import parse_qs, urlparse

from argon2 import PasswordHasher
from rune_bench.api_backend import (
    get_cost_estimate,
    list_backend_models,
    list_vastai_models,
    run_agentic_agent,
    run_benchmark,
    run_llm_instance,
)
from rune_bench.api_contracts import (
    CostEstimationRequest,
    RunAgenticAgentRequest,
    RunBenchmarkRequest,
    RunLLMInstanceRequest,
)
from rune_bench.job_store import JobRecord, JobStore
from rune_bench.metrics import SQLiteMetricsCollector, clear_collector, set_collector, set_job_id, span

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
        tenant_tokens: dict[str, str] = {}
        if tokens_raw:
            for item in tokens_raw.split(","):
                tenant, _, token = item.partition(":")
                tenant = tenant.strip()
                token = token.strip()
                if tenant and token:
                    tenant_tokens[tenant] = hashlib.sha256(token.encode("utf-8")).hexdigest()
        if not auth_disabled and not tenant_tokens:
            raise RuntimeError(
                "RUNE API auth is enabled but no tenants are configured. "
                "Set RUNE_API_TOKENS='tenant-a:token-a' or RUNE_API_AUTH_DISABLED=1 for development."
            )
        return cls(auth_disabled=auth_disabled, tenant_tokens=tenant_tokens)


class RuneApiApplication:
    def __init__(
        self,
        *,
        store: JobStore,
        security: ApiSecurityConfig,
        backend_functions: dict[str, BackendHandler] | None = None,
    ) -> None:
        self.store = store
        self.security = security
        self.backend_functions = backend_functions or {
            "agentic-agent": _run_agentic_backend,
            "benchmark": _run_benchmark_backend,
            "ollama-instance": _run_llm_instance_backend,
            "cost-estimate": _get_cost_estimate_backend,
        }
        self.store.mark_incomplete_jobs_failed()
        self.auth_failures: dict[str, list[float]] = {}
        self.auth_lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "RuneApiApplication":
        db_path = os.environ.get("RUNE_API_DB_PATH", ".rune-api/jobs.db")
        return cls(store=JobStore(Path(db_path)), security=ApiSecurityConfig.from_env())

    def create_handler(self):
        app = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):  # noqa: A003
                return

            def _write_json(self, status_code: int, payload: dict) -> None:
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _read_json(self) -> dict:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8") or "{}")
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON payload: {exc}") from exc
                if not isinstance(payload, dict):
                    raise ValueError("JSON request body must be an object")
                return payload

            def _authenticate(self) -> str:
                client_ip = self.client_address[0]
                now = time.time()
                
                with app.auth_lock:
                    failures = app.auth_failures.get(client_ip, [])
                    failures = [t for t in failures if now - t < 60]
                    app.auth_failures[client_ip] = failures
                    if len(failures) >= 10:
                        logging.warning(f"Auth blocked: Rate limit exceeded for IP {client_ip}")
                        raise PermissionError("rate limit exceeded")

                tenant_id = self.headers.get("X-Tenant-ID", "").strip() or "default"
                if app.security.auth_disabled:
                    return tenant_id

                auth_header = self.headers.get("Authorization", "").strip()
                api_key = self.headers.get("X-API-Key", "").strip()
                token = api_key
                if auth_header.lower().startswith("bearer "):
                    token = auth_header[7:].strip()

                expected_hash = app.security.tenant_tokens.get(tenant_id)
                if expected_hash:
                    actual_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
                    if hmac.compare_digest(actual_hash, expected_hash):
                        logging.info(f"Auth success: IP {client_ip} authenticated as tenant '{tenant_id}'")
                        return tenant_id
                
                with app.auth_lock:
                    app.auth_failures[client_ip].append(now)
                logging.warning(f"Auth failure: IP {client_ip} failed to authenticate as tenant '{tenant_id}'")
                raise PermissionError("invalid tenant/token combination")

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path

                if path == "/healthz":
                    self._write_json(200, {"status": "ok"})
                    return

                try:
                    tenant_id = self._authenticate()
                except PermissionError as exc:
                    self._write_json(401, {"error": str(exc)})
                    return

                if path == "/v1/catalog/vastai-models":
                    self._write_json(200, {"models": list_vastai_models()})
                    return

                if path == "/v1/ollama/models":
                    query = parse_qs(parsed.query)
                    backend_url = query.get("backend_url", [""])[0]
                    if not backend_url:
                        self._write_json(400, {"error": "missing required query parameter: backend_url"})
                        return
                    try:
                        payload = list_backend_models(backend_url)
                    except RuntimeError as exc:
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

                if path.startswith("/v1/jobs/"):
                    job_id = path.removeprefix("/v1/jobs/").strip()
                    if job_id.endswith("/events"):
                        raw_job_id = job_id.removesuffix("/events").strip()
                        job = app.store.get_job(raw_job_id, tenant_id=tenant_id)
                        if job is None:
                            self._write_json(404, {"error": f"job not found: {raw_job_id}"})
                            return
                        events = app.store.get_events_for_job(raw_job_id)
                        self._write_json(200, {"job_id": raw_job_id, "events": events})
                        return
                    job = app.store.get_job(job_id, tenant_id=tenant_id)
                    if job is None:
                        self._write_json(404, {"error": f"job not found: {job_id}"})
                        return
                    self._write_json(200, _job_to_payload(job))
                    return

                self._write_json(404, {"error": f"unknown path: {path}"})

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                try:
                    tenant_id = self._authenticate()
                except PermissionError as exc:
                    self._write_json(401, {"error": str(exc)})
                    return

                try:
                    payload = self._read_json()
                except ValueError as exc:
                    self._write_json(400, {"error": str(exc)})
                    return

                if path == "/v1/estimates":
                    try:
                        result = app._dispatch("cost-estimate", payload)
                        self._write_json(200, result)
                    except Exception as exc:
                        self._write_json(400, {"error": str(exc)})
                    return

                endpoint_to_kind = {
                    "/v1/jobs/agentic-agent": "agentic-agent",
                    "/v1/jobs/benchmark": "benchmark",
                    "/v1/jobs/ollama-instance": "ollama-instance",
                }
                kind = endpoint_to_kind.get(path)
                if not kind:
                    self._write_json(404, {"error": f"unknown path: {path}"})
                    return

                idempotency_key = self.headers.get("Idempotency-Key", "").strip() or None
                try:
                    job_id, created = app.store.create_job(
                        tenant_id=tenant_id,
                        kind=kind,
                        request_payload=payload,
                        idempotency_key=idempotency_key,
                    )
                except Exception as exc:
                    self._write_json(500, {"error": f"failed to persist job: {exc}"})
                    return

                if created:
                    thread = threading.Thread(
                        target=app._execute_job,
                        args=(job_id, kind, payload),
                        daemon=True,
                    )
                    thread.start()

                self._write_json(202, {"job_id": job_id, "status": "accepted"})

        return Handler

    def _execute_job(self, job_id: str, kind: str, payload: dict) -> None:
        set_collector(SQLiteMetricsCollector(self.store))
        set_job_id(job_id)
        self.store.update_job(job_id, status="running", message="job is running")
        try:
            with span("job.execute", kind=kind):
                result = self._dispatch(kind, payload)
        except Exception as exc:  # noqa: BLE001
            self.store.update_job(job_id, status="failed", error=str(exc), message=str(exc))
            return
        finally:
            clear_collector()
            set_job_id(None)
        self.store.update_job(job_id, status="succeeded", result_payload=result, message="job completed")

    def _dispatch(self, kind: str, payload: dict) -> dict:
        request: BackendRequest
        if kind == "agentic-agent":
            request = RunAgenticAgentRequest(**payload)
        elif kind == "benchmark":
            request = RunBenchmarkRequest(**payload)
        elif kind == "ollama-instance":
            request = RunLLMInstanceRequest(**payload)
        elif kind == "cost-estimate":
            request = CostEstimationRequest(**payload)
        else:
            raise RuntimeError(f"unsupported job kind: {kind}")

        handler = self.backend_functions.get(kind)
        if handler is None:
            raise RuntimeError(f"no backend function registered for job kind: {kind}")
        return handler(request)

    def serve(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        server = ThreadingHTTPServer((host, port), self.create_handler())
        try:
            server.serve_forever()
        finally:
            server.server_close()


def _job_to_payload(job: JobRecord) -> dict:
    payload: dict[str, object] = {
        "job_id": job.job_id,
        "tenant_id": job.tenant_id,
        "kind": job.kind,
        "status": job.status,
        "message": job.message,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }
    if job.result_payload is not None:
        payload["result"] = job.result_payload
    if job.error is not None:
        payload["error"] = job.error
    return payload
