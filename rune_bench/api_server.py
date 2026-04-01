"""HTTP server for RUNE API mode."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

from rune_bench.api_backend import (
    list_ollama_models,
    list_vastai_models,
    run_agentic_agent,
    run_benchmark,
    run_ollama_instance,
)
from rune_bench.api_contracts import (
    RunAgenticAgentRequest,
    RunBenchmarkRequest,
    RunOllamaInstanceRequest,
)
from rune_bench.job_store import JobRecord, JobStore


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
                    tenant_tokens[tenant] = token
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
        backend_functions: dict[str, Callable[[object], dict]] | None = None,
    ) -> None:
        self.store = store
        self.security = security
        self.backend_functions = backend_functions or {
            "agentic-agent": lambda request: run_agentic_agent(request),
            "benchmark": lambda request: run_benchmark(request),
            "ollama-instance": lambda request: run_ollama_instance(request),
        }
        self.store.mark_incomplete_jobs_failed()

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
                tenant_id = self.headers.get("X-Tenant-ID", "").strip() or "default"
                if app.security.auth_disabled:
                    return tenant_id

                auth_header = self.headers.get("Authorization", "").strip()
                api_key = self.headers.get("X-API-Key", "").strip()
                token = api_key
                if auth_header.lower().startswith("bearer "):
                    token = auth_header[7:].strip()

                expected = app.security.tenant_tokens.get(tenant_id)
                if not expected or token != expected:
                    raise PermissionError("invalid tenant/token combination")
                return tenant_id

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
                    ollama_url = query.get("ollama_url", [""])[0]
                    if not ollama_url:
                        self._write_json(400, {"error": "missing required query parameter: ollama_url"})
                        return
                    try:
                        payload = list_ollama_models(ollama_url)
                    except RuntimeError as exc:
                        self._write_json(400, {"error": str(exc)})
                        return
                    self._write_json(200, payload)
                    return

                if path.startswith("/v1/jobs/"):
                    job_id = path.removeprefix("/v1/jobs/").strip()
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
        self.store.update_job(job_id, status="running", message="job is running")
        try:
            result = self._dispatch(kind, payload)
        except Exception as exc:  # noqa: BLE001
            self.store.update_job(job_id, status="failed", error=str(exc), message=str(exc))
            return
        self.store.update_job(job_id, status="succeeded", result_payload=result, message="job completed")

    def _dispatch(self, kind: str, payload: dict) -> dict:
        if kind == "agentic-agent":
            request = RunAgenticAgentRequest(**payload)
        elif kind == "benchmark":
            request = RunBenchmarkRequest(**payload)
        elif kind == "ollama-instance":
            request = RunOllamaInstanceRequest(**payload)
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
    payload = {
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
