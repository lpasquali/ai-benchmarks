# SPDX-License-Identifier: Apache-2.0
"""HTTP server for RUNE API mode."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, TypeAlias
from urllib.parse import parse_qs, urlparse

import structlog

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
    CreateProfileRequest,
    RunAgenticAgentRequest,
    RunBenchmarkRequest,
    RunLLMInstanceRequest,
    SettingsResponse,
    UpdateSettingsRequest,
)
from rune_bench.common import (
    create_profile,
    get_raw_config,
    load_config,
    peek_profile_from_argv,
    update_settings,
)
from rune_bench.metrics import SQLiteMetricsCollector, clear_collector, set_collector, set_job_id, span
from rune_bench.storage import StoragePort, make_storage, resolve_storage_url
from rune_bench.storage.sqlite import JobRecord, SQLiteStorageAdapter

# Back-compat alias: legacy tests and callers still reference
# ``rune_bench.api_server.JobStore``. The class is now
# ``SQLiteStorageAdapter`` under ``rune_bench.storage.sqlite``.
JobStore = SQLiteStorageAdapter

BackendRequest: TypeAlias = (
    RunAgenticAgentRequest | RunBenchmarkRequest | RunLLMInstanceRequest | CostEstimationRequest
)
BackendHandler: TypeAlias = Callable[[BackendRequest], dict]

# SR-Q-005: token-bucket — burst 20, sustained 100 requests / 60s
_REQUEST_RATE_BUCKET_CAPACITY = 20.0
_REQUEST_RATE_REFILL_PER_SEC = 100.0 / 60.0

_MIN_API_TOKEN_LEN = 32  # SR-Q-016 / SR-001 (256-bit secret as printable string)

# SR-Q-008: bound time waiting for request bytes on each accepted connection
_HTTP_REQUEST_SOCKET_TIMEOUT_S = float(os.environ.get("RUNE_API_REQUEST_SOCKET_TIMEOUT", "30"))

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    cache_logger_on_first_use=True,
)

_audit_log = structlog.get_logger("rune.api.audit")


class RequestRateLimited(Exception):
    """Raised when per-IP API request rate exceeds SR-Q-005 budget."""


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
                    if len(token) < _MIN_API_TOKEN_LEN:
                        raise RuntimeError(
                            f"RUNE API token for tenant {tenant!r} must be at least "
                            f"{_MIN_API_TOKEN_LEN} characters (SR-Q-016)."
                        )
                    tenant_tokens[tenant] = hashlib.sha256(token.encode("utf-8")).hexdigest()
        if not auth_disabled and not tenant_tokens:
            raise RuntimeError(
                "RUNE API auth is enabled but no tenants are configured. "
                f"Set RUNE_API_TOKENS='tenant-a:<{_MIN_API_TOKEN_LEN}+-char-secret>' "
                "or RUNE_API_AUTH_DISABLED=1 for development."
            )
        return cls(auth_disabled=auth_disabled, tenant_tokens=tenant_tokens)


class RuneApiApplication:
    def __init__(
        self,
        *,
        store: StoragePort,
        security: ApiSecurityConfig,
        backend_functions: dict[str, BackendHandler] | None = None,
    ) -> None:
        self.store = store
        self.security = security
        self.backend_functions = backend_functions or {
            "agentic-agent": _run_agentic_backend,
            "benchmark": _run_benchmark_backend,
            "llm-instance": _run_llm_instance_backend,
            "ollama-instance": _run_llm_instance_backend,  # deprecated alias
            "cost-estimate": _get_cost_estimate_backend,
        }
        self.store.mark_incomplete_jobs_failed()
        self.auth_failures: dict[str, list[float]] = {}
        self.auth_lock = threading.Lock()
        self._request_rate_buckets: dict[str, tuple[float, float]] = {}

    def _consume_api_request_budget(self, client_ip: str) -> None:
        """SR-Q-005: token-bucket per IP (burst 20, refill 100/min)."""
        now = time.time()
        with self.auth_lock:
            if client_ip not in self._request_rate_buckets:
                self._request_rate_buckets[client_ip] = (_REQUEST_RATE_BUCKET_CAPACITY, now)
            tokens, last_ts = self._request_rate_buckets[client_ip]
            elapsed = now - last_ts
            tokens = min(
                _REQUEST_RATE_BUCKET_CAPACITY,
                tokens + elapsed * _REQUEST_RATE_REFILL_PER_SEC,
            )
            if tokens < 1.0:
                self._request_rate_buckets[client_ip] = (tokens, now)
                raise RequestRateLimited("too many requests")
            tokens -= 1.0
            self._request_rate_buckets[client_ip] = (tokens, now)

    @classmethod
    def from_env(cls, *, db_url: str | None = None) -> "RuneApiApplication":
        resolved_db_url = resolve_storage_url(db_url)
        return cls(
            store=make_storage(resolved_db_url),
            security=ApiSecurityConfig.from_env(),
        )

    def create_handler(self):
        app = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):  # noqa: A003
                return

            def setup(self) -> None:
                super().setup()
                try:
                    self.request.settimeout(_HTTP_REQUEST_SOCKET_TIMEOUT_S)
                except (AttributeError, OSError):
                    pass

            def _write_json(self, status_code: int, payload: dict) -> None:
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _read_json(self) -> dict:
                length = int(self.headers.get("Content-Length", "0"))
                MAX_BODY_SIZE = 10 * 1024 * 1024  # 10 MiB (SR-Q-004)
                if length > MAX_BODY_SIZE:
                    raise ValueError(f"request body exceeds maximum size ({MAX_BODY_SIZE // 1024 // 1024} MiB)")
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
                        _audit_log.warning(
                            "auth_rate_limited",
                            client_ip=client_ip,
                            reason="too_many_failed_attempts",
                            threshold=10,
                            window_seconds=60,
                        )
                        raise PermissionError("rate limit exceeded")

                tenant_id = self.headers.get("X-Tenant-ID", "").strip() or "default"
                if app.security.auth_disabled:
                    return tenant_id

                auth_header = self.headers.get("Authorization", "").strip()
                api_key = self.headers.get("X-API-Key", "").strip()
                token = api_key
                if auth_header.lower().startswith("bearer "):
                    token = auth_header[7:].strip()

                if len(token) < _MIN_API_TOKEN_LEN:
                    with app.auth_lock:
                        app.auth_failures[client_ip].append(now)
                    _audit_log.warning(
                        "auth_failure_token_too_short",
                        client_ip=client_ip,
                        tenant_id=tenant_id,
                        endpoint=self.path,
                        min_length=_MIN_API_TOKEN_LEN,
                    )
                    raise PermissionError("invalid tenant/token combination")

                expected_hash = app.security.tenant_tokens.get(tenant_id)
                if expected_hash:
                    actual_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
                    if hmac.compare_digest(actual_hash, expected_hash):
                        _audit_log.info(
                            "auth_success",
                            client_ip=client_ip,
                            tenant_id=tenant_id,
                            endpoint=self.path,
                            auth_result="success",
                        )
                        return tenant_id

                with app.auth_lock:
                    app.auth_failures[client_ip].append(now)
                _audit_log.warning(
                    "auth_failure_invalid_credentials",
                    client_ip=client_ip,
                    tenant_id=tenant_id,
                    endpoint=self.path,
                    auth_result="failure",
                )
                raise PermissionError("invalid tenant/token combination")

            def _enforce_request_rate_limit(self, path: str) -> None:
                if path == "/healthz":
                    return
                app._consume_api_request_budget(self.client_address[0])

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path

                if path == "/healthz":
                    self._write_json(
                        200,
                        {
                            "status": "ok",
                            "active_threads": threading.active_count(),
                        },
                    )
                    return

                try:
                    self._enforce_request_rate_limit(path)
                except RequestRateLimited as exc:
                    self._write_json(429, {"error": str(exc)})
                    return

                try:
                    tenant_id = self._authenticate()
                except PermissionError as exc:
                    self._write_json(401, {"error": str(exc)})
                    return

                if path == "/v1/settings":
                    raw = get_raw_config()
                    active_profile = peek_profile_from_argv()
                    effective = load_config(active_profile)
                    
                    def redact(d: dict) -> dict:
                        """Recursively redact sensitive keys (token, key, secret)."""
                        redacted = {}
                        for k, v in d.items():
                            if isinstance(v, dict):
                                redacted[k] = redact(v)
                            elif any(s in k.lower() for s in ("token", "key", "secret")):
                                redacted[k] = "[REDACTED]"
                            else:
                                redacted[k] = v
                        return redacted

                    response = SettingsResponse(
                        defaults=redact(raw.get("defaults") or {}),
                        profiles=redact(raw.get("profiles") or {}),
                        active_profile=active_profile,
                        effective_config=redact(effective),
                    )
                    self._write_json(200, response.to_dict())
                    return

                if path == "/v1/catalog/vastai-models":
                    self._write_json(200, {"models": list_vastai_models()})
                    return

                if path in ("/v1/llm/models", "/v1/ollama/models"):
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

                if path.startswith("/v1/chains/") and path.endswith("/state"):
                    run_id = path.removeprefix("/v1/chains/").removesuffix("/state").strip()
                    if not run_id:
                        self._write_json(400, {"error": "missing run_id in /v1/chains/{run_id}/state"})
                        return
                    job = app.store.get_job(run_id, tenant_id=tenant_id)
                    if job is None:
                        self._write_json(404, {"error": f"chain run not found: {run_id}"})
                        return
                    state = app.store.get_chain_state(run_id)
                    if state is None:
                        # Job exists but chain state hasn't been initialized yet — return an
                        # empty shell so the dashboard can render a placeholder.
                        self._write_json(
                            200,
                            {
                                "run_id": run_id,
                                "nodes": [],
                                "edges": [],
                                "overall_status": "pending",
                            },
                        )
                        return
                    self._write_json(200, {"run_id": run_id, **state})
                    return

                if path.startswith("/v1/audits/") and "/artifacts" in path:
                    # Two shapes:
                    #   /v1/audits/{run_id}/artifacts            → list metadata
                    #   /v1/audits/{run_id}/artifacts/{aid}      → stream bytes
                    rest = path.removeprefix("/v1/audits/")
                    if "/artifacts" not in rest:
                        self._write_json(404, {"error": f"unknown path: {path}"})
                        return
                    run_id, _, tail = rest.partition("/artifacts")
                    run_id = run_id.strip()
                    if not run_id:
                        self._write_json(400, {"error": "missing run_id in /v1/audits/{run_id}/artifacts"})
                        return
                    job = app.store.get_job(run_id, tenant_id=tenant_id)
                    if job is None:
                        self._write_json(404, {"error": f"audit run not found: {run_id}"})
                        return

                    if tail in ("", "/"):
                        # List metadata
                        artifacts = app.store.list_audit_artifacts(run_id)
                        for a in artifacts:
                            a["download_url"] = (
                                f"/v1/audits/{run_id}/artifacts/{a['artifact_id']}"
                            )
                        kinds_present = sorted({a["kind"] for a in artifacts})
                        self._write_json(
                            200,
                            {
                                "run_id": run_id,
                                "artifacts": artifacts,
                                "summary": {
                                    "total_count": len(artifacts),
                                    "kinds_present": kinds_present,
                                },
                            },
                        )
                        return

                    # tail is "/<artifact_id>"
                    artifact_id = tail.lstrip("/").strip()
                    if not artifact_id:
                        self._write_json(400, {"error": "missing artifact_id"})
                        return
                    record = app.store.get_audit_artifact(
                        job_id=run_id, artifact_id=artifact_id
                    )
                    if record is None:
                        self._write_json(404, {"error": f"artifact not found: {artifact_id}"})
                        return
                    content, name, kind = record
                    content_type = _audit_artifact_content_type(kind)
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(content)))
                    self.send_header(
                        "Content-Disposition",
                        f'attachment; filename="{name}"',
                    )
                    self.end_headers()
                    self.wfile.write(content)
                    return

                self._write_json(404, {"error": f"unknown path: {path}"})

            def do_PUT(self) -> None:
                self._handle_update()

            def do_PATCH(self) -> None:
                self._handle_update()

            def _handle_update(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                try:
                    _ = self._authenticate()
                except PermissionError as exc:
                    self._write_json(401, {"error": str(exc)})
                    return

                try:
                    payload = self._read_json()
                except ValueError as exc:
                    self._write_json(400, {"error": str(exc)})
                    return

                if path == "/v1/settings":
                    try:
                        req = UpdateSettingsRequest(**payload)
                        path_updated = update_settings(req.settings, req.profile)
                        # Reload config for the current process to pick up changes
                        load_config(peek_profile_from_argv())
                        self._write_json(200, {
                            "status": "updated",
                            "profile": req.profile or "defaults",
                            "file": str(path_updated),
                        })
                    except Exception as exc:
                        self._write_json(400, {"error": str(exc)})
                    return
                self._write_json(404, {"error": f"unknown path: {path}"})

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                try:
                    self._enforce_request_rate_limit(path)
                except RequestRateLimited as exc:
                    self._write_json(429, {"error": str(exc)})
                    return

                try:
                    tenant_id = self._authenticate()
                except PermissionError as exc:
                    self._write_json(401, {"error": str(exc)})
                    return

                try:
                    payload = self._read_json()
                except ValueError as exc:
                    error_msg = str(exc)
                    # SR-Q-004: Return 413 for request size violations
                    if "exceeds maximum size" in error_msg:
                        self._write_json(413, {"error": error_msg})
                    else:
                        self._write_json(400, {"error": error_msg})
                    return

                if path == "/v1/settings/profiles":
                    try:
                        req = CreateProfileRequest(**payload)
                        path_updated = create_profile(req.name, req.settings)
                        # Reload config for the current process to pick up changes
                        load_config(peek_profile_from_argv())
                        self._write_json(201, {
                            "status": "created",
                            "profile": req.name,
                            "file": str(path_updated),
                        })
                    except Exception as exc:
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
                    "/v1/jobs/llm-instance": "llm-instance",
                    "/v1/jobs/ollama-instance": "llm-instance",  # deprecated alias
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
        elif kind in ("llm-instance", "ollama-instance"):
            request = RunLLMInstanceRequest(**payload)
        elif kind == "cost-estimate":
            request = CostEstimationRequest(**payload)
        else:
            raise RuntimeError(f"unsupported job kind: {kind}")

        handler = self.backend_functions.get(kind)
        if handler is None:
            raise RuntimeError(f"no backend function registered for job kind: {kind}")
        return handler(request)

    def serve(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        server = ThreadingHTTPServer((host, port), self.create_handler())
        server.timeout = _HTTP_REQUEST_SOCKET_TIMEOUT_S
        try:
            server.serve_forever()
        finally:
            server.server_close()


_AUDIT_ARTIFACT_CONTENT_TYPES: dict[str, str] = {
    "slsa_provenance": "application/json",
    "sbom": "application/json",
    "tla_report": "text/plain; charset=utf-8",
    "rekor_entry": "application/json",
    "sigstore_bundle": "application/octet-stream",
    "tpm_attestation": "application/octet-stream",
}


def _audit_artifact_content_type(kind: str) -> str:
    """Map an audit artifact kind to its HTTP Content-Type for download responses."""
    return _AUDIT_ARTIFACT_CONTENT_TYPES.get(kind, "application/octet-stream")


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
