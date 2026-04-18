# SPDX-License-Identifier: Apache-2.0
import asyncio
import hmac
import inspect
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from rune_bench import debug_pprof
from rune_bench.api_backend import (
    get_cost_estimate,
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
from rune_bench.common.backend_utils import (
    list_backend_models,
)
from rune_bench.common.config import create_profile, get_raw_config, load_config, update_settings
from rune_bench.metrics.pricing import PricingSoothSayer
from rune_bench.storage.sqlite import SQLiteStorageAdapter

JobStore = SQLiteStorageAdapter
_HTTP_REQUEST_SOCKET_TIMEOUT_S = 30.0


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured audit logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Merge extra context if present
        if hasattr(record, "tenant_id"):
            log_data["tenant_id"] = record.tenant_id
        if hasattr(record, "job_id"):
            log_data["job_id"] = record.job_id
            
        return json.dumps(log_data)


def setup_logging(level: int = logging.INFO, json_format: bool = True) -> None:
    """Configure the root logger for RUNE."""
    root = logging.getLogger()
    # Clear existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        
    handler = logging.StreamHandler()
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        
    root.addHandler(handler)
    root.setLevel(level)


class RequestRateLimited(RuntimeError):
    """Raised when a tenant exceeds the allowed request rate."""


def _job_to_payload(job: object) -> dict:
    """Helper to convert a job database record to an API payload."""
    return {
        "job_id": str(getattr(job, "job_id", job)),
        "status": str(getattr(job, "status", "pending")),
        "kind": str(getattr(job, "kind", "unknown")),
        "created_at": str(getattr(job, "created_at", "")),
        "updated_at": str(getattr(job, "updated_at", "")),
        "result": getattr(job, "result_payload", None),
        "error": getattr(job, "error", None),
    }


def _audit_artifact_content_type(kind: str) -> str:
    """Return an appropriate Content-Type for an audit artifact kind."""
    if kind == "screenshot":
        return "image/png"
    if kind in ("sbom", "slsa_provenance", "rekor_entry"):
        return "application/json"
    if kind == "tla_report":
        return "text/plain; charset=utf-8"
    if kind == "log":
        return "text/plain"
    return "application/octet-stream"


async def _run_agentic_backend(request: object) -> dict:
    if not isinstance(request, RunAgenticAgentRequest):
        raise RuntimeError("invalid request type for agentic-agent backend")
    return await run_agentic_agent(request)


async def _run_benchmark_backend(request: object) -> dict:
    if not isinstance(request, RunBenchmarkRequest):
        raise RuntimeError("invalid request type for benchmark backend")
    return await run_benchmark(request)


async def _run_llm_instance_backend(request: object) -> dict:
    if not isinstance(request, RunLLMInstanceRequest):
        raise RuntimeError("invalid request type for ollama-instance backend")
    return await run_llm_instance(request)


async def _get_cost_estimate_backend(request: object) -> dict:
    if not isinstance(request, CostEstimationRequest):
        raise RuntimeError("invalid request type for cost-estimate backend")
    return await get_cost_estimate(request)


class ApiSecurityConfig:
    auth_disabled: bool
    tenant_tokens: dict[str, str]

    def __init__(self, auth_disabled: bool, tenant_tokens: dict[str, str]) -> None:
        object.__setattr__(self, "auth_disabled", auth_disabled)
        object.__setattr__(self, "tenant_tokens", tenant_tokens)

    @classmethod
    def from_env(cls) -> "ApiSecurityConfig":
        auth_disabled = os.environ.get("RUNE_API_AUTH_DISABLED", "").lower() in ("1", "true", "yes", "on")
        raw_tokens = os.environ.get("RUNE_API_TOKENS", "").strip()
        
        tenant_tokens: dict[str, str] = {}
        if raw_tokens:
            for pair in raw_tokens.split(","):
                if ":" in pair:
                    tenant, secret = pair.split(":", 1)
                    secret = secret.strip()
                    if len(secret) < 32:
                        raise RuntimeError(
                            f"RUNE API token for tenant '{tenant.strip()}' is too short "
                            f"({len(secret)} chars); minimum length is 32 characters."
                        )
                    tenant_tokens[tenant.strip()] = secret
        
        if not auth_disabled and not tenant_tokens:
            raise RuntimeError(
                "RUNE API auth is enabled but no tenants are configured. "
                "Set RUNE_API_TOKENS='tenant-a:<secret>' "
                "or RUNE_API_AUTH_DISABLED=1 for development."
            )
        
        return cls(auth_disabled=auth_disabled, tenant_tokens=tenant_tokens)


class RuneApiApplication:
    def __init__(
        self, 
        store: SQLiteStorageAdapter, 
        security: ApiSecurityConfig,
        backend_functions: dict[str, object] | None = None
    ):
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
            history = [t for t in history if now - t < 60]
            if len(history) >= 100:
                raise RequestRateLimited("rate limit exceeded")
            history.append(now)
            self._rate_limits[tenant_id] = history

    async def _dispatch(self, kind: str, payload: dict) -> dict:
        """Internal synchronous dispatch for local/unit testing of job logic."""
        supported_kinds = ("agentic-agent", "benchmark", "ollama-instance", "llm-instance", "cost-estimate")
        if kind not in supported_kinds:
            raise RuntimeError(f"unsupported job kind: {kind}")

        handler = self.backend_functions.get(kind)
        if not handler:
            raise RuntimeError(f"no backend function registered for: {kind}")

        if kind == "agentic-agent":
            req = RunAgenticAgentRequest.from_dict(payload) if hasattr(RunAgenticAgentRequest, "from_dict") else RunAgenticAgentRequest(**payload)
        elif kind == "benchmark":
            req = RunBenchmarkRequest.from_dict(payload) if hasattr(RunBenchmarkRequest, "from_dict") else RunBenchmarkRequest(**payload)
        elif kind in ("ollama-instance", "llm-instance"):
            req = RunLLMInstanceRequest.from_dict(payload) if hasattr(RunLLMInstanceRequest, "from_dict") else RunLLMInstanceRequest(**payload)
        elif kind == "cost-estimate":
            req = CostEstimationRequest.from_dict(payload) if hasattr(CostEstimationRequest, "from_dict") else CostEstimationRequest(**payload)

        res = handler(req)
        if inspect.isawaitable(res):
            return await res
        return res

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
                if expected_token:
                    if hmac.compare_digest(token, expected_token):
                        return tenant_id
                    else:
                        logging.error(f"Auth failed for {tenant_id}", extra={"tenant_id": tenant_id})
                else:
                    logging.error(f"Auth failed for {tenant_id}: tenant not found", extra={"tenant_id": tenant_id})
                return None

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path

                if path == "/v1/healthz" or path == "/healthz":
                    self._write_json(200, {"status": "ok", "active_threads": threading.active_count()})
                    return

                tenant_id_hint = self.headers.get("X-Tenant-ID", "default").strip()
                try:
                    app._enforce_request_rate_limit(tenant_id_hint)
                except RequestRateLimited:
                    self._write_json(401, {"error": "rate limit exceeded"})
                    return

                tenant_id = self._authenticate()
                if not tenant_id:
                    self._write_json(401, {"error": "unauthorized"})
                    return

                if path in ("/v1/catalog/models", "/v1/catalog/vastai-models", "/v1/ollama/models", "/v1/llm/models"):
                    query = parse_qs(parsed.query)
                    backend_url = query.get("backend_url", [""])[0]
                    backend_type = query.get("backend_type", ["ollama"])[0]
                    if path == "/v1/catalog/vastai-models":
                        backend_type = "vastai"
                    
                    try:
                        if backend_type == "vastai":
                            payload = list_vastai_models()
                        else:
                            if not backend_url:
                                self._write_json(400, {"error": "missing required query parameter: backend_url"})
                                return
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
                    agent = query.get("agent", [""])[0]
                    model = query.get("model", [""])[0]
                    gpu = query.get("gpu", ["RTX 4090"])[0]
                    suite = query.get("suite", [""])[0]
                    
                    try:
                        soothsayer = PricingSoothSayer(store=app.store)
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        projection = loop.run_until_complete(soothsayer.simulate(
                            tenant_id=tenant_id, agent=agent, model=model, gpu=gpu, suite=suite
                        ))
                        loop.close()
                        self._write_json(200, projection)
                    except Exception as exc:
                        logging.exception("FinOps simulation failed")
                        self._write_json(400, {"error": str(exc)})
                    return

                if path == "/v1/settings":
                    raw = get_raw_config()
                    effective = load_config(os.environ.get("RUNE_PROFILE"))
                    
                    if "profiles" in raw:
                        for p in raw["profiles"].values():
                            if isinstance(p, dict) and "api_token" in p:
                                p["api_token"] = "[REDACTED]"
                    
                    resp = SettingsResponse(
                        defaults=raw.get("defaults", {}),
                        profiles=raw.get("profiles", {}),
                        active_profile=os.environ.get("RUNE_PROFILE"),
                        effective_config=effective
                    )
                    self._write_json(200, resp.to_dict())
                    return

                if path.startswith("/v1/runs/") and path.endswith("/trace"):
                    run_id = path.split("/")[3]
                    job = app.store.get_job(run_id)
                    if not job or (not app.security.auth_disabled and job.tenant_id != tenant_id):
                        self._write_json(404, {"error": "job not found"})
                        return
                    
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.end_headers()
                    
                    last_event_id = 0
                    while True:
                        job = app.store.get_job(run_id)
                        if not job:
                            return

                        events = app.store.get_events_for_job(run_id, after_id=last_event_id)
                        for e in events:
                            data = json.dumps(e)
                            try:
                                self.wfile.write(f"event: log\ndata: {data}\n\n".encode())
                                self.wfile.flush()
                                last_event_id = max(last_event_id, e.get("id", 0))
                            except Exception:
                                return
                        
                        if job.status in ("succeeded", "failed", "cancelled"):
                            try:
                                self.wfile.write(b"event: end\ndata: {}\n\n")
                            except Exception:
                                pass
                            return
                        
                        time.sleep(1)
                    return

                if (path.startswith("/v1/runs/") or path.startswith("/v1/audits/")) and "/artifacts" in path:
                    parts = path.split("/")
                    if len(parts) < 5:
                        self._write_json(404, {"error": "not found"})
                        return
                    run_id = parts[3]
                    if not run_id:
                        self._write_json(400, {"error": "missing run_id"})
                        return
                    job = app.store.get_job(run_id)
                    if not job or (not app.security.auth_disabled and job.tenant_id != tenant_id):
                        msg = "audit run not found" if path.startswith("/v1/audits/") else "job not found"
                        self._write_json(404, {"error": msg})
                        return
                    
                    if parts[4] != "artifacts":
                        self._write_json(404, {"error": "not found"})
                        return

                    if len(parts) == 5 or (len(parts) == 6 and not parts[5]):
                        artifacts = app.store.list_audit_artifacts(run_id)
                        prefix = "/v1/audits" if path.startswith("/v1/audits/") else "/v1/runs"
                        for a in artifacts:
                            a["download_url"] = f"{prefix}/{run_id}/artifacts/{a['artifact_id']}"
                        
                        self._write_json(200, {
                            "run_id": run_id,
                            "artifacts": artifacts,
                            "summary": {
                                "total_count": len(artifacts),
                                "kinds_present": sorted(list(set(a["kind"] for a in artifacts)))
                            }
                        })
                        return
                    elif len(parts) == 6:
                        artifact_id = parts[5]
                        res = app.store.get_audit_artifact(job_id=run_id, artifact_id=artifact_id)
                        if not res:
                            self._write_json(404, {"error": "artifact not found"})
                            return
                        content, name, kind = res
                        self.send_response(200)
                        self.send_header("Content-Type", _audit_artifact_content_type(kind))
                        self.send_header("Content-Length", str(len(content)))
                        self.send_header("Content-Disposition", f'attachment; filename="{name}"')
                        self.end_headers()
                        self.wfile.write(content)
                        return

                if path.startswith("/v1/chains/") and path.endswith("/state"):
                    run_id = path.split("/")[3]
                    if not run_id:
                        self._write_json(400, {"error": "missing run_id"})
                        return
                    job = app.store.get_job(run_id)
                    if not job or (not app.security.auth_disabled and job.tenant_id != tenant_id):
                        self._write_json(404, {"error": "chain run not found"})
                        return
                    state = app.store.get_chain_state(run_id)
                    if state is None:
                        self._write_json(200, {"run_id": run_id, "overall_status": "pending", "nodes": [], "edges": []})
                        return
                    self._write_json(200, state)
                    return

                if path.startswith("/v1/jobs/"):
                    job_id = path.split("/")[-1]
                    job = app.store.get_job(job_id)
                    if not job or (not app.security.auth_disabled and job.tenant_id != tenant_id):
                        self._write_json(404, {"error": "job not found"})
                        return
                    self._write_json(200, _job_to_payload(job))
                    return

                self._write_json(404, {"error": "not found"})

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path

                tenant_id_hint = self.headers.get("X-Tenant-ID", "default").strip()
                try:
                    app._enforce_request_rate_limit(tenant_id_hint)
                except RequestRateLimited:
                    self._write_json(401, {"error": "rate limit exceeded"})
                    return

                tenant_id = self._authenticate()
                if not tenant_id:
                    self._write_json(401, {"error": "unauthorized"})
                    return

                length = int(self.headers.get("Content-Length", 0))
                if length > 10 * 1024 * 1024:
                    self._write_json(413, {"error": "request too large (exceeds maximum size of 10 MiB)"})
                    return
                
                body = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    self._write_json(400, {"error": "invalid JSON"})
                    return

                if path == "/v1/settings":
                    req = UpdateSettingsRequest(**data)
                    update_settings(req.settings, req.profile)
                    self._write_json(200, {"status": "updated"})
                    return
                
                if path == "/v1/settings/profiles":
                    req = CreateProfileRequest(**data)
                    create_profile(req.name, req.settings)
                    self._write_json(201, {"status": "created"})
                    return

                if path.startswith("/v1/jobs/"):
                    kind = path.split("/")[-1]
                    handler = app.backend_functions.get(kind)
                    if not handler:
                        self._write_json(404, {"error": f"unknown job kind: {kind}"})
                        return
                    
                    if not isinstance(data, dict):
                        self._write_json(400, {"error": "request body must be a JSON object"})
                        return

                    idem_key = self.headers.get("Idempotency-Key") or self.headers.get("X-Idempotency-Key")
                    job_id, created = app.store.create_job(
                        tenant_id=tenant_id, 
                        kind=kind, 
                        request_payload=data,
                        idempotency_key=idem_key
                    )
                    
                    if created:
                        def _run_in_thread(job_id_inner, handler_inner, kind_inner, data_inner):
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                loop.run_until_complete(app._execute_job(job_id_inner, handler_inner, kind_inner, data_inner))
                            finally:
                                loop.close()

                        threading.Thread(
                            target=_run_in_thread,
                            args=(job_id, handler, kind, data),
                            daemon=True
                        ).start()
                    
                    self._write_json(202, {"job_id": job_id, "status": "accepted"})
                    return

                self._write_json(404, {"error": "not found"})

            def do_PATCH(self) -> None:
                self.do_POST()

            def do_PUT(self) -> None:
                self.do_POST()

        return RuneApiHandler

    async def _execute_job(self, job_id: str, handler: object, kind: str, payload: dict) -> None:
        self.store.update_job(job_id, status="running")
        try:
            supported_kinds = ("agentic-agent", "benchmark", "ollama-instance", "llm-instance", "cost-estimate")
            if kind not in supported_kinds:
                raise RuntimeError(f"unsupported job kind: {kind}")

            if kind == "agentic-agent":
                req = RunAgenticAgentRequest.from_dict(payload) if hasattr(RunAgenticAgentRequest, "from_dict") else RunAgenticAgentRequest(**payload)
            elif kind == "benchmark":
                req = RunBenchmarkRequest.from_dict(payload) if hasattr(RunBenchmarkRequest, "from_dict") else RunBenchmarkRequest(**payload)
            elif kind in ("ollama-instance", "llm-instance"):
                req = RunLLMInstanceRequest.from_dict(payload) if hasattr(RunLLMInstanceRequest, "from_dict") else RunLLMInstanceRequest(**payload)
            elif kind == "cost-estimate":
                req = CostEstimationRequest.from_dict(payload) if hasattr(CostEstimationRequest, "from_dict") else CostEstimationRequest(**payload)

            res = handler(req)
            if inspect.isawaitable(res):
                result = await res
            else:
                result = res
            
            self.store.update_job(job_id, status="succeeded", result_payload=result)
        except Exception as exc:
            job = self.store.get_job(job_id)
            tenant_id = getattr(job, "tenant_id", "unknown") if job else "unknown"
            logging.exception("Job %s failed", job_id, extra={"job_id": job_id, "tenant_id": tenant_id})
            self.store.update_job(job_id, status="failed", error=str(exc))

    def serve(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        setup_logging(json_format=True)
        debug_pprof.start_background_server_if_configured()
        server = ThreadingHTTPServer((host, port), self.create_handler())
        server.timeout = _HTTP_REQUEST_SOCKET_TIMEOUT_S
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            self.store.close()
