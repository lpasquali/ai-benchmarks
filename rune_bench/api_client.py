"""HTTP client for RUNE API backend mode."""

import os
from dataclasses import dataclass
import time
from urllib.parse import urlencode

from rune_bench.common import make_http_request, normalize_url


@dataclass
class RuneApiClient:
    base_url: str
    api_token: str | None = None
    tenant_id: str | None = None
    verify_ssl: bool = True

    def __post_init__(self) -> None:
        self.base_url = normalize_url(self.base_url, service_name="RUNE API")
        self.api_token = (self.api_token or os.environ.get("RUNE_API_TOKEN") or "").strip() or None
        self.tenant_id = (self.tenant_id or os.environ.get("RUNE_API_TENANT") or "default").strip() or "default"

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, str] | None = None,
        body: dict | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        url = self.base_url + path
        if query:
            url += "?" + urlencode(query)

        headers: dict[str, str] = {"X-Tenant-ID": self.tenant_id or "default"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
            headers["X-API-Key"] = self.api_token
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        return make_http_request(
            url,
            method=method,
            payload=body,
            action=f"{method} {path}",
            timeout_seconds=20,
            headers=headers,
            debug_prefix="RUNE API",
            verify_ssl=self.verify_ssl,
        )

    def get_vastai_models(self) -> list[dict]:
        payload = self._request("GET", "/v1/catalog/vastai-models")
        models = payload.get("models")
        if not isinstance(models, list):
            raise RuntimeError("API payload missing 'models' list for Vast.ai model catalog")
        return [m for m in models if isinstance(m, dict)]

    def get_ollama_models(self, ollama_url: str) -> dict:
        payload = self._request("GET", "/v1/ollama/models", query={"ollama_url": ollama_url})
        if not isinstance(payload.get("models"), list):
            raise RuntimeError("API payload missing 'models' list for Ollama models endpoint")
        if not isinstance(payload.get("running_models"), list):
            raise RuntimeError("API payload missing 'running_models' list for Ollama models endpoint")
        return payload

    def submit_agentic_agent_job(self, request_payload: dict, *, idempotency_key: str | None = None) -> str:
        payload = self._request(
            "POST",
            "/v1/jobs/agentic-agent",
            body=request_payload,
            idempotency_key=idempotency_key,
        )
        job_id = payload.get("job_id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise RuntimeError("API response missing 'job_id' for agentic-agent job")
        return job_id

    def submit_benchmark_job(self, request_payload: dict, *, idempotency_key: str | None = None) -> str:
        payload = self._request(
            "POST",
            "/v1/jobs/benchmark",
            body=request_payload,
            idempotency_key=idempotency_key,
        )
        job_id = payload.get("job_id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise RuntimeError("API response missing 'job_id' for benchmark job")
        return job_id

    def submit_ollama_instance_job(self, request_payload: dict, *, idempotency_key: str | None = None) -> str:
        payload = self._request(
            "POST",
            "/v1/jobs/ollama-instance",
            body=request_payload,
            idempotency_key=idempotency_key,
        )
        job_id = payload.get("job_id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise RuntimeError("API response missing 'job_id' for ollama-instance job")
        return job_id

    def get_job_status(self, job_id: str) -> dict:
        payload = self._request("GET", f"/v1/jobs/{job_id}")
        status = payload.get("status")
        if not isinstance(status, str) or not status.strip():
            raise RuntimeError(f"API response missing 'status' for job {job_id}")
        return payload

    def wait_for_job(
        self,
        job_id: str,
        *,
        timeout_seconds: int = 3600,
        poll_interval_seconds: float = 2.0,
        on_update: callable | None = None,
    ) -> dict:
        deadline = time.monotonic() + timeout_seconds
        last_status = ""

        while time.monotonic() < deadline:
            payload = self.get_job_status(job_id)
            status = str(payload.get("status", "unknown")).strip().lower()
            message = payload.get("message")
            if isinstance(message, str) and on_update is not None and status != last_status:
                on_update(status, message)
            elif on_update is not None and status != last_status:
                on_update(status, None)

            last_status = status
            if status in {"succeeded", "success", "completed"}:
                return payload
            if status in {"failed", "error", "cancelled", "canceled"}:
                detail = payload.get("error") or payload.get("message") or f"status={status}"
                raise RuntimeError(f"Job {job_id} failed: {detail}")

            time.sleep(poll_interval_seconds)

        raise RuntimeError(f"Timed out waiting for job {job_id} after {timeout_seconds}s")
