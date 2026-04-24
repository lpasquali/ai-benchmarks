# SPDX-License-Identifier: Apache-2.0
"""HttpTransport — drives a remote driver server via the rune job-poll protocol.

The server must implement:
    POST /v1/actions/{action}  body: {"params": {...}}
        → 202 {"job_id": "..."}
    GET  /v1/jobs/{job_id}
        → {"status": "...", "result": {...}} | {"status": "failed", "error": "..."}
"""

from __future__ import annotations

import time

from rune_bench.common import make_async_http_request, make_http_request, normalize_url
from rune_bench.debug import debug_log
from rune_bench.drivers.timeouts import driver_invocation_timeout_seconds

_POLL_INTERVAL_S = 2.0
_POLL_TIMEOUT_S = 3600.0
_TERMINAL_STATUSES = frozenset(
    {"succeeded", "success", "completed", "failed", "error", "cancelled"}
)


class HttpTransport:
    """Submit a driver job over HTTP and poll until it reaches a terminal status."""

    def __init__(
        self,
        base_url: str,
        *,
        api_token: str = "",
        tenant: str = "default",
    ) -> None:
        self._base_url = normalize_url(base_url, service_name="Driver HTTP").rstrip("/")
        self._api_token = api_token
        self._tenant = tenant

    def _build_headers(self) -> dict[str, str]:
        """Build auth/tenant request headers mirroring RuneApiClient's convention."""
        headers: dict[str, str] = {"X-Tenant-ID": self._tenant or "default"}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"
            headers["X-API-Key"] = self._api_token
        return headers

    def call(self, action: str, params: dict) -> dict:
        debug_log(f"HttpTransport → {self._base_url} action={action!r}")
        timeout = driver_invocation_timeout_seconds()

        response = make_http_request(
            f"{self._base_url}/v1/actions/{action}",
            method="POST",
            payload={"params": params},
            action=f"submit driver action {action!r}",
            timeout_seconds=int(timeout),
            headers=self._build_headers(),
            debug_prefix="Driver HTTP",
        )
        job_id = response.get("job_id")
        if not job_id:
            raise RuntimeError(
                f"Driver HTTP server did not return a job_id for action {action!r}"
            )

        debug_log(f"HttpTransport polling job_id={job_id}")
        deadline = time.monotonic() + _POLL_TIMEOUT_S
        while time.monotonic() < deadline:
            poll = make_http_request(
                f"{self._base_url}/v1/jobs/{job_id}",
                method="GET",
                payload=None,
                action=f"poll driver job {job_id}",
                timeout_seconds=int(timeout),
                headers=self._build_headers(),
                debug_prefix="Driver HTTP",
            )
            status = str(poll.get("status", ""))
            if status in _TERMINAL_STATUSES:
                if status in ("failed", "error", "cancelled"):
                    raise RuntimeError(
                        f"Driver job {job_id} failed: {poll.get('error', status)}"
                    )
                return poll.get("result", {})
            time.sleep(_POLL_INTERVAL_S)

        raise RuntimeError(
            f"Driver job {job_id} timed out after {_POLL_TIMEOUT_S:.0f}s"
        )


class AsyncHttpTransport:
    """Submit a driver job over HTTP asynchronously and poll until it reaches a terminal status."""

    def __init__(
        self,
        base_url: str,
        *,
        api_token: str = "",
        tenant: str = "default",
    ) -> None:
        self._base_url = normalize_url(base_url, service_name="Driver HTTP").rstrip("/")
        self._api_token = api_token
        self._tenant = tenant

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"X-Tenant-ID": self._tenant or "default"}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"
            headers["X-API-Key"] = self._api_token
        return headers

    async def call_async(self, action: str, params: dict) -> dict:
        import asyncio

        debug_log(f"AsyncHttpTransport → {self._base_url} action={action!r}")
        timeout = driver_invocation_timeout_seconds()

        response = await make_async_http_request(
            f"{self._base_url}/v1/actions/{action}",
            method="POST",
            payload={"params": params},
            action=f"submit driver action {action!r}",
            timeout_seconds=int(timeout),
            headers=self._build_headers(),
            debug_prefix="Driver HTTP",
        )
        job_id = response.get("job_id")
        if not job_id:
            raise RuntimeError(
                f"Driver HTTP server did not return a job_id for action {action!r}"
            )

        debug_log(f"AsyncHttpTransport polling job_id={job_id}")
        deadline = asyncio.get_event_loop().time() + _POLL_TIMEOUT_S
        while asyncio.get_event_loop().time() < deadline:
            poll = await make_async_http_request(
                f"{self._base_url}/v1/jobs/{job_id}",
                method="GET",
                payload=None,
                action=f"poll driver job {job_id}",
                timeout_seconds=int(timeout),
                headers=self._build_headers(),
                debug_prefix="Driver HTTP",
            )
            status = str(poll.get("status", ""))
            if status in _TERMINAL_STATUSES:
                if status in ("failed", "error", "cancelled"):
                    raise RuntimeError(
                        f"Driver job {job_id} failed: {poll.get('error', status)}"
                    )
                return poll.get("result", {})
            await asyncio.sleep(_POLL_INTERVAL_S)

        raise RuntimeError(
            f"Driver job {job_id} timed out after {_POLL_TIMEOUT_S:.0f}s"
        )
