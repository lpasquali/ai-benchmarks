"""Shared HTTP client utilities for URL normalization and request handling."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from rune_bench.debug import debug_log


def normalize_url(url: str | None, service_name: str = "service") -> str:
    """Validate and normalize an HTTP(S) URL.
    
    Adds ``http://`` when no recognized scheme is present.
    
    Args:
        url: URL string to normalize
        service_name: Name of the service (for error messages)
        
    Returns:
        Normalized URL (scheme://netloc/path format)
        
    Raises:
        RuntimeError: If URL is missing, invalid scheme, or missing host.
    """
    if not url:
        raise RuntimeError(
            f"Missing {service_name} URL. Provide a valid URL or set the appropriate environment variable."
        )

    # Normalize: if no recognized scheme, prepend http://
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        # No valid scheme found; treat entire input as host:port
        url = f"http://{url}"
        parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(
            f"Invalid {service_name} URL. Expected format like http://host:port"
        )

    return url


def make_http_request(
    url: str,
    *,
    method: str,
    payload: dict[str, Any] | None = None,
    action: str = "perform request",
    timeout_seconds: int = 30,
    headers: dict[str, str] | None = None,
    debug_prefix: str = "HTTP",
) -> dict[str, Any]:
    """Execute an HTTP request and return parsed JSON response.
    
    Args:
        url: Full URL to request
        method: HTTP method (GET, POST, etc.)
        payload: Optional JSON payload for request body
        action: Human-readable description of operation (for error messages)
        timeout_seconds: Request timeout in seconds
        headers: Optional custom headers (Content-Type is auto-added for payloads)
        debug_prefix: Prefix for debug log messages
        
    Returns:
        Parsed JSON response as dict
        
    Raises:
        RuntimeError: If the request fails or response is invalid
    """
    request_headers: dict[str, str] = headers or {}
    data = None
    
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=request_headers, method=method)

    debug_log(
        f"{debug_prefix} request: method={method} url={url} action={action} "
        f"payload={json.dumps(payload, sort_keys=True) if payload is not None else '<none>'}"
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            debug_log(
                f"{debug_prefix} response: method={method} url={url} status={getattr(response, 'status', '<unknown>')} "
                f"body={raw}"
            )
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        debug_log(f"{debug_prefix} HTTP error: method={method} url={url} status={exc.code} detail={detail}")
        if detail:
            raise RuntimeError(f"Failed to {action}: {detail}") from exc
        raise RuntimeError(f"Failed to {action}: HTTP {exc.code}") from exc
    except (URLError, TimeoutError) as exc:
        debug_log(f"{debug_prefix} transport error: method={method} url={url} error={exc}")
        raise RuntimeError(f"Failed to {action}: {exc}") from exc

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response while attempting to {action}") from exc

    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected JSON payload while attempting to {action}")
    
    return result
