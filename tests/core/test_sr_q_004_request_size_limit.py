# SPDX-License-Identifier: Apache-2.0
"""Tests for SR-Q-004: Request Body Size Limit."""

import http.client
import json
import socket
import threading
import time

import pytest

from rune_bench.api_server import ApiSecurityConfig, RuneApiApplication
from rune_bench.storage.sqlite import SQLiteStorageAdapter


@pytest.fixture
def test_storage(tmp_path):
    """Create a temporary storage adapter for testing."""
    db_path = tmp_path / "test.db"
    storage = SQLiteStorageAdapter(f"sqlite:///{db_path}")
    yield storage


@pytest.fixture
def test_app(test_storage):
    """Create a test API application with auth disabled."""
    security = ApiSecurityConfig(auth_disabled=True, tenant_tokens={})
    app = RuneApiApplication(store=test_storage, security=security)
    return app


@pytest.fixture
def api_server(test_app):
    """Start API server in background thread."""
    # Find a free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server_thread = threading.Thread(
        target=test_app.serve, args=("127.0.0.1", port), daemon=True
    )
    server_thread.start()
    time.sleep(0.5)  # Give server time to start
    try:
        yield f"127.0.0.1:{port}"
    finally:
        # shutdown() is not implemented in RuneApiApplication, but the server
        # will exit when the process ends because it's a daemon thread.
        # We MUST close the store though.
        test_app.store.close()


def test_request_size_limit_enforced(api_server):
    """SR-Q-004: Requests with Content-Length > 10 MiB should return HTTP 413."""
    host = api_server

    # Send headers indicating oversized body (don't actually send 10MB)
    conn = http.client.HTTPConnection(host, timeout=5)
    oversized_length = 10 * 1024 * 1024 + 1000  # 10 MiB + 1000 bytes

    # Send small payload but with oversized Content-Length header
    small_payload = {"test": "data"}
    payload_bytes = json.dumps(small_payload).encode("utf-8")

    conn.request(
        "POST",
        "/v1/jobs/benchmark",
        body=payload_bytes,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(oversized_length),  # Lie about size
        },
    )
    response = conn.getresponse()
    data = json.loads(response.read().decode("utf-8"))
    conn.close()

    assert response.status == 413, f"Expected 413, got {response.status}"
    assert "exceeds maximum size" in data["error"]
    assert "10 MiB" in data["error"]


def test_request_under_limit_accepted(api_server):
    """SR-Q-004: Requests under 10 MiB should be accepted."""
    host = api_server

    # Create a small valid payload
    payload = {
        "model": "test-model",
        "question": "test question",
        "backend_url": "http://localhost:11434",
    }
    payload_bytes = json.dumps(payload).encode("utf-8")

    conn = http.client.HTTPConnection(host)
    conn.request(
        "POST",
        "/v1/jobs/benchmark",
        body=payload_bytes,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(payload_bytes)),
        },
    )
    response = conn.getresponse()
    response.read()
    conn.close()

    # Should get 202 (accepted) or another valid response, not 413
    assert response.status != 413, "Small request should not be rejected"
    assert response.status in (200, 202, 400), (
        f"Expected valid status, got {response.status}"
    )


def test_request_at_limit_boundary(api_server):
    """SR-Q-004: Request exactly at 10 MiB should be accepted."""
    host = api_server

    # Create a payload at exactly 10 MiB
    # Account for JSON overhead
    max_size = 10 * 1024 * 1024
    data_size = max_size - 100  # Leave room for JSON structure
    payload = {"data": "x" * data_size}
    payload_bytes = json.dumps(payload).encode("utf-8")

    # Ensure we're under the limit
    assert len(payload_bytes) < max_size

    conn = http.client.HTTPConnection(host)
    conn.request(
        "POST",
        "/v1/jobs/benchmark",
        body=payload_bytes,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(payload_bytes)),
        },
    )
    response = conn.getresponse()
    response.read()
    conn.close()

    # Should not be rejected for size
    assert response.status != 413, "Request at limit should not be rejected"
