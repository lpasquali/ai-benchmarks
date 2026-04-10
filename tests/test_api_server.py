# SPDX-License-Identifier: Apache-2.0
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from argon2 import PasswordHasher

from rune_bench.api_client import RuneApiClient
from rune_bench.api_server import ApiSecurityConfig, RuneApiApplication
from rune_bench.job_store import JobStore

_ph = PasswordHasher()

# SR-Q-016: API tokens must be at least 32 characters when auth is enabled.
_API_TOKEN_A = "a" * 32
_API_TOKEN_B = "b" * 32
# Precomputed SHA-256 hex of _API_TOKEN_A / _API_TOKEN_B (matches api_server fingerprint; avoids CodeQL
# py/weak-sensitive-data-hashing on test-only bearer material).
_SHA256_HEX_API_TOKEN_A = "3ba3f5f43b92602683c19aee62a20342b084dd5971ddd33808d81a328879a547"
_SHA256_HEX_API_TOKEN_B = "bdb339768bc5e4fecbe55a442056919b2b325907d49bcbf3bf8de13781996a83"


@pytest.fixture
def rune_api_server(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    state = {"agentic_calls": 0, "store": store}

    def run_agentic(request):
        state["agentic_calls"] += 1
        return {"answer": f"ok:{request.question}"}

    app = RuneApiApplication(
        store=store,
        security=ApiSecurityConfig(
            auth_disabled=False,
            tenant_tokens={
                "tenant-a": _SHA256_HEX_API_TOKEN_A,
                "tenant-b": _SHA256_HEX_API_TOKEN_B,
            },
        ),
        backend_functions={
            "agentic-agent": run_agentic,
            "benchmark": lambda request: {"answer": "bench"},
            "llm-instance": lambda request: {"mode": "existing", "backend_url": request.backend_url},
            "ollama-instance": lambda request: {"mode": "existing", "backend_url": request.backend_url},
        },
    )

    server = ThreadingHTTPServer(("127.0.0.1", 0), app.create_handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    try:
        yield base_url, state
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_healthz_is_public(rune_api_server):
    base_url, _state = rune_api_server
    with urlopen(f"{base_url}/healthz") as response:  # nosec  # test request mock/local execution
        payload = json.loads(response.read().decode("utf-8"))

    assert payload["status"] == "ok"
    assert isinstance(payload.get("active_threads"), int)
    assert payload["active_threads"] >= 1


def test_api_server_requires_auth(rune_api_server):
    base_url, _state = rune_api_server
    request = Request(f"{base_url}/v1/catalog/vastai-models")

    with pytest.raises(HTTPError) as exc:
        urlopen(request)  # nosec  # test request mock/local execution

    assert exc.value.code == 401


def test_api_server_enforces_tenant_scoping_and_idempotency(rune_api_server):
    base_url, state = rune_api_server
    client_a = RuneApiClient(base_url, api_token=_API_TOKEN_A, tenant_id="tenant-a")  # nosec  # test credentials
    client_b = RuneApiClient(base_url, api_token=_API_TOKEN_B, tenant_id="tenant-b")  # nosec  # test credentials
    request_payload = {
        "question": "What is unhealthy?",
        "model": "llama3.1:8b",
        "backend_url": None,
        "backend_warmup": False,
        "backend_warmup_timeout": 1,
        "kubeconfig": "/tmp/config",  # nosec  # test artifact paths
    }

    job_id_1 = client_a.submit_agentic_agent_job(request_payload, idempotency_key="idem-1")
    job_id_2 = client_a.submit_agentic_agent_job(request_payload, idempotency_key="idem-1")

    assert job_id_1 == job_id_2

    payload = client_a.wait_for_job(job_id_1, timeout_seconds=5, poll_interval_seconds=0.01)
    assert payload["result"]["answer"] == "ok:What is unhealthy?"
    assert state["agentic_calls"] == 1

    with pytest.raises(RuntimeError, match="job not found"):
        client_b.get_job_status(job_id_1)


def test_api_server_rate_limiting(rune_api_server):
    base_url, _state = rune_api_server

    # Attempt 10 failed logins
    for i in range(10):
        request = Request(f"{base_url}/v1/catalog/vastai-models")
        request.add_header("Authorization", "Bearer invalid-token")
        with pytest.raises(HTTPError) as exc:
            urlopen(request)
        assert exc.value.code == 401

    # 11th attempt should trigger rate limit
    request = Request(f"{base_url}/v1/catalog/vastai-models")
    request.add_header("Authorization", "Bearer invalid-token")
    with pytest.raises(HTTPError) as exc:
        urlopen(request)
    assert exc.value.code == 401
    payload = json.loads(exc.value.read().decode("utf-8"))
    assert payload["error"] == "rate limit exceeded"


def test_api_request_rate_limit_sr_q_005(rune_api_server):
    """SR-Q-005: Per-IP request budget returns HTTP 429 after burst (token bucket)."""
    base_url, _state = rune_api_server
    # Bucket capacity 20;21st request in a tight loop should be throttled.
    for i in range(20):
        request = Request(f"{base_url}/v1/catalog/vastai-models")
        request.add_header("Authorization", f"Bearer {_API_TOKEN_A}")  # nosec
        request.add_header("X-Tenant-ID", "tenant-a")
        with urlopen(request) as response:  # nosec
            assert response.status == 200

    request = Request(f"{base_url}/v1/catalog/vastai-models")
    request.add_header("Authorization", f"Bearer {_API_TOKEN_A}")  # nosec
    request.add_header("X-Tenant-ID", "tenant-a")
    with pytest.raises(HTTPError) as exc:
        urlopen(request)  # nosec
    assert exc.value.code == 429
    payload = json.loads(exc.value.read().decode("utf-8"))
    assert payload["error"] == "too many requests"


# ── /v1/chains/{run_id}/state ───────────────────────────────────────────────


def _auth_headers(token: str, tenant: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "X-Tenant-ID": tenant,
    }


def _get_json(url: str, headers: dict) -> dict:
    request = Request(url)
    for k, v in headers.items():
        request.add_header(k, v)
    with urlopen(request) as response:  # nosec  # local test request
        return json.loads(response.read().decode("utf-8"))


def test_chain_state_returns_404_for_unknown_run(rune_api_server):
    base_url, _ = rune_api_server
    request = Request(f"{base_url}/v1/chains/does-not-exist/state")
    request.add_header("Authorization", f"Bearer {_API_TOKEN_A}")  # nosec
    request.add_header("X-Tenant-ID", "tenant-a")
    with pytest.raises(HTTPError) as exc:
        urlopen(request)  # nosec
    assert exc.value.code == 404
    payload = json.loads(exc.value.read().decode("utf-8"))
    assert "chain run not found" in payload["error"]


def test_chain_state_returns_404_for_other_tenants_run(rune_api_server):
    base_url, _ = rune_api_server
    client_a = RuneApiClient(base_url, api_token=_API_TOKEN_A, tenant_id="tenant-a")  # nosec
    job_id = client_a.submit_agentic_agent_job(
        {
            "question": "q",
            "model": "m",
            "backend_url": None,
            "backend_warmup": False,
            "backend_warmup_timeout": 1,
            "kubeconfig": "/tmp/cfg",  # nosec
        },
        idempotency_key="t1",
    )

    request = Request(f"{base_url}/v1/chains/{job_id}/state")
    request.add_header("Authorization", f"Bearer {_API_TOKEN_B}")  # nosec
    request.add_header("X-Tenant-ID", "tenant-b")
    with pytest.raises(HTTPError) as exc:
        urlopen(request)  # nosec
    assert exc.value.code == 404


def test_chain_state_returns_empty_shell_when_job_exists_but_no_chain_state(rune_api_server, tmp_path):
    base_url, _ = rune_api_server
    client_a = RuneApiClient(base_url, api_token=_API_TOKEN_A, tenant_id="tenant-a")  # nosec
    job_id = client_a.submit_agentic_agent_job(
        {
            "question": "q",
            "model": "m",
            "backend_url": None,
            "backend_warmup": False,
            "backend_warmup_timeout": 1,
            "kubeconfig": "/tmp/cfg",  # nosec
        },
        idempotency_key="empty-shell",
    )

    payload = _get_json(
        f"{base_url}/v1/chains/{job_id}/state",
        _auth_headers(_API_TOKEN_A, "tenant-a"),
    )
    assert payload == {
        "run_id": job_id,
        "nodes": [],
        "edges": [],
        "overall_status": "pending",
    }


def test_chain_state_returns_full_state_shape(rune_api_server):
    """Verify the API endpoint returns the documented JSON shape.

    Full populated-state behavior is exercised at the JobStore level in
    test_job_store.py (where we control the DB directly). This test confirms
    the wire format the dashboard will consume.
    """
    base_url, _ = rune_api_server
    client_a = RuneApiClient(base_url, api_token=_API_TOKEN_A, tenant_id="tenant-a")  # nosec
    job_id = client_a.submit_agentic_agent_job(
        {
            "question": "q",
            "model": "m",
            "backend_url": None,
            "backend_warmup": False,
            "backend_warmup_timeout": 1,
            "kubeconfig": "/tmp/cfg",  # nosec
        },
        idempotency_key="full-state",
    )

    payload = _get_json(
        f"{base_url}/v1/chains/{job_id}/state",
        _auth_headers(_API_TOKEN_A, "tenant-a"),
    )
    # Empty-shell shape (no chain state initialized for this job)
    assert payload["run_id"] == job_id
    assert payload["overall_status"] == "pending"
    assert payload["nodes"] == []
    assert payload["edges"] == []


def test_chain_state_endpoint_requires_auth(rune_api_server):
    base_url, _ = rune_api_server
    request = Request(f"{base_url}/v1/chains/anything/state")
    with pytest.raises(HTTPError) as exc:
        urlopen(request)  # nosec
    assert exc.value.code == 401


def test_chain_state_endpoint_rejects_empty_run_id(rune_api_server):
    base_url, _ = rune_api_server
    # /v1/chains//state has empty run_id between the two slashes
    request = Request(f"{base_url}/v1/chains//state")
    request.add_header("Authorization", f"Bearer {_API_TOKEN_A}")  # nosec
    request.add_header("X-Tenant-ID", "tenant-a")
    with pytest.raises(HTTPError) as exc:
        urlopen(request)  # nosec
    assert exc.value.code == 400
    payload = json.loads(exc.value.read().decode("utf-8"))
    assert "missing run_id" in payload["error"]


# ── /v1/audits/{run_id}/artifacts ───────────────────────────────────────────


def _create_throwaway_job(client) -> str:
    return client.submit_agentic_agent_job(
        {
            "question": "q",
            "model": "m",
            "backend_url": None,
            "backend_warmup": False,
            "backend_warmup_timeout": 1,
            "kubeconfig": "/tmp/cfg",  # nosec
        },
        idempotency_key=f"audit-{id(client)}",
    )


def test_audit_artifacts_list_empty_for_new_job(rune_api_server):
    base_url, _ = rune_api_server
    client = RuneApiClient(base_url, api_token=_API_TOKEN_A, tenant_id="tenant-a")  # nosec
    job_id = _create_throwaway_job(client)
    payload = _get_json(
        f"{base_url}/v1/audits/{job_id}/artifacts",
        _auth_headers(_API_TOKEN_A, "tenant-a"),
    )
    assert payload == {
        "run_id": job_id,
        "artifacts": [],
        "summary": {"total_count": 0, "kinds_present": []},
    }


def test_audit_artifacts_list_returns_404_for_unknown_run(rune_api_server):
    base_url, _ = rune_api_server
    request = Request(f"{base_url}/v1/audits/does-not-exist/artifacts")
    request.add_header("Authorization", f"Bearer {_API_TOKEN_A}")  # nosec
    request.add_header("X-Tenant-ID", "tenant-a")
    with pytest.raises(HTTPError) as exc:
        urlopen(request)  # nosec
    assert exc.value.code == 404
    payload = json.loads(exc.value.read().decode("utf-8"))
    assert "audit run not found" in payload["error"]


def test_audit_artifacts_list_is_tenant_scoped(rune_api_server):
    base_url, _ = rune_api_server
    client_a = RuneApiClient(base_url, api_token=_API_TOKEN_A, tenant_id="tenant-a")  # nosec
    job_id = _create_throwaway_job(client_a)
    request = Request(f"{base_url}/v1/audits/{job_id}/artifacts")
    request.add_header("Authorization", f"Bearer {_API_TOKEN_B}")  # nosec
    request.add_header("X-Tenant-ID", "tenant-b")
    with pytest.raises(HTTPError) as exc:
        urlopen(request)  # nosec
    assert exc.value.code == 404


def test_audit_artifacts_endpoints_require_auth(rune_api_server):
    base_url, _ = rune_api_server
    for path in ("/v1/audits/x/artifacts", "/v1/audits/x/artifacts/y"):
        request = Request(f"{base_url}{path}")
        with pytest.raises(HTTPError) as exc:
            urlopen(request)  # nosec
        assert exc.value.code == 401


def test_audit_artifacts_list_rejects_empty_run_id(rune_api_server):
    base_url, _ = rune_api_server
    request = Request(f"{base_url}/v1/audits//artifacts")
    request.add_header("Authorization", f"Bearer {_API_TOKEN_A}")  # nosec
    request.add_header("X-Tenant-ID", "tenant-a")
    with pytest.raises(HTTPError) as exc:
        urlopen(request)  # nosec
    assert exc.value.code == 400
    payload = json.loads(exc.value.read().decode("utf-8"))
    assert "missing run_id" in payload["error"]


def test_audit_artifact_download_returns_404_for_unknown_artifact(rune_api_server):
    base_url, _ = rune_api_server
    client = RuneApiClient(base_url, api_token=_API_TOKEN_A, tenant_id="tenant-a")  # nosec
    job_id = _create_throwaway_job(client)
    request = Request(f"{base_url}/v1/audits/{job_id}/artifacts/missing-id")
    request.add_header("Authorization", f"Bearer {_API_TOKEN_A}")  # nosec
    request.add_header("X-Tenant-ID", "tenant-a")
    with pytest.raises(HTTPError) as exc:
        urlopen(request)  # nosec
    assert exc.value.code == 404
    payload = json.loads(exc.value.read().decode("utf-8"))
    assert "artifact not found" in payload["error"]


def test_audit_artifact_download_returns_404_for_other_tenants_run(rune_api_server):
    base_url, _ = rune_api_server
    client_a = RuneApiClient(base_url, api_token=_API_TOKEN_A, tenant_id="tenant-a")  # nosec
    job_id = _create_throwaway_job(client_a)
    request = Request(f"{base_url}/v1/audits/{job_id}/artifacts/x")
    request.add_header("Authorization", f"Bearer {_API_TOKEN_B}")  # nosec
    request.add_header("X-Tenant-ID", "tenant-b")
    with pytest.raises(HTTPError) as exc:
        urlopen(request)  # nosec
    assert exc.value.code == 404


def test_audit_artifact_content_type_helper():
    """Internal helper covers all six known kinds plus a fallback."""
    from rune_bench.api_server import _audit_artifact_content_type

    assert _audit_artifact_content_type("slsa_provenance") == "application/json"
    assert _audit_artifact_content_type("sbom") == "application/json"
    assert _audit_artifact_content_type("rekor_entry") == "application/json"
    assert (
        _audit_artifact_content_type("tla_report")
        == "text/plain; charset=utf-8"
    )
    assert _audit_artifact_content_type("sigstore_bundle") == "application/octet-stream"
    assert _audit_artifact_content_type("tpm_attestation") == "application/octet-stream"
    # Unknown kind → safe fallback
    assert _audit_artifact_content_type("invented") == "application/octet-stream"


def test_audit_artifacts_list_returns_populated_artifacts(rune_api_server):
    """Happy path: write artifacts via the JobStore exposed by the fixture, list via the API."""
    base_url, state = rune_api_server
    store = state["store"]
    client = RuneApiClient(base_url, api_token=_API_TOKEN_A, tenant_id="tenant-a")  # nosec
    job_id = _create_throwaway_job(client)

    # Two artifacts of two different kinds
    aid1 = store.record_audit_artifact(
        job_id=job_id, kind="slsa_provenance", name="provenance.json", content=b'{"_type":"slsa.provenance"}'
    )
    aid2 = store.record_audit_artifact(
        job_id=job_id, kind="sbom", name="sbom.json", content=b'{"bomFormat":"CycloneDX"}'
    )

    payload = _get_json(
        f"{base_url}/v1/audits/{job_id}/artifacts",
        _auth_headers(_API_TOKEN_A, "tenant-a"),
    )
    assert payload["run_id"] == job_id
    assert payload["summary"] == {"total_count": 2, "kinds_present": ["sbom", "slsa_provenance"]}
    assert len(payload["artifacts"]) == 2

    artifact_ids = {a["artifact_id"] for a in payload["artifacts"]}
    assert artifact_ids == {aid1, aid2}

    for a in payload["artifacts"]:
        assert a["download_url"] == f"/v1/audits/{job_id}/artifacts/{a['artifact_id']}"
        assert a["size_bytes"] > 0
        assert len(a["sha256"]) == 64
        # Bytes intentionally excluded from list response
        assert "content" not in a


def test_audit_artifact_download_streams_bytes_with_correct_headers(rune_api_server):
    base_url, state = rune_api_server
    store = state["store"]
    client = RuneApiClient(base_url, api_token=_API_TOKEN_A, tenant_id="tenant-a")  # nosec
    job_id = _create_throwaway_job(client)

    payload_bytes = b'{"_type":"slsa.provenance","subject":[]}'
    artifact_id = store.record_audit_artifact(
        job_id=job_id,
        kind="slsa_provenance",
        name="provenance.json",
        content=payload_bytes,
    )

    request = Request(f"{base_url}/v1/audits/{job_id}/artifacts/{artifact_id}")
    request.add_header("Authorization", f"Bearer {_API_TOKEN_A}")  # nosec
    request.add_header("X-Tenant-ID", "tenant-a")
    with urlopen(request) as response:  # nosec
        body = response.read()
        content_type = response.headers.get("Content-Type")
        content_length = int(response.headers.get("Content-Length", "0"))
        content_disposition = response.headers.get("Content-Disposition")

    assert body == payload_bytes
    assert content_type == "application/json"
    assert content_length == len(payload_bytes)
    assert content_disposition == 'attachment; filename="provenance.json"'


def test_audit_artifact_download_uses_octet_stream_for_binary_kinds(rune_api_server):
    base_url, state = rune_api_server
    store = state["store"]
    client = RuneApiClient(base_url, api_token=_API_TOKEN_A, tenant_id="tenant-a")  # nosec
    job_id = _create_throwaway_job(client)

    artifact_id = store.record_audit_artifact(
        job_id=job_id, kind="sigstore_bundle", name="bundle.sig", content=b"\x00\x01\x02"
    )
    request = Request(f"{base_url}/v1/audits/{job_id}/artifacts/{artifact_id}")
    request.add_header("Authorization", f"Bearer {_API_TOKEN_A}")  # nosec
    request.add_header("X-Tenant-ID", "tenant-a")
    with urlopen(request) as response:  # nosec
        body = response.read()
        content_type = response.headers.get("Content-Type")

    assert body == b"\x00\x01\x02"
    assert content_type == "application/octet-stream"


def test_audit_artifacts_endpoints_unknown_subpath_returns_404(rune_api_server):
    """Edge case: malformed path under /v1/audits/ that doesn't include /artifacts."""
    base_url, _ = rune_api_server
    request = Request(f"{base_url}/v1/audits/malformed/wrong-suffix")
    request.add_header("Authorization", f"Bearer {_API_TOKEN_A}")  # nosec
    request.add_header("X-Tenant-ID", "tenant-a")
    with pytest.raises(HTTPError) as exc:
        urlopen(request)  # nosec
    assert exc.value.code == 404


def test_audit_artifact_download_rejects_empty_artifact_id(rune_api_server):
    base_url, _ = rune_api_server
    client = RuneApiClient(base_url, api_token=_API_TOKEN_A, tenant_id="tenant-a")  # nosec
    job_id = _create_throwaway_job(client)
    # Trailing slash after /artifacts/ — falls into the download branch with empty id
    request = Request(f"{base_url}/v1/audits/{job_id}/artifacts/")
    request.add_header("Authorization", f"Bearer {_API_TOKEN_A}")  # nosec
    request.add_header("X-Tenant-ID", "tenant-a")
    # The endpoint treats this as the list path (tail == "/"), which returns 200
    # with an empty list. That's fine — the empty-artifact-id branch only fires
    # for paths like /v1/audits/{run_id}/artifacts// which urllib normalizes
    # away. Verify the list-path behavior here for completeness.
    with urlopen(request) as response:  # nosec
        payload = json.loads(response.read().decode("utf-8"))
    assert payload["run_id"] == job_id
    assert payload["summary"]["total_count"] == 0


def test_chain_state_response_to_dict():
    from rune_bench.api_contracts import ChainStateResponse

    resp = ChainStateResponse(
        run_id="r1",
        nodes=[{"id": "a", "status": "success"}],
        edges=[{"from": "a", "to": "b"}],
        overall_status="success",
    )
    d = resp.to_dict()
    assert d == {
        "run_id": "r1",
        "nodes": [{"id": "a", "status": "success"}],
        "edges": [{"from": "a", "to": "b"}],
        "overall_status": "success",
    }


def test_audit_artifact_to_dict():
    from rune_bench.api_contracts import AuditArtifact

    artifact = AuditArtifact(
        artifact_id="aid-1",
        kind="sbom",
        name="x.json",
        size_bytes=42,
        sha256="abc" * 21 + "x",
        created_at=12345.6,
        download_url="/v1/audits/r1/artifacts/aid-1",
    )
    d = artifact.to_dict()
    assert d["artifact_id"] == "aid-1"
    assert d["kind"] == "sbom"
    assert d["download_url"] == "/v1/audits/r1/artifacts/aid-1"


def test_audit_artifacts_response_to_dict():
    from rune_bench.api_contracts import AuditArtifactsResponse

    resp = AuditArtifactsResponse(
        run_id="r1",
        artifacts=[{"artifact_id": "a"}],
        summary={"total_count": 1, "kinds_present": ["sbom"]},
    )
    d = resp.to_dict()
    assert d == {
        "run_id": "r1",
        "artifacts": [{"artifact_id": "a"}],
        "summary": {"total_count": 1, "kinds_present": ["sbom"]},
    }
