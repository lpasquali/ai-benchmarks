# SPDX-License-Identifier: Apache-2.0
import json
import threading
from unittest.mock import MagicMock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import rune_bench.api_server as api_server
from rune_bench.job_store import JobStore
from rune_bench.resources.vastai import InstanceManager


class DummySDK:
    def __init__(self):
        self.instances = []
        self.volumes = []

    def create_instance(self, **kwargs):
        return {"new_contract": 55, "kwargs": kwargs}

    def show_instances(self, raw=True):
        return self.instances

    def destroy_instance(self, id, raw=True):
        return {"id": id}

    def show_volumes(self, raw=True):
        return self.volumes

    def destroy_volume(self, id, raw=True):
        return {"id": id}


def test_instance_manager_create_wait_list_and_stop(monkeypatch):
    sdk = DummySDK()
    manager = InstanceManager(sdk)
    template = type("Template", (), {"env": "FOO=1", "image": "img"})()
    model = type("Model", (), {"required_disk_gb": 40})()

    assert manager.create(10, model, template) == 55

    sdk.instances = [{"id": 55, "actual_status": "running"}]
    monkeypatch.setattr("rune_bench.resources.vastai.instance.time.sleep", lambda *_: None)
    assert manager.wait_until_running(55)["id"] == 55
    assert manager.list_instances() == sdk.instances
    manager.stop_instance(55)


def test_instance_manager_error_and_helper_paths(monkeypatch):
    sdk = MagicMock()
    sdk.create_instance.side_effect = Exception("nope")
    manager = InstanceManager(sdk)
    with pytest.raises(RuntimeError, match="Instance creation failed"):
        manager.create(1, type("M", (), {"required_disk_gb": 1})(), type("T", (), {"env": "e", "image": None})())

    sdk = MagicMock()
    sdk.create_instance.return_value = {}
    with pytest.raises(RuntimeError, match="Could not parse contract id"):
        InstanceManager(sdk).create(1, type("M", (), {"required_disk_gb": 1})(), type("T", (), {"env": "e", "image": None})())

    sdk = DummySDK()
    manager = InstanceManager(sdk)
    monkeypatch.setattr("rune_bench.resources.vastai.instance._POLL_MAX_ATTEMPTS", 1)
    monkeypatch.setattr("rune_bench.resources.vastai.instance.time.sleep", lambda *_: None)
    with pytest.raises(RuntimeError, match="did not reach 'running'"):
        manager.wait_until_running(2)

    sdk = MagicMock()
    sdk.show_instances.side_effect = Exception("boom")
    with pytest.raises(RuntimeError, match="Failed to list Vast.ai instances"):
        InstanceManager(sdk).list_instances()

    sdk = DummySDK()
    manager = InstanceManager(sdk)
    sdk.instances = [{"id": "1"}]
    assert manager._fetch_instance(1)["id"] == "1"
    sdk.show_instances = lambda raw=True: {"not": "a-list"}
    assert manager._fetch_instance(1) is None
    assert manager._list_volumes_optional() == []

    sdk.show_volumes = lambda raw=True: [{"id": "v1"}]
    assert manager._list_volumes_optional() == [{"id": "v1"}]
    sdk.show_volumes = lambda raw=True: {"volumes": [{"id": "v1"}]}
    assert manager._list_volumes_optional() == [{"id": "v1"}]

    sdk.destroy_volume = MagicMock(side_effect=Exception("bad"))
    with pytest.raises(RuntimeError, match="Failed to destroy volume"):
        manager._destroy_volume("v1")

    monkeypatch.setattr(manager, "_list_volumes_optional", lambda: None)
    assert manager._verify_volumes_deleted(["v1"])[0] is False
    monkeypatch.setattr(manager, "_list_volumes_optional", lambda: [{"id": "v2"}])
    assert manager._verify_volumes_deleted(["v1"])[0] is True
    monkeypatch.setattr(manager, "_list_volumes_optional", lambda: [{"id": "v1"}])
    assert manager._verify_volumes_deleted(["v1"])[0] is False
    assert manager._verify_volumes_deleted([]) == (True, "no related volumes detected")

    assert InstanceManager._extract_related_volume_ids("bad") == []
    assert InstanceManager._first_float({"a": "3.5"}, ("a",)) == 3.5
    assert InstanceManager._first_float({"a": "x"}, ("a",)) is None


def test_instance_manager_destroy_related_storage_partial_failures(monkeypatch):
    manager = InstanceManager(DummySDK())
    monkeypatch.setattr(manager, "_fetch_instance", lambda _cid: {"volume_id": "v1", "volumes": [{"id": "v2"}]})
    monkeypatch.setattr(manager, "_destroy_instance", lambda _cid: None)
    destroyed = []

    def destroy_volume(volume_id):
        destroyed.append(volume_id)
        if volume_id == "v2":
            raise RuntimeError("nope")

    monkeypatch.setattr(manager, "_destroy_volume", destroy_volume)
    monkeypatch.setattr(manager, "_wait_until_instance_absent", lambda _cid: False)
    monkeypatch.setattr(manager, "_verify_volumes_deleted", lambda vids: (False, f"remaining {','.join(vids)}"))

    result = manager.destroy_instance_and_related_storage(9)
    assert result.destroyed_instance is True
    assert result.destroyed_volume_ids == ["v1"]
    assert result.verification_ok is False


def test_build_connection_details_without_machine_id():
    details = InstanceManager.build_connection_details(
        1,
        {"state": "created", "ports": {"svc": [{"HostIp": "1.2.3.4", "HostPort": "8080"}]}, "ssh_host": None, "ssh_port": None},
    )
    assert details.status == "created"
    assert details.service_urls[0]["proxy"] is None


@pytest.fixture
def misc_server(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNE_API_AUTH_DISABLED", "1")
    monkeypatch.delenv("RUNE_DB_URL", raising=False)
    monkeypatch.setenv("RUNE_API_DB_PATH", str(tmp_path / "jobs.db"))
    app = api_server.RuneApiApplication.from_env()
    server = api_server.ThreadingHTTPServer(("127.0.0.1", 0), app.create_handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", app
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_api_security_from_env(monkeypatch):
    monkeypatch.delenv("RUNE_API_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("RUNE_API_TOKENS", raising=False)
    with pytest.raises(RuntimeError, match="no tenants are configured"):
        api_server.ApiSecurityConfig.from_env()

    monkeypatch.setenv("RUNE_API_AUTH_DISABLED", "1")
    assert api_server.ApiSecurityConfig.from_env().auth_disabled is True

    monkeypatch.setenv("RUNE_API_AUTH_DISABLED", "0")
    long_a, long_b = "a" * 32, "b" * 32
    monkeypatch.setenv("RUNE_API_TOKENS", f"tenant-a:{long_a},tenant-b:{long_b}")
    cfg = api_server.ApiSecurityConfig.from_env()
    assert cfg.tenant_tokens["tenant-b"] == "bdb339768bc5e4fecbe55a442056919b2b325907d49bcbf3bf8de13781996a83"

    monkeypatch.setenv("RUNE_API_TOKENS", "tenant-a:short")
    with pytest.raises(RuntimeError, match="SR-Q-016"):
        api_server.ApiSecurityConfig.from_env()


def test_api_server_misc_paths(misc_server):
    base_url, app = misc_server

    with urlopen(f"{base_url}/v1/catalog/vastai-models") as response:  # nosec  # test request mock/local execution
        payload = json.loads(response.read().decode("utf-8"))
    assert "models" in payload

    req = Request(f"{base_url}/v1/ollama/models")
    with pytest.raises(HTTPError) as exc:
        urlopen(req)  # nosec  # test request mock/local execution
    assert exc.value.code == 400

    bad_req = Request(
        f"{base_url}/v1/jobs/agentic-agent",
        data=b"[]",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(HTTPError) as exc:
        urlopen(bad_req)  # nosec  # test request mock/local execution
    assert exc.value.code == 400

    unknown_req = Request(f"{base_url}/nope")
    with pytest.raises(HTTPError) as exc:
        urlopen(unknown_req)  # nosec  # test request mock/local execution
    assert exc.value.code == 404

    job = app.store.create_job(tenant_id="default", kind="agentic-agent", request_payload={})[0]
    with urlopen(f"{base_url}/v1/jobs/{job}") as response:  # nosec  # test request mock/local execution
        payload = json.loads(response.read().decode("utf-8"))
    assert payload["job_id"] == job


def test_api_server_internal_dispatch_and_failures(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    app = api_server.RuneApiApplication(
        store=store,
        security=api_server.ApiSecurityConfig(auth_disabled=True, tenant_tokens={}),
        backend_functions={"agentic-agent": lambda request: {"ok": True}},
    )

    with pytest.raises(RuntimeError, match="unsupported job kind"):
        app._dispatch("nope", {})

    with pytest.raises(RuntimeError, match="no backend function registered"):
        app._dispatch(
            "benchmark",
            {
                "vastai": False,
                "template_hash": "t",
                "min_dph": 1,
                "max_dph": 2,
                "reliability": 0.9,
                "backend_url": None,
                "question": "q",
                "model": "m",
                "backend_warmup": False,
                "backend_warmup_timeout": 1,
                "kubeconfig": "/tmp/k",  # nosec  # test artifact paths
                "vastai_stop_instance": False,
            },
        )

    app._execute_job("missing", "agentic-agent", {"question": "q", "model": "m", "backend_url": None, "backend_warmup": False, "backend_warmup_timeout": 1, "kubeconfig": "/tmp/k"})  # nosec  # test artifact paths
    assert store.get_job("missing") is None


def test_job_to_payload_fields(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    job_id, _ = store.create_job(tenant_id="t", kind="agentic-agent", request_payload={"x": 1})
    store.update_job(job_id, status="failed", error="boom", message="bad")
    job = store.get_job(job_id)
    payload = api_server._job_to_payload(job)
    assert payload["error"] == "boom"
    assert payload["kind"] == "agentic-agent"
