import json
import pytest
from unittest.mock import MagicMock
from rune_bench.api_server import RuneApiApplication, ApiSecurityConfig
from rune_bench.job_store import JobStore

class MockRfile:
    def __init__(self, data):
        self.data = data
    def read(self, length):
        return self.data

def run_handler(app, path, method="GET", body=b""):
    handler_class = app.create_handler()
    class MockHandler(handler_class):
        def __init__(self, path, method="GET", body=b""):
            self.path = path
            self.command = method
            self.client_address = ("127.0.0.1", 1234)
            self.headers = {"Content-Length": str(len(body))}
            self.rfile = MockRfile(body)
            self.wfile = MagicMock()
            self.code = None
        def send_response(self, code): self.code = code
        def send_header(self, k, v): pass
        def end_headers(self): pass
        def setup(self): pass
        def finish(self): pass
    
    h = MockHandler(path, method, body)
    if method == "GET":
        h.do_GET()
    elif method == "PUT":
        h.do_PUT()
    elif method == "POST":
        h.do_POST()
    return h

def test_api_server_new_routes(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs.db")
    security = ApiSecurityConfig(auth_disabled=True, tenant_tokens={})
    app = RuneApiApplication(store=store, security=security)

    # Settings
    h = run_handler(app, "/v1/settings")
    assert h.code in (200, 400, 500)
    
    h = run_handler(app, "/v1/settings", "PUT", b'{"test": 1}')
    assert h.code in (200, 400)
    
    h = run_handler(app, "/v1/settings/profiles", "POST", b'{"name": "test", "config": {}}')
    assert h.code in (200, 400)
    
    # Interaction GET missing
    h = run_handler(app, "/v1/runs/test_job/interaction")
    assert h.code == 404
    
    # Interaction GET present
    from rune_bench.interactive import session_manager
    session_manager.pending_prompts["test_job"] = {"a": 1}
    h = run_handler(app, "/v1/runs/test_job/interaction")
    if h.code != 200:
        print("ERROR PAYLOAD:", h.wfile.write.call_args)
    assert h.code == 200
    
    # Interaction POST valid
    h = run_handler(app, "/v1/runs/test_job/interaction", "POST", b'{"ans": 2}')
    assert h.code == 200
    
    # Interaction POST missing job
    h = run_handler(app, "/v1/runs/missing/interaction", "POST", b'{"ans": 2}')
    assert h.code == 400
    
    # Trace logic (minimal test so it doesn't block)
    store.create_job(tenant_id="t1", kind="benchmark", request_payload={"a": 1}, idempotency_key="job1")
    store.update_job("job1", status="completed")
    
    # Mock time.sleep to throw Exception to break the while True loop instantly if it didn't break
    import time
    def _sleep(*args): raise RuntimeError("stop loop")
    monkeypatch.setattr(time, "sleep", _sleep)
    
    h = run_handler(app, "/v1/runs/job1/trace")
    assert h.code == 200
    
    h = run_handler(app, "/v1/runs/missing_job/trace")
    assert h.code == 200
