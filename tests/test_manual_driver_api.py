import pytest
from unittest.mock import patch, MagicMock
from rune_bench.drivers.manual import ManualDriverTransport
import os

def test_manual_driver_api_mode(monkeypatch):
    from rune_bench.interactive import session_manager
    transport = ManualDriverTransport()
    
    # Mock sys.stdout.isatty to False to trigger API mode
    import sys
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    
    from rune_bench.metrics import set_job_id, _tls
    set_job_id("job-123")
    
    def fake_request(job_id, prompt):
        assert job_id == "job-123"
        return {"result": "test"}
    
    monkeypatch.setattr(session_manager, "request_input", fake_request)
    
    res = transport.call("do_thing", {"a": 1})
    assert res == {"result": "test"}
    
    set_job_id(None)
