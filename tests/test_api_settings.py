# SPDX-License-Identifier: Apache-2.0
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from rune_bench.api_server import RuneApiApplication, ApiSecurityConfig
from rune_bench.storage.sqlite import SQLiteStorageAdapter

@pytest.fixture
def temp_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "rune.yaml"
        config_path.write_text("""
version: "1"
defaults:
  model: llama3.1:8b
  backend: local
profiles:
  test:
    model: llama3.1:70b
""")
        # Mock the candidates in rune_bench.common.config
        with patch("rune_bench.common.config._PROJECT_CANDIDATES", [config_path]), \
             patch("rune_bench.common.config._GLOBAL_CANDIDATES", []):
            yield config_path

@pytest.fixture
def api_app():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmpdb:
        storage = SQLiteStorageAdapter(Path(tmpdb.name))
        security = ApiSecurityConfig(auth_disabled=True, tenant_tokens={})
        app = RuneApiApplication(store=storage, security=security)
        yield app

class MockRequest:
    def __init__(self, path, headers=None, body=None):
        self.path = path
        self.headers = headers or {}
        self.body = body or b""
        self.client_address = ("127.0.0.1", 12345)
        self.rfile = tempfile.TemporaryFile()
        self.rfile.write(self.body)
        self.rfile.seek(0)
        self.wfile = tempfile.TemporaryFile()
        self.status_code = None
        self.response_headers = {}

    def send_response(self, code):
        self.status_code = code

    def send_header(self, key, value):
        self.response_headers[key] = value

    def end_headers(self):
        pass

def test_get_settings(api_app, temp_config):
    handler_class = api_app.create_handler()
    mock_req = MockRequest("/v1/settings")
    
    handler = handler_class.__new__(handler_class)
    handler.path = mock_req.path
    handler.headers = mock_req.headers
    handler.rfile = mock_req.rfile
    handler.wfile = mock_req.wfile
    handler.client_address = mock_req.client_address
    handler.server = type('Server', (), {'app': api_app})
    # Inject helper methods from BaseHTTPRequestHandler or mock them
    handler.send_response = mock_req.send_response
    handler.send_header = mock_req.send_header
    handler.end_headers = mock_req.end_headers

    handler.do_GET()
    
    mock_req.wfile.seek(0)
    response_body = json.loads(mock_req.wfile.read().decode("utf-8"))
    
    assert mock_req.status_code == 200
    assert response_body["defaults"]["model"] == "llama3.1:8b"
    assert "test" in response_body["profiles"]
    assert response_body["effective_config"]["model"] == "llama3.1:8b"

def test_patch_settings(api_app, temp_config):
    handler_class = api_app.create_handler()
    update_payload = json.dumps({
        "settings": {"model": "llama3.1:405b"},
        "profile": "test"
    }).encode("utf-8")
    mock_req = MockRequest("/v1/settings", headers={"Content-Length": str(len(update_payload))}, body=update_payload)
    
    handler = handler_class.__new__(handler_class)
    handler.path = mock_req.path
    handler.headers = mock_req.headers
    handler.rfile = mock_req.rfile
    handler.wfile = mock_req.wfile
    handler.client_address = mock_req.client_address
    handler.server = type('Server', (), {'app': api_app})
    handler.send_response = mock_req.send_response
    handler.send_header = mock_req.send_header
    handler.end_headers = mock_req.end_headers

    handler.do_PATCH()
    
    assert mock_req.status_code == 200
    
    # Verify file was updated
    with open(temp_config, "r") as f:
        import yaml
        data = yaml.safe_load(f)
        assert data["profiles"]["test"]["model"] == "llama3.1:405b"

def test_post_profile(api_app, temp_config):
    handler_class = api_app.create_handler()
    profile_payload = json.dumps({
        "name": "new-profile",
        "settings": {"vastai": True}
    }).encode("utf-8")
    mock_req = MockRequest("/v1/settings/profiles", headers={"Content-Length": str(len(profile_payload))}, body=profile_payload)
    
    handler = handler_class.__new__(handler_class)
    handler.path = mock_req.path
    handler.headers = mock_req.headers
    handler.rfile = mock_req.rfile
    handler.wfile = mock_req.wfile
    handler.client_address = mock_req.client_address
    handler.server = type('Server', (), {'app': api_app})
    handler.send_response = mock_req.send_response
    handler.send_header = mock_req.send_header
    handler.end_headers = mock_req.end_headers

    handler.do_POST()
    
    assert mock_req.status_code == 201
    
def test_put_settings_defaults(api_app, temp_config):
    handler_class = api_app.create_handler()
    update_payload = json.dumps({
        "settings": {"new_default": "value"},
        "profile": None
    }).encode("utf-8")
    mock_req = MockRequest("/v1/settings", headers={"Content-Length": str(len(update_payload))}, body=update_payload)
    
    handler = handler_class.__new__(handler_class)
    handler.path = mock_req.path
    handler.headers = mock_req.headers
    handler.rfile = mock_req.rfile
    handler.wfile = mock_req.wfile
    handler.client_address = mock_req.client_address
    handler.server = type('Server', (), {'app': api_app})
    handler.send_response = mock_req.send_response
    handler.send_header = mock_req.send_header
    handler.end_headers = mock_req.end_headers

    handler.do_PUT()
    
    assert mock_req.status_code == 200
    
    # Verify file was updated
    with open(temp_config, "r") as f:
        import yaml
        data = yaml.safe_load(f)
        assert data["defaults"]["new_default"] == "value"

def test_put_settings_new_profile(api_app, temp_config):
    handler_class = api_app.create_handler()
    update_payload = json.dumps({
        "settings": {"foo": "bar"},
        "profile": "nonexistent"
    }).encode("utf-8")
    mock_req = MockRequest("/v1/settings", headers={"Content-Length": str(len(update_payload))}, body=update_payload)
    
    handler = handler_class.__new__(handler_class)
    handler.path = mock_req.path
    handler.headers = mock_req.headers
    handler.rfile = mock_req.rfile
    handler.wfile = mock_req.wfile
    handler.client_address = mock_req.client_address
    handler.server = type('Server', (), {'app': api_app})
    handler.send_response = mock_req.send_response
    handler.send_header = mock_req.send_header
    handler.end_headers = mock_req.end_headers

    handler.do_PUT()
    
    assert mock_req.status_code == 200
    
    # Verify file was updated
    with open(temp_config, "r") as f:
        import yaml
        data = yaml.safe_load(f)
        assert data["profiles"]["nonexistent"]["foo"] == "bar"
