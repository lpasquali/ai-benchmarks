# SPDX-License-Identifier: Apache-2.0
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
from typer.testing import CliRunner

import rune


@pytest.fixture
def mock_rune_api_server():
    state = {
        "agent_polls": 0,
        "bench_polls": 0,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A003
            return

        def _write_json(self, status: int, payload: dict):
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/v1/catalog/vastai-models":
                self._write_json(
                    200,
                    {
                        "models": [
                            {
                                "name": "llama3.1:8b",
                                "vram_mb": 8000,
                                "required_disk_gb": 41,
                            }
                        ]
                    },
                )
                return

            if path == "/v1/ollama/models":
                query = parse_qs(parsed.query)
                backend_url = query.get("backend_url", ["http://localhost:11434"])[0]
                self._write_json(
                    200,
                    {
                        "backend_url": backend_url,
                        "models": ["llama3.1:8b"],
                        "running_models": ["llama3.1:8b"],
                    },
                )
                return

            if path == "/v1/jobs/agent-1":
                state["agent_polls"] += 1
                if state["agent_polls"] < 2:
                    self._write_json(200, {"status": "running", "message": "analyzing cluster"})
                else:
                    self._write_json(200, {"status": "succeeded", "result": {"answer": "agent-http-answer"}})
                return

            if path == "/v1/jobs/bench-1":
                state["bench_polls"] += 1
                if state["bench_polls"] < 2:
                    self._write_json(200, {"status": "running", "message": "phase 2"})
                else:
                    self._write_json(200, {"status": "succeeded", "result": {"answer": "benchmark-http-answer"}})
                return

            self._write_json(404, {"error": f"Unknown path: {path}"})

        def do_POST(self):
            parsed = urlparse(self.path)
            path = parsed.path

            length = int(self.headers.get("Content-Length", "0"))
            _body = self.rfile.read(length) if length else b""

            if path == "/v1/jobs/agentic-agent":
                self._write_json(200, {"job_id": "agent-1"})
                return

            if path == "/v1/jobs/benchmark":
                self._write_json(200, {"job_id": "bench-1"})
                return

            self._write_json(404, {"error": f"Unknown path: {path}"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    host, port = server.server_address
    base_url = f"http://{host}:{port}"
    try:
        yield base_url
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_cli_http_ollama_list_models(mock_rune_api_server):
    runner = CliRunner()
    result = runner.invoke(
        rune.app,
        [
            "--backend",
            "http",
            "--api-base-url",
            mock_rune_api_server,
            "ollama-list-models",
            "--backend-url",
            "http://example:11434",
        ],
    )

    assert result.exit_code == 0
    assert "llama3.1:8b" in result.stdout
    assert "http://example:11434" in result.stdout


def test_cli_http_run_agentic_agent_job_flow(mock_rune_api_server):
    runner = CliRunner()
    result = runner.invoke(
        rune.app,
        [
            "--backend",
            "http",
            "--api-base-url",
            mock_rune_api_server,
            "run-agentic-agent",
            "--question",
            "What is unhealthy?",
            "--model",
            "llama3.1:8b",
        ],
    )

    assert result.exit_code == 0
    assert "agent-http-answer" in result.stdout


def test_cli_http_run_benchmark_job_flow(mock_rune_api_server):
    runner = CliRunner()
    result = runner.invoke(
        rune.app,
        [
            "--backend",
            "http",
            "--api-base-url",
            mock_rune_api_server,
            "run-benchmark",
            "--question",
            "Why degraded?",
            "--model",
            "llama3.1:8b",
        ],
    )

    assert result.exit_code == 0
    assert "benchmark-http-answer" in result.stdout
