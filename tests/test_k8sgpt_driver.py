# SPDX-License-Identifier: Apache-2.0
"""Tests for rune_bench.drivers.k8sgpt — the driver entry point and client.

The driver process calls ``k8sgpt analyze`` as a subprocess.
subprocess.run and shutil.which are monkeypatched throughout so no k8sgpt
installation is required.
"""

from __future__ import annotations

import io
import json
import subprocess  # nosec  # tests require subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import rune_bench.drivers.k8sgpt.__main__ as k8sgpt_main
from rune_bench.agents.sre.k8sgpt import K8sGPTRunner
from rune_bench.drivers.k8sgpt import K8sGPTDriverClient


# ---------------------------------------------------------------------------
# Sample k8sgpt output
# ---------------------------------------------------------------------------

_SAMPLE_RESULTS = {
    "results": [
        {
            "kind": "Pod",
            "name": "default/nginx-broken",
            "error": [{"text": "Back-off pulling image"}],
            "details": "The image 'nginx:nonexistent' cannot be found.",
            "parent_object": "Deployment/nginx",
        },
        {
            "kind": "Service",
            "name": "default/my-svc",
            "error": [{"text": "No endpoints"}],
            "details": "Service has no matching pods.",
            "parent_object": "",
        },
    ]
}


# ---------------------------------------------------------------------------
# _handle_ask
# ---------------------------------------------------------------------------


def test_handle_ask_calls_k8sgpt_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_run(
        cmd: list, env: dict, capture_output: bool, text: bool, check: bool
    ) -> subprocess.CompletedProcess:
        captured["cmd"] = cmd
        captured["env"] = env
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(_SAMPLE_RESULTS), stderr=""
        )

    monkeypatch.setattr(k8sgpt_main.shutil, "which", lambda _: "/usr/bin/k8sgpt")
    monkeypatch.setattr(k8sgpt_main.subprocess, "run", fake_run)

    result = k8sgpt_main._handle_ask(
        {
            "question": "What is wrong?",
            "model": "llama3.1:8b",
            "kubeconfig_path": "/tmp/kubeconfig",  # nosec  # test artifact paths
            "backend_url": "http://ollama:11434",
        }
    )

    assert "answer" in result
    assert "findings" in result
    assert len(result["findings"]) == 2
    assert "nginx-broken" in result["answer"]
    assert "k8sgpt" in captured["cmd"]
    assert captured["env"]["KUBECONFIG"] == "/tmp/kubeconfig"  # nosec  # test artifact paths
    assert "--base-url" in captured["cmd"]
    assert "http://ollama:11434" in captured["cmd"]


def test_handle_ask_without_backend_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_run(cmd: list, **kw) -> subprocess.CompletedProcess:
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(_SAMPLE_RESULTS), stderr=""
        )

    monkeypatch.setattr(k8sgpt_main.shutil, "which", lambda _: "/usr/bin/k8sgpt")
    monkeypatch.setattr(k8sgpt_main.subprocess, "run", fake_run)

    result = k8sgpt_main._handle_ask(
        {
            "question": "q",
            "model": "m",
            "kubeconfig_path": "/tmp/kc",  # nosec  # test artifact paths
        }
    )
    assert result["answer"]
    assert "--base-url" not in captured["cmd"]


def test_handle_ask_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(k8sgpt_main.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match="k8sgpt binary not found"):
        k8sgpt_main._handle_ask(
            {
                "question": "q",
                "model": "m",
                "kubeconfig_path": "/tmp/kc",  # nosec  # test artifact paths
            }
        )


def test_handle_ask_empty_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(k8sgpt_main.shutil, "which", lambda _: "/usr/bin/k8sgpt")
    monkeypatch.setattr(
        k8sgpt_main.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(
            [], 0, stdout=json.dumps({"results": None}), stderr=""
        ),
    )

    result = k8sgpt_main._handle_ask(
        {
            "question": "q",
            "model": "m",
            "kubeconfig_path": "/tmp/kc",  # nosec  # test artifact paths
        }
    )
    assert result["answer"] == "No issues detected"
    assert result["findings"] == []


def test_handle_ask_empty_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(k8sgpt_main.shutil, "which", lambda _: "/usr/bin/k8sgpt")
    monkeypatch.setattr(
        k8sgpt_main.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess([], 0, stdout="", stderr=""),
    )

    result = k8sgpt_main._handle_ask(
        {
            "question": "q",
            "model": "m",
            "kubeconfig_path": "/tmp/kc",  # nosec  # test artifact paths
        }
    )
    assert result["answer"] == "No issues detected"


def test_handle_ask_raises_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(k8sgpt_main.shutil, "which", lambda _: "/usr/bin/k8sgpt")
    monkeypatch.setattr(
        k8sgpt_main.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(
            [], 1, stdout="", stderr="k8sgpt error"
        ),
    )

    with pytest.raises(RuntimeError, match="k8sgpt error"):
        k8sgpt_main._handle_ask(
            {"question": "q", "model": "m", "kubeconfig_path": "/tmp/kc"}
        )  # nosec  # test artifact paths


def test_handle_ask_resource_kind_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_run(cmd: list, **kw) -> subprocess.CompletedProcess:
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps({"results": []}), stderr=""
        )

    monkeypatch.setattr(k8sgpt_main.shutil, "which", lambda _: "/usr/bin/k8sgpt")
    monkeypatch.setattr(k8sgpt_main.subprocess, "run", fake_run)

    k8sgpt_main._handle_ask(
        {
            "question": "Pod",
            "model": "m",
            "kubeconfig_path": "/tmp/kc",  # nosec  # test artifact paths
        }
    )
    assert "--filter" in captured["cmd"]
    assert "Pod" in captured["cmd"]


# ---------------------------------------------------------------------------
# _format_findings
# ---------------------------------------------------------------------------


def test_format_findings_produces_readable_output() -> None:
    results = _SAMPLE_RESULTS["results"]
    formatted = k8sgpt_main._format_findings(results)

    assert "Finding 1" in formatted
    assert "Pod/default/nginx-broken" in formatted
    assert "Back-off pulling image" in formatted
    assert "Parent: Deployment/nginx" in formatted
    assert "Finding 2" in formatted
    assert "Service/default/my-svc" in formatted


def test_format_findings_empty_list() -> None:
    assert k8sgpt_main._format_findings([]) == ""


# ---------------------------------------------------------------------------
# _handle_info
# ---------------------------------------------------------------------------


def test_handle_info_returns_driver_metadata() -> None:
    result = k8sgpt_main._handle_info({})
    assert result["name"] == "k8sgpt"
    assert "ask" in result["actions"]
    assert "info" in result["actions"]


# ---------------------------------------------------------------------------
# main() — full stdin/stdout loop
# ---------------------------------------------------------------------------


def test_main_processes_ask_request(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        k8sgpt_main, "_handle_ask", lambda p: {"answer": "great answer", "findings": []}
    )
    monkeypatch.setattr(
        k8sgpt_main.sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "action": "ask",
                    "params": {
                        "question": "q",
                        "model": "m",
                        "kubeconfig_path": "/tmp/kc",
                    },  # nosec  # test artifact paths
                    "id": "test-id",
                }
            )
            + "\n"
        ),
    )

    k8sgpt_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "ok"
    assert response["result"]["answer"] == "great answer"
    assert response["id"] == "test-id"


def test_main_returns_error_for_unknown_action(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        k8sgpt_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "unknown", "params": {}, "id": "u1"}) + "\n"),
    )

    k8sgpt_main.main()

    response = json.loads(capsys.readouterr().out.strip())
    assert response["status"] == "error"
    assert "unknown" in response["error"].lower()


# ---------------------------------------------------------------------------
# K8sGPTDriverClient
# ---------------------------------------------------------------------------


def test_client_init_requires_existing_kubeconfig(tmp_path: Path) -> None:
    missing = tmp_path / "missing-kubeconfig"
    with pytest.raises(FileNotFoundError):
        K8sGPTDriverClient(missing)


def test_client_ask_calls_transport(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")

    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": "the answer", "findings": []}

    client = K8sGPTDriverClient(kubeconfig, transport=mock_transport)
    answer = client.ask(
        "What is wrong?", "llama3.1:8b", backend_url="http://ollama:11434"
    )

    assert answer == "the answer"
    mock_transport.call.assert_called_once()
    action, params = mock_transport.call.call_args[0]
    assert action == "ask"
    assert params["question"] == "What is wrong?"
    assert params["model"] == "llama3.1:8b"
    assert params["kubeconfig_path"] == str(kubeconfig)
    assert params["backend_url"] == "http://ollama:11434"


def test_client_ask_strips_model_whitespace(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")

    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": "ok"}

    client = K8sGPTDriverClient(kubeconfig, transport=mock_transport)
    client.ask("q", "  llama3.1:8b  ")

    _, params = mock_transport.call.call_args[0]
    assert params["model"] == "llama3.1:8b"


def test_client_ask_raises_on_missing_answer(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")

    mock_transport = MagicMock()
    mock_transport.call.return_value = {"findings": []}

    client = K8sGPTDriverClient(kubeconfig, transport=mock_transport)
    with pytest.raises(RuntimeError, match="did not include an answer"):
        client.ask("q", "m")


def test_client_ask_raises_on_none_answer(tmp_path: Path) -> None:
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\n")

    mock_transport = MagicMock()
    mock_transport.call.return_value = {"answer": None}

    client = K8sGPTDriverClient(kubeconfig, transport=mock_transport)
    with pytest.raises(RuntimeError, match="empty answer"):
        client.ask("q", "m")


def test_runner_alias() -> None:
    assert K8sGPTRunner is K8sGPTDriverClient


def test_format_findings_with_string_error() -> None:
    """_format_findings must not iterate character-by-character when error is a string."""
    results = {
        "results": [
            {
                "kind": "Pod",
                "name": "default/broken",
                "error": "Back-off pulling image",
                "details": "",
                "parent_object": "",
            }
        ]
    }
    output = k8sgpt_main._format_findings(results["results"])
    assert "Back-off pulling image" in output
    # If iterated char-by-char the result would have "B", "a", "c", ... on separate lines
    assert "Error: B\n" not in output
