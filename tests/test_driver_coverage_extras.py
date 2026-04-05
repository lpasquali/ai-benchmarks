
"""Extra tests for driver coverage gaps."""

from __future__ import annotations

import io
import json
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# PentestGPT _normalize_model prefix stripping (line 53)
# ---------------------------------------------------------------------------

import rune_bench.drivers.pentestgpt.__main__ as ptgpt_main


def test_pentestgpt_normalize_model_strips_prefix() -> None:
    assert ptgpt_main._normalize_model("ollama/llama3.1:8b") == "llama3.1:8b"
    assert ptgpt_main._normalize_model("ollama_chat/mistral:7b") == "mistral:7b"
    assert ptgpt_main._normalize_model("llama3.1:8b") == "llama3.1:8b"


# ---------------------------------------------------------------------------
# PentestGPT _check_authorization (lines 66-72)
# ---------------------------------------------------------------------------


def test_pentestgpt_authorization_blocks_unlisted_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_PENTESTGPT_ALLOWED_TARGETS", "example.com")
    with pytest.raises(RuntimeError, match="not in RUNE_PENTESTGPT_ALLOWED_TARGETS"):
        ptgpt_main._check_authorization("scan https://evil.com/path")


def test_pentestgpt_authorization_allows_listed_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNE_PENTESTGPT_ALLOWED_TARGETS", "example.com")
    # Should not raise
    ptgpt_main._check_authorization("scan https://example.com/path")


def test_pentestgpt_authorization_skips_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNE_PENTESTGPT_ALLOWED_TARGETS", raising=False)
    # Should not raise
    ptgpt_main._check_authorization("scan https://evil.com")


# ---------------------------------------------------------------------------
# BurpGPT driver uncovered lines
# ---------------------------------------------------------------------------

import rune_bench.drivers.burpgpt.__main__ as burp_main


def test_burpgpt_handle_info() -> None:
    info = burp_main._handle_info({})
    assert info["name"] == "burpgpt"
    assert "ask" in info["actions"]


# ---------------------------------------------------------------------------
# Dagger driver — cover _load_pipeline_command and more of _handle_ask
# ---------------------------------------------------------------------------

import rune_bench.drivers.dagger.__main__ as dagger_main


def test_dagger_handle_info() -> None:
    info = dagger_main._handle_info({})
    assert info["name"] == "dagger"
    assert "ask" in info["actions"]


def test_dagger_load_pipeline_missing() -> None:
    with pytest.raises(FileNotFoundError, match="Pipeline template"):
        dagger_main._load_pipeline_command("nonexistent_pipeline", "test")


def test_dagger_main_loop_info(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(
        dagger_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "info", "params": {}, "id": "d1"}) + "\n"),
    )
    dagger_main.main()
    resp = json.loads(capsys.readouterr().out.strip())
    assert resp["status"] == "ok"
    assert resp["result"]["name"] == "dagger"


def test_dagger_main_loop_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(
        dagger_main.sys,
        "stdin",
        io.StringIO(json.dumps({"action": "bad", "id": "d2"}) + "\n"),
    )
    dagger_main.main()
    resp = json.loads(capsys.readouterr().out.strip())
    assert resp["status"] == "error"
