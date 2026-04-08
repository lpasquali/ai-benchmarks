# SPDX-License-Identifier: Apache-2.0
"""Tests for CrewAI and LangGraph driver clients (__init__.py coverage)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rune_bench.drivers.crewai import CrewAIDriverClient
from rune_bench.drivers.langgraph import LangGraphDriverClient


# ---------------------------------------------------------------------------
# CrewAIDriverClient
# ---------------------------------------------------------------------------


class TestCrewAIDriverClient:
    def test_ask_success(self) -> None:
        transport = MagicMock()
        transport.call.return_value = {"answer": "crew result"}
        client = CrewAIDriverClient(transport=transport)

        result = client.ask("What happened?", "llama3.1:8b", backend_url="http://localhost:11434")

        assert result == "crew result"
        transport.call.assert_called_once_with("ask", {
            "question": "What happened?",
            "model": "llama3.1:8b",
            "backend_url": "http://localhost:11434",
        })

    def test_ask_without_backend_url(self) -> None:
        transport = MagicMock()
        transport.call.return_value = {"answer": "ok"}
        client = CrewAIDriverClient(transport=transport)

        result = client.ask("q", "m")

        assert result == "ok"
        transport.call.assert_called_once_with("ask", {
            "question": "q",
            "model": "m",
        })

    def test_ask_missing_answer_key(self) -> None:
        transport = MagicMock()
        transport.call.return_value = {"other": "data"}
        client = CrewAIDriverClient(transport=transport)

        with pytest.raises(RuntimeError, match="did not include an answer"):
            client.ask("q", "m")

    def test_ask_none_answer(self) -> None:
        transport = MagicMock()
        transport.call.return_value = {"answer": None}
        client = CrewAIDriverClient(transport=transport)

        with pytest.raises(RuntimeError, match="empty answer"):
            client.ask("q", "m")

    def test_ask_empty_answer(self) -> None:
        transport = MagicMock()
        transport.call.return_value = {"answer": ""}
        client = CrewAIDriverClient(transport=transport)

        with pytest.raises(RuntimeError, match="empty answer"):
            client.ask("q", "m")

    def test_model_whitespace_stripped(self) -> None:
        transport = MagicMock()
        transport.call.return_value = {"answer": "ok"}
        client = CrewAIDriverClient(transport=transport)

        client.ask("q", "  llama3.1:8b  ")

        call_args = transport.call.call_args[0][1]
        assert call_args["model"] == "llama3.1:8b"


# ---------------------------------------------------------------------------
# LangGraphDriverClient
# ---------------------------------------------------------------------------


class TestLangGraphDriverClient:
    def test_ask_success(self) -> None:
        transport = MagicMock()
        transport.call.return_value = {"answer": "lg result"}
        client = LangGraphDriverClient(transport=transport)

        result = client.ask("Research topic", "llama3.1:8b", backend_url="http://localhost:11434")

        assert result == "lg result"
        transport.call.assert_called_once_with("ask", {
            "question": "Research topic",
            "model": "llama3.1:8b",
            "backend_url": "http://localhost:11434",
        })

    def test_ask_without_backend_url(self) -> None:
        transport = MagicMock()
        transport.call.return_value = {"answer": "ok"}
        client = LangGraphDriverClient(transport=transport)

        result = client.ask("q", "m")

        assert result == "ok"
        transport.call.assert_called_once_with("ask", {
            "question": "q",
            "model": "m",
        })

    def test_ask_missing_answer_key(self) -> None:
        transport = MagicMock()
        transport.call.return_value = {"other": "data"}
        client = LangGraphDriverClient(transport=transport)

        with pytest.raises(RuntimeError, match="did not include an answer"):
            client.ask("q", "m")

    def test_ask_none_answer(self) -> None:
        transport = MagicMock()
        transport.call.return_value = {"answer": None}
        client = LangGraphDriverClient(transport=transport)

        with pytest.raises(RuntimeError, match="empty answer"):
            client.ask("q", "m")

    def test_ask_empty_answer(self) -> None:
        transport = MagicMock()
        transport.call.return_value = {"answer": ""}
        client = LangGraphDriverClient(transport=transport)

        # Empty string is valid — driver returns it as-is
        assert client.ask("q", "m") == ""

    def test_model_whitespace_stripped(self) -> None:
        transport = MagicMock()
        transport.call.return_value = {"answer": "ok"}
        client = LangGraphDriverClient(transport=transport)

        client.ask("q", "  llama3.1:8b  ")

        call_args = transport.call.call_args[0][1]
        assert call_args["model"] == "llama3.1:8b"
