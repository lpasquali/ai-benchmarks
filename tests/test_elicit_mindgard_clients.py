# SPDX-License-Identifier: Apache-2.0
"""Tests for Elicit and Mindgard driver clients (__init__.py coverage)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rune_bench.drivers.elicit import ElicitDriverClient
from rune_bench.drivers.mindgard import MindgardDriverClient


class TestElicitDriverClient:
    def test_ask_success(self) -> None:
        transport = MagicMock()
        transport.call.return_value = {"answer": "literature review result"}
        client = ElicitDriverClient(transport=transport)

        result = client.ask("What is X?", "m", backend_url="http://localhost:11434")
        assert result == "literature review result"
        transport.call.assert_called_once()

    def test_ask_without_backend_url(self) -> None:
        transport = MagicMock()
        transport.call.return_value = {"answer": "ok"}
        client = ElicitDriverClient(transport=transport)

        result = client.ask("q", "m")
        assert result == "ok"

    def test_ask_missing_answer(self) -> None:
        transport = MagicMock()
        transport.call.return_value = {"other": "data"}
        client = ElicitDriverClient(transport=transport)

        with pytest.raises(RuntimeError, match="did not include an answer"):
            client.ask("q", "m")

    def test_ask_none_answer(self) -> None:
        transport = MagicMock()
        transport.call.return_value = {"answer": None}
        client = ElicitDriverClient(transport=transport)

        with pytest.raises(RuntimeError, match="empty answer"):
            client.ask("q", "m")

    def test_ask_empty_answer(self) -> None:
        transport = MagicMock()
        transport.call.return_value = {"answer": ""}
        client = ElicitDriverClient(transport=transport)

        with pytest.raises(RuntimeError, match="empty answer"):
            client.ask("q", "m")


class TestMindgardDriverClient:
    def test_ask_success(self) -> None:
        transport = MagicMock()
        transport.call.return_value = {"answer": "security assessment"}
        client = MindgardDriverClient(transport=transport)

        result = client.ask(
            "test the model", "llama3:8b", backend_url="http://target:11434"
        )
        assert result == "security assessment"
        transport.call.assert_called_once()

    def test_ask_without_backend_url(self) -> None:
        transport = MagicMock()
        transport.call.return_value = {"answer": "ok"}
        client = MindgardDriverClient(transport=transport)

        result = client.ask("q", "m")
        assert result == "ok"

    def test_ask_missing_answer(self) -> None:
        transport = MagicMock()
        transport.call.return_value = {"other": "data"}
        client = MindgardDriverClient(transport=transport)

        with pytest.raises(RuntimeError, match="did not include an answer"):
            client.ask("q", "m")

    def test_ask_none_answer(self) -> None:
        transport = MagicMock()
        transport.call.return_value = {"answer": None}
        client = MindgardDriverClient(transport=transport)

        with pytest.raises(RuntimeError, match="empty answer"):
            client.ask("q", "m")

    def test_ask_empty_answer(self) -> None:
        transport = MagicMock()
        transport.call.return_value = {"answer": ""}
        client = MindgardDriverClient(transport=transport)

        with pytest.raises(RuntimeError, match="empty answer"):
            client.ask("q", "m")
