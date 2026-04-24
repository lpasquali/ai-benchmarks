# SPDX-License-Identifier: Apache-2.0
"""Tests for new transport types (Manual, Browser), async factories,
ChainExecutionEngine, and driver ask_structured/ask_async methods."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rune_bench.agents.base import AgentResult


# ---------------------------------------------------------------------------
# ChainExecutionEngine
# ---------------------------------------------------------------------------


class TestChainExecutionEngine:
    def test_simple_chain_executes(self) -> None:
        from rune_bench.agents.chain import ChainExecutionEngine, ChainStep

        agent = MagicMock()
        agent.ask_async = AsyncMock(return_value=AgentResult(answer="result"))

        steps = [
            ChainStep(name="step1", agent=agent, question_template="What is {topic}?")
        ]
        engine = ChainExecutionEngine(steps)

        result = asyncio.run(engine.execute({"topic": "AI"}, model="m"))
        assert "step1" in result.steps
        assert result.steps["step1"].answer == "result"

    def test_chain_with_dependencies(self) -> None:
        from rune_bench.agents.chain import ChainExecutionEngine, ChainStep

        agent1 = MagicMock()
        agent1.ask_async = AsyncMock(return_value=AgentResult(answer="first"))

        agent2 = MagicMock()
        agent2.ask_async = AsyncMock(return_value=AgentResult(answer="second"))

        steps = [
            ChainStep(name="a", agent=agent1, question_template="Start {topic}"),
            ChainStep(
                name="b",
                agent=agent2,
                question_template="Continue: {a}",
                dependencies=["a"],
            ),
        ]
        engine = ChainExecutionEngine(steps)

        result = asyncio.run(
            engine.execute({"topic": "AI"}, model="m", backend_url="http://x")
        )
        assert result.steps["b"].answer == "second"

    def test_cycle_detection(self) -> None:
        from rune_bench.agents.chain import ChainExecutionEngine, ChainStep

        agent = MagicMock()
        steps = [
            ChainStep(name="a", agent=agent, question_template="q", dependencies=["b"]),
            ChainStep(name="b", agent=agent, question_template="q", dependencies=["a"]),
        ]
        with pytest.raises(ValueError, match="Cycle detected"):
            ChainExecutionEngine(steps)

    def test_unknown_dependency(self) -> None:
        from rune_bench.agents.chain import ChainExecutionEngine, ChainStep

        agent = MagicMock()
        steps = [
            ChainStep(
                name="a",
                agent=agent,
                question_template="q",
                dependencies=["nonexistent"],
            )
        ]
        with pytest.raises(ValueError, match="unknown step"):
            ChainExecutionEngine(steps)

    def test_template_missing_context(self) -> None:
        from rune_bench.agents.chain import ChainExecutionEngine, ChainStep

        agent = MagicMock()
        agent.ask_async = AsyncMock(return_value=AgentResult(answer="ok"))

        steps = [ChainStep(name="a", agent=agent, question_template="{missing_key}")]
        engine = ChainExecutionEngine(steps)

        with pytest.raises(RuntimeError, match="template missing context"):
            asyncio.run(engine.execute({}, model="m"))

    def test_chain_result_dataclass(self) -> None:
        from rune_bench.agents.chain import ChainResult

        r = ChainResult(steps={"a": AgentResult(answer="ok")})
        assert r.steps["a"].answer == "ok"
        assert r.metadata == {}

    def test_chain_step_dataclass(self) -> None:
        from rune_bench.agents.chain import ChainStep

        agent = MagicMock()
        step = ChainStep(name="s", agent=agent, question_template="q")
        assert step.name == "s"
        assert step.dependencies == []

    # ── Recorder integration ────────────────────────────────────────────

    def test_engine_without_recorder_does_not_raise(self) -> None:
        """Existing call sites that pass no recorder must continue to work."""
        from rune_bench.agents.chain import ChainExecutionEngine, ChainStep

        agent = MagicMock()
        agent.ask_async = AsyncMock(return_value=AgentResult(answer="ok"))
        engine = ChainExecutionEngine(
            [ChainStep(name="a", agent=agent, question_template="{topic}")],
        )
        # No recorder, no job_id → no exceptions
        result = asyncio.run(engine.execute({"topic": "x"}, model="m"))
        assert result.steps["a"].answer == "ok"

    def test_engine_with_recorder_emits_initialize_and_transitions(self) -> None:
        from rune_bench.agents.chain import ChainExecutionEngine, ChainStep

        agent = MagicMock()
        agent.ask_async = AsyncMock(return_value=AgentResult(answer="ok"))

        recorder = MagicMock()
        engine = ChainExecutionEngine(
            [ChainStep(name="step1", agent=agent, question_template="{topic}")],
            recorder=recorder,
            job_id="job-42",
        )

        asyncio.run(engine.execute({"topic": "x"}, model="m"))

        # initialize called once with the full DAG shell
        recorder.initialize.assert_called_once()
        init_kwargs = recorder.initialize.call_args.kwargs
        assert init_kwargs["job_id"] == "job-42"
        assert [n["id"] for n in init_kwargs["nodes"]] == ["step1"]
        assert init_kwargs["edges"] == []
        assert all(n["status"] == "pending" for n in init_kwargs["nodes"])

        # transition called with running then success for the step
        statuses = [c.kwargs["status"] for c in recorder.transition.call_args_list]
        assert statuses == ["running", "success"]
        for call in recorder.transition.call_args_list:
            assert call.kwargs["job_id"] == "job-42"
            assert call.kwargs["node_id"] == "step1"

    def test_engine_recorder_records_dependency_edges(self) -> None:
        from rune_bench.agents.chain import ChainExecutionEngine, ChainStep

        agent = MagicMock()
        agent.ask_async = AsyncMock(return_value=AgentResult(answer="ok"))
        recorder = MagicMock()

        steps = [
            ChainStep(name="a", agent=agent, question_template="{topic}"),
            ChainStep(
                name="b", agent=agent, question_template="{a}", dependencies=["a"]
            ),
        ]
        engine = ChainExecutionEngine(steps, recorder=recorder, job_id="job-1")
        asyncio.run(engine.execute({"topic": "x"}, model="m"))

        edges = recorder.initialize.call_args.kwargs["edges"]
        assert edges == [{"from": "a", "to": "b"}]

    def test_engine_recorder_records_failure(self) -> None:
        from rune_bench.agents.chain import ChainExecutionEngine, ChainStep

        agent = MagicMock()
        agent.ask_async = AsyncMock(side_effect=RuntimeError("boom"))
        recorder = MagicMock()
        engine = ChainExecutionEngine(
            [ChainStep(name="a", agent=agent, question_template="{topic}")],
            recorder=recorder,
            job_id="job-1",
        )
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(engine.execute({"topic": "x"}, model="m"))

        statuses = [c.kwargs["status"] for c in recorder.transition.call_args_list]
        assert "failed" in statuses
        failed_call = next(
            c
            for c in recorder.transition.call_args_list
            if c.kwargs["status"] == "failed"
        )
        assert failed_call.kwargs["error"] == "boom"

    def test_engine_recorder_records_template_failure(self) -> None:
        from rune_bench.agents.chain import ChainExecutionEngine, ChainStep

        agent = MagicMock()
        agent.ask_async = AsyncMock(return_value=AgentResult(answer="ok"))
        recorder = MagicMock()
        engine = ChainExecutionEngine(
            [ChainStep(name="a", agent=agent, question_template="{missing}")],
            recorder=recorder,
            job_id="job-1",
        )
        with pytest.raises(RuntimeError, match="template missing context"):
            asyncio.run(engine.execute({}, model="m"))

        statuses = [c.kwargs["status"] for c in recorder.transition.call_args_list]
        assert statuses == ["failed"]

    def test_engine_with_recorder_but_no_job_id_skips_calls(self) -> None:
        """Defensive: if job_id is None the recorder is silently ignored."""
        from rune_bench.agents.chain import ChainExecutionEngine, ChainStep

        agent = MagicMock()
        agent.ask_async = AsyncMock(return_value=AgentResult(answer="ok"))
        recorder = MagicMock()
        engine = ChainExecutionEngine(
            [ChainStep(name="a", agent=agent, question_template="{topic}")],
            recorder=recorder,
            job_id=None,
        )
        asyncio.run(engine.execute({"topic": "x"}, model="m"))
        recorder.initialize.assert_not_called()
        recorder.transition.assert_not_called()


# ---------------------------------------------------------------------------
# ManualDriverTransport
# ---------------------------------------------------------------------------


class TestManualDriverTransport:
    def test_call_returns_valid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from rune_bench.drivers.manual import ManualDriverTransport

        console = MagicMock()
        transport = ManualDriverTransport(console=console)

        with patch("rune_bench.drivers.manual.Prompt") as mock_prompt:
            mock_prompt.ask.return_value = '{"answer": "manual result"}'
            result = transport.call("ask", {"question": "q"})

        assert result == {"answer": "manual result"}

    def test_call_abort(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from rune_bench.drivers.manual import ManualDriverTransport

        console = MagicMock()
        transport = ManualDriverTransport(console=console)

        with patch("rune_bench.drivers.manual.Prompt") as mock_prompt:
            mock_prompt.ask.return_value = "abort"
            with pytest.raises(RuntimeError, match="aborted"):
                transport.call("ask", {})

    def test_call_retries_on_invalid_json(self) -> None:
        from rune_bench.drivers.manual import ManualDriverTransport

        console = MagicMock()
        transport = ManualDriverTransport(console=console)

        with patch("rune_bench.drivers.manual.Prompt") as mock_prompt:
            mock_prompt.ask.side_effect = ["not-json", '{"answer": "ok"}']
            result = transport.call("ask", {})

        assert result == {"answer": "ok"}
        # First call returned invalid JSON, console should print error
        assert console.print.call_count >= 1

    def test_call_retries_on_non_dict_json(self) -> None:
        from rune_bench.drivers.manual import ManualDriverTransport

        console = MagicMock()
        transport = ManualDriverTransport(console=console)

        with patch("rune_bench.drivers.manual.Prompt") as mock_prompt:
            mock_prompt.ask.side_effect = ['"just a string"', '{"answer": "ok"}']
            result = transport.call("ask", {})

        assert result == {"answer": "ok"}

    def test_call_async_delegates_to_sync(self) -> None:
        from rune_bench.drivers.manual import ManualDriverTransport

        console = MagicMock()
        transport = ManualDriverTransport(console=console)

        with patch("rune_bench.drivers.manual.Prompt") as mock_prompt:
            mock_prompt.ask.return_value = '{"answer": "async"}'
            result = asyncio.run(transport.call_async("ask", {}))

        assert result == {"answer": "async"}


# ---------------------------------------------------------------------------
# BrowserDriverTransport
# ---------------------------------------------------------------------------


class TestBrowserDriverTransport:
    def test_call_raises_without_playwright(self) -> None:
        from rune_bench.drivers.browser import BrowserDriverTransport

        transport = BrowserDriverTransport(driver_name="test")

        with patch("rune_bench.drivers.browser.async_playwright", None):
            with pytest.raises(ImportError, match="playwright"):
                transport.call("ask", {"url": "http://example.com"})

    def test_call_raises_without_url(self) -> None:
        from rune_bench.drivers.browser import BrowserDriverTransport

        transport = BrowserDriverTransport(driver_name="test")

        # Mock playwright so it doesn't fail on import
        mock_pw = MagicMock()
        with patch("rune_bench.drivers.browser.async_playwright", mock_pw):
            with pytest.raises(ValueError, match="requires a URL"):
                transport.call("ask", {})

    def test_call_raises_for_unknown_action(self) -> None:
        from rune_bench.drivers.browser import BrowserDriverTransport

        transport = BrowserDriverTransport(driver_name="test")

        mock_pw = MagicMock()
        with patch("rune_bench.drivers.browser.async_playwright", mock_pw):
            with pytest.raises(NotImplementedError, match="not implemented"):
                transport.call("unknown_action", {})

    def test_get_default_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from rune_bench.drivers.browser import BrowserDriverTransport

        monkeypatch.setenv("RUNE_MIDJOURNEY_DRIVER_URL", "http://mj.local")
        transport = BrowserDriverTransport(driver_name="midjourney")
        assert transport._get_default_url() == "http://mj.local"

    def test_get_default_url_none_without_driver_name(self) -> None:
        from rune_bench.drivers.browser import BrowserDriverTransport

        transport = BrowserDriverTransport()
        assert transport._get_default_url() is None

    def test_url_from_params(self) -> None:
        from rune_bench.drivers.browser import BrowserDriverTransport

        transport = BrowserDriverTransport(driver_name="test")

        # Mock the full playwright chain
        mock_page = AsyncMock()
        mock_page.evaluate.return_value = "page text"

        mock_browser = AsyncMock()
        mock_browser.new_page.return_value = mock_page

        mock_pw_instance = AsyncMock()
        mock_pw_instance.chromium.launch.return_value = mock_browser

        mock_pw_cm = AsyncMock()
        mock_pw_cm.__aenter__ = AsyncMock(return_value=mock_pw_instance)
        mock_pw_cm.__aexit__ = AsyncMock(return_value=False)

        mock_pw = MagicMock(return_value=mock_pw_cm)
        with patch("rune_bench.drivers.browser.async_playwright", mock_pw):
            result = transport.call("ask", {"url": "http://test.com", "question": "q"})

        assert result["answer"] == "page text"
        assert result["metadata"]["url"] == "http://test.com"

    def test_url_from_env_without_driver_name_fails(self) -> None:
        from rune_bench.drivers.browser import BrowserDriverTransport

        transport = BrowserDriverTransport()

        mock_pw = MagicMock()
        with patch("rune_bench.drivers.browser.async_playwright", mock_pw):
            with pytest.raises(ValueError, match="requires a 'url' parameter"):
                transport.call("ask", {})


# ---------------------------------------------------------------------------
# Driver factory: make_driver_transport / make_async_driver_transport
# ---------------------------------------------------------------------------


class TestDriverFactories:
    def test_make_driver_transport_manual(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rune_bench.drivers import make_driver_transport
        from rune_bench.drivers.manual import ManualDriverTransport

        monkeypatch.setenv("RUNE_TEST_DRIVER_MODE", "manual")
        transport = make_driver_transport("test")
        assert isinstance(transport, ManualDriverTransport)

    def test_make_driver_transport_browser(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rune_bench.drivers import make_driver_transport
        from rune_bench.drivers.browser import BrowserDriverTransport

        monkeypatch.setenv("RUNE_TEST_DRIVER_MODE", "browser")
        transport = make_driver_transport("test")
        assert isinstance(transport, BrowserDriverTransport)

    def test_make_driver_transport_http(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from rune_bench.drivers import make_driver_transport
        from rune_bench.drivers.http import HttpTransport

        monkeypatch.setenv("RUNE_TEST_DRIVER_MODE", "http")
        monkeypatch.setenv("RUNE_TEST_DRIVER_URL", "http://localhost:9999")
        transport = make_driver_transport("test")
        assert isinstance(transport, HttpTransport)

    def test_make_driver_transport_http_missing_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rune_bench.drivers import make_driver_transport

        monkeypatch.setenv("RUNE_TEST_DRIVER_MODE", "http")
        monkeypatch.delenv("RUNE_TEST_DRIVER_URL", raising=False)
        with pytest.raises(RuntimeError, match="not set"):
            make_driver_transport("test")

    def test_make_async_driver_transport_manual(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rune_bench.drivers import make_async_driver_transport
        from rune_bench.drivers.manual import ManualDriverTransport

        monkeypatch.setenv("RUNE_TEST_DRIVER_MODE", "manual")
        transport = make_async_driver_transport("test")
        assert isinstance(transport, ManualDriverTransport)

    def test_make_async_driver_transport_browser(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rune_bench.drivers import make_async_driver_transport
        from rune_bench.drivers.browser import BrowserDriverTransport

        monkeypatch.setenv("RUNE_TEST_DRIVER_MODE", "browser")
        transport = make_async_driver_transport("test")
        assert isinstance(transport, BrowserDriverTransport)

    def test_make_async_driver_transport_http(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rune_bench.drivers import make_async_driver_transport
        from rune_bench.drivers.http import AsyncHttpTransport

        monkeypatch.setenv("RUNE_TEST_DRIVER_MODE", "http")
        monkeypatch.setenv("RUNE_TEST_DRIVER_URL", "http://localhost:9999")
        transport = make_async_driver_transport("test")
        assert isinstance(transport, AsyncHttpTransport)

    def test_make_async_driver_transport_http_missing_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rune_bench.drivers import make_async_driver_transport

        monkeypatch.setenv("RUNE_TEST_DRIVER_MODE", "http")
        monkeypatch.delenv("RUNE_TEST_DRIVER_URL", raising=False)
        with pytest.raises(RuntimeError, match="not set"):
            make_async_driver_transport("test")

    def test_make_async_driver_transport_stdio_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rune_bench.drivers import make_async_driver_transport
        from rune_bench.drivers.stdio import AsyncStdioTransport

        monkeypatch.delenv("RUNE_TEST_DRIVER_MODE", raising=False)
        monkeypatch.delenv("RUNE_TEST_DRIVER_CMD", raising=False)
        transport = make_async_driver_transport("test")
        assert isinstance(transport, AsyncStdioTransport)

    def test_make_async_driver_transport_stdio_custom_cmd(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rune_bench.drivers import make_async_driver_transport
        from rune_bench.drivers.stdio import AsyncStdioTransport

        monkeypatch.delenv("RUNE_TEST_DRIVER_MODE", raising=False)
        monkeypatch.setenv("RUNE_TEST_DRIVER_CMD", "python3 -m custom_driver")
        transport = make_async_driver_transport("test")
        assert isinstance(transport, AsyncStdioTransport)


# ---------------------------------------------------------------------------
# Driver client ask_structured and ask_async coverage
# ---------------------------------------------------------------------------


class TestDriverClientAskStructured:
    """Cover ask_structured on drivers that delegate ask() -> ask_structured()."""

    def test_holmes_ask_structured(self, tmp_path) -> None:
        from rune_bench.drivers.holmes import HolmesDriverClient

        kc = tmp_path / "kubeconfig"
        kc.write_text("apiVersion: v1\n")
        transport = MagicMock()
        transport.call.return_value = {"answer": "diag result", "result_type": "text"}
        client = HolmesDriverClient(kc, transport=transport)

        result = client.ask_structured("q", "m")
        assert isinstance(result, AgentResult)
        assert result.answer == "diag result"

    def test_holmes_ask_async(self, tmp_path) -> None:
        from rune_bench.drivers.holmes import HolmesDriverClient

        kc = tmp_path / "kubeconfig"
        kc.write_text("apiVersion: v1\n")
        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {"answer": "async diag"}
        client = HolmesDriverClient(kc, transport=transport)
        client._async_transport = async_transport

        result = asyncio.run(client.ask_async("q", "m"))
        assert isinstance(result, AgentResult)
        assert result.answer == "async diag"

    def test_holmes_ask_async_missing_answer(self, tmp_path) -> None:
        from rune_bench.drivers.holmes import HolmesDriverClient

        kc = tmp_path / "kubeconfig"
        kc.write_text("apiVersion: v1\n")
        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {}
        client = HolmesDriverClient(kc, transport=transport)
        client._async_transport = async_transport

        with pytest.raises(RuntimeError, match="did not include an answer"):
            asyncio.run(client.ask_async("q", "m"))

    def test_holmes_ask_async_none_answer(self, tmp_path) -> None:
        from rune_bench.drivers.holmes import HolmesDriverClient

        kc = tmp_path / "kubeconfig"
        kc.write_text("apiVersion: v1\n")
        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {"answer": None}
        client = HolmesDriverClient(kc, transport=transport)
        client._async_transport = async_transport

        with pytest.raises(RuntimeError, match="empty answer"):
            asyncio.run(client.ask_async("q", "m"))

    def test_holmes_ask_async_empty_string_answer(self, tmp_path) -> None:
        from rune_bench.drivers.holmes import HolmesDriverClient

        kc = tmp_path / "kubeconfig"
        kc.write_text("apiVersion: v1\n")
        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {"answer": ""}
        client = HolmesDriverClient(kc, transport=transport)
        client._async_transport = async_transport

        with pytest.raises(RuntimeError, match="empty answer"):
            asyncio.run(client.ask_async("q", "m"))

    def test_holmes_ask_async_with_backend_url(self, tmp_path) -> None:
        from rune_bench.drivers.holmes import HolmesDriverClient

        kc = tmp_path / "kubeconfig"
        kc.write_text("apiVersion: v1\n")
        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {"answer": "with-url"}
        client = HolmesDriverClient(kc, transport=transport)
        client._async_transport = async_transport

        result = asyncio.run(client.ask_async("q", "m", backend_url="http://x"))
        assert result.answer == "with-url"

    def test_k8sgpt_ask_structured(self, tmp_path) -> None:
        from rune_bench.drivers.k8sgpt import K8sGPTDriverClient

        kc = tmp_path / "kubeconfig"
        kc.write_text("apiVersion: v1\n")
        transport = MagicMock()
        transport.call.return_value = {"answer": "k8s result"}
        client = K8sGPTDriverClient(kc, transport=transport)

        result = client.ask_structured("q", "m")
        assert isinstance(result, AgentResult)
        assert result.answer == "k8s result"

    def test_k8sgpt_ask_async(self, tmp_path) -> None:
        from rune_bench.drivers.k8sgpt import K8sGPTDriverClient

        kc = tmp_path / "kubeconfig"
        kc.write_text("apiVersion: v1\n")
        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {"answer": "async k8s"}
        client = K8sGPTDriverClient(kc, transport=transport)
        client._async_transport = async_transport

        result = asyncio.run(client.ask_async("q", "m"))
        assert result.answer == "async k8s"

    def test_k8sgpt_ask_async_missing_answer(self, tmp_path) -> None:
        from rune_bench.drivers.k8sgpt import K8sGPTDriverClient

        kc = tmp_path / "kubeconfig"
        kc.write_text("apiVersion: v1\n")
        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {}
        client = K8sGPTDriverClient(kc, transport=transport)
        client._async_transport = async_transport

        with pytest.raises(RuntimeError, match="did not include"):
            asyncio.run(client.ask_async("q", "m"))

    def test_k8sgpt_ask_async_none_answer(self, tmp_path) -> None:
        from rune_bench.drivers.k8sgpt import K8sGPTDriverClient

        kc = tmp_path / "kubeconfig"
        kc.write_text("apiVersion: v1\n")
        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {"answer": None}
        client = K8sGPTDriverClient(kc, transport=transport)
        client._async_transport = async_transport

        with pytest.raises(RuntimeError, match="empty answer"):
            asyncio.run(client.ask_async("q", "m"))

    def test_k8sgpt_ask_async_empty_answer(self, tmp_path) -> None:
        from rune_bench.drivers.k8sgpt import K8sGPTDriverClient

        kc = tmp_path / "kubeconfig"
        kc.write_text("apiVersion: v1\n")
        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {"answer": ""}
        client = K8sGPTDriverClient(kc, transport=transport)
        client._async_transport = async_transport

        with pytest.raises(RuntimeError, match="empty answer"):
            asyncio.run(client.ask_async("q", "m"))

    def test_crewai_ask_structured(self) -> None:
        from rune_bench.drivers.crewai import CrewAIDriverClient

        transport = MagicMock()
        transport.call.return_value = {"answer": "crew result", "result_type": "report"}
        client = CrewAIDriverClient(transport=transport)

        result = client.ask_structured("q", "m")
        assert isinstance(result, AgentResult)
        assert result.result_type == "report"

    def test_crewai_ask_async(self) -> None:
        from rune_bench.drivers.crewai import CrewAIDriverClient

        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {"answer": "async crew"}
        client = CrewAIDriverClient(transport=transport)
        client._async_transport = async_transport

        result = asyncio.run(client.ask_async("q", "m"))
        assert result.answer == "async crew"

    def test_crewai_ask_async_with_backend_url(self) -> None:
        from rune_bench.drivers.crewai import CrewAIDriverClient

        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {"answer": "ok"}
        client = CrewAIDriverClient(transport=transport)
        client._async_transport = async_transport

        result = asyncio.run(client.ask_async("q", "m", backend_url="http://x"))
        assert result.answer == "ok"

    def test_crewai_ask_async_missing_answer(self) -> None:
        from rune_bench.drivers.crewai import CrewAIDriverClient

        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {}
        client = CrewAIDriverClient(transport=transport)
        client._async_transport = async_transport

        with pytest.raises(RuntimeError, match="did not include"):
            asyncio.run(client.ask_async("q", "m"))

    def test_crewai_ask_async_none_answer(self) -> None:
        from rune_bench.drivers.crewai import CrewAIDriverClient

        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {"answer": None}
        client = CrewAIDriverClient(transport=transport)
        client._async_transport = async_transport

        with pytest.raises(RuntimeError, match="empty answer"):
            asyncio.run(client.ask_async("q", "m"))

    def test_crewai_ask_async_empty_answer(self) -> None:
        from rune_bench.drivers.crewai import CrewAIDriverClient

        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {"answer": ""}
        client = CrewAIDriverClient(transport=transport)
        client._async_transport = async_transport

        with pytest.raises(RuntimeError, match="empty answer"):
            asyncio.run(client.ask_async("q", "m"))

    def test_dagger_ask_structured(self) -> None:
        from rune_bench.drivers.dagger import DaggerDriverClient

        transport = MagicMock()
        transport.call.return_value = {"answer": "dagger result"}
        client = DaggerDriverClient(transport=transport)

        result = client.ask_structured("q", "m")
        assert isinstance(result, AgentResult)

    def test_dagger_ask_async(self) -> None:
        from rune_bench.drivers.dagger import DaggerDriverClient

        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {"answer": "async dagger"}
        client = DaggerDriverClient(transport=transport)
        client._async_transport = async_transport

        result = asyncio.run(client.ask_async("q", "m"))
        assert result.answer == "async dagger"

    def test_dagger_ask_async_missing_answer(self) -> None:
        from rune_bench.drivers.dagger import DaggerDriverClient

        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {}
        client = DaggerDriverClient(transport=transport)
        client._async_transport = async_transport

        with pytest.raises(RuntimeError, match="did not include"):
            asyncio.run(client.ask_async("q", "m"))

    def test_dagger_ask_async_none_answer(self) -> None:
        from rune_bench.drivers.dagger import DaggerDriverClient

        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {"answer": None}
        client = DaggerDriverClient(transport=transport)
        client._async_transport = async_transport

        with pytest.raises(RuntimeError, match="empty answer"):
            asyncio.run(client.ask_async("q", "m"))

    def test_dagger_ask_async_empty_answer(self) -> None:
        from rune_bench.drivers.dagger import DaggerDriverClient

        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {"answer": ""}
        client = DaggerDriverClient(transport=transport)
        client._async_transport = async_transport

        with pytest.raises(RuntimeError, match="empty answer"):
            asyncio.run(client.ask_async("q", "m"))

    def test_pentestgpt_ask_structured(self) -> None:
        from rune_bench.drivers.pentestgpt import PentestGPTDriverClient

        transport = MagicMock()
        transport.call.return_value = {"answer": "pentest result"}
        client = PentestGPTDriverClient(transport=transport)

        result = client.ask_structured("q", "m", backend_url="http://x")
        assert isinstance(result, AgentResult)

    def test_pentestgpt_ask_async(self) -> None:
        from rune_bench.drivers.pentestgpt import PentestGPTDriverClient

        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {"answer": "async pentest"}
        client = PentestGPTDriverClient(transport=transport)
        client._async_transport = async_transport

        result = asyncio.run(client.ask_async("q", "m", backend_url="http://x"))
        assert result.answer == "async pentest"

    def test_pentestgpt_ask_async_missing_answer(self) -> None:
        from rune_bench.drivers.pentestgpt import PentestGPTDriverClient

        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {}
        client = PentestGPTDriverClient(transport=transport)
        client._async_transport = async_transport

        with pytest.raises(RuntimeError, match="did not include"):
            asyncio.run(client.ask_async("q", "m", backend_url="http://x"))

    def test_pentestgpt_ask_async_none_answer(self) -> None:
        from rune_bench.drivers.pentestgpt import PentestGPTDriverClient

        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {"answer": None}
        client = PentestGPTDriverClient(transport=transport)
        client._async_transport = async_transport

        with pytest.raises(RuntimeError, match="empty answer"):
            asyncio.run(client.ask_async("q", "m", backend_url="http://x"))

    def test_langgraph_ask_async(self) -> None:
        from rune_bench.drivers.langgraph import LangGraphDriverClient

        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {"answer": "async lg"}
        client = LangGraphDriverClient(transport=transport)
        client._async_transport = async_transport

        result = asyncio.run(client.ask_async("q", "m"))
        assert result.answer == "async lg"

    def test_langgraph_ask_async_missing_answer(self) -> None:
        from rune_bench.drivers.langgraph import LangGraphDriverClient

        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {}
        client = LangGraphDriverClient(transport=transport)
        client._async_transport = async_transport

        with pytest.raises(RuntimeError, match="did not include an answer"):
            asyncio.run(client.ask_async("q", "m"))

    def test_langgraph_ask_async_none_answer(self) -> None:
        from rune_bench.drivers.langgraph import LangGraphDriverClient

        transport = MagicMock()
        async_transport = AsyncMock()
        async_transport.call_async.return_value = {"answer": None}
        client = LangGraphDriverClient(transport=transport)
        client._async_transport = async_transport

        with pytest.raises(RuntimeError, match="empty answer"):
            asyncio.run(client.ask_async("q", "m"))


# ---------------------------------------------------------------------------
# Agent aliases (thin wrappers — import coverage)
# ---------------------------------------------------------------------------


class TestAgentAliases:
    def test_invokeai_runner_exists(self) -> None:
        from rune_bench.agents.art.invokeai import InvokeAIRunner
        from rune_bench.drivers.invokeai import InvokeAIDriverClient

        assert InvokeAIRunner is InvokeAIDriverClient

    def test_browseruse_runner_exists(self) -> None:
        from rune_bench.agents.ops.browseruse import BrowserUseRunner
        from rune_bench.drivers.browseruse import BrowserUseDriverClient

        assert BrowserUseRunner is BrowserUseDriverClient
