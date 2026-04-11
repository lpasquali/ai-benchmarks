# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import rune_bench.workflows as workflows


def test_use_existing_backend_server(monkeypatch):
    monkeypatch.setattr(workflows, "normalize_backend_url", lambda _: "http://localhost:11434")
    server = workflows.use_existing_backend_server("localhost:11434", "model:1")
    assert server.url == "http://localhost:11434"
    assert server.model_name == "model:1"


def test_list_backend_models(monkeypatch):
    fake_backend = MagicMock()
    fake_backend.list_models.return_value = ["a", "b"]
    monkeypatch.setattr(workflows, "normalize_backend_url", lambda u: u)
    mock_backend_class = MagicMock()
    mock_backend_class.return_value = fake_backend
    mock_backend_class.normalize_url.side_effect = lambda u: u
    monkeypatch.setattr("rune_bench.common.backend_utils.OllamaBackend", mock_backend_class)
    assert workflows.list_backend_models("http://localhost:11434") == ["a", "b"]


def test_list_running_backend_models(monkeypatch):
    fake_backend = MagicMock()
    fake_backend.list_running_models.return_value = ["a"]
    monkeypatch.setattr(workflows, "normalize_backend_url", lambda u: u)
    mock_backend_class = MagicMock()
    mock_backend_class.return_value = fake_backend
    mock_backend_class.normalize_url.side_effect = lambda u: u
    monkeypatch.setattr("rune_bench.common.backend_utils.OllamaBackend", mock_backend_class)
    assert workflows.list_running_backend_models("http://localhost:11434") == ["a"]


def test_warmup_backend_model_normalizes_before_warmup(monkeypatch):
    fake_backend = MagicMock()
    fake_backend.warmup.return_value = "foo:1"
    monkeypatch.setattr(workflows, "normalize_backend_url", lambda u: u)
    mock_backend_class = MagicMock()
    mock_backend_class.return_value = fake_backend
    mock_backend_class.normalize_url.side_effect = lambda u: u
    monkeypatch.setattr("rune_bench.common.backend_utils.OllamaBackend", mock_backend_class)

    out = workflows.warmup_backend_model("http://localhost:11434", "ollama_chat/foo:1")

    assert out == "foo:1"
    fake_backend.warmup.assert_called_once_with(
        "ollama_chat/foo:1",
        timeout_seconds=120,
        poll_interval_seconds=2.0,
        keep_alive="30m",
    )


def test_extract_ollama_service_url_prefers_direct_then_proxy():
    details = workflows.ConnectionDetails(
        contract_id=1,
        status="running",
        ssh_host=None,
        ssh_port=None,
        machine_id=None,
        service_urls=[
            {"name": "something", "direct": "http://x:8080", "proxy": "https://proxy/x/8080"},
            {"name": "ollama", "direct": "http://x:11434", "proxy": None},
        ],
    )
    assert workflows._extract_ollama_service_url(details) == "http://x:11434"

    details2 = workflows.ConnectionDetails(
        contract_id=1,
        status="running",
        ssh_host=None,
        ssh_port=None,
        machine_id=None,
        service_urls=[
            {"name": "ollama", "direct": "http://x:8080", "proxy": "https://proxy:11434/x"},
        ],
    )
    assert workflows._extract_ollama_service_url(details2) == "https://proxy:11434/x"


# Backward-compatible aliases still work
def test_backward_compatible_aliases():
    assert workflows.use_existing_ollama_server is workflows.use_existing_backend_server
    assert workflows.list_existing_ollama_models is workflows.list_backend_models
    assert workflows.list_running_ollama_models is workflows.list_running_backend_models
    assert workflows.warmup_existing_ollama_model is workflows.warmup_backend_model
    assert workflows.provision_vastai_ollama is workflows.provision_vastai_backend


# ── run_chain_workflow + JobStoreChainRecorder ──────────────────────────────


import asyncio  # noqa: E402

from unittest.mock import AsyncMock  # noqa: E402

from rune_bench.agents.base import AgentResult  # noqa: E402
from rune_bench.job_store import JobStore  # noqa: E402


def test_jobstore_chain_recorder_initialize_delegates_to_store(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    try:
        recorder = workflows.JobStoreChainRecorder(store)

        recorder.initialize(
            job_id="job-1",
            nodes=[{"id": "a", "agent_name": "X"}],
            edges=[],
        )
        state = store.get_chain_state("job-1")
        assert state is not None
        assert [n["id"] for n in state["nodes"]] == ["a"]
    finally:
        store.close()


def test_jobstore_chain_recorder_transition_delegates_to_store(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    try:
        recorder = workflows.JobStoreChainRecorder(store)
        recorder.initialize(
            job_id="job-1",
            nodes=[{"id": "a", "agent_name": "X"}],
            edges=[],
        )
        recorder.transition(
            job_id="job-1",
            node_id="a",
            status="success",
            started_at=1.0,
            finished_at=2.0,
        )
        state = store.get_chain_state("job-1")
        assert state is not None
        assert state["nodes"][0]["status"] == "success"
        assert state["nodes"][0]["started_at"] == 1.0
        assert state["nodes"][0]["finished_at"] == 2.0
        assert state["overall_status"] == "success"
    finally:
        store.close()


def test_run_chain_workflow_persists_full_state_via_store(tmp_path):
    """End-to-end: run_chain_workflow runs the engine and the JobStore reflects the final DAG state."""
    from rune_bench.agents.chain import ChainStep

    store = JobStore(tmp_path / "jobs.db")
    try:
        agent = MagicMock()
        agent.ask_async = AsyncMock(return_value=AgentResult(answer="ok"))
        steps = [
            ChainStep(name="draft", agent=agent, question_template="{topic}"),
            ChainStep(name="review", agent=agent, question_template="{draft}", dependencies=["draft"]),
        ]

        result = asyncio.run(
            workflows.run_chain_workflow(
                steps=steps,
                initial_context={"topic": "ai"},
                model="m",
                job_id="job-xyz",
                store=store,
            )
        )

        assert "draft" in result.steps
        assert "review" in result.steps

        state = store.get_chain_state("job-xyz")
        assert state is not None
        assert state["overall_status"] == "success"
        assert {n["id"] for n in state["nodes"]} == {"draft", "review"}
        assert state["edges"] == [{"from": "draft", "to": "review"}]
        for node in state["nodes"]:
            assert node["status"] == "success"
            assert node["started_at"] is not None
            assert node["finished_at"] is not None
    finally:
        store.close()


def test_run_chain_workflow_records_failure_in_store(tmp_path):
    from rune_bench.agents.chain import ChainStep
    import pytest as _pytest

    store = JobStore(tmp_path / "jobs.db")
    try:
        failing_agent = MagicMock()
        failing_agent.ask_async = AsyncMock(side_effect=RuntimeError("boom"))
        steps = [ChainStep(name="only", agent=failing_agent, question_template="{topic}")]

        with _pytest.raises(RuntimeError, match="boom"):
            asyncio.run(
                workflows.run_chain_workflow(
                    steps=steps,
                    initial_context={"topic": "ai"},
                    model="m",
                    job_id="job-fail",
                    store=store,
                )
            )

        state = store.get_chain_state("job-fail")
        assert state is not None
        assert state["overall_status"] == "failed"
        assert state["nodes"][0]["error"] == "boom"
    finally:
        store.close()
