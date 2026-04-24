# SPDX-License-Identifier: Apache-2.0
"""Tests for Tier 1 agent runner modules that delegate to drivers.

These modules are thin wrappers / re-exports; we verify they import cleanly
and expose the expected public interface.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# ComfyUI (Art/Creative — Tier 1)
# ---------------------------------------------------------------------------


def test_comfyui_runner_instantiates():
    from rune_bench.agents.art.comfyui import ComfyUIRunner

    runner = ComfyUIRunner()
    assert runner._base_url == "http://127.0.0.1:8188"


def test_comfyui_runner_custom_base_url():
    from rune_bench.agents.art.comfyui import ComfyUIRunner

    runner = ComfyUIRunner(base_url="http://localhost:9999/")
    assert runner._base_url == "http://localhost:9999"


def test_comfyui_runner_ask_raises_not_implemented():
    from rune_bench.agents.art.comfyui import ComfyUIRunner

    runner = ComfyUIRunner()
    with pytest.raises(
        NotImplementedError, match="ComfyUIRunner is not yet implemented"
    ):
        runner.ask("a cat in space", model="sd-xl")


# ---------------------------------------------------------------------------
# PentestGPT (Cybersec — Tier 1)
# ---------------------------------------------------------------------------


def test_pentestgpt_runner_instantiates():
    from rune_bench.agents.cybersec.pentestgpt import PentestGPTRunner

    runner = PentestGPTRunner()
    assert hasattr(runner, "_client")


def test_pentestgpt_runner_delegates_ask(monkeypatch):
    from rune_bench.agents.cybersec.pentestgpt import PentestGPTRunner

    runner = PentestGPTRunner()
    monkeypatch.setattr(runner._client, "ask", lambda q, m, o=None: f"result:{q}")
    assert runner.ask("scan target", model="qwen3:32b") == "result:scan target"


# ---------------------------------------------------------------------------
# CrewAI (Ops — Tier 1)
# ---------------------------------------------------------------------------


def test_crewai_runner_is_driver_client():
    from rune_bench.agents.ops.crewai import CrewAIRunner
    from rune_bench.drivers.crewai import CrewAIDriverClient

    assert CrewAIRunner is CrewAIDriverClient


# ---------------------------------------------------------------------------
# Dagger (Ops — Tier 1)
# ---------------------------------------------------------------------------


def test_dagger_runner_is_driver_client():
    from rune_bench.agents.ops.dagger import DaggerRunner
    from rune_bench.drivers.dagger import DaggerDriverClient

    assert DaggerRunner is DaggerDriverClient


# ---------------------------------------------------------------------------
# LangGraph (Research — Tier 1)
# ---------------------------------------------------------------------------


def test_langgraph_runner_is_driver_client():
    from rune_bench.agents.research.langgraph import LangGraphRunner
    from rune_bench.drivers.langgraph import LangGraphDriverClient

    assert LangGraphRunner is LangGraphDriverClient
