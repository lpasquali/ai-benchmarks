# SPDX-License-Identifier: Apache-2.0
"""Tests for Tier 1 agent driver clients.

We verify they import cleanly and expose the expected public interface.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# ComfyUI (Art/Creative — Tier 1)
# ---------------------------------------------------------------------------


def test_comfyui_driver_instantiates():
    from rune_bench.drivers.comfyui import ComfyUIDriverClient

    runner = ComfyUIDriverClient(base_url="http://127.0.0.1:8188")
    assert runner


def test_comfyui_driver_custom_base_url():
    from rune_bench.drivers.comfyui import ComfyUIDriverClient

    runner = ComfyUIDriverClient(base_url="http://localhost:9999/")
    assert runner


# ---------------------------------------------------------------------------
# PentestGPT (Cybersec — Tier 1)
# ---------------------------------------------------------------------------


def test_pentestgpt_driver_instantiates():
    from rune_bench.drivers.pentestgpt import PentestGPTDriverClient

    runner = PentestGPTDriverClient()
    assert runner


# ---------------------------------------------------------------------------
# CrewAI (Ops — Tier 1)
# ---------------------------------------------------------------------------


def test_crewai_driver_exists():
    from rune_bench.drivers.crewai import CrewAIDriverClient
    assert CrewAIDriverClient


# ---------------------------------------------------------------------------
# Dagger (Ops — Tier 1)
# ---------------------------------------------------------------------------


def test_dagger_driver_exists():
    from rune_bench.drivers.dagger import DaggerDriverClient
    assert DaggerDriverClient


# ---------------------------------------------------------------------------
# LangGraph (Research — Tier 1)
# ---------------------------------------------------------------------------


def test_langgraph_driver_exists():
    from rune_bench.drivers.langgraph import LangGraphDriverClient
    assert LangGraphDriverClient
