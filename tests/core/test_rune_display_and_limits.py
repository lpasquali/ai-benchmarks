# SPDX-License-Identifier: Apache-2.0
import os

from rich.console import Console

import rune
from rune_bench.backends.ollama import OllamaModelCapabilities
from rune_bench.resources.vastai import ConnectionDetails
from rune_bench.workflows import ExistingOllamaServer, VastAIProvisioningResult


def test_apply_model_limits_sets_only_missing(monkeypatch):
    monkeypatch.delenv("OVERRIDE_MAX_CONTENT_SIZE", raising=False)
    monkeypatch.setenv("OVERRIDE_MAX_OUTPUT_TOKEN", "777")

    rune._apply_model_limits(
        OllamaModelCapabilities(
            model_name="x",
            context_window=262144,
            max_output_tokens=52428,
        )
    )

    assert os.environ["OVERRIDE_MAX_CONTENT_SIZE"] == "262144"
    # existing value preserved
    assert os.environ["OVERRIDE_MAX_OUTPUT_TOKEN"] == "777"


def test_print_existing_ollama_displays_override_rows(monkeypatch):
    test_console = Console(record=True, width=180)
    monkeypatch.setattr(rune, "console", test_console)

    rune._print_existing_ollama(
        ExistingOllamaServer(
            url="http://localhost:11434", model_name="kavai/qwen3.5-GPT5:9b"
        ),
        capabilities=OllamaModelCapabilities(
            model_name="kavai/qwen3.5-GPT5:9b",
            context_window=262144,
            max_output_tokens=52428,
        ),
    )

    text = test_console.export_text()
    assert "OVERRIDE_MAX_CONTENT_SIZE" in text
    assert "OVERRIDE_MAX_OUTPUT_TOKEN" in text
    assert "262,144 tokens" in text
    assert "52,428 tokens" in text


def test_print_vastai_result_displays_override_rows(monkeypatch):
    test_console = Console(record=True, width=180)
    monkeypatch.setattr(rune, "console", test_console)

    details = ConnectionDetails(
        contract_id=123,
        status="running",
        ssh_host="1.2.3.4",
        ssh_port=22,
        machine_id="m1",
        service_urls=[
            {"name": "ollama", "direct": "http://1.2.3.4:11434", "proxy": None}
        ],
    )

    result = VastAIProvisioningResult(
        offer_id=1,
        total_vram_mb=24000,
        model_name="kavai/qwen3.5-GPT5:9b",
        model_vram_mb=18000,
        required_disk_gb=60,
        template_env="FOO=bar",
        contract_id=123,
        details=details,
        backend_url="http://1.2.3.4:11434",
    )

    rune._print_vastai_result(
        result,
        capabilities=OllamaModelCapabilities(
            model_name="kavai/qwen3.5-GPT5:9b",
            context_window=262144,
            max_output_tokens=52428,
        ),
    )

    text = test_console.export_text()
    assert "OVERRIDE_MAX_CONTENT_SIZE" in text
    assert "OVERRIDE_MAX_OUTPUT_TOKEN" in text
