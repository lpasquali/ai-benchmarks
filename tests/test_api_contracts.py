from pathlib import Path

from rune_bench.api_contracts import (
    RunAgenticAgentRequest,
    RunBenchmarkRequest,
    RunOllamaInstanceRequest,
)


def test_run_ollama_instance_request_to_dict():
    request = RunOllamaInstanceRequest(
        vastai=True,
        template_hash="tmpl",
        min_dph=2.0,
        max_dph=3.0,
        reliability=0.99,
        ollama_url=None,
    )

    payload = request.to_dict()
    assert payload["vastai"] is True
    assert payload["template_hash"] == "tmpl"


def test_agentic_request_from_cli_converts_kubeconfig_to_string():
    request = RunAgenticAgentRequest.from_cli(
        question="q",
        model="m",
        ollama_url="http://localhost:11434",
        ollama_warmup=True,
        ollama_warmup_timeout=90,
        kubeconfig=Path("/tmp/kubeconfig"),
    )

    payload = request.to_dict()
    assert payload["kubeconfig"] == "/tmp/kubeconfig"


def test_benchmark_request_from_cli_converts_kubeconfig_to_string():
    request = RunBenchmarkRequest.from_cli(
        vastai=False,
        template_hash="hash",
        min_dph=2.3,
        max_dph=3.0,
        reliability=0.99,
        ollama_url="http://localhost:11434",
        question="what is unhealthy",
        model="llama3.1:8b",
        ollama_warmup=True,
        ollama_warmup_timeout=90,
        kubeconfig=Path("/home/user/.kube/config"),
        vastai_stop_instance=True,
    )

    payload = request.to_dict()
    assert payload["kubeconfig"] == "/home/user/.kube/config"
    assert payload["vastai_stop_instance"] is True
