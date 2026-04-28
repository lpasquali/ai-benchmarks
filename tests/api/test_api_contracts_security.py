# SPDX-License-Identifier: Apache-2.0
import pytest
from rune_bench.api_contracts import (
    RunBenchmarkRequest,
    RunAgenticAgentRequest,
    RunLLMInstanceRequest,
    CostEstimationRequest,
)


def test_run_benchmark_request_string_limits():
    # model too long
    with pytest.raises(ValueError, match="model exceeds maximum length"):
        RunBenchmarkRequest(
            provisioning=None,
            backend_url="http://api",
            question="test",
            model="a" * 129,
            backend_warmup=True,
            backend_warmup_timeout=30,
            kubeconfig="/path/to/config",
        )

    # question too long
    with pytest.raises(ValueError, match="question exceeds maximum length"):
        RunBenchmarkRequest(
            provisioning=None,
            backend_url="http://api",
            question="a" * 100001,
            model="llama3",
            backend_warmup=True,
            backend_warmup_timeout=30,
            kubeconfig="/path/to/config",
        )


def test_run_agentic_agent_request_string_limits():
    # agent name too long
    with pytest.raises(ValueError, match="agent exceeds maximum length"):
        RunAgenticAgentRequest(
            question="test",
            model="llama3",
            backend_url="http://api",
            backend_warmup=True,
            backend_warmup_timeout=30,
            agent="a" * 65,
        )


def test_run_llm_instance_request_string_limits():
    # backend_url too long
    with pytest.raises(ValueError, match="backend_url exceeds maximum length"):
        RunLLMInstanceRequest(
            backend_url="http://" + ("a" * 2048), backend_type="ollama"
        )


def test_cost_estimation_request_string_limits():
    # model too long
    with pytest.raises(ValueError, match="model exceeds maximum length"):
        CostEstimationRequest(model="a" * 129)
