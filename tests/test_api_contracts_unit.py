# SPDX-License-Identifier: Apache-2.0
import pytest
from rune_bench.api_contracts import RunBenchmarkRequest, RunAgenticAgentRequest, CostEstimationRequest, Provisioning, VastAIProvisioning

def test_benchmark_request_to_dict():
    req = RunBenchmarkRequest(
        question="q", model="m", backend_url="u", backend_warmup=True,
        backend_warmup_timeout=10, kubeconfig="k", 
        provisioning=Provisioning(vastai=VastAIProvisioning(template_hash="h", min_dph=0.1, max_dph=1.0, reliability=0.9))
    )
    d = req.to_dict()
    assert d["question"] == "q"
    assert d["provisioning"]["vastai"]["template_hash"] == "h"

def test_agentic_request_to_dict():
    req = RunAgenticAgentRequest(
        question="q", model="m", backend_url="u", backend_warmup=True,
        backend_warmup_timeout=10, kubeconfig="k", agent="holmes"
    )
    d = req.to_dict()
    assert d["agent"] == "holmes"

def test_cost_request_to_dict():
    req = CostEstimationRequest(estimated_duration_seconds=3600, aws=True)
    d = req.to_dict()
    assert d["aws"] is True
