import json
import pytest

from rune_bench.agents.experimental.memory_provider import MemoryProvider


def test_episodic_memory():
    provider = MemoryProvider()
    provider.append_episodic("kubectl get pods", "pod/web Running")
    provider.append_episodic("kubectl logs web", "Error: failed to start")
    
    context = provider.get_episodic_context(last_n=1)
    assert len(context) == 1
    assert context[0]["action"] == "kubectl logs web"
    
    context_all = provider.get_episodic_context(last_n=5)
    assert len(context_all) == 2


def test_semantic_memory():
    provider = MemoryProvider()
    provider.store_semantic("CrashLoopBackOff", "Usually indicates a missing env var or failing startup script.")
    
    assert provider.retrieve_semantic("CrashLoopBackOff") == "Usually indicates a missing env var or failing startup script."
    assert provider.retrieve_semantic("OOMKilled") is None


def test_procedural_memory():
    provider = MemoryProvider()
    provider.cache_procedure("rollback_deployment", ["kubectl rollout undo deploy/web", "kubectl get pods -w"])
    
    steps = provider.get_procedure("rollback_deployment")
    assert steps is not None
    assert len(steps) == 2
    assert steps[0] == "kubectl rollout undo deploy/web"
    
    assert provider.get_procedure("scale_up") is None


def test_dump_memory_state():
    provider = MemoryProvider()
    provider.append_episodic("ls", "file.txt")
    provider.store_semantic("error_404", "Not Found")
    provider.cache_procedure("fix_it", ["reboot"])
    
    state_str = provider.dump_memory_state()
    state = json.loads(state_str)
    
    assert len(state["episodic"]) == 1
    assert state["semantic"]["error_404"] == "Not Found"
    assert state["procedural"]["fix_it"] == ["reboot"]
