
from rune_bench.agents.experimental.reflection_agent import ReflectionAgentRunner


def test_reflection_agent_ask():
    runner = ReflectionAgentRunner(max_reflections=1)
    answer = runner.ask("What is 2+2?", model="dummy", ollama_url=None)
    assert answer == "Reflected and improved: Draft response to: What is 2+2?"


def test_reflection_agent_ask_multiple_reflections():
    runner = ReflectionAgentRunner(max_reflections=2)
    answer = runner.ask("What is 2+2?", model="dummy", ollama_url=None)
    assert answer == "Reflected and improved: Reflected and improved: Draft response to: What is 2+2?"


def test_reflection_agent_ask_structured():
    runner = ReflectionAgentRunner(max_reflections=1)
    result = runner.ask_structured("Test query", model="dummy", ollama_url=None)
    
    assert result.answer == "Reflected and improved: Draft response to: Test query"
    assert result.metadata is not None
    assert result.metadata["draft"] == "Draft response to: Test query"
    assert len(result.metadata["reflections"]) == 1
    assert result.metadata["reflections"][0] == result.answer
    assert result.metadata["reflection_count"] == 1
