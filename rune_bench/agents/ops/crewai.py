"""CrewAI agentic runner stub.

Scope:      Ops/Misc  |  Rank 5  |  Rating 4.0
Capability: Orchestrates groups of agents to complete complex tasks.
Docs:       https://docs.crewai.com/
            https://docs.crewai.com/concepts/crews
            https://docs.crewai.com/concepts/agents
            https://docs.crewai.com/concepts/tasks
Ecosystem:  OSS Framework

Implementation notes:
- Install:  pip install crewai crewai-tools
- Auth:     No API key for local use; OPENAI_API_KEY for cloud LLM backend.
            For Ollama: set OPENAI_API_BASE=<ollama_url>/v1 + OPENAI_API_KEY=ollama
- SDK:      Official Python SDK
    from crewai import Agent, Task, Crew
    agent = Agent(role="SRE", goal=question, llm=ollama_model)
    task  = Task(description=question, agent=agent)
    crew  = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
- For Ollama backend:
    from langchain_community.llms import Ollama
    llm = Ollama(model=model, base_url=ollama_url)
    agent = Agent(..., llm=llm)
- `question` maps to the crew objective / task description.
- `model` and `ollama_url` configure the LLM for each agent in the crew.
"""


class CrewAIRunner:
    """Ops/Misc agent: multi-agent task orchestration via CrewAI."""

    def __init__(self) -> None:
        pass

    def ask(self, question: str, model: str, ollama_url: str | None = None) -> str:
        """Kick off a CrewAI multi-agent workflow and return the final output."""
        raise NotImplementedError(
            "CrewAIRunner is not yet implemented. "
            "See https://docs.crewai.com/ for framework and Ollama integration details."
        )
