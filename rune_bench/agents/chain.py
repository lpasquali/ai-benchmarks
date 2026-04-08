# SPDX-License-Identifier: Apache-2.0
"""Multi-agent chain execution engine for RUNE."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from rune_bench.agents.base import AgentResult, AgentRunner
from rune_bench.debug import debug_log

@dataclass
class ChainStep:
    """A single step in a multi-agent chain."""
    name: str
    agent: AgentRunner
    question_template: str
    dependencies: list[str] = field(default_factory=list)

@dataclass
class ChainResult:
    """The outcome of a chain execution."""
    steps: dict[str, AgentResult]
    metadata: dict[str, Any] = field(default_factory=dict)

class ChainExecutionEngine:
    """Orchestrates the execution of a Directed Acyclic Graph (DAG) of agents."""

    def __init__(self, steps: list[ChainStep]) -> None:
        self._steps = {s.name: s for s in steps}
        self._validate_dag()

    def _validate_dag(self) -> None:
        """Ensure the chain is a valid DAG (no cycles)."""
        visited: set[str] = set()
        path: set[str] = set()

        def visit(name: str) -> None:
            if name in path:
                raise ValueError(f"Cycle detected in agent chain at '{name}'")
            if name in visited:
                return
            path.add(name)
            for dep in self._steps[name].dependencies:
                if dep not in self._steps:
                    raise ValueError(f"Step '{name}' depends on unknown step '{dep}'")
                visit(dep)
            path.remove(name)
            visited.add(name)

        for name in self._steps:
            visit(name)

    async def execute(
        self,
        initial_context: dict[str, Any],
        model: str,
        backend_url: str | None = None,
    ) -> ChainResult:
        """Execute the chain asynchronously."""
        results: dict[str, AgentResult] = {}
        tasks: dict[str, asyncio.Task] = {}

        async def run_step(step_name: str) -> AgentResult:
            step = self._steps[step_name]
            
            # Wait for dependencies
            await asyncio.gather(*(tasks[dep] for dep in step.dependencies))
            
            # Prepare context for template
            context = initial_context.copy()
            for dep_name in step.dependencies:
                context[dep_name] = (await tasks[dep_name]).answer

            # Format question
            try:
                question = step.question_template.format(**context)
            except KeyError as exc:
                raise RuntimeError(f"Step '{step_name}' template missing context: {exc}")

            debug_log(f"ChainEngine: running step '{step_name}'")
            return await step.agent.ask_async(
                question=question,
                model=model,
                backend_url=backend_url,
            )

        # Create tasks for all steps (dependencies will be awaited inside run_step)
        for name in self._steps:
            tasks[name] = asyncio.create_task(run_step(name))

        # Wait for all tasks to complete
        await asyncio.gather(*tasks.values())
        
        for name, task in tasks.items():
            results[name] = await task

        return ChainResult(steps=results)
