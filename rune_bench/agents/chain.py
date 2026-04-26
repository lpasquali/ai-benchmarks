# SPDX-License-Identifier: Apache-2.0
"""Multi-agent chain execution engine for RUNE."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

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


class ChainStateRecorder(Protocol):
    """Optional callback interface invoked by ``ChainExecutionEngine`` on every state change.

    A recorder lets the engine remain transport-agnostic while still emitting
    enough information for callers (e.g. ``JobStore``) to persist a live DAG
    state for dashboard rendering. All methods MUST be safe to call from any
    asyncio task; implementations are responsible for thread-safety.
    """

    def initialize(
        self, *, job_id: str, nodes: list[dict], edges: list[dict]
    ) -> None: ...

    def transition(
        self,
        *,
        job_id: str,
        node_id: str,
        status: str,
        started_at: float | None = None,
        finished_at: float | None = None,
        error: str | None = None,
    ) -> None: ...


class ChainExecutionEngine:
    """Orchestrates the execution of a Directed Acyclic Graph (DAG) of agents."""

    def __init__(
        self,
        steps: list[ChainStep],
        *,
        recorder: ChainStateRecorder | None = None,
        job_id: str | None = None,
    ) -> None:
        self._steps = {s.name: s for s in steps}
        self._recorder = recorder
        self._job_id = job_id
        self._validate_dag()

    def _initial_nodes_and_edges(self) -> tuple[list[dict], list[dict]]:
        nodes = [
            {
                "id": name,
                "agent_name": getattr(
                    step.agent, "__class__", type(step.agent)
                ).__name__,
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "error": None,
            }
            for name, step in self._steps.items()
        ]
        edges = [
            {"from": dep, "to": name}
            for name, step in self._steps.items()
            for dep in step.dependencies
        ]
        return nodes, edges

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

        # Notify the recorder before any step runs so the dashboard can show the
        # full DAG with every node in `pending` state.
        if self._recorder is not None and self._job_id is not None:
            nodes, edges = self._initial_nodes_and_edges()
            self._recorder.initialize(job_id=self._job_id, nodes=nodes, edges=edges)

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
                self._notify(
                    step_name, status="failed", error=str(exc), finished_at=time.time()
                )
                raise RuntimeError(
                    f"Step '{step_name}' template missing context: {exc}"
                )

            debug_log(f"ChainEngine: running step '{step_name}'")
            self._notify(step_name, status="running", started_at=time.time())
            try:
                result = await step.agent.ask_async(
                    question=question,
                    model=model,
                    backend_url=backend_url,
                )
            except Exception as exc:  # noqa: BLE001 — record then re-raise
                self._notify(
                    step_name, status="failed", error=str(exc), finished_at=time.time()
                )
                raise
            self._notify(step_name, status="success", finished_at=time.time())
            return result

        # Create tasks for all steps (dependencies will be awaited inside run_step)
        for name in self._steps:
            tasks[name] = asyncio.create_task(run_step(name))

        # Wait for all tasks to complete
        await asyncio.gather(*tasks.values())

        for name, task in tasks.items():
            results[name] = await task

        return ChainResult(steps=results)

    def _notify(
        self,
        node_id: str,
        *,
        status: str,
        started_at: float | None = None,
        finished_at: float | None = None,
        error: str | None = None,
    ) -> None:
        if self._recorder is None or self._job_id is None:
            return
        self._recorder.transition(
            job_id=self._job_id,
            node_id=node_id,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            error=error,
        )
