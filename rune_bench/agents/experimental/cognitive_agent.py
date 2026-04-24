# SPDX-License-Identifier: Apache-2.0
"""Experimental Unified Cognitive Agent Runner.

Integrates MemoryProvider, SafetyInterceptor, MCP Tool Routing,
and a ReAct/Reflection loop to serve as the prototype for Tier 3 autonomous agents.
"""

import logging
from typing import Any, Dict, List, Optional

from rune_bench.drivers.mcp_poc import MCPClientDriver
from rune_bench.agents.experimental.memory_provider import MemoryProvider
from rune_bench.agents.experimental.safety_interceptor import (
    SafetyInterceptor,
    SafetyViolation,
)

logger = logging.getLogger(__name__)


class CognitiveAgentRunner:
    """Unified Cognitive Agent running a Plan -> Act -> Observe -> Reflect loop."""

    def __init__(
        self,
        mcp_client: MCPClientDriver,
        max_iterations: int = 3,
        whitelisted_commands: Optional[List[str]] = None,
    ) -> None:
        self.mcp_client = mcp_client
        self.max_iterations = max_iterations
        self.memory = MemoryProvider()
        self.safety = SafetyInterceptor(whitelisted_commands=whitelisted_commands)

    def _plan(
        self, question: str, context: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Simulate generating a plan based on the question and current episodic memory."""
        # A real implementation would prompt the LLM here, providing the available tools
        # from self.mcp_client.discover_tools() and the episodic context.
        logger.info("Generating plan...")

        # For prototype, we generate a static mock plan if the question matches known ones.
        if "kubectl get pods" in question:
            return [{"tool": "kubectl_get_pods", "args": {}}]
        elif "echo" in question:
            return [{"tool": "echo", "args": {"message": "Cognitive test"}}]
        elif "rm -rf" in question:
            return [{"tool": "shell", "args": {"command": "rm -rf /"}}]

        return [{"tool": "echo", "args": {"message": "Unknown objective"}}]

    def _reflect(self, goal: str, outcomes: List[str]) -> str:
        """Simulate reflecting on the execution outcomes."""
        logger.info("Reflecting on outcomes...")
        # A real implementation would ask the LLM if the goal was met based on outcomes.
        if any("failed" in out.lower() or "blocked" in out.lower() for out in outcomes):
            return f"Reflection: Failed to achieve goal '{goal}'. Needs replanning."
        return f"Reflection: Successfully achieved goal '{goal}'."

    def ask(
        self,
        question: str,
        model: str,
        backend_url: str | None = None,
        backend_type: str = "ollama",
    ) -> str:
        """Run the full cognitive ReAct loop."""
        self.memory.append_episodic(action="Receive Objective", result=question)

        all_outcomes = []

        for iteration in range(self.max_iterations):
            # 1. Plan
            context = self.memory.get_episodic_context()
            plan = self._plan(question, context)

            iteration_outcomes = []

            # 2. Act & Observe
            for step in plan:
                tool = step.get("tool", "")
                args = step.get("args", {})

                try:
                    # Guard the execution with SafetyInterceptor
                    # Map the 'command' arg if the tool is generic, otherwise the tool name
                    self.safety.evaluate(tool, args)

                    # Execute via MCP
                    output = self.mcp_client.run_tool(tool, **args)
                    outcome = f"Step {tool} succeeded: {output}"

                except SafetyViolation as e:
                    outcome = f"Step {tool} blocked by safety policy: {e}"
                except Exception as e:
                    outcome = f"Step {tool} failed: {e}"

                iteration_outcomes.append(outcome)
                all_outcomes.append(outcome)
                self.memory.append_episodic(action=f"Execute {tool}", result=outcome)

            # 3. Reflect
            reflection = self._reflect(question, iteration_outcomes)
            self.memory.append_episodic(action="Reflect", result=reflection)
            all_outcomes.append(reflection)

            # If successful, we can break early
            if "Successfully" in reflection:
                # Procedural memory caching (mock)
                self.memory.cache_procedure(
                    question, [step.get("tool", "") for step in plan]
                )
                break

        return "\n".join(all_outcomes)
