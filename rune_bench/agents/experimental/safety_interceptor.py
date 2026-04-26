# SPDX-License-Identifier: Apache-2.0
"""Experimental Safety Interceptor (Observer Agent).

Acts as middleware to evaluate tool execution requests against safety policies
before they reach the host system.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SafetyViolation(Exception):
    """Raised when an action violates safety policies."""

    pass


class SafetyInterceptor:
    """Evaluates agentic tool execution requests for destructive patterns."""

    def __init__(self, whitelisted_commands: Optional[List[str]] = None) -> None:
        self.whitelisted_commands = set(whitelisted_commands or [])
        # Default-deny patterns for destructive shell/system actions
        self.blacklisted_patterns = [
            re.compile(r"\brm\s+-r[fF]?\b"),  # Recursive remove
            re.compile(r"\bmv\s+.*?\s+/dev/null"),  # Move to null
            re.compile(r">\s*/dev/(sd|hd|nvme)"),  # Overwrite block devices
            re.compile(r"\bchmod\s+-R\s+777\b"),  # Global writable
            re.compile(r"\bchown\s+-R\b"),  # Recursive chown
            re.compile(r"\baws\s+iam\s+(create|delete|update)\b"),  # Destructive IAM
            re.compile(r"\bkubectl\s+delete\s+(namespace|ns)\b"),  # Delete NS
        ]

    def add_whitelist(self, command_prefix: str) -> None:
        """Allow a specific command prefix."""
        self.whitelisted_commands.add(command_prefix)

    def evaluate(self, tool_name: str, params: Dict[str, Any]) -> bool:
        """
        Evaluate the tool execution request.

        Raises SafetyViolation if the action is deemed destructive and not whitelisted.
        Returns True if safe to proceed.
        """
        # If it's an explicitly allowed tool/prefix, let it pass
        if tool_name in self.whitelisted_commands:
            logger.info(f"SafetyInterceptor: Tool '{tool_name}' is whitelisted.")
            return True

        command_string = str(params.get("command", ""))

        # If the tool isn't explicitly a 'command', we just check the tool name
        # and any stringified parameters against the blacklist.
        full_context = f"{tool_name} {command_string}"

        for pattern in self.blacklisted_patterns:
            if pattern.search(full_context):
                logger.warning(
                    f"SafetyInterceptor: Blocked destructive action matching {pattern.pattern}"
                )
                raise SafetyViolation(
                    f"Action '{full_context}' was blocked by Safety Interceptor: "
                    "matches destructive pattern."
                )

        return True
