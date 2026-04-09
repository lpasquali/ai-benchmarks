# SPDX-License-Identifier: Apache-2.0
"""ManualDriverTransport — human-in-the-loop transport for agents with no public API."""

from __future__ import annotations

import json

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from rune_bench.debug import debug_log

class ManualDriverTransport:
    """Blocks and prompts a human user to provide the driver response via CLI."""

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    def call(self, action: str, params: dict) -> dict:
        from rune_bench.metrics import _tls
        from rune_bench.interactive import session_manager
        import sys
        
        job_id = getattr(_tls, "job_id", None)
        
        # If running in a background job (not an interactive TTY) or API mode, yield to session manager
        if job_id and not sys.stdout.isatty():
            prompt_data = {
                "action": action,
                "params": params,
                "message": "Please perform this action manually and provide the result JSON."
            }
            debug_log(f"ManualDriverTransport suspending for API interaction on job_id {job_id}")
            result = session_manager.request_input(job_id, prompt_data)
            return result

        self._console.print(Panel(
            f"[bold yellow]MANUAL ACTION REQUIRED[/bold yellow]\n\n"
            f"[bold]Action:[/bold] {action}\n"
            f"[bold]Params:[/bold]\n{json.dumps(params, indent=2)}\n\n"
            f"Please perform this action manually and provide the result JSON.",
            title="RUNE Human-in-the-Loop",
            border_style="yellow"
        ))

        while True:
            response_str = Prompt.ask(
                "[bold cyan]Result JSON (or 'abort')[/bold cyan]",
                console=self._console
            ).strip()

            if response_str.lower() == "abort":
                raise RuntimeError("Manual action aborted by user.")

            try:
                result = json.loads(response_str)
                if not isinstance(result, dict):
                    self._console.print("[red]Error: Result must be a JSON object (dict).[/red]")
                    continue
                return result
            except json.JSONDecodeError as exc:
                self._console.print(f"[red]Error parsing JSON: {exc}[/red]")

    async def call_async(self, action: str, params: dict) -> dict:
        """Async variant — currently just delegates to sync call as CLI input is blocking."""
        debug_log(f"ManualDriverTransport.call_async: delegating to sync call for {action!r}")
        return self.call(action, params)
