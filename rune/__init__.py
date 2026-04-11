# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""rune.py — CLI entry point for RUNE.

RUNE = Reliability Use-case Numeric Evaluator.
All business logic lives in the `rune_bench` package.
This file only handles CLI argument parsing, Rich output, and orchestration.
"""

from pathlib import Path
import os
import sys
import socket
import asyncio
from typing import Callable

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

try:
    from rune_bench.resources.vastai.sdk import VastAI
except ImportError:
    VastAI = None  # type: ignore[assignment,misc]

from rune_bench.metrics.cost import calculate_run_cost
from rune_bench.api_client import RuneApiClient
from rune_bench.api_contracts import (
    RunAgenticAgentRequest,
    RunBenchmarkRequest,
    RunLLMInstanceRequest,
)
from rune_bench.common import (
    INIT_TEMPLATE,
    FailClosedError,
    ModelSelector,
    get_loaded_config_files,
    load_config,
    peek_profile_from_argv,
)
from rune_bench.debug import set_debug
from rune_bench.metrics import InMemoryCollector, set_collector, clear_collector
from rune_bench.backends import get_backend
from rune_bench.backends.base import ModelCapabilities
from rune_bench.agents.registry import get_agent
from rune_bench.workflows import (
    ExistingOllamaServer,
    SpendGateAction,
    UserAbortedError,
    VastAIProvisioningResult,
    DEFAULT_SPEND_THRESHOLD,
    evaluate_spend_gate,
    list_backend_models,
    list_running_backend_models,
    provision_vastai_backend,
    run_preflight_cost_check,
    stop_vastai_instance,
    warmup_backend_model,
    use_existing_backend_server,
)

# Backward-compatible aliases for renamed workflow functions.
use_existing_ollama_server = use_existing_backend_server
list_existing_ollama_models = list_backend_models
list_running_ollama_models = list_running_backend_models
warmup_existing_ollama_model = warmup_backend_model
provision_vastai_ollama = provision_vastai_backend

app = typer.Typer(help="RUNE — Reliability Use-case Numeric Evaluator", add_completion=False)
db_app = typer.Typer(
    help="Database maintenance utilities",
    add_completion=False,
    no_args_is_help=True,
)
app.add_typer(db_app, name="db")
console = Console()

# Load rune.yaml (project) / ~/.rune/config.yaml (global) before reading env vars.
# Values are injected into os.environ only when the env var is not already set,
# preserving: CLI flags > env vars > yaml config > built-in defaults.
load_config(peek_profile_from_argv())

DEFAULT_VASTAI_TEMPLATE = "c166c11f035d3a97871a23bd32ca6aba"
BACKEND_MODE = os.environ.get("RUNE_BACKEND", "local").strip().lower() or "local"
API_BASE_URL = os.environ.get("RUNE_API_BASE_URL", "http://localhost:8080").strip() or "http://localhost:8080"
API_TOKEN = os.environ.get("RUNE_API_TOKEN", "").strip() or None
API_TENANT = os.environ.get("RUNE_API_TENANT", "default").strip() or "default"
VERIFY_SSL = os.environ.get("RUNE_INSECURE", "").strip().lower() not in {"1", "true", "yes", "on"}

# Active profile (sourced from --profile argv peek or RUNE_PROFILE env var).
_ACTIVE_PROFILE: str | None = peek_profile_from_argv()


@app.callback()
def main(
    profile: str | None = typer.Option(
        None,
        "--profile",
        envvar="RUNE_PROFILE",
        help="Activate a named profile from rune.yaml (e.g. production, ci, test)",
    ),
    backend: str = typer.Option(
        BACKEND_MODE,
        "--backend",
        envvar="RUNE_BACKEND",
        help="Execution backend: local (default) or http",
    ),
    api_base_url: str = typer.Option(
        API_BASE_URL,
        "--api-base-url",
        envvar="RUNE_API_BASE_URL",
        help="Base URL for HTTP backend mode",
    ),
    api_token: str = typer.Option(
        API_TOKEN or "",
        "--api-token",
        envvar="RUNE_API_TOKEN",
        help="Bearer/API token for HTTP backend mode",
    ),
    api_tenant: str = typer.Option(
        API_TENANT,
        "--api-tenant",
        envvar="RUNE_API_TENANT",
        help="Tenant identifier for HTTP backend mode",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        envvar="RUNE_DEBUG",
        help="Show outbound client requests and API calls",
        is_eager=True,
    ),
    insecure: bool = typer.Option(
        False,
        "--insecure",
        envvar="RUNE_INSECURE",
        help="Skip TLS certificate verification (for self-signed certs, e.g. in-cluster)",
        is_eager=True,
    ),
) -> None:
    """Configure global CLI options."""
    global BACKEND_MODE, API_BASE_URL, API_TOKEN, API_TENANT, VERIFY_SSL, _ACTIVE_PROFILE
    normalized_backend = backend.strip().lower()
    if normalized_backend not in {"local", "http"}:
        raise typer.BadParameter("--backend must be either 'local' or 'http'")
    BACKEND_MODE = normalized_backend
    API_BASE_URL = api_base_url.strip() or "http://localhost:8080"
    API_TOKEN = api_token.strip() or None
    API_TENANT = api_tenant.strip() or "default"
    VERIFY_SSL = not insecure
    _ACTIVE_PROFILE = profile
    set_debug(debug)


def _print_error_and_exit(message: str, code: int = 1) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(code)


def _is_containerized() -> bool:
    """Return True when running inside a Docker container or Kubernetes pod."""
    return (
        os.environ.get("KUBERNETES_SERVICE_HOST") is not None
        or Path("/.dockerenv").exists()
    )


def _find_free_port() -> int:
    """Bind to port 0 and let the OS pick a free ephemeral port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _resolve_serve_port() -> int:
    """Return the port the API server should bind to.

    In a container (Docker / Kubernetes) use the conventional 8080 so
    Service/Ingress rules work without surprises.  On a developer
    workstation 8080 is often occupied, so we pick a random free port
    instead.
    """
    if _is_containerized():
        return 8080
    port = _find_free_port()
    console.print(f"[dim]Local mode: using random free port {port}[/dim]")
    return port


def _enable_debug_if_requested(debug: bool) -> None:
    if debug:
        set_debug(True)


_SUPPORTED_BACKEND_TYPES = {"ollama"}


def _resolve_backend_type(backend_type: str | None = None) -> str:
    """Return the effective LLM backend type.

    Resolution order:
      1. Explicit *backend_type* argument (from CLI flag).
      2. ``RUNE_BACKEND_TYPE`` environment variable.
      3. Default: ``"ollama"``.

    Raises ``RuntimeError`` if the resolved value is not a recognised backend.
    """
    resolved = (
        backend_type
        or os.environ.get("RUNE_BACKEND_TYPE", "").strip()
        or "ollama"
    )
    if resolved not in _SUPPORTED_BACKEND_TYPES:
        raise RuntimeError(
            f"Unsupported backend_type '{resolved}'. "
            f"Supported: {sorted(_SUPPORTED_BACKEND_TYPES)}"
        )
    return resolved


def _confirm_instance_creation() -> bool:
    console.print()
    ack = console.input(
        "[bold magenta]Create instance now? Type 'yes' to proceed (default no): [/bold magenta]"
    ).strip()
    return ack == "yes"


def _fetch_model_capabilities(
    backend_url: str, model: str, backend_type: str = "ollama",
) -> ModelCapabilities | None:
    """Try to fetch model capabilities from the backend; return None on any failure."""
    try:
        backend = get_backend(backend_type, backend_url)
        normalized = backend.normalize_model_name(model)
        return backend.get_model_capabilities(normalized)
    except RuntimeError:
        return None


def _vastai_sdk() -> "VastAI":
    """Instantiate VastAI SDK reading the API key from the environment."""
    if VastAI is None:
        raise RuntimeError(
            "The 'vastai' package is required for Vast.ai provisioning. "
            "Install it with: pip install 'rune-bench[vastai]'"
        )
    api_key = os.environ.get("VAST_API_KEY", "")
    return VastAI(api_key=api_key, raw=True)


def _apply_model_limits(capabilities: ModelCapabilities) -> None:
    """Pre-set LiteLLM override env vars so the agent skips the redundant re-fetch."""
    import os

    for env_name, value in (
        ("OVERRIDE_MAX_CONTENT_SIZE", capabilities.context_window),
        ("OVERRIDE_MAX_OUTPUT_TOKEN", capabilities.max_output_tokens),
    ):
        if value and value > 0 and not os.environ.get(env_name):
            os.environ[env_name] = str(value)


def _print_existing_ollama(server: ExistingOllamaServer, capabilities: ModelCapabilities | None = None) -> None:
    table = Table(title="Existing Ollama Server", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="dim")
    table.add_column("Value")
    table.add_row("Mode", "Existing server (Vast.ai disabled)")
    table.add_row("URL", server.url)
    table.add_row("Model", server.model_name)
    if capabilities:
        if capabilities.context_window is not None:
            table.add_row("Context window (OVERRIDE_MAX_CONTENT_SIZE)", f"{capabilities.context_window:,} tokens")
        if capabilities.max_output_tokens is not None:
            table.add_row("Max output tokens (OVERRIDE_MAX_OUTPUT_TOKEN)", f"{capabilities.max_output_tokens:,} tokens")
    console.print(table)


def _print_vastai_result(result: VastAIProvisioningResult, capabilities: ModelCapabilities | None = None) -> None:
    console.print(f"[green]Best offer:[/green] id={result.offer_id}, gpu_total_ram={result.total_vram_mb} MB")
    console.print(f"[green]Selected model:[/green] {result.model_name} (~{result.model_vram_mb} MB VRAM)")
    console.print(f"[green]Required disk:[/green] {result.required_disk_gb} GB")
    console.print(f"[dim]Template env:[/dim] {result.template_env}")
    action = "Reused running contract" if result.reused_existing_instance else "Provisioned contract"
    console.print(f"[green]{action}:[/green] {result.contract_id}")

    table = Table(title="Instance External URLs & Connection Details", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="dim")
    table.add_column("Value")

    table.add_row("Contract", str(result.details.contract_id))
    table.add_row("Status", result.details.status)

    if result.details.ssh_host and result.details.ssh_port:
        table.add_row("SSH", f"ssh -p {result.details.ssh_port} root@{result.details.ssh_host}")

    for svc in result.details.service_urls:
        table.add_row(f"Direct ({svc['name']})", svc["direct"])
        if svc["proxy"]:
            table.add_row(f"Vast Proxy ({svc['name']})", svc["proxy"])

    if capabilities:
        if capabilities.context_window is not None:
            table.add_row("Context window (OVERRIDE_MAX_CONTENT_SIZE)", f"{capabilities.context_window:,} tokens")
        if capabilities.max_output_tokens is not None:
            table.add_row("Max output tokens (OVERRIDE_MAX_OUTPUT_TOKEN)", f"{capabilities.max_output_tokens:,} tokens")

    console.print(table)
    console.print("\n[dim]Monitor with: python -m vastai show instances[/dim]")

    if result.backend_url:
        console.print(f"[dim]Detected Ollama endpoint:[/dim] {result.backend_url}")

    if result.pull_warning:
        console.print(f"[orange1]Warning:[/orange1] {result.pull_warning}")


def _print_vastai_models() -> None:
    table = Table(title="Configured Vast.ai Models", show_header=True, header_style="bold magenta")
    table.add_column("Model")
    table.add_column("VRAM (MB)", justify="right")
    table.add_column("Required Disk (GB)", justify="right")

    for model in ModelSelector().list_models():
        table.add_row(model.name, str(model.vram_mb), str(model.required_disk_gb))

    console.print(table)


def _print_ollama_models(backend_url: str, models: list[str], running_models: set[str]) -> None:
    table = Table(title="Existing Ollama Models", show_header=True, header_style="bold magenta")
    table.add_column("Ollama URL")
    table.add_column("Model")
    table.add_column("Running")

    if not models:
        table.add_row(backend_url, "<no models found>", "n/a")
    else:
        for index, model in enumerate(models):
            table.add_row(backend_url if index == 0 else "", model, "yes" if model in running_models else "no")

    console.print(table)


def _http_client() -> RuneApiClient:
    base_url = os.environ.get("RUNE_API_BASE_URL", API_BASE_URL)
    token = os.environ.get("RUNE_API_TOKEN", API_TOKEN)
    tenant = os.environ.get("RUNE_API_TENANT", API_TENANT)
    return RuneApiClient(base_url, api_token=token, tenant_id=tenant, verify_ssl=VERIFY_SSL)


async def _run_preflight_cost_check(
    *,
    vastai: bool,
    max_dph: float,
    min_dph: float,
    yes: bool,
    estimated_duration_seconds: int = 3600,
) -> None:
    """Estimate projected spend and display a Rich warning panel before execution.

    Raises typer.Exit(1) on FailClosedError, estimation failure without --yes,
    or if the user declines to proceed.
    Does nothing when no cloud cost driver is active (local/existing Ollama server).
    """
    if not vastai:
        return

    try:
        result = await run_preflight_cost_check(
            vastai=vastai,
            max_dph=max_dph,
            min_dph=min_dph,
            estimated_duration_seconds=estimated_duration_seconds,
            backend_mode=BACKEND_MODE,
            http_client=_http_client() if BACKEND_MODE == "http" else None,
        )
    except FailClosedError as exc:
        console.print(f"[red]Cost estimation error:[/red] {exc}")
        raise typer.Exit(1)
    except RuntimeError as exc:
        if yes:
            console.print(f"[yellow]Cost estimation unavailable:[/yellow] {exc}")
            return
        console.print(
            f"[red]Cost estimation failed:[/red] {exc}\n"
            "Pass --yes / -y to skip cost check and proceed."
        )
        raise typer.Exit(1)

    if not result:
        return

    projected_cost: float = float(result.get("projected_cost_usd", 0.0))
    driver: str = str(result.get("cost_driver", "unknown"))
    impact: str = str(result.get("resource_impact", "low"))
    warning: str | None = result.get("warning")  # type: ignore[assignment]

    panel_lines = [
        f"Projected cost: [bold]${projected_cost:.2f}[/bold] (driver: {driver})",
        f"Resource impact: {impact}",
    ]
    if warning:
        panel_lines.append(f"Warning: {warning}")

    console.print(Panel(
        "\n".join(panel_lines),
        title="[bold yellow]Spend Warning[/bold yellow]",
        border_style="yellow",
    ))

    try:
        threshold = float(os.environ.get("RUNE_SPEND_WARNING_THRESHOLD", str(DEFAULT_SPEND_THRESHOLD)))
    except (ValueError, TypeError):
        console.print("[yellow]Warning: Invalid RUNE_SPEND_WARNING_THRESHOLD value; using default $5.00.[/yellow]")
        threshold = DEFAULT_SPEND_THRESHOLD

    action = evaluate_spend_gate(projected_cost, threshold=threshold, yes=yes)

    if action is SpendGateAction.ALLOW:
        return

    if action is SpendGateAction.BLOCK:
        console.print(
            f"[red]Spend threshold exceeded (${projected_cost:.2f} > ${threshold:.2f}). "
            "Pass --yes / -y to proceed in CI.[/red]"
        )
        raise typer.Exit(1)

    if action is SpendGateAction.PROMPT:
        if not os.isatty(sys.stdin.fileno()):
            console.print(
                "[red]Confirm-to-spend prompt required, but environment is non-interactive. "
                "Use --yes / -y to proceed.[/red]"
            )
            raise typer.Exit(1)

        ack = console.input("\n[bold magenta]Proceed with benchmark? [y/N]: [/bold magenta]").strip().lower()
        if ack not in {"y", "yes"}:
            console.print("Aborted.")
            raise typer.Exit(1)


async def _run_http_job_with_progress(
    *,
    submit_description: str,
    wait_description: str,
    submit_job: Callable[[], str],
    client: RuneApiClient,
    timeout_seconds: int = 3600,
    poll_interval_seconds: float = 2.0,
) -> dict:
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task(submit_description, total=None)
        job_id = submit_job()
        progress.update(task, description=f"{wait_description} (job={job_id})")

        def on_update(status: str, message: str | None) -> None:
            detail = f": {message}" if message else ""
            progress.update(task, description=f"{wait_description} [{status}]{detail}")

        return await asyncio.to_thread(
            client.wait_for_job,
            job_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            on_update=on_update,
        )


def _print_metrics_summary(collector: InMemoryCollector) -> None:
    """Print a workflow lifecycle metrics table to the console."""
    rows = collector.summary_rows()
    if not rows:
        return
    table = Table(title="Workflow Lifecycle Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Event", style="dim")
    table.add_column("Total", justify="right")
    table.add_column("OK", justify="right", style="green")
    table.add_column("Error", justify="right", style="red")
    table.add_column("Avg (ms)", justify="right")
    table.add_column("Min (ms)", justify="right")
    table.add_column("Max (ms)", justify="right")
    for row in rows:
        table.add_row(
            row["event"],
            str(row["total"]),
            str(row["ok"]),
            str(row["error"]),
            f"{row['avg_ms']:.0f}",
            f"{row['min_ms']:.0f}",
            f"{row['max_ms']:.0f}",
        )
    console.print(table)


def _warmup_ollama_model(*, backend_url: str, model_name: str, timeout_seconds: int) -> None:
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        progress.add_task(f"Loading Ollama model {model_name} and waiting until it is ready...", total=None)
        try:
            warmed_model = warmup_backend_model(
                backend_url,
                model_name,
                timeout_seconds=timeout_seconds,
            )
        except RuntimeError as exc:
            _print_error_and_exit(str(exc))

    console.print(f"[green]Ollama model ready:[/green] {warmed_model}")


def _run_vastai_provisioning(
    *,
    template_hash: str,
    min_dph: float,
    max_dph: float,
    reliability: float,
) -> VastAIProvisioningResult:
    sdk = _vastai_sdk()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as p:
        task = p.add_task("Provisioning on Vast.ai...", total=None)

        def on_poll(status: str) -> None:
            p.update(task, description=f"Waiting for running status: {status}")

        try:
            return provision_vastai_backend(
                sdk,
                template_hash=template_hash,
                min_dph=min_dph,
                max_dph=max_dph,
                reliability=reliability,
                confirm_create=_confirm_instance_creation,
                on_poll=on_poll,
            )
        except UserAbortedError:
            console.print("Aborted.")
            raise typer.Exit(0)
        except RuntimeError as exc:
            _print_error_and_exit(str(exc))
    raise AssertionError("unreachable")  # pragma: no cover


@app.command("serve")
def serve_api(
    api_host: str = typer.Option(
        "127.0.0.1",
        "--host",
        envvar="RUNE_API_HOST",
        help="Host to bind API server to (set RUNE_API_HOST=0.0.0.0 for container deployments)",
    ),
    api_port: int | None = typer.Option(
        None,
        "--port",
        envvar="RUNE_API_PORT",
        help="Port to bind API server to (default: 8080 in containers, random free port locally)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        envvar="RUNE_DEBUG",
        help="Show outbound client requests and API calls",
    ),
) -> None:
    """Start the standalone RUNE API server."""
    from rune_bench.api_server import RuneApiApplication

    _enable_debug_if_requested(debug)

    resolved_port = api_port if api_port is not None else _resolve_serve_port()
    console.print(Panel.fit(
        f"[bold blue]RUNE API Server[/bold blue]\nStarting on {api_host}:{resolved_port}"
        + ("  [yellow](TLS verification disabled)[/yellow]" if not VERIFY_SSL else "")
    ))

    try:
        app_server = RuneApiApplication.from_env()
        app_server.serve(host=api_host, port=resolved_port)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")
        raise typer.Exit(0)
    except Exception as exc:
        _print_error_and_exit(f"Server error: {exc}")


@app.command("run-llm-instance")
async def run_llm_instance(
    debug: bool = typer.Option(
        False,
        "--debug",
        envvar="RUNE_DEBUG",
        help="Show outbound client requests and API calls",
    ),
    vastai: bool = typer.Option(
        False,
        "--vastai",
        envvar="RUNE_VASTAI",
        help="Enable Vast.ai provisioning flow",
    ),
    template_hash: str = typer.Option(
        DEFAULT_VASTAI_TEMPLATE,
        "--vastai-template",
        envvar="RUNE_VASTAI_TEMPLATE",
        help="Vast.ai template hash to use",
    ),
    max_dph: float = typer.Option(3.0, "--vastai-max-dph", envvar="RUNE_VASTAI_MAX_DPH", help="Maximum dollars per hour"),
    min_dph: float = typer.Option(2.3, "--vastai-min-dph", envvar="RUNE_VASTAI_MIN_DPH", help="Minimum dollars per hour"),
    reliability: float = typer.Option(0.99, "--vastai-reliability", envvar="RUNE_VASTAI_RELIABILITY", help="Minimum reliability score"),
    backend_url: str | None = typer.Option(
        None,
        "--backend-url",
        envvar="RUNE_BACKEND_URL",
        help="Use an already running Ollama server URL when --vastai is not enabled",
    ),
    idempotency_key: str | None = typer.Option(
        None,
        "--idempotency-key",
        envvar="RUNE_IDEMPOTENCY_KEY",
        help="Optional idempotency key when using the HTTP backend",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        envvar="RUNE_YES",
        help="Skip interactive spend confirmation (auto-approve)",
    ),
) -> None:
    """Provision an Ollama instance on Vast.ai, or use an existing server."""
    _enable_debug_if_requested(debug)
    provisioning = None
    if vastai:
        from rune_bench.api_contracts import Provisioning, VastAIProvisioning
        provisioning = Provisioning(
            vastai=VastAIProvisioning(
                template_hash=template_hash,
                min_dph=min_dph,
                max_dph=max_dph,
                reliability=reliability,
                stop_instance=False,
            )
        )
    _request = RunLLMInstanceRequest(
        provisioning=provisioning,
        backend_url=backend_url,
    )
    console.print(Panel.fit("[bold blue]RUNE — Reliability Use-case Numeric Evaluator[/bold blue]"))

    await _run_preflight_cost_check(
        vastai=vastai,
        max_dph=max_dph,
        min_dph=min_dph,
        yes=yes,
    )

    if BACKEND_MODE == "http":
        try:
            client = _http_client()
            payload = await _run_http_job_with_progress(
                submit_description="Submitting ollama-instance job to HTTP backend...",
                wait_description="Waiting for ollama-instance job",
                submit_job=lambda: client.submit_ollama_instance_job(
                    _request.to_dict(),
                    idempotency_key=idempotency_key,
                ),
                client=client,
            )
        except RuntimeError as exc:
            _print_error_and_exit(str(exc))

        result_obj = payload.get("result")
        http_result: dict[str, object] = result_obj if isinstance(result_obj, dict) else {}
        mode = http_result.get("mode")
        if mode == "existing":
            server = ExistingOllamaServer(
                url=str(http_result.get("backend_url", backend_url or "")),
                model_name="<user-selected>",
            )
            _print_existing_ollama(server)
            return
        if mode == "vastai":
            console.print(f"[green]Provisioned contract:[/green] {http_result.get('contract_id')}")
            if http_result.get("backend_url"):
                console.print(f"[dim]Detected Ollama endpoint:[/dim] {http_result.get('backend_url')}")
            if http_result.get("model_name"):
                console.print(f"[green]Selected model:[/green] {http_result.get('model_name')}")
            return
        _print_error_and_exit("HTTP backend finished but did not return an Ollama instance result")

    if not vastai:
        try:
            server = use_existing_backend_server(backend_url, model_name="<user-selected>")
        except RuntimeError as exc:
            _print_error_and_exit(str(exc))
        _print_existing_ollama(server)
        return

    _cli_metrics = InMemoryCollector()
    set_collector(_cli_metrics)
    result = _run_vastai_provisioning(
        template_hash=template_hash,
        min_dph=min_dph,
        max_dph=max_dph,
        reliability=reliability,
    )
    _print_vastai_result(result)
    _print_metrics_summary(_cli_metrics)
    clear_collector()


@app.command("vastai-list-models")
def vastai_list_models() -> None:
    """List the configured models used by the Vast.ai provisioning flow."""
    console.print(Panel.fit("[bold blue]RUNE — Configured Vast.ai Models[/bold blue]"))

    if BACKEND_MODE == "http":
        try:
            models_payload = _http_client().get_vastai_models()
        except RuntimeError as exc:
            _print_error_and_exit(str(exc))

        table = Table(title="Configured Vast.ai Models", show_header=True, header_style="bold magenta")
        table.add_column("Model")
        table.add_column("VRAM (MB)", justify="right")
        table.add_column("Required Disk (GB)", justify="right")

        for model in models_payload:
            table.add_row(
                str(model.get("name", "<unknown>")),
                str(model.get("vram_mb", "<unknown>")),
                str(model.get("required_disk_gb", "<unknown>")),
            )

        console.print(table)
        return

    _print_vastai_models()


@app.command("ollama-list-models")
def ollama_list_models(
    debug: bool = typer.Option(
        False,
        "--debug",
        envvar="RUNE_DEBUG",
        help="Show outbound client requests and API calls",
    ),
    backend_url: str = typer.Option(
        ...,
        "--backend-url",
        envvar="RUNE_BACKEND_URL",
        help="Ollama server URL to query for available models",
    ),
) -> None:
    """List the models exposed by an existing Ollama server."""
    _enable_debug_if_requested(debug)
    console.print(Panel.fit("[bold blue]RUNE — Existing Ollama Models[/bold blue]"))

    if BACKEND_MODE == "http":
        try:
            payload = _http_client().get_ollama_models(backend_url)
            normalized_url = str(payload.get("backend_url", backend_url))
            models = [str(m) for m in payload.get("models", [])]
            running_models = {str(m) for m in payload.get("running_models", [])}
        except RuntimeError as exc:
            _print_error_and_exit(str(exc))

        _print_ollama_models(normalized_url, models, running_models)
        return

    try:
        normalized_url = use_existing_backend_server(backend_url, model_name="<n/a>").url
        models = list_backend_models(normalized_url)
        running_models = set(list_running_backend_models(normalized_url))
    except RuntimeError as exc:
        _print_error_and_exit(str(exc))

    _print_ollama_models(normalized_url, models, running_models)


@app.command("run-agentic-agent")
async def run_agentic_agent(
    debug: bool = typer.Option(
        False,
        "--debug",
        envvar="RUNE_DEBUG",
        help="Show outbound client requests and API calls",
    ),
    question: str = typer.Option(
        "What is unhealthy in this Kubernetes cluster?",
        "--question",
        "-q",
        envvar="RUNE_QUESTION",
        help="Question to ask the agentic system",
    ),
    model: str = typer.Option(
        "llama3.1:8b",
        "--model",
        "-m",
        envvar="RUNE_MODEL",
        help="Model to use for the agent",
    ),
    backend_url: str | None = typer.Option(
        None,
        "--backend-url",
        envvar="RUNE_BACKEND_URL",
        help="Ollama server URL (used for Ollama-backed models)",
    ),
    backend_warmup: bool = typer.Option(
        True,
        "--backend-warmup/--no-backend-warmup",
        envvar="RUNE_BACKEND_WARMUP",
        help="Load the selected model into the Ollama server before starting the agent",
    ),
    backend_warmup_timeout: int = typer.Option(
        90,
        "--backend-warmup-timeout",
        envvar="RUNE_BACKEND_WARMUP_TIMEOUT",
        min=1,
        help="Seconds to wait for the Ollama model to become ready",
    ),
    kubeconfig: Path = typer.Option(
        Path.home() / ".kube" / "config",
        "--kubeconfig",
        envvar="RUNE_KUBECONFIG",
        help="Path to kubeconfig file",
    ),
    idempotency_key: str | None = typer.Option(
        None,
        "--idempotency-key",
        envvar="RUNE_IDEMPOTENCY_KEY",
        help="Optional idempotency key when using the HTTP backend",
    ),
) -> None:
    """Run an agentic agent against a Kubernetes cluster."""
    _enable_debug_if_requested(debug)
    _request = RunAgenticAgentRequest.from_cli(
        question=question,
        model=model,
        backend_url=backend_url,
        backend_warmup=backend_warmup,
        backend_warmup_timeout=backend_warmup_timeout,
        kubeconfig=kubeconfig,
    )
    console.print(Panel.fit("[bold blue]RUNE — Agentic Agent Runner[/bold blue]"))

    if BACKEND_MODE == "http":
        try:
            client = _http_client()
            payload = await _run_http_job_with_progress(
                submit_description="Submitting agentic-agent job to HTTP backend...",
                wait_description="Waiting for agentic-agent job",
                submit_job=lambda: client.submit_agentic_agent_job(
                    _request.to_dict(),
                    idempotency_key=idempotency_key,
                ),
                client=client,
            )
        except RuntimeError as exc:
            _print_error_and_exit(str(exc))

        result_obj = payload.get("result")
        http_result: dict[str, object] = result_obj if isinstance(result_obj, dict) else {}
        answer = http_result.get("answer") or payload.get("answer")
        result_type = str(http_result.get("result_type", payload.get("result_type", "text")))
        artifacts = http_result.get("artifacts") or payload.get("artifacts")

        if not isinstance(answer, str) or not answer.strip():
            _print_error_and_exit("HTTP backend finished but did not return an agent answer")

        console.print(f"\n[bold green]Agent Answer ({result_type})[/bold green]")
        console.print(answer)
        if artifacts:
            console.print(f"\n[bold blue]Artifacts ({len(artifacts)})[/bold blue]")
            console.print(artifacts)
        return

    _cli_metrics = InMemoryCollector()
    set_collector(_cli_metrics)

    if backend_url and backend_warmup:
        _warmup_ollama_model(
            backend_url=backend_url,
            model_name=model,
            timeout_seconds=backend_warmup_timeout,
        )

    # Block 10 — Run agentic agent
    try:
        from rune_bench.metrics import span as _span
        runner = get_agent(_request.agent, kubeconfig=kubeconfig)
        with _span("agent.ask", model=model, backend="existing"):
            result = await runner.ask_structured(
                question=question,
                model=model,
                backend_url=backend_url,
                backend_type=_resolve_backend_type(None),
            )
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]Agent error:[/red] {exc}")
        raise typer.Exit(1)

    _print_metrics_summary(_cli_metrics)
    clear_collector()
    console.print(f"\n[bold green]Agent Answer ({result.result_type})[/bold green]")
    console.print(result.answer)
    if result.artifacts:
        console.print(f"\n[bold blue]Artifacts ({len(result.artifacts)})[/bold blue]")
        console.print(result.artifacts)


@app.command("run-benchmark")
async def run_benchmark(
    debug: bool = typer.Option(
        False,
        "--debug",
        envvar="RUNE_DEBUG",
        help="Show outbound client requests and API calls",
    ),
    vastai: bool = typer.Option(
        False,
        "--vastai",
        envvar="RUNE_VASTAI",
        help="Enable Vast.ai provisioning flow (Phase 1)",
    ),
    template_hash: str = typer.Option(
        DEFAULT_VASTAI_TEMPLATE,
        "--vastai-template",
        envvar="RUNE_VASTAI_TEMPLATE",
        help="Vast.ai template hash to use",
    ),
    max_dph: float = typer.Option(3.0, "--vastai-max-dph", envvar="RUNE_VASTAI_MAX_DPH", help="Maximum dollars per hour"),
    min_dph: float = typer.Option(2.3, "--vastai-min-dph", envvar="RUNE_VASTAI_MIN_DPH", help="Minimum dollars per hour"),
    reliability: float = typer.Option(0.99, "--vastai-reliability", envvar="RUNE_VASTAI_RELIABILITY", help="Minimum reliability score"),
    backend_url: str | None = typer.Option(
        None,
        "--backend-url",
        envvar="RUNE_BACKEND_URL",
        help="Existing Ollama server URL when --vastai is not enabled",
    ),
    question: str = typer.Option(
        "What is unhealthy in this Kubernetes cluster?",
        "--question",
        "-q",
        envvar="RUNE_QUESTION",
        help="Question to ask the agentic system",
    ),
    model: str = typer.Option(
        "llama3.1:8b",
        "--model",
        "-m",
        envvar="RUNE_MODEL",
        help="Model to use for the agent when --vastai is not enabled",
    ),
    backend_warmup: bool = typer.Option(
        True,
        "--backend-warmup/--no-backend-warmup",
        envvar="RUNE_BACKEND_WARMUP",
        help="Load the selected model into the Ollama server before the agent starts when using --backend-url",
    ),
    backend_warmup_timeout: int = typer.Option(
        90,
        "--backend-warmup-timeout",
        envvar="RUNE_BACKEND_WARMUP_TIMEOUT",
        min=1,
        help="Seconds to wait for the Ollama model to become ready",
    ),
    kubeconfig: Path = typer.Option(
        Path.home() / ".kube" / "config",
        "--kubeconfig",
        envvar="RUNE_KUBECONFIG",
        help="Path to kubeconfig file",
    ),
    vastai_stop_instance: bool = typer.Option(
        True,
        "--vastai-stop-instance/--no-vastai-stop-instance",
        envvar="RUNE_VASTAI_STOP_INSTANCE",
        help="Destroy Vast.ai instance + related storage after agent execution and verify cleanup (enabled by default)",
    ),
    idempotency_key: str | None = typer.Option(
        None,
        "--idempotency-key",
        envvar="RUNE_IDEMPOTENCY_KEY",
        help="Optional idempotency key when using the HTTP backend",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        envvar="RUNE_YES",
        help="Skip interactive spend confirmation (auto-approve)",
    ),
) -> None:
    """Run full benchmark: provision Ollama instance, then run agentic agent."""
    _enable_debug_if_requested(debug)
    _request = RunBenchmarkRequest.from_cli(
        vastai=vastai,
        template_hash=template_hash,
        min_dph=min_dph,
        max_dph=max_dph,
        reliability=reliability,
        backend_url=backend_url,
        question=question,
        model=model,
        backend_warmup=backend_warmup,
        backend_warmup_timeout=backend_warmup_timeout,
        kubeconfig=kubeconfig,
        vastai_stop_instance=vastai_stop_instance,
    )
    console.print(Panel.fit("[bold blue]RUNE — Full Benchmark Workflow[/bold blue]"))

    await _run_preflight_cost_check(
        vastai=vastai,
        max_dph=max_dph,
        min_dph=min_dph,
        yes=yes,
    )

    if BACKEND_MODE == "http":
        try:
            client = _http_client()
            payload = await _run_http_job_with_progress(
                submit_description="Submitting benchmark job to HTTP backend...",
                wait_description="Waiting for benchmark job",
                submit_job=lambda: client.submit_benchmark_job(
                    _request.to_dict(),
                    idempotency_key=idempotency_key,
                ),
                client=client,
            )
        except RuntimeError as exc:
            _print_error_and_exit(str(exc))

        result_obj = payload.get("result")
        http_result: dict[str, object] = result_obj if isinstance(result_obj, dict) else {}
        answer = http_result.get("answer") or payload.get("answer")
        result_type = str(http_result.get("result_type", payload.get("result_type", "text")))
        artifacts = http_result.get("artifacts") or payload.get("artifacts")

        if not isinstance(answer, str) or not answer.strip():
            _print_error_and_exit("HTTP backend finished but did not return a benchmark answer")

        console.print(f"\n[bold green]Agent Answer ({result_type})[/bold green]")
        console.print(answer)
        if artifacts:
            console.print(f"\n[bold blue]Artifacts ({len(artifacts)})[/bold blue]")
            console.print(artifacts)
        return

    _cli_metrics = InMemoryCollector()
    set_collector(_cli_metrics)
    selected_model_name = model
    selected_backend_url = backend_url
    vastai_contract_to_stop: int | str | None = None

    if vastai:
        console.print("\n[bold cyan]PHASE 1: Provisioning Ollama Instance[/bold cyan]")
        result = _run_vastai_provisioning(
            template_hash=template_hash,
            min_dph=min_dph,
            max_dph=max_dph,
            reliability=reliability,
        )
        selected_model_name = result.model_name
        selected_backend_url = result.backend_url
        vastai_contract_to_stop = result.contract_id
        vastai_capabilities = _fetch_model_capabilities(result.backend_url, selected_model_name) if result.backend_url else None
        if vastai_capabilities:
            _apply_model_limits(vastai_capabilities)
        _print_vastai_result(result, capabilities=vastai_capabilities)
        if not selected_backend_url:
            if vastai_stop_instance and vastai_contract_to_stop is not None:
                try:
                    teardown = stop_vastai_instance(_vastai_sdk(), vastai_contract_to_stop)
                    status = "verified" if teardown.verification_ok else "not fully verified"
                    console.print(f"[green]Destroyed Vast.ai contract:[/green] {vastai_contract_to_stop} ({status})")
                    if teardown.destroyed_volume_ids:
                        console.print(
                            "[green]Destroyed related volumes:[/green] "
                            + ", ".join(teardown.destroyed_volume_ids)
                        )
                    console.print(f"[dim]{teardown.verification_message}[/dim]")
                except RuntimeError as stop_exc:
                    console.print(
                        f"[orange1]Warning:[/orange1] Failed to destroy Vast.ai contract "
                        f"{vastai_contract_to_stop}: {stop_exc}"
                    )
            _print_error_and_exit(
                "Could not determine Ollama URL from the Vast.ai instance service mappings. "
                "Ensure port 11434 is exposed in the template."
            )
    else:
        try:
            server = use_existing_backend_server(backend_url, model_name=selected_model_name)
        except RuntimeError as exc:
            _print_error_and_exit(str(exc))
        console.print("\n[bold cyan]PHASE 1: Using Existing Ollama Server[/bold cyan]")
        capabilities = _fetch_model_capabilities(server.url, selected_model_name)
        if capabilities:
            _apply_model_limits(capabilities)
        _print_existing_ollama(server, capabilities=capabilities)
        selected_backend_url = server.url

    # ========== PHASE 2: Agentic Agent Execution ==========
    console.print("\n[bold cyan]PHASE 2: Running Agentic Agent[/bold cyan]")

    if selected_backend_url and backend_warmup:
        _warmup_ollama_model(
            backend_url=selected_backend_url,
            model_name=selected_model_name,
            timeout_seconds=backend_warmup_timeout,
        )

    # Block 10 — Run agentic agent
    try:
        runner = get_agent("holmes", kubeconfig=kubeconfig)
        result = await runner.ask_structured(
            question=question,
            model=selected_model_name,
            backend_url=selected_backend_url,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]Agent error:[/red] {exc}")
        raise typer.Exit(1)
    finally:
        if vastai and vastai_stop_instance and vastai_contract_to_stop is not None:
            try:
                teardown = stop_vastai_instance(_vastai_sdk(), vastai_contract_to_stop)
                status = "verified" if teardown.verification_ok else "not fully verified"
                console.print(f"[green]Destroyed Vast.ai contract:[/green] {vastai_contract_to_stop} ({status})")
                if teardown.destroyed_volume_ids:
                    console.print(
                        "[green]Destroyed related volumes:[/green] "
                        + ", ".join(teardown.destroyed_volume_ids)
                    )
                console.print(f"[dim]{teardown.verification_message}[/dim]")
            except RuntimeError as stop_exc:
                console.print(
                    f"[orange1]Warning:[/orange1] Failed to destroy Vast.ai contract "
                    f"{vastai_contract_to_stop}: {stop_exc}"
                )

    _print_metrics_summary(_cli_metrics)
    clear_collector()
    console.print(f"\n[bold green]Agent Answer ({result.result_type})[/bold green]")
    console.print(result.answer)
    if result.artifacts:
        console.print(f"\n[bold blue]Artifacts ({len(result.artifacts)})[/bold blue]")
        console.print(result.artifacts)


if __name__ == "__main__":
    app()


@app.command("info")
def show_info() -> None:
    """Show installed extras and environment information."""
    import importlib.metadata

    try:
        version = importlib.metadata.version("rune-bench")
    except importlib.metadata.PackageNotFoundError:
        version = "(dev — not installed as package)"

    table = Table(title="RUNE Environment", show_header=True, header_style="bold blue")
    table.add_column("Extra / Component", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Install command", style="dim")

    # Check vastai
    try:
        import vastai  # type: ignore[import-untyped, import-not-found]  # noqa: F401  # Reason: missing stubs
        vastai_status = "[green]✓ installed[/green]"
        vastai_cmd = ""
    except ImportError:
        vastai_status = "[yellow]✗ not installed[/yellow]"
        vastai_cmd = 'pip install "rune-bench[vastai]"'

    # Check holmesgpt
    try:
        import holmes  # type: ignore[import-untyped, import-not-found]  # noqa: F401  # Reason: missing stubs
        holmes_status = "[green]✓ installed[/green]"
        holmes_cmd = ""
    except ImportError:
        holmes_status = "[yellow]✗ not installed[/yellow]"
        holmes_cmd = 'pip install "rune-bench[holmes]"'

    table.add_row("[bold]vastai[/bold]  (Vast.ai GPU provisioning)", vastai_status, vastai_cmd)
    table.add_row("[bold]holmes[/bold]  (SRE agent driver)", holmes_status, holmes_cmd)

    console.print(Panel.fit(f"[bold blue]rune-bench[/bold blue] v{version}"))
    console.print(table)
    if vastai_cmd or holmes_cmd:
        console.print(
            "\n[dim]Install all extras:[/dim] [cyan]pip install \"rune-bench[all]\"[/cyan]"
        )


@app.command("init")
def init_config(
    force: bool = typer.Option(False, "--force", help="Overwrite existing rune.yaml"),
) -> None:
    """Scaffold a rune.yaml configuration file in the current directory."""
    target = Path("rune.yaml")
    if target.exists() and not force:
        console.print(
            "[yellow]rune.yaml already exists.[/yellow] Use [cyan]--force[/cyan] to overwrite."
        )
        raise typer.Exit(0)
    target.write_text(INIT_TEMPLATE)
    console.print(f"[green]✓ Created[/green] {target.resolve()}")
    console.print(
        "\n[dim]Edit the file to set your defaults and profiles, then activate a profile with:[/dim]\n"
        "  [cyan]rune --profile production run-benchmark[/cyan]\n"
        "  [cyan]RUNE_PROFILE=ci rune run-agentic-agent[/cyan]\n"
        "\n[dim]Tip:[/dim] Add [cyan]rune.yaml[/cyan] to [dim].gitignore[/dim] if it contains "
        "environment-specific paths."
    )


@app.command("config")
def show_config() -> None:
    """Show the effective merged configuration (profile + defaults + files loaded)."""
    effective = load_config(_ACTIVE_PROFILE)
    config_files = get_loaded_config_files()

    # Header
    profile_label = f"[bold cyan]{_ACTIVE_PROFILE}[/bold cyan]" if _ACTIVE_PROFILE else "[dim](none — using defaults)[/dim]"
    console.print(Panel.fit(
        f"Active profile: {profile_label}",
        title="[bold blue]RUNE Configuration[/bold blue]",
    ))

    # Config files loaded
    if config_files:
        console.print("\n[bold]Config files loaded[/bold] (later overrides earlier):")
        for cf in config_files:
            console.print(f"  [green]✓[/green] {cf.resolve()}")
    else:
        console.print("\n[dim]No rune.yaml found — using built-in defaults and env vars.[/dim]")
        console.print(
            "[dim]Run [cyan]rune init[/cyan] to scaffold a rune.yaml in the current directory.[/dim]"
        )

    if not effective:
        return

    # Effective values table
    table = Table(title="\nEffective configuration", show_header=True, header_style="bold blue")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="bold")
    table.add_column("Env var", style="dim")

    from rune_bench.common.config import _FIELD_ENV_MAP
    for key, env_var in _FIELD_ENV_MAP.items():
        if key in effective:
            env_val = os.environ.get(env_var, "")
            table.add_row(key, str(effective[key]), f"{env_var}={env_val}" if env_val else env_var)

    console.print(table)
    console.print(
        "\n[dim]Secrets (RUNE_API_TOKEN, VAST_API_KEY) are never shown here — "
        "check your environment directly.[/dim]"
    )
