#!/usr/bin/env python3
"""rune.py — CLI entry point for RUNE.

RUNE = Reliability Use-case Numeric Evaluator.
All business logic lives in the `rune_bench` package.
This file only handles CLI argument parsing, Rich output, and orchestration.
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from vastai import VastAI

from rune_bench import HolmesRunner
from rune_bench.common import ModelSelector
from rune_bench.debug import set_debug
from rune_bench.ollama import OllamaClient, OllamaModelCapabilities, OllamaModelManager
from rune_bench.workflows import (
    ExistingOllamaServer,
    UserAbortedError,
    VastAIProvisioningResult,
    list_existing_ollama_models,
    list_running_ollama_models,
    provision_vastai_ollama,
    stop_vastai_instance,
    warmup_existing_ollama_model,
    use_existing_ollama_server,
)

app = typer.Typer(help="RUNE — Reliability Use-case Numeric Evaluator", add_completion=False)
console = Console()

DEFAULT_VASTAI_TEMPLATE = "c166c11f035d3a97871a23bd32ca6aba"


@app.callback()
def main(
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show outbound client requests and API calls",
        is_eager=True,
    ),
) -> None:
    """Configure global CLI options."""
    set_debug(debug)


def _print_error_and_exit(message: str, code: int = 1) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(code)


def _enable_debug_if_requested(debug: bool) -> None:
    if debug:
        set_debug(True)


def _confirm_instance_creation() -> bool:
    console.print()
    ack = console.input(
        "[bold magenta]Create instance now? Type 'yes' to proceed (default no): [/bold magenta]"
    ).strip()
    return ack == "yes"


def _fetch_model_capabilities(ollama_url: str, model: str) -> OllamaModelCapabilities | None:
    """Try to fetch model capabilities from Ollama; return None on any failure."""
    try:
        normalized = OllamaModelManager.create(ollama_url).normalize_model_name(model)
        return OllamaClient(ollama_url).get_model_capabilities(normalized)
    except RuntimeError:
        return None


def _apply_model_limits(capabilities: OllamaModelCapabilities) -> None:
    """Pre-set Holmes/LiteLLM override env vars so Holmes skips the redundant re-fetch."""
    import os

    for env_name, value in (
        ("OVERRIDE_MAX_CONTENT_SIZE", capabilities.context_window),
        ("OVERRIDE_MAX_OUTPUT_TOKEN", capabilities.max_output_tokens),
    ):
        if value and value > 0 and not os.environ.get(env_name):
            os.environ[env_name] = str(value)


def _print_existing_ollama(server: ExistingOllamaServer, capabilities: OllamaModelCapabilities | None = None) -> None:
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


def _print_vastai_result(result: VastAIProvisioningResult, capabilities: OllamaModelCapabilities | None = None) -> None:
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

    if result.ollama_url:
        console.print(f"[dim]Detected Ollama endpoint:[/dim] {result.ollama_url}")

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


def _print_ollama_models(ollama_url: str, models: list[str], running_models: set[str]) -> None:
    table = Table(title="Existing Ollama Models", show_header=True, header_style="bold magenta")
    table.add_column("Ollama URL")
    table.add_column("Model")
    table.add_column("Running")

    if not models:
        table.add_row(ollama_url, "<no models found>", "n/a")
    else:
        for index, model in enumerate(models):
            table.add_row(ollama_url if index == 0 else "", model, "yes" if model in running_models else "no")

    console.print(table)


def _warmup_ollama_model(*, ollama_url: str, model_name: str, timeout_seconds: int) -> None:
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        progress.add_task(f"Loading Ollama model {model_name} and waiting until it is ready...", total=None)
        try:
            warmed_model = warmup_existing_ollama_model(
                ollama_url,
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
    sdk = VastAI(raw=True)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as p:
        task = p.add_task("Provisioning on Vast.ai...", total=None)

        def on_poll(status: str) -> None:
            p.update(task, description=f"Waiting for running status: {status}")

        try:
            return provision_vastai_ollama(
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


@app.command("run-ollama-instance")
def run_ollama_instance(
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show outbound client requests and API calls",
    ),
    vastai: bool = typer.Option(
        False,
        "--vastai",
        help="Enable Vast.ai provisioning flow",
    ),
    template_hash: str = typer.Option(
        DEFAULT_VASTAI_TEMPLATE,
        "--vastai-template",
        help="Vast.ai template hash to use",
    ),
    max_dph: float = typer.Option(3.0, "--vastai-max-dph", help="Maximum dollars per hour"),
    min_dph: float = typer.Option(2.3, "--vastai-min-dph", help="Minimum dollars per hour"),
    reliability: float = typer.Option(0.99, "--vastai-reliability", help="Minimum reliability score"),
    ollama_url: str | None = typer.Option(
        None,
        "--ollama-url",
        help="Use an already running Ollama server URL when --vastai is not enabled",
    ),
) -> None:
    """Provision an Ollama instance on Vast.ai, or use an existing server."""
    _enable_debug_if_requested(debug)
    console.print(Panel.fit("[bold blue]RUNE — Reliability Use-case Numeric Evaluator[/bold blue]"))

    if not vastai:
        try:
            server = use_existing_ollama_server(ollama_url, model_name="<user-selected>")
        except RuntimeError as exc:
            _print_error_and_exit(str(exc))
        _print_existing_ollama(server)
        return

    result = _run_vastai_provisioning(
        template_hash=template_hash,
        min_dph=min_dph,
        max_dph=max_dph,
        reliability=reliability,
    )
    _print_vastai_result(result)


@app.command("vastai-list-models")
def vastai_list_models() -> None:
    """List the configured models used by the Vast.ai provisioning flow."""
    console.print(Panel.fit("[bold blue]RUNE — Configured Vast.ai Models[/bold blue]"))
    _print_vastai_models()


@app.command("ollama-list-models")
def ollama_list_models(
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show outbound client requests and API calls",
    ),
    ollama_url: str = typer.Option(
        ...,
        "--ollama-url",
        help="Ollama server URL to query for available models",
    ),
) -> None:
    """List the models exposed by an existing Ollama server."""
    _enable_debug_if_requested(debug)
    console.print(Panel.fit("[bold blue]RUNE — Existing Ollama Models[/bold blue]"))

    try:
        normalized_url = use_existing_ollama_server(ollama_url, model_name="<n/a>").url
        models = list_existing_ollama_models(normalized_url)
        running_models = set(list_running_ollama_models(normalized_url))
    except RuntimeError as exc:
        _print_error_and_exit(str(exc))

    _print_ollama_models(normalized_url, models, running_models)


@app.command("run-agentic-agent")
def run_agentic_agent(
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show outbound client requests and API calls",
    ),
    question: str = typer.Option(
        "What is unhealthy in this Kubernetes cluster?",
        "--question",
        "-q",
        help="Question to ask the agentic system",
    ),
    model: str = typer.Option(
        "llama3.1:8b",
        "--model",
        "-m",
        help="Model to use for the agent",
    ),
    ollama_url: str | None = typer.Option(
        None,
        "--ollama-url",
        help="Ollama server URL (used for Ollama-backed models)",
    ),
    ollama_warmup: bool = typer.Option(
        True,
        "--ollama-warmup/--no-ollama-warmup",
        help="Load the selected model into the Ollama server before starting Holmes",
    ),
    ollama_warmup_timeout: int = typer.Option(
        90,
        "--ollama-warmup-timeout",
        min=1,
        help="Seconds to wait for the Ollama model to become ready",
    ),
    kubeconfig: Path = typer.Option(
        Path.home() / ".kube" / "config",
        "--kubeconfig",
        help="Path to kubeconfig file",
    ),
) -> None:
    """Run an agentic system (HolmesGPT) against a Kubernetes cluster."""
    _enable_debug_if_requested(debug)
    console.print(Panel.fit("[bold blue]RUNE — Agentic Agent Runner[/bold blue]"))

    if ollama_url and ollama_warmup:
        _warmup_ollama_model(
            ollama_url=ollama_url,
            model_name=model,
            timeout_seconds=ollama_warmup_timeout,
        )

    # Block 10 — Run HolmesGPT agent
    try:
        runner = HolmesRunner(kubeconfig)
        answer = runner.ask(question=question, model=model, ollama_url=ollama_url)
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]Agent error:[/red] {exc}")
        raise typer.Exit(1)

    console.print("\n[bold green]Agent Answer[/bold green]")
    console.print(answer)


@app.command("run-benchmark")
def run_benchmark(
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show outbound client requests and API calls",
    ),
    vastai: bool = typer.Option(
        False,
        "--vastai",
        help="Enable Vast.ai provisioning flow (Phase 1)",
    ),
    template_hash: str = typer.Option(
        DEFAULT_VASTAI_TEMPLATE,
        "--vastai-template",
        help="Vast.ai template hash to use",
    ),
    max_dph: float = typer.Option(3.0, "--vastai-max-dph", help="Maximum dollars per hour"),
    min_dph: float = typer.Option(2.3, "--vastai-min-dph", help="Minimum dollars per hour"),
    reliability: float = typer.Option(0.99, "--vastai-reliability", help="Minimum reliability score"),
    ollama_url: str | None = typer.Option(
        None,
        "--ollama-url",
        help="Existing Ollama server URL when --vastai is not enabled",
    ),
    question: str = typer.Option(
        "What is unhealthy in this Kubernetes cluster?",
        "--question",
        "-q",
        help="Question to ask the agentic system",
    ),
    model: str = typer.Option(
        "llama3.1:8b",
        "--model",
        "-m",
        help="Model to use for the agent when --vastai is not enabled",
    ),
    ollama_warmup: bool = typer.Option(
        True,
        "--ollama-warmup/--no-ollama-warmup",
        help="Load the selected model into the Ollama server before Holmes starts when using --ollama-url",
    ),
    ollama_warmup_timeout: int = typer.Option(
        90,
        "--ollama-warmup-timeout",
        min=1,
        help="Seconds to wait for the Ollama model to become ready",
    ),
    kubeconfig: Path = typer.Option(
        Path.home() / ".kube" / "config",
        "--kubeconfig",
        help="Path to kubeconfig file",
    ),
    vastai_stop_instance: bool = typer.Option(
        True,
        "--vastai-stop-instance/--no-vastai-stop-instance",
        help="Destroy Vast.ai instance + related storage after agent execution and verify cleanup (enabled by default)",
    ),
) -> None:
    """Run full benchmark: provision Ollama instance, then run agentic agent."""
    _enable_debug_if_requested(debug)
    console.print(Panel.fit("[bold blue]RUNE — Full Benchmark Workflow[/bold blue]"))

    selected_model_name = model
    selected_ollama_url = ollama_url
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
        selected_ollama_url = result.ollama_url
        vastai_contract_to_stop = result.contract_id
        vastai_capabilities = _fetch_model_capabilities(result.ollama_url, selected_model_name) if result.ollama_url else None
        if vastai_capabilities:
            _apply_model_limits(vastai_capabilities)
        _print_vastai_result(result, capabilities=vastai_capabilities)
        if not selected_ollama_url:
            if vastai_stop_instance and vastai_contract_to_stop is not None:
                try:
                    teardown = stop_vastai_instance(VastAI(raw=True), vastai_contract_to_stop)
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
            server = use_existing_ollama_server(ollama_url, model_name=selected_model_name)
        except RuntimeError as exc:
            _print_error_and_exit(str(exc))
        console.print("\n[bold cyan]PHASE 1: Using Existing Ollama Server[/bold cyan]")
        capabilities = _fetch_model_capabilities(server.url, selected_model_name)
        if capabilities:
            _apply_model_limits(capabilities)
        _print_existing_ollama(server, capabilities=capabilities)
        selected_ollama_url = server.url

    # ========== PHASE 2: Agentic Agent Execution ==========
    console.print("\n[bold cyan]PHASE 2: Running Agentic Agent[/bold cyan]")

    if selected_ollama_url and ollama_warmup:
        _warmup_ollama_model(
            ollama_url=selected_ollama_url,
            model_name=selected_model_name,
            timeout_seconds=ollama_warmup_timeout,
        )

    # Block 10 — Run agentic agent
    try:
        runner = HolmesRunner(kubeconfig)
        answer = runner.ask(
            question=question,
            model=selected_model_name,
            ollama_url=selected_ollama_url,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]Agent error:[/red] {exc}")
        raise typer.Exit(1)
    finally:
        if vastai and vastai_stop_instance and vastai_contract_to_stop is not None:
            try:
                teardown = stop_vastai_instance(VastAI(raw=True), vastai_contract_to_stop)
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

    console.print("\n[bold green]Agent Answer[/bold green]")
    console.print(answer)


if __name__ == "__main__":
    app()
