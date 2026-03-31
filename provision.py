#!/usr/bin/env python3
"""provision.py — CLI entry point for the AI Benchmarks Vast.ai provisioner.

All business logic lives in the `provisioner` package.
This file only handles CLI argument parsing, Rich output, and orchestration.
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from vastai import VastAI

from provisioner import HolmesRunner, InstanceManager, ModelSelector, OfferFinder, TemplateLoader

app = typer.Typer(help="AI Benchmarks Vast.ai Provisioning CLI", add_completion=False)
console = Console()

DEFAULT_TEMPLATE = "c166c11f035d3a97871a23bd32ca6aba"


@app.command("provision")
def provision_instance(
    template_hash: str = typer.Option(DEFAULT_TEMPLATE, "--template", "-t", help="Template hash to use"),
    max_dph: float = typer.Option(3.0, help="Maximum dollars per hour"),
    min_dph: float = typer.Option(2.3, help="Minimum dollars per hour"),
    reliability: float = typer.Option(0.99, help="Minimum reliability score"),
    run_holmes: bool = typer.Option(False, "--run-holmes", help="Run HolmesGPT after provisioning"),
    holmes_question: str = typer.Option(
        "What is unhealthy in this Kubernetes cluster?",
        "--holmes-question",
        help="Question to ask HolmesGPT",
    ),
    kubeconfig: Path = typer.Option(
        Path.home() / ".kube" / "config",
        "--kubeconfig",
        help="Path to kubeconfig file",
    ),
) -> None:
    console.print(Panel.fit("[bold blue]Vast.ai Auto-Provisioner[/bold blue]"))

    sdk = VastAI(raw=True)

    # Block 1 — Find the best offer
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as p:
        p.add_task(f"Searching offers (DPH {min_dph}–{max_dph}, reliability>{reliability})...", total=None)
        try:
            offer = OfferFinder(sdk).find_best(min_dph=min_dph, max_dph=max_dph, reliability=reliability)
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)

    console.print(f"[green]Best offer:[/green] id={offer.offer_id}, gpu_total_ram={offer.total_vram_mb} MB")

    # Block 2+3 — Select model and calculate disk size
    try:
        model = ModelSelector().select(offer.total_vram_mb)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Selected model:[/green] {model.name} (~{model.vram_mb} MB VRAM)")
    console.print(f"[green]Required disk:[/green] {model.required_disk_gb} GB")

    # Block 4 — Load template
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as p:
        p.add_task(f"Loading template {template_hash}...", total=None)
        try:
            template = TemplateLoader(sdk).load(template_hash)
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)

    console.print(f"[dim]Template env:[/dim] {template.env}")

    # Block 5 — Explicit user confirmation
    console.print()
    ack = console.input(
        "[bold magenta]Create instance now? Type 'yes' to proceed (default no): [/bold magenta]"
    ).strip()
    if ack != "yes":
        console.print("Aborted.")
        raise typer.Exit(0)

    manager = InstanceManager(sdk)

    # Block 6 — Create instance
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as p:
        p.add_task("Creating instance...", total=None)
        try:
            contract_id = manager.create(offer.offer_id, model, template)
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)

    console.print(f"[green]Provisioned contract:[/green] {contract_id}")

    # Block 7 — Wait until running
    console.print("Waiting for instance to become running...")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as p:
        task = p.add_task("Polling status...", total=None)

        def on_poll(status: str) -> None:
            p.update(task, description=f"Status: {status}")

        try:
            instance_info = manager.wait_until_running(contract_id, on_poll=on_poll)
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)

    # Block 8 — Pull model on instance
    console.print(Panel.fit(f"Pulling model on instance: [bold]{model.name}[/bold]"))
    try:
        manager.pull_model(contract_id, model.name)
    except RuntimeError as exc:
        console.print(f"[orange1]Warning:[/orange1] {exc}")

    # Block 9 — Print connection details
    details = InstanceManager.build_connection_details(contract_id, instance_info)

    table = Table(title="Instance External URLs & Connection Details", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="dim")
    table.add_column("Value")

    table.add_row("Contract", str(details.contract_id))
    table.add_row("Status", details.status)

    if details.ssh_host and details.ssh_port:
        table.add_row("SSH", f"ssh -p {details.ssh_port} root@{details.ssh_host}")

    for svc in details.service_urls:
        table.add_row(f"Direct ({svc['name']})", svc["direct"])
        if svc["proxy"]:
            table.add_row(f"Vast Proxy ({svc['name']})", svc["proxy"])

    console.print(table)
    console.print("\n[dim]Monitor with: python -m vastai show instances[/dim]")

    # Block 10 — Optional HolmesGPT
    if run_holmes:
        try:
            runner = HolmesRunner(kubeconfig)
            answer = runner.ask(question=holmes_question, model=model.name)
        except (FileNotFoundError, RuntimeError) as exc:
            console.print(f"[red]HolmesGPT error:[/red] {exc}")
            raise typer.Exit(1)

        console.print("\n[bold green]HolmesGPT Answer[/bold green]")
        console.print(answer)


if __name__ == "__main__":
    app()
