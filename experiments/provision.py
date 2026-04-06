#!/usr/bin/env python3

import math
import time
from pathlib import Path

import holmesgpt  # type: ignore
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from vastai import VastAI  # type: ignore[import-untyped, import-not-found]  # Reason: vastai SDK does not provide type hints

app = typer.Typer(help="AI Benchmarks Vast.ai Provisioning CLI", add_completion=False)
console = Console()
vast_sdk = VastAI(raw=True)

DEFAULT_VASTAI_TEMPLATE = "c166c11f035d3a97871a23bd32ca6aba"

MODELS = [
    {"name": "llama3.1:405b", "vram_mb": 260000},
    {"name": "mixtral:8x22b", "vram_mb": 95000},
    {"name": "command-r-plus:104b", "vram_mb": 75000},
    {"name": "qwen2.5-coder:72b", "vram_mb": 55000},
    {"name": "llama3.1:70b", "vram_mb": 50000},
    {"name": "mixtral:8x7b", "vram_mb": 32000},
    {"name": "command-r:35b", "vram_mb": 28000},
    {"name": "llama3.1:8b", "vram_mb": 8000},
]


def _pick_template(templates: list[dict], template_hash: str) -> dict | None:
    for t in templates:
        if str(t.get("id")) == template_hash:
            return t
        if str(t.get("hash_id")) == template_hash:
            return t
        if str(t.get("hash")) == template_hash:
            return t
        if str(t.get("template_hash")) == template_hash:
            return t
    return None


def _run_holmesgpt_after_provision(selected_model: str, question: str, kubeconfig: Path) -> None:
    if not kubeconfig.exists():
        raise typer.BadParameter(f"kubeconfig not found: {kubeconfig}")

    console.print(
        f"[cyan]Running HolmesGPT SDK[/cyan] "
        f"(model={selected_model}, kubeconfig={kubeconfig})"
    )

    answer = None

    if hasattr(holmesgpt, "ask"):
        try:
            answer = holmesgpt.ask(
                question=question,
                model=selected_model,
                kubeconfig=str(kubeconfig),
            )
        except TypeError:
            pass

    if answer is None and hasattr(holmesgpt, "HolmesGPT"):
        try:
            client = holmesgpt.HolmesGPT(model=selected_model, kubeconfig=str(kubeconfig))
            answer = client.ask(question=question)
        except TypeError:
            client = holmesgpt.HolmesGPT(kubeconfig=str(kubeconfig))
            answer = client.ask(question=question, model=selected_model)

    if answer is None:
        raise RuntimeError(
            "Unsupported HolmesGPT SDK API shape. "
            "Expected holmesgpt.ask(...) or holmesgpt.HolmesGPT(...).ask(...)."
        )

    console.print("\n[bold green]HolmesGPT Answer[/bold green]")
    console.print(str(answer))


@app.command("provision")
def provision_instance(
    template_hash: str = typer.Option(
        DEFAULT_VASTAI_TEMPLATE,
        "--template",
        "-t",
        help="Vast.ai template hash to use",
    ),
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
):
    console.print(Panel.fit("[bold blue]Vast.ai Auto-Provisioner[/bold blue]"))

    # 1) Search offer
    query = f"reliability > {reliability} verified=True dph>={min_dph} dph<={max_dph}"
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(f"Searching offers: {query}", total=None)
        try:
            offers = vast_sdk.search_offers(
                query=query,
                order="gpu_total_ram-,dlperf-,total_flops-",
                disable_bundling=True,
                raw=True,
            )
        except Exception as e:
            console.print(f"[red]Error searching offers:[/red] {e}")
            raise typer.Exit(1)

    if not isinstance(offers, list) or not offers:
        console.print("[red]No matching offers found.[/red]")
        raise typer.Exit(1)

    best_offer = offers[0]
    offer_id = best_offer.get("id")
    total_vram = int(best_offer.get("gpu_total_ram", 0))
    if not offer_id or total_vram <= 0:
        console.print("[red]Offer response missing id/gpu_total_ram.[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Best offer:[/green] id={offer_id}, gpu_total_ram={total_vram} MB")

    # 2) Select model by VRAM
    selected_model = None
    required_vram_mb = 0
    for m in MODELS:
        if total_vram >= m["vram_mb"]:
            selected_model = m["name"]
            required_vram_mb = m["vram_mb"]
            break

    if not selected_model:
        console.print("[red]No configured model fits available VRAM.[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Selected model:[/green] {selected_model} (~{required_vram_mb} MB VRAM)")

    # 3) Disk sizing: model-size +15% + 32GB buffer
    required_disk_gb = math.ceil((required_vram_mb / 1024) * 1.15) + 32
    console.print(f"[green]Disk size:[/green] {required_disk_gb} GB")

    # 4) Load template and extract env/image
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task(f"Loading template {template_hash}", total=None)
        try:
            templates = vast_sdk.show_templates(raw=True)
        except Exception:
            templates = []

    if not isinstance(templates, list):
        templates = []

    template = _pick_template(templates, template_hash)
    if not template:
        console.print(f"[red]Template not found:[/red] {template_hash}")
        raise typer.Exit(1)

    template_env = str(template.get("env", "")).strip()
    template_image = template.get("image") or template.get("docker_image")

    # Keep existing env and ensure /workspace mount flag exists
    final_env = f"{template_env} -v /workspace".strip()
    console.print(f"[dim]Template env:[/dim] {final_env}")

    # 5) Explicit confirmation: only exact "yes" proceeds
    console.print()
    ack = console.input("[bold magenta]Create instance now? Type 'yes' to proceed (default no): [/bold magenta]").strip()
    if ack != "yes":
        console.print("Aborted.")
        raise typer.Exit(0)

    # 6) Create instance
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        progress.add_task("Creating instance...", total=None)
        try:
            kwargs = {
                "id": str(offer_id),
                "disk": float(required_disk_gb),
                "env": final_env,
                "raw": True,
            }
            if template_image:
                kwargs["image"] = template_image
            create_res = vast_sdk.create_instance(**kwargs)
        except Exception as e:
            console.print(f"[red]Error creating instance:[/red] {e}")
            raise typer.Exit(1)

    new_contract = None
    if isinstance(create_res, dict):
        new_contract = create_res.get("new_contract") or create_res.get("id")

    if not new_contract:
        console.print("[red]Failed to parse new contract id from create response.[/red]")
        console.print(str(create_res))
        raise typer.Exit(1)

    console.print(f"[green]Provisioned contract:[/green] {new_contract}")

    # 7) Wait until running
    console.print("Waiting for instance to become running...")
    instance_info = None
    status = "unknown"

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Polling status...", total=None)
        for _ in range(36):  # ~6 minutes
            try:
                instances = vast_sdk.show_instances(raw=True)
            except Exception:
                instances = []

            if isinstance(instances, list):
                instance_info = next((i for i in instances if str(i.get("id")) == str(new_contract)), None)
                if instance_info:
                    status = instance_info.get("actual_status", instance_info.get("state", "unknown"))
                    progress.update(task, description=f"Status: {status}")
                    if str(status).lower() == "running":
                        break
            time.sleep(10)

    if not instance_info or str(status).lower() != "running":
        console.print("[red]Instance did not reach running state in time.[/red]")
        raise typer.Exit(1)

    # 8) Pull selected model on instance and wait completion
    console.print(Panel.fit(f"Pulling model on instance: [bold]{selected_model}[/bold]"))

    command = f"ollama pull '{selected_model}' && echo 'Model pulled successfully'"
    try:
        # Try common SDK argument spellings
        try:
            vast_sdk.execute(id=str(new_contract), command=command, raw=True)
        except TypeError:
            vast_sdk.execute(id=str(new_contract), COMMAND=command, raw=True)
    except Exception as e:
        console.print(f"[orange1]Warning:[/orange1] model pull execution failed/timed out: {e}")

    # 9) Print connection details + external URLs
    ssh_host = instance_info.get("ssh_host")
    ssh_port = instance_info.get("ssh_port")
    machine_id = instance_info.get("machine_id")
    ports = instance_info.get("ports", {}) or {}

    table = Table(title="Instance External URLs & Connection Details", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="dim")
    table.add_column("Value")

    table.add_row("Contract", str(new_contract))
    table.add_row("Status", str(status))

    if ssh_host and ssh_port:
        table.add_row("SSH", f"ssh -p {ssh_port} root@{ssh_host}")

    if isinstance(ports, dict):
        for svc, mappings in ports.items():
            if not mappings or not isinstance(mappings, list):
                continue
            first = mappings[0]
            host_ip = first.get("HostIp")
            host_port = first.get("HostPort")
            if host_ip and host_port:
                table.add_row(f"Direct ({svc})", f"http://{host_ip}:{host_port}")
                if machine_id:
                    table.add_row(f"Vast Proxy ({svc})", f"https://server-{machine_id}.vast.ai:{host_port}")

    console.print(table)
    console.print("\n[dim]Monitor with: python -m vastai show instances[/dim]")

    # 10) Optional HolmesGPT SDK run
    if run_holmes:
        _run_holmesgpt_after_provision(
            selected_model=selected_model,
            question=holmes_question,
            kubeconfig=kubeconfig,
        )


if __name__ == "__main__":
    app()