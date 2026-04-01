# ai-benchmarks

A collection of benchmarks, evaluation scripts, and reproducible test suites for comparing AI models, LLMs, and inference frameworks.

## Setup & Provisioning

`ai-benchmarks` includes **RUNE** — Reliability Use-case Numeric Evaluator.

RUNE orchestrates benchmarkable DevOps/SRE operations, with optional Vast.ai provisioning for Ollama and agentic investigation via HolmesGPT.

## Repository Layout

```text
ai-benchmarks/
├── rune.py                  # Thin Typer CLI (commands, prompts, Rich output)
├── provision.py             # CLI shim forwarding to rune.py
├── rune_bench/
│   ├── __init__.py
│   ├── workflows.py         # Reusable orchestration workflows (no Typer/Rich)
│   ├── vastai/
│   │   ├── offer.py         # OfferFinder
│   │   ├── template.py      # TemplateLoader
│   │   ├── instance.py      # InstanceManager + ConnectionDetails
│   │   └── __init__.py
│   ├── common/
│   │   ├── models.py        # ModelSelector + MODELS
│   │   └── __init__.py
│   ├── agents/
│   │   ├── holmes.py        # HolmesRunner
│   │   └── __init__.py
│   └── ollama/              # NEW: Modular Ollama integration
│       ├── client.py        # OllamaClient (HTTP transport)
│       ├── models.py        # OllamaModelManager (business logic)
│       └── __init__.py
├── docs/
│   ├── architecture.md
│   ├── OLLAMA_REFACTORING.md    # Details on class-based redesign
│   └── ARCHITECTURE_COMPARISON.md
├── experiments/
│   └── provision.py
├── requirements.txt
└── Dockerfile
```

See [docs/architecture.md](docs/architecture.md) for workflow details, including the Ollama module design.
See [docs/API_COMPATIBILITY_PLAN.md](docs/API_COMPATIBILITY_PLAN.md) for the CLI-to-API compatibility roadmap.

## RUNE Commands

`rune.py` provides five commands:

- `run-ollama-instance`: `--vastai` enabled runs the Vast.ai provisioning workflow; without `--vastai`, use `--ollama-url` existing server mode.
- `run-agentic-agent`: run HolmesGPT-only analysis against Kubernetes.
- `run-benchmark`: phase 1 selects an Ollama source (Vast.ai provisioning or existing server), then phase 2 runs HolmesGPT analysis.
- `vastai-list-models`: print the configured model catalog used for Vast.ai auto-selection.
- `ollama-list-models`: list the models currently exposed by an existing Ollama server URL.

## CLI Options Summary

### Backend selection

- `--backend local|http` (or `RUNE_BACKEND` env var)
- `--api-base-url http://host:port` (or `RUNE_API_BASE_URL` env var)

Default mode is `local`, preserving the current in-process CLI behavior.
In `http` mode, model-list commands can query a remote RUNE API.

### Shared agent options

- `--question`, `-q`
- `--model`, `-m` (used by `run-agentic-agent`, and by `run-benchmark` when `--vastai` is disabled)
- `--ollama-warmup`, `--no-ollama-warmup`
- `--ollama-warmup-timeout`
- `--kubeconfig`

### Vast.ai options (enabled only when `--vastai` is set)

- `--vastai`
- `--vastai-template`
- `--vastai-min-dph`
- `--vastai-max-dph`
- `--vastai-reliability`

Use `vastai-list-models` to inspect the configured Vast.ai model shortlist.

### Existing server mode

- `--ollama-url` (required when `--vastai` is not enabled)

Use `ollama-list-models --ollama-url ...` to inspect the exact model names exposed by your existing server.

## Running RUNE

### Option A: Docker

```bash
# Build image
docker build -t ai-benchmark-rune .

# Existing server mode (default)
docker run -it --rm \
  ai-benchmark-rune run-ollama-instance \
  --ollama-url http://host.docker.internal:11434

# Vast.ai mode
docker run -it --rm \
  -v ~/.vast_api_key:/root/.vast_api_key \
  ai-benchmark-rune run-ollama-instance \
  --vastai

# Agent-only mode
docker run -it --rm \
  -v ~/.kube:/root/.kube \
  ai-benchmark-rune run-agentic-agent \
  --question "What is unhealthy?"

# Full benchmark with Vast.ai phase 1
docker run -it --rm \
  -v ~/.vast_api_key:/root/.vast_api_key \
  -v ~/.kube:/root/.kube \
  ai-benchmark-rune run-benchmark \
  --vastai \
  --question "Why is the cluster degraded?"
```

### Option B: Local

```bash
pip install -r requirements.txt

# Existing server mode
./rune.py run-ollama-instance --ollama-url http://localhost:11434

# Vast.ai mode
./rune.py run-ollama-instance --vastai

# Show the configured Vast.ai model shortlist
./rune.py vastai-list-models

# Show models exposed by an existing Ollama server
./rune.py ollama-list-models --ollama-url http://localhost:11434

# Agent-only mode
./rune.py run-agentic-agent --question "What is unhealthy?"

# Full benchmark (existing server phase 1)
./rune.py run-benchmark --ollama-url http://localhost:11434 --model llama3.1:8b

# Full benchmark without pre-loading the Ollama model
./rune.py run-benchmark --ollama-url http://localhost:11434 --model llama3.1:8b --no-ollama-warmup

# Full benchmark (Vast.ai phase 1)
./rune.py run-benchmark --vastai --question "What is unhealthy?"
```

## Testing

### Automated tests (safe/offline)

Automated tests are designed to run anywhere without creating cloud resources.
They mock Ollama and Vast.ai boundaries.

```bash
pip install -r requirements.txt
python -m pytest -q
```

### Manual tests (cost-incurring)

Vast.ai instance creation/destruction paths should be validated manually,
because they can incur real costs.

Example manual run:

```bash
./rune.py run-benchmark --vastai --question "What is unhealthy?"
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

See [SECURITY.md](SECURITY.md).

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
