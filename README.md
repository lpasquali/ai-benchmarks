# rune

A collection of benchmarks, evaluation scripts, and reproducible test suites for comparing AI models, LLMs, and inference frameworks.

## Setup & Provisioning

`rune` includes **RUNE** — Reliability Use-case Numeric Evaluator.

RUNE orchestrates benchmarkable DevOps/SRE operations, with optional Vast.ai provisioning for Ollama and agentic investigation via HolmesGPT.

## Repository Layout

```text
rune/
├── rune/
│   ├── __init__.py          # Thin Typer CLI (commands, prompts, Rich output)
│   ├── __main__.py          # Package entrypoint (python -m rune)
│   └── api.py               # API server entrypoint (python -m rune.api)
├── provision.py             # CLI shim forwarding to rune package
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
Kubernetes operator orchestration now lives in the dedicated `lpasquali/rune-operator` repository.

## RUNE Commands

`python -m rune` provides five commands:

- `run-ollama-instance`: `--vastai` enabled runs the Vast.ai provisioning workflow; without `--vastai`, use `--ollama-url` existing server mode.
- `run-agentic-agent`: run HolmesGPT-only analysis against Kubernetes.
- `run-benchmark`: phase 1 selects an Ollama source (Vast.ai provisioning or existing server), then phase 2 runs HolmesGPT analysis.
- `vastai-list-models`: print the configured model catalog used for Vast.ai auto-selection.
- `ollama-list-models`: list the models currently exposed by an existing Ollama server URL.

## CLI Options Summary

### Backend selection

- `--backend local|http` (or `RUNE_BACKEND` env var)
- `--api-base-url http://host:port` (or `RUNE_API_BASE_URL` env var)
- `--api-token ...` (or `RUNE_API_TOKEN` env var)
- `--api-tenant ...` (or `RUNE_API_TENANT` env var)
- `--idempotency-key ...` on async HTTP job commands

Default mode is `local`, preserving the current in-process CLI behavior.
In `http` mode, the following commands can query/execute against a remote RUNE API:

- `vastai-list-models`
- `ollama-list-models`
- `run-ollama-instance` (job submit/poll)
- `run-agentic-agent` (job submit/poll)
- `run-benchmark` (job submit/poll)

### API server mode

Run the in-repo server with persistent SQLite-backed jobs:

```bash
export RUNE_API_TOKENS='default:dev-token'
export RUNE_API_DB_PATH=.rune-api/jobs.db
python -m rune.api
```

Development-only unauthenticated mode is also available:

```bash
export RUNE_API_AUTH_DISABLED=1
python -m rune.api
```

Server-side controls:

- persistent async jobs in SQLite
- tenant-scoped job lookup via `X-Tenant-ID`
- token auth via `Authorization: Bearer ...` or `X-API-Key`
- idempotent POST job creation via `Idempotency-Key`

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
python -m rune run-ollama-instance --ollama-url http://localhost:11434

# Vast.ai mode
python -m rune run-ollama-instance --vastai

# Show the configured Vast.ai model shortlist
python -m rune vastai-list-models

# Show models exposed by an existing Ollama server
python -m rune ollama-list-models --ollama-url http://localhost:11434

# Agent-only mode
python -m rune run-agentic-agent --question "What is unhealthy?"

# Full benchmark (existing server phase 1)
python -m rune run-benchmark --ollama-url http://localhost:11434 --model llama3.1:8b

# Full benchmark without pre-loading the Ollama model
python -m rune run-benchmark --ollama-url http://localhost:11434 --model llama3.1:8b --no-ollama-warmup

# Full benchmark (Vast.ai phase 1)
python -m rune run-benchmark --vastai --question "What is unhealthy?"
```

## Testing

### Automated tests (safe/offline)

Automated tests are designed to run anywhere without creating cloud resources.
They mock Ollama and Vast.ai boundaries.

```bash
pip install -r requirements.txt
python -m pytest -q
```

Coverage is enforced at a minimum of 97% via pytest configuration.

Coverage table columns mean:

- `Stmts`: executable Python statements in the file
- `Miss`: statements not executed by tests
- `Cover`: percentage covered (`(Stmts - Miss) / Stmts`)
- `Missing`: uncovered line numbers/ranges (for example `144-146` means lines 144, 145, 146)

For a more graphical report, open the generated HTML output at:

- `htmlcov/index.html`

### Manual tests (cost-incurring)

Vast.ai instance creation/destruction paths should be validated manually,
because they can incur real costs.

Example manual run:

```bash
python -m rune run-benchmark --vastai --question "What is unhealthy?"
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

See [SECURITY.md](SECURITY.md).
See [docs/compliance-targets.md](docs/compliance-targets.md) for the repository's explicit security and compliance targets.

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
