# Copilot instructions for RUNE

## Project overview

RUNE (Reliability Use-case Numeric Evaluator) benchmarks AI models and inference frameworks.
It orchestrates 20+ agent drivers across 5 domains (SRE, Research, Legal, Art, Cybersec),
provisions GPU compute via Vast.ai, and exposes both a CLI and an HTTP API.

## Architecture layers

| Layer | Location | Rule |
|---|---|---|
| CLI (Typer + Rich) | `rune/` | Thin shell only — no business logic |
| Orchestration | `rune_bench/workflows.py` | All business flow lives here |
| Agent drivers | `rune_bench/drivers/<name>/driver.py` | One subdirectory per driver; all satisfy `DriverTransport` protocol (`drivers/base.py`) |
| Agent runners | `rune_bench/agents/<domain>/` | Grouped by domain (sre, research, legal, art, cybersec, ops); satisfy `AgentRunner` protocol (`agents/base.py`) |
| LLM backends | `rune_bench/backends/` | Ollama, OpenAI, Bedrock; satisfy `LLMBackend` protocol (`backends/base.py`) |
| Resource providers | `rune_bench/resources/` | Vast.ai and existing-Ollama; satisfy `LLMResourceProvider` protocol (`resources/base.py`) |
| Catalog | `rune_bench/catalog/` | Agent metadata, chain definitions (`chains.yaml`), scope registry (`scopes.csv`) |
| HTTP API | `rune_bench/api_server.py`, `api_backend.py`, `api_contracts.py`, `job_store.py` | stdlib `ThreadingHTTPServer` + SQLite job store |
| Config | `rune_bench/common/config.py` | YAML profiles; precedence: CLI flags > env vars > `./rune.yaml` > `~/.rune/config.yaml` > defaults |

## Key protocols

Four `typing.Protocol` interfaces define the extension points — new integrations implement one of these:

- `DriverTransport` — send action + params to a driver process (stdio or HTTP)
- `AgentRunner` — execute an agent investigation and return results
- `LLMBackend` — talk to an LLM inference endpoint
- `LLMResourceProvider` — provision or locate compute for LLM inference

## Conventions

- Raise `RuntimeError` with user-facing messages at integration boundaries.
- Normalize URLs (add `http://` if missing) in client/workflow helpers; reuse existing normalizers.
- Strip LiteLLM prefixes (`ollama/`, `ollama_chat/`) via `OllamaModelManager.normalize_model_name()` before Ollama API calls.
- Warmup path intentionally unloads other running Ollama models for deterministic memory state.
- For Vast.ai, prefer reusing matching running instances (`find_reusable_running_instance`) before provisioning new ones.
- Secrets (API tokens, `VAST_API_KEY`) must stay in env vars — never in `rune.yaml`.

## Cost safety

Fail-closed cost estimation gates GPU provisioning. If estimation confidence drops below 95%, the operation is rejected. Pre-flight spend warnings surface in the CLI with configurable thresholds (default $5.00).

## Local vs HTTP backend

CLI defaults to local in-process execution; `--backend http` / `RUNE_BACKEND=http` switches to the API server.
When adding a CLI feature, mirror it across `api_contracts.py` / client / server / backend so both modes stay compatible.

## Auth

API auth is tenant-scoped (`X-Tenant-ID` + bearer/API key) unless `RUNE_API_AUTH_DISABLED=1`.
Async job creation supports `Idempotency-Key`; keep this wired through client and server endpoints.

## Developer workflow

```bash
pip install -r requirements.txt        # install deps (or: uv sync)
python -m pytest -q                     # run tests (97% coverage gate)
python -m rune ...                      # CLI entrypoint

# API server
export RUNE_API_TOKENS='default:dev-token'
export RUNE_API_DB_PATH=.rune-api/jobs.db
python -m rune.api
```

## Testing rules

- Tests must stay offline — mock Ollama/Vast.ai/network boundaries; see patterns under `tests/`.
- HTTP-mode changes need client/server flow tests (e.g. `tests/test_cli_http_mode.py`).
- Never introduce automated tests that create real cloud resources (Vast.ai lifecycle is manual).
- Coverage threshold is 97% (`pytest.ini`); avoid adding untested branches.

## CI / release

- Quality gates run in GitHub Actions with path-change detection and parallel jobs.
- CVE scanning (Grype/Trivy, CVSS >= 7.0 blocks release), SAST (Bandit), type checking (mypy), lint (ruff).
- Docker images are multi-arch (amd64 + arm64), built once and pushed to GHCR.
- Formal correctness specs live in `spec/` (TLA+).
