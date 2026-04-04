# Copilot instructions for rune (RUNE)

## Architecture and boundaries
- Treat [rune/__init__.py](../rune/__init__.py) as a thin CLI layer only (Typer args, Rich output, progress UI).
- Put orchestration/business flow in [rune_bench/workflows.py](../rune_bench/workflows.py), not in CLI command handlers.
- Keep provider-specific logic in domain modules:
  - Ollama transport/logic: [rune_bench/backends/ollama.py](../rune_bench/backends/ollama.py)
  - Vast.ai offer/template/instance lifecycle: [rune_bench/resources/vastai/offer.py](../rune_bench/resources/vastai/offer.py), [rune_bench/resources/vastai/template.py](../rune_bench/resources/vastai/template.py), [rune_bench/resources/vastai/instance.py](../rune_bench/resources/vastai/instance.py)
  - Agent execution: [rune_bench/agents/sre/holmes.py](../rune_bench/agents/sre/holmes.py)

## Local vs HTTP backend pattern
- CLI defaults to local in-process execution; HTTP mode is opt-in via `--backend http` / `RUNE_BACKEND`.
- Preserve request shapes via dataclass contracts in [rune_bench/api_contracts.py](../rune_bench/api_contracts.py).
- HTTP server is stdlib `ThreadingHTTPServer` + SQLite job store:
  - API server: [rune_bench/api_server.py](../rune_bench/api_server.py)
  - Job persistence/idempotency: [rune_bench/job_store.py](../rune_bench/job_store.py)
  - Local backend used by server jobs: [rune_bench/api_backend.py](../rune_bench/api_backend.py)
- When adding a CLI feature, mirror it across contracts/client/server/backend so local and HTTP behavior stay compatible.

## Project-specific conventions
- Raise `RuntimeError` with clear user-facing messages at integration boundaries (Ollama/Vast.ai/API).
- URL inputs are normalized (add `http://` if missing) in client/workflow helpers; reuse existing normalizers.
- Model names may include LiteLLM prefixes (`ollama/`, `ollama_chat/`); normalize before Ollama API calls using `OllamaModelManager.normalize_model_name()`.
- Warmup path intentionally unloads other running Ollama models for deterministic memory state before polling readiness.
- For Vast.ai, prefer reusing matching running instances before provisioning a new one (`find_reusable_running_instance`).

## External integration points
- Vast.ai SDK (`VastAI(raw=True)`) is the only cloud control plane used by workflows.
- HolmesGPT runner supports multiple SDK shapes and CLI fallback in [rune_bench/agents/sre/holmes.py](../rune_bench/agents/sre/holmes.py).
- API auth is tenant-scoped (`X-Tenant-ID` + bearer/API key) unless `RUNE_API_AUTH_DISABLED=1`.
- Async job creation supports `Idempotency-Key`; keep this wired through client and server endpoints.

## Developer workflows
- Install deps: `pip install -r requirements.txt`
- Run tests (coverage gate is strict): `python -m pytest -q`
- Coverage threshold is 97% (`pytest.ini`); avoid adding untested branches.
- Start API server locally:
  - `export RUNE_API_TOKENS='default:dev-token'`
  - `export RUNE_API_DB_PATH=.rune-api/jobs.db`
  - `python -m rune.api`
- CLI entrypoint is `python -m rune ...`; package entrypoint file is [rune/__main__.py](../rune/__main__.py).

## Testing expectations
- Keep tests offline and mock boundaries (Ollama/Vast.ai/network); see patterns under [tests/](../tests/).
- For HTTP mode changes, update client/server flow tests (example: [tests/test_cli_http_mode.py](../tests/test_cli_http_mode.py)).
- Real Vast.ai lifecycle validation is manual/cost-incurring; do not introduce automated tests that create cloud resources.
