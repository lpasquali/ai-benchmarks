# RUNE Client/Server Compatibility Plan

Goal: keep the current CLI UX unchanged while enabling a cloud-native API backend suitable for Kubernetes Operators and CRDs.

## Command Compatibility Map

Current CLI commands in [rune/__init__.py](../rune/__init__.py):

- `run-ollama-instance`
- `run-agentic-agent`
- `run-benchmark`
- `vastai-list-models`
- `ollama-list-models`

Target API operations:

- `POST /v1/jobs/ollama-instance`
- `POST /v1/jobs/agentic-agent`
- `POST /v1/jobs/benchmark`
- `GET /v1/catalog/vastai-models`
- `GET /v1/ollama/models`

## Delivery Phases

### Phase 1 — Contracts + CLI adapter seam (first implementation)

Status: completed.

1. Add stable request/response contracts for each CLI operation.
2. Add a local backend adapter that executes the existing in-process workflow.
3. Keep CLI behavior identical; transport stays local by default.
4. Add tests for contracts and mapping from CLI inputs.

### Phase 2 — HTTP backend (opt-in)

Status: completed.

1. Add API transport client implementation using the same contracts.
2. Add CLI option/environment switch to use local or remote backend.
3. Preserve Rich output and progress style in CLI by streaming job events.

Implemented so far:

- HTTP mode for `vastai-list-models` and `ollama-list-models`
- HTTP mode for `run-ollama-instance` via async job submit/poll flow
- HTTP mode for `run-agentic-agent` via async job submit/poll flow
- HTTP mode for `run-benchmark` via async job submit/poll flow
- In-repo HTTP server implementation with persistent SQLite job storage
- Tenant-scoped auth and idempotent job submission controls

### Phase 3 — Kubernetes readiness

1. Add status/event endpoints for operator reconciliation loops.
2. Add finer-grained authz guardrails for cost-incurring operations.
3. Add explicit reconciliation/resource APIs beyond the current job model.

## Non-goals (for now)

- Replacing current CLI UX.
- Enabling automated tests that create real Vast.ai instances.
- Forcing users to run a server in local mode.
