# Rune Operator (Go)

Kubernetes operator to orchestrate short-lifecycle `rune_bench` workflows via CRDs and expose observability for cloud-native stacks.

## Features

- CRD: `RuneBenchmark` (`bench.rune.ai/v1alpha1`)
- Schedules and triggers benchmark jobs against RUNE API
- Status history, conditions, failure counters
- Prometheus metrics on `/metrics`
- OpenTelemetry trace emission (OTLP) when configured
- Panic recovery with stack traces logged for ingestion by observability backends

## CRD

CRD manifest:
- `config/crd/bases/bench.rune.ai_runebenchmarks.yaml`

Sample:
- `config/samples/bench_v1alpha1_runebenchmark.yaml`

## Metrics

- `rune_operator_reconcile_total{result=...}`
- `rune_operator_run_duration_millis`
- `rune_operator_active_schedules`

## Build

```bash
cd rune-operator
go mod tidy
go build ./...
```

## Build operator container image (separate from RUNE image)

```bash
cd rune-operator
docker build -t rune:rune-operator .
```

## Run locally

```bash
cd rune-operator
go run . --leader-elect=false
```

## Deploy outline

1. Apply CRD.
2. Create RBAC + ServiceAccount.
3. Run deployment for this operator.
4. Create `RuneBenchmark` resources.

The operator calls RUNE API endpoint `/v1/jobs` using `spec.apiBaseUrl`.
