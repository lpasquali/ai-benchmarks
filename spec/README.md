# TLA+ Specifications

This directory contains formal TLA+ specifications for two critical correctness properties of the RUNE platform.

## Files

| File | What it specifies |
|------|-------------------|
| `JobStateMachine.tla` | Job lifecycle state machine + idempotency keys |
| `FailClosedEstimation.tla` | Fail-Closed safety invariant for cost-estimation |

---

## JobStateMachine.tla

### What it models

- Every job follows a strict linear path: `queued → running → succeeded | failed`.
- No job may jump directly from `queued` to a terminal state.
- No job may regress from a terminal state (`succeeded` / `failed`) back to an earlier state.
- `(tenant, idempotency_key)` pairs are unique — submitting the same key twice is a no-op.

### Key invariants

| Invariant | Meaning |
|-----------|---------|
| `TypeInvariant` | All job records are well-typed |
| `NoRegression` | Terminal jobs stay terminal |
| `IdempotencyInvariant` | No duplicate idempotency-key pairs |

---

## FailClosedEstimation.tla

### What it models

API reachability confidence is represented as an integer in `[0, 100]`
(percentage × 100, to avoid floating-point).  The fail-closed threshold is **95** (= 0.95).

- Whenever confidence drops below the threshold it **atomically** transitions the system to `halted`.
- `ExecuteAction` (cost-estimation / provisioning) is only enabled when `systemState = "active"`.
- Recovery to `active` requires an explicit operator action **after** confidence has recovered to ≥ threshold.

### Key invariants

| Invariant | Meaning |
|-----------|---------|
| `TypeInvariant` | `confidence` and `systemState` are well-typed |
| `FailClosedInvariant` | `confidence < 95 ⇒ systemState = "halted"` |
| `ActiveOnlyWhenSafe` | `systemState = "active" ⇒ confidence ≥ 95` |

---

## Checking the specs with TLC

### Prerequisites

Install the [TLA+ Toolbox](https://lamport.azurewebsites.net/tla/toolbox.html)
or use the standalone TLC jar:

```bash
# Download TLC (adjust version as needed)
curl -L https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar \
     -o tla2tools.jar
```

### Run TLC on JobStateMachine

Create a model file `JobStateMachine.cfg`:

```
SPECIFICATION Spec
CONSTANTS MaxJobs = 3
INVARIANTS TypeInvariant IdempotencyInvariant
```

Then run:

```bash
java -jar tla2tools.jar -config spec/JobStateMachine.cfg spec/JobStateMachine.tla
```

### Run TLC on FailClosedEstimation

Create a model file `FailClosedEstimation.cfg`:

```
SPECIFICATION Spec
CONSTANTS
    MinConfidence = 0
    MaxConfidence = 100
    Threshold = 95
INVARIANTS TypeInvariant FailClosedInvariant ActiveOnlyWhenSafe
```

Then run:

```bash
java -jar tla2tools.jar -config spec/FailClosedEstimation.cfg spec/FailClosedEstimation.tla
```

### Using the TLA+ VS Code Extension

Install the [TLA+ extension](https://marketplace.visualstudio.com/items?itemName=alygin.vscode-tlaplus)
for VS Code. Open any `.tla` file and use the **"TLA+: Check model with TLC"** command.

---

## CI integration (optional)

A Docker-based syntax check can be added to `quality-gates.yml` using the
[tla-web](https://github.com/will62794/tla-web) image:

```yaml
- name: Validate TLA+ syntax
  run: |
    docker pull ghcr.io/will62794/tla-web:latest
    docker run --rm -v ${{ github.workspace }}/spec:/spec \
      ghcr.io/will62794/tla-web:latest \
      tla2tools.jar -tool SANY /spec/JobStateMachine.tla
    docker run --rm -v ${{ github.workspace }}/spec:/spec \
      ghcr.io/will62794/tla-web:latest \
      tla2tools.jar -tool SANY /spec/FailClosedEstimation.tla
```
