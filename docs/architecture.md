# Architecture: provision.py

## Overview

`provision.py` is a Python CLI tool that automates the full lifecycle of:
1. Finding the best available GPU instance on Vast.ai
2. Selecting the largest LLM model that fits its VRAM
3. Provisioning the instance with the correct disk size and template
4. Pulling the selected model via Ollama on the remote instance
5. Optionally running HolmesGPT against a Kubernetes cluster using the provisioned model

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        provision.py                             │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  1. Search Offers   │  vast_sdk.search_offers(query, order)
│                     │  filters: reliability, dph min/max
│                     │  sort:    gpu_total_ram, dlperf, flops
└────────┬────────────┘
         │ best_offer → offer_id, total_vram
         ▼
┌─────────────────────┐
│  2. Select Model    │  Iterate MODELS list (sorted largest→smallest)
│                     │  Pick first model where vram_mb <= total_vram
└────────┬────────────┘
         │ selected_model, required_vram_mb
         ▼
┌─────────────────────┐
│  3. Disk Sizing     │  required_disk_gb = ceil(vram_mb / 1024 * 1.15) + 32
│                     │  +15% model buffer + 32 GB OS/container buffer
└────────┬────────────┘
         │ required_disk_gb
         ▼
┌─────────────────────┐
│  4. Load Template   │  vast_sdk.show_templates()
│                     │  Match by: id, hash_id, hash, template_hash
│                     │  Extract: env flags, docker image
└────────┬────────────┘
         │ template_env, template_image, final_env
         ▼
┌─────────────────────┐
│  5. User            │  Print summary of: offer, model, disk, template
│     Confirmation    │  Require exact input "yes" to proceed
│                     │  Any other input → abort immediately
└────────┬────────────┘
         │ confirmed
         ▼
┌─────────────────────┐
│  6. Create Instance │  vast_sdk.create_instance(id, disk, env, image)
│                     │  Returns: new_contract (instance id)
└────────┬────────────┘
         │ new_contract
         ▼
┌─────────────────────┐
│  7. Wait for        │  vast_sdk.show_instances() polling loop
│     Running State   │  Poll every 10s, max 36 attempts (~6 min)
│                     │  Check actual_status / state == "running"
└────────┬────────────┘
         │ instance_info (ssh_host, ssh_port, machine_id, ports)
         ▼
┌─────────────────────┐
│  8. Pull Model      │  vast_sdk.execute(id, command)
│     via Ollama      │  command: ollama pull '<model>'
│                     │  Runs inside the provisioned container
└────────┬────────────┘
         │ model ready
         ▼
┌─────────────────────┐
│  9. Print           │  Rich table output:
│     Connection      │  - SSH command
│     Details         │  - Direct HTTP URLs (host:port per service)
│                     │  - Vast HTTPS proxy URLs (server-<id>.vast.ai)
└────────┬────────────┘
         │ (optional --run-holmes flag)
         ▼
┌─────────────────────┐
│ 10. HolmesGPT       │  holmesgpt.ask() or holmesgpt.HolmesGPT().ask()
│     (optional)      │  Uses: selected_model, kubeconfig, question
│                     │  Prints answer to terminal
└─────────────────────┘
```

---

## Logical Blocks Detail

### Block 1 — Search Offers
**Location:** `provision_instance()` step 1  
**SDK call:** `vast_sdk.search_offers(query, order, disable_bundling, raw)`  
**Purpose:** Query the Vast.ai marketplace for GPU instances matching the configured reliability and price range, sorted by descending VRAM, then DL performance, then total FLOPS. The first result is the best available machine.

**Key parameters (CLI options):**

| Option | Default | Description |
|--------|---------|-------------|
| `--min-dph` | `2.3` | Minimum dollars per hour |
| `--max-dph` | `3.0` | Maximum dollars per hour |
| `--reliability` | `0.99` | Minimum reliability score |

---

### Block 2 — Model Selection
**Location:** `provision_instance()` step 2  
**Purpose:** Match the largest model from the `MODELS` list that fits within the instance's total VRAM. Models are defined in descending VRAM order so the first match is always the most capable one that fits.

**Model list (descending VRAM):**

| Model | VRAM Required |
|-------|--------------|
| `llama3.1:405b` | ~254 GB |
| `mixtral:8x22b` | ~93 GB |
| `command-r-plus:104b` | ~73 GB |
| `qwen2.5-coder:72b` | ~54 GB |
| `llama3.1:70b` | ~49 GB |
| `mixtral:8x7b` | ~31 GB |
| `command-r:35b` | ~27 GB |
| `llama3.1:8b` | ~8 GB |

---

### Block 3 — Disk Sizing
**Location:** `provision_instance()` step 3  
**Formula:**
```
required_disk_gb = ceil((vram_mb / 1024) * 1.15) + 32
```
- `/ 1024`: Convert MB → GB
- `* 1.15`: Add 15% overhead for model weight files on disk
- `+ 32`: Fixed buffer for OS, container layers, Ollama daemon

---

### Block 4 — Template Loading
**Location:** `_pick_template()` + `provision_instance()` step 4  
**SDK call:** `vast_sdk.show_templates(raw=True)`  
**Purpose:** Fetch all saved Vast.ai templates and match by any known hash field (`id`, `hash_id`, `hash`, `template_hash`). Extracts the pre-configured `env` flags and Docker image from the template so the instance is created with the exact same configuration as defined in the Vast.ai UI.

**Default template hash:** `c166c11f035d3a97871a23bd32ca6aba`  
Configurable via `--template` / `-t` CLI option.

---

### Block 5 — User Confirmation
**Location:** `provision_instance()` step 5  
**Purpose:** Display a summary of the planned action and require explicit `yes` input before spending money. Hitting Enter or typing anything else aborts immediately with no side effects.

---

### Block 6 — Instance Creation
**Location:** `provision_instance()` step 6  
**SDK call:** `vast_sdk.create_instance(id, disk, env, image, raw)`  
**Purpose:** Provision the selected offer as a running container using the template image and env flags. The `/workspace` volume mount is appended to the template env to ensure model weights persist in the correct path.

---

### Block 7 — Polling for Running State
**Location:** `provision_instance()` step 7  
**SDK call:** `vast_sdk.show_instances(raw=True)`  
**Purpose:** Poll the instance state every 10 seconds until it reaches `running` or a 6-minute timeout is hit. Uses a Rich spinner to provide live status feedback.

---

### Block 8 — Model Pull via Ollama
**Location:** `provision_instance()` step 8  
**SDK call:** `vast_sdk.execute(id, command, raw)`  
**Purpose:** Run `ollama pull <model>` inside the provisioned container via the Vast.ai execute API. Waits for completion before proceeding to print connection details.

---

### Block 9 — Connection Details Output
**Location:** `provision_instance()` step 9  
**Purpose:** Print a Rich-formatted table with all external access points for the running instance:
- **SSH:** Direct shell access
- **Direct HTTP:** `http://<host_ip>:<host_port>` per exposed service port
- **Vast HTTPS Proxy:** `https://server-<machine_id>.vast.ai:<port>` per service

---

### Block 10 — HolmesGPT (Optional)
**Location:** `_run_holmesgpt_after_provision()` + `provision_instance()` step 10  
**Triggered by:** `--run-holmes` flag  
**SDK calls:** `holmesgpt.ask(...)` or `holmesgpt.HolmesGPT(...).ask(...)`  
**Purpose:** After the model is running on the provisioned instance, use HolmesGPT to investigate the Kubernetes cluster pointed to by `--kubeconfig`, using the provisioned model as the LLM backend.

**CLI options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--run-holmes` | `False` | Enable HolmesGPT step |
| `--holmes-question` | `"What is unhealthy in this Kubernetes cluster?"` | Question to ask |
| `--kubeconfig` | `~/.kube/config` | Path to kubeconfig |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `vastai` | Vast.ai Python SDK (offers, templates, instances, execute) |
| `holmesgpt` | HolmesGPT SDK (optional Kubernetes investigation) |
| `typer` | CLI argument parsing and help generation |
| `rich` | Terminal UI (tables, spinners, panels, styled output) |

---

## Configuration Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_TEMPLATE` | `c166c11f035d3a97871a23bd32ca6aba` | Default Vast.ai template hash |
| `MODELS` | See Block 2 table | Ordered list of models with VRAM requirements |