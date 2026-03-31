# ai-benchmarks

A collection of benchmarks, evaluation scripts, and reproducible test suites for comparing AI models, LLMs, and inference frameworks.

## Setup & Provisioning

`ai-benchmarks` includes an automated provisioning CLI tool built to orchestrate GPU instances on Vast.ai, deploy Ollama with the best-fitting LLM model, and optionally run HolmesGPT for Kubernetes cluster investigation.

### Repository Layout

```text
ai-benchmarks/
├── provision.py          # CLI entry point (Typer) — orchestration only
├── provisioner/          # Business logic module
│   ├── offer.py          # Block 1:   OfferFinder — search best Vast.ai GPU offer
│   ├── models.py         # Block 2+3: ModelSelector — VRAM matching + disk sizing
│   ├── template.py       # Block 4:   TemplateLoader — resolve Vast.ai template config
│   ├── instance.py       # Block 6-9: InstanceManager — create, poll, exec, report
│   └── holmes.py         # Block 10:  HolmesRunner — HolmesGPT SDK integration
├── docs/
│   └── architecture.md   # Full architecture flow diagram and block descriptions
├── experiments/
│   └── provision.py      # Backup of original monolithic script
├── requirements.txt
└── Dockerfile
```

See [docs/architecture.md](docs/architecture.md) for the full flow diagram and per-block documentation.

### Provisioning Flow

`provision.py` is a thin CLI that drives the `provisioner` module through the following steps:

1. Search Vast.ai offers by price and reliability
2. Select the largest Ollama model fitting available VRAM
3. Calculate required disk space (VRAM × 1.15 + 32 GB buffer)
4. Load env and image from a Vast.ai template
5. Prompt for explicit confirmation before spending money
6. Create the instance
7. Poll until running
8. Pull the selected model via Ollama on the remote container
9. Print SSH and HTTPS proxy connection details
10. Optionally run HolmesGPT against a Kubernetes cluster using the provisioned model

### Running the Provisioner

#### Option A: Docker (Recommended)

Build and run completely containerized, without affecting your local Python environment.

```bash
# Build the image
docker build -t ai-benchmark-provisioner .

# Run (mounts your Vast.ai API key)
docker run -it --rm \
  -v ~/.vast_api_key:/root/.vast_api_key \
  ai-benchmark-provisioner provision

# With HolmesGPT (also mount your kubeconfig)
docker run -it --rm \
  -v ~/.vast_api_key:/root/.vast_api_key \
  -v ~/.kube:/root/.kube \
  ai-benchmark-provisioner provision \
  --run-holmes \
  --holmes-question "Why is my cluster degraded?"

# Show all available flags
docker run -it --rm ai-benchmark-provisioner provision --help
```

*The `-it` flag is required — the CLI prompts for confirmation before creating instances.*

#### Option B: Local

```bash
# Install dependencies
pip install -r requirements.txt

# Run
./provision.py provision

# With HolmesGPT
./provision.py provision --run-holmes --holmes-question "What is unhealthy?"

# All flags
./provision.py provision --help
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Security

See [SECURITY.md](SECURITY.md) for the responsible disclosure process.

## License

This project is released under the **GNU General Public License v3.0**.
See [LICENSE](LICENSE) for the full text.
