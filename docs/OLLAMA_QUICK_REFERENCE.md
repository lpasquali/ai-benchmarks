# Ollama Module Quick Reference

## Quick Start

### List available models
```python
from rune_bench.ollama import OllamaModelManager

manager = OllamaModelManager.create("http://localhost:11434")
models = manager.list_available_models()
print(models)
```

### Check running models
```python
manager = OllamaModelManager.create("http://localhost:11434")
running = manager.list_running_models()
print(f"Currently running: {running}")
```

### Load a model with automatic cleanup
```python
manager = OllamaModelManager.create("http://localhost:11434")
loaded = manager.warmup_model(
    "mistral:latest",
    timeout_seconds=120,
    unload_others=True,  # Unload conflicting models
)
print(f"Ready: {loaded}")
```

### Direct HTTP client access (advanced)
```python
from rune_bench.ollama import OllamaClient

client = OllamaClient("http://localhost:11434")
models = client.get_available_models()
running = client.get_running_models()
client.load_model("mistral:latest")
client.unload_model("llama2:latest")
```

### Model name normalization
```python
manager = OllamaModelManager.create("http://localhost:11434")

# Strips LiteLLM provider prefixes
plain = manager.normalize_model_name("ollama/mistral:latest")
# Result: "mistral:latest"
```

## CLI Usage

### List available models on a server
```bash
python -m rune ollama-list-models --ollama-url http://localhost:11434
```

### List Vast.ai models
```bash
python -m rune vastai-list-models
```

### Run benchmark with warm-up
```bash
python -m rune run-benchmark \
    --ollama-url http://localhost:11434 \
    --model kavai/qwen3.5-GPT5:9b \
    --ollama-warmup \
    --ollama-warmup-timeout 90
```

## Module Structure

```
rune_bench.ollama
├── OllamaClient          # Low-level HTTP client
│   ├── base_url: str
│   ├── get_available_models() → list[str]
│   ├── get_running_models() → set[str]
│   ├── load_model(model_name, keep_alive)
│   └── unload_model(model_name)
│
└── OllamaModelManager    # High-level operations
    ├── client: OllamaClient
    ├── create(base_url) → OllamaModelManager
    ├── list_available_models() → list[str]
    ├── list_running_models() → list[str]
    ├── warmup_model(model_name, timeout, ...) → str
    └── normalize_model_name(model_name) → str
```

## Common Patterns

### Pattern 1: Simple Model Query
```python
manager = OllamaModelManager.create("http://localhost:11434")
available = manager.list_available_models()

if "mistral:latest" in available:
    print("Model is available")
else:
    print("Model not found")
```

### Pattern 2: Warm-up Before Use
```python
manager = OllamaModelManager.create("http://localhost:11434")

try:
    model = manager.warmup_model(
        "mistral:latest",
        timeout_seconds=60,
        unload_others=True,
    )
    print(f"Model loaded: {model}")
except RuntimeError as e:
    print(f"Failed to load model: {e}")
```

### Pattern 3: Batch Operations
```python
manager = OllamaModelManager.create("http://localhost:11434")

for model in ["mistral:latest", "llama2:latest"]:
    try:
        manager.warmup_model(model, timeout_seconds=30)
        print(f"✓ {model}")
    except RuntimeError as e:
        print(f"✗ {model}: {e}")
```

### Pattern 4: Testing with Mocks
```python
from unittest.mock import Mock
from rune_bench.ollama import OllamaModelManager

# Create a mock client
mock_client = Mock()
mock_client.get_available_models.return_value = ["mistral", "llama2"]

# Test with mock
manager = OllamaModelManager(client=mock_client)
assert manager.list_available_models() == ["mistral", "llama2"]
```

## Error Handling

All methods raise `RuntimeError` with descriptive messages:

```python
from rune_bench.ollama import OllamaModelManager

manager = OllamaModelManager.create("http://invalid:99999")

try:
    models = manager.list_available_models()
except RuntimeError as e:
    # Error messages include:
    # - What operation failed
    # - The server URL
    # - Details from the server (if available)
    print(f"Error: {e}")
```

## Performance Notes

- **Connection timeout**: 30 seconds (configurable via OllamaClient)
- **Model polling**: Default 2-second intervals
- **Model warm-up**: Default 120-second timeout
- **No caching**: Each call queries the server fresh

## Backward Compatibility

The old function-based API in `workflows.py` still works:

```python
from rune_bench.workflows import (
    list_existing_ollama_models,
    warmup_existing_ollama_model,
)

# These are thin wrappers around the new classes
models = list_existing_ollama_models("http://localhost:11434")
warmup_existing_ollama_model("http://localhost:11434", "mistral:latest")
```

## Extending the Module

### Adding a new method to OllamaModelManager

```python
# In rune_bench/ollama/models.py
class OllamaModelManager:
    def get_model_info(self, model_name: str) -> dict:
        """Get detailed info about a specific model."""
        models = self.client.get_available_models()
        for m in models:
            if m == model_name or m.startswith(model_name + ":"):
                # Find full model info from /api/tags
                # This would require extending OllamaClient.get_available_models()
                # to return full model objects instead of just names
                return m
        raise RuntimeError(f"Model {model_name} not found")
```

### Adding a new method to OllamaClient

```python
# In rune_bench/ollama/client.py
class OllamaClient:
    def get_model_details(self, model_name: str) -> dict:
        """Get raw model info from /api/show endpoint."""
        return self._make_request(
            f"/api/show",
            method="POST",
            payload={"model": model_name},
            action=f"show model details for {model_name}",
        )
```

Then use in OllamaModelManager:

```python
def get_model_details(self, model_name: str) -> dict:
    """Get detailed info about a specific model."""
    return self.client.get_model_details(model_name)
```

## Troubleshooting

### "Failed to query available models"
- Check server is running: `curl http://localhost:11434/api/tags`
- Check URL format: should be `http://hostname:port` or `https://hostname:port`
- Check network connectivity: `ping hostname`

### "Timed out waiting for model"
- Model might be very large, increase `timeout_seconds`
- Check server has enough memory to load model
- Check server logs for errors

### "Model not found"
- Verify model name with `manager.list_available_models()`
- Model names are case-sensitive and include tags (e.g., `mistral:latest`)

## Documentation

- **API Details**: [docs/architecture.md](architecture.md)
- **Design Comparison**: [docs/ARCHITECTURE_COMPARISON.md](ARCHITECTURE_COMPARISON.md)
- **Refactoring Details**: [docs/OLLAMA_REFACTORING.md](OLLAMA_REFACTORING.md)
