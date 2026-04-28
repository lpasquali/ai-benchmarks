# RUNE Test Suite

This directory contains the automated test suite for the RUNE platform.
Tests are organized by functional area to maintain readability and modularity.

## Directory Structure

- **`api/`**: Tests for the HTTP API server, request/response contracts, and security middleware.
- **`backends/`**: Unit and integration tests for LLM backends (Ollama, Bedrock, etc.).
- **`core/`**: Core logic tests including agent registries, configuration, telemetry, and CLI commands.
- **`drivers/`**: Tests for agent driver clients and their subprocess entry points.
- **`integration/`**: Heavyweight integration tests requiring external services (e.g., PostgreSQL).
- **`uat/`**: User Acceptance Tests (UAT) that typically require live credentials.

## Running Tests

Tests are run using `pytest`:

```bash
# Run all unit and core tests
python -m pytest tests/

# Run a specific category
python -m pytest tests/drivers/

# Run with coverage report
python -m pytest --cov=rune_bench --cov=rune tests/
```

## Contribution Guidelines

1. **New Tests**: Place new tests in the corresponding subdirectory.
2. **Naming**: Follow the `test_*.py` pattern for discovery.
3. **Mocking**: Use `unittest.mock` or `pytest-mock` to isolate unit tests from external APIs.
