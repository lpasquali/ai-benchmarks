"""Live integration tests against a real Ollama instance with TinyLlama.

These tests exercise ``OllamaClient`` and ``OllamaModelManager`` end-to-end
against a real Ollama HTTP server.  They are automatically skipped unless
the ``OLLAMA_TEST_URL`` environment variable is set.

In CI the ``RuneGate/Ollama-Integration/TinyLlama`` job sets both env vars
and runs only these tests::

    pytest -m integration -p no:cov -o addopts='' tests/test_ollama_integration.py

Locally, point at any running Ollama instance::

    OLLAMA_TEST_URL=http://localhost:11434 OLLAMA_TEST_MODEL=tinyllama \\
        pytest -m integration -p no:cov -o addopts='' -v tests/test_ollama_integration.py
"""

import os

import pytest

from rune_bench.backends.ollama import OllamaClient, OllamaModelManager

# ---------------------------------------------------------------------------
# Configuration (overridable via environment)
# ---------------------------------------------------------------------------

_OLLAMA_URL: str = os.getenv("OLLAMA_TEST_URL", "")
_OLLAMA_MODEL: str = os.getenv("OLLAMA_TEST_MODEL", "tinyllama")

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_matches(needle: str, haystack: "list[str] | set[str]") -> bool:
    """Return True when *needle*'s base name (before ``:``) appears in *haystack*."""
    base = needle.split(":")[0]
    return any(m.split(":")[0] == base for m in haystack)


# ---------------------------------------------------------------------------
# Module-scoped fixtures (one shared client per test session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ollama_url() -> str:
    if not _OLLAMA_URL:
        pytest.skip("OLLAMA_TEST_URL not set — skipping live Ollama integration tests")
    return _OLLAMA_URL


@pytest.fixture(scope="module")
def ollama_client(ollama_url: str) -> OllamaClient:
    return OllamaClient(ollama_url)


@pytest.fixture(scope="module")
def ollama_manager(ollama_client: OllamaClient) -> OllamaModelManager:
    return OllamaModelManager(client=ollama_client)


# ---------------------------------------------------------------------------
# OllamaClient — live HTTP tests
# ---------------------------------------------------------------------------


class TestOllamaClientLive:
    """Exercise OllamaClient against a real Ollama instance."""

    def test_get_available_models_includes_test_model(self, ollama_client: OllamaClient) -> None:
        models = ollama_client.get_available_models()
        assert isinstance(models, list), "get_available_models must return a list"
        assert _model_matches(_OLLAMA_MODEL, models), (
            f"Expected '{_OLLAMA_MODEL}' to be pulled into Ollama; available: {models}"
        )

    def test_get_model_capabilities_returns_context_window(self, ollama_client: OllamaClient) -> None:
        caps = ollama_client.get_model_capabilities(_OLLAMA_MODEL)
        assert caps.context_window is not None, "context_window must be populated for tinyllama"
        assert caps.context_window > 0
        assert caps.max_output_tokens is not None
        assert caps.max_output_tokens > 0

    def test_load_model_does_not_raise(self, ollama_client: OllamaClient) -> None:
        # POST /api/generate with keep_alive loads the model into VRAM/RAM.
        # Should complete without error for a pulled model.
        ollama_client.load_model(_OLLAMA_MODEL, keep_alive="2m")

    def test_get_running_models_after_load(self, ollama_client: OllamaClient) -> None:
        ollama_client.load_model(_OLLAMA_MODEL, keep_alive="5m")
        running = ollama_client.get_running_models()
        assert isinstance(running, set)
        assert _model_matches(_OLLAMA_MODEL, running), (
            f"Expected '{_OLLAMA_MODEL}' in running models after load; running: {running}"
        )

    def test_unload_model_removes_from_running(self, ollama_client: OllamaClient) -> None:
        ollama_client.load_model(_OLLAMA_MODEL, keep_alive="5m")
        ollama_client.unload_model(_OLLAMA_MODEL)
        running = ollama_client.get_running_models()
        assert not _model_matches(_OLLAMA_MODEL, running), (
            f"Expected '{_OLLAMA_MODEL}' to be unloaded; still running: {running}"
        )


# ---------------------------------------------------------------------------
# OllamaModelManager — live orchestration tests
# ---------------------------------------------------------------------------


class TestOllamaModelManagerLive:
    """Exercise OllamaModelManager orchestration against a real Ollama instance."""

    def test_list_running_models_returns_sorted_list(self, ollama_manager: OllamaModelManager) -> None:
        result = ollama_manager.list_running_models()
        assert isinstance(result, list)
        assert result == sorted(result), "list_running_models must return a sorted list"

    def test_warmup_model_loads_and_confirms(self, ollama_manager: OllamaModelManager) -> None:
        loaded = ollama_manager.warmup_model(
            _OLLAMA_MODEL,
            timeout_seconds=120,
            poll_interval_seconds=2,
        )
        assert loaded == _OLLAMA_MODEL, (
            f"warmup_model must return the requested model name; got: {loaded!r}"
        )

    def test_normalize_model_name_strips_prefix(self, ollama_manager: OllamaModelManager) -> None:
        assert ollama_manager.normalize_model_name(f"ollama/{_OLLAMA_MODEL}") == _OLLAMA_MODEL
        assert ollama_manager.normalize_model_name(f"ollama_chat/{_OLLAMA_MODEL}") == _OLLAMA_MODEL
        assert ollama_manager.normalize_model_name(f"  {_OLLAMA_MODEL}  ") == _OLLAMA_MODEL
