"""Tests for the LLM backend factory, registry, and OllamaBackend wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rune_bench.backends import get_backend, list_backends, register_backend
from rune_bench.backends import _BACKEND_REGISTRY
from rune_bench.backends.base import LLMBackend, ModelCapabilities


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockBackend:
    """Minimal backend that satisfies the LLMBackend protocol for testing."""

    def __init__(self, base_url: str, **kwargs: object) -> None:
        self._base_url = base_url
        self._kwargs = kwargs

    @property
    def base_url(self) -> str:
        return self._base_url

    def get_model_capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities(model_name=model)

    def list_models(self) -> list[str]:
        return ["mock-model"]

    def list_running_models(self) -> list[str]:
        return []

    def normalize_model_name(self, model_name: str) -> str:
        return model_name

    def warmup(
        self,
        model_name: str,
        *,
        timeout_seconds: int = 120,
        poll_interval_seconds: float = 2.0,
        keep_alive: str = "30m",
    ) -> str:
        return model_name


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure a pristine custom registry for each test."""
    saved = dict(_BACKEND_REGISTRY)
    _BACKEND_REGISTRY.clear()
    yield
    _BACKEND_REGISTRY.clear()
    _BACKEND_REGISTRY.update(saved)


# ---------------------------------------------------------------------------
# Factory: get_backend
# ---------------------------------------------------------------------------

class TestGetBackend:
    """Tests for the get_backend() factory function."""

    def test_builtin_ollama_returns_ollama_backend(self):
        """get_backend('ollama', ...) returns an OllamaBackend instance."""
        with patch("rune_bench.backends.ollama.OllamaClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.base_url = "http://localhost:11434"
            mock_client_cls.return_value = mock_client

            backend = get_backend("ollama", "http://localhost:11434")

        from rune_bench.backends.ollama import OllamaBackend
        assert isinstance(backend, OllamaBackend)

    def test_unknown_backend_raises_value_error(self):
        """get_backend() raises ValueError for unregistered backend types."""
        with pytest.raises(ValueError, match="Unknown backend type 'nonexistent'"):
            get_backend("nonexistent", "http://localhost:1234")

    def test_unknown_backend_error_lists_available(self):
        """The ValueError message includes available backend names."""
        with pytest.raises(ValueError, match="ollama"):
            get_backend("nonexistent", "http://localhost:1234")

    def test_custom_registry_returns_custom_backend(self):
        """register_backend() makes the backend available via get_backend()."""
        register_backend("mock", _MockBackend)
        backend = get_backend("mock", "http://example.com:8080")
        assert isinstance(backend, _MockBackend)
        assert backend.base_url == "http://example.com:8080"

    def test_custom_registry_shadows_builtin(self):
        """Custom registration takes precedence over built-in entries."""
        register_backend("ollama", _MockBackend)
        backend = get_backend("ollama", "http://localhost:11434")
        assert isinstance(backend, _MockBackend)

    def test_kwargs_forwarded_to_custom_backend(self):
        """Extra kwargs are forwarded to the backend constructor."""
        register_backend("mock", _MockBackend)
        backend = get_backend("mock", "http://example.com", extra_param="value")
        assert isinstance(backend, _MockBackend)
        assert backend._kwargs == {"extra_param": "value"}


# ---------------------------------------------------------------------------
# Registry: register_backend / list_backends
# ---------------------------------------------------------------------------

class TestRegistry:
    """Tests for register_backend() and list_backends()."""

    def test_list_backends_default(self):
        """list_backends() includes built-in backends."""
        backends = list_backends()
        assert "ollama" in backends

    def test_list_backends_includes_custom(self):
        """list_backends() includes custom registrations."""
        register_backend("custom-test", _MockBackend)
        backends = list_backends()
        assert "custom-test" in backends
        assert "ollama" in backends

    def test_list_backends_sorted(self):
        """list_backends() returns a sorted list."""
        register_backend("zzz-backend", _MockBackend)
        register_backend("aaa-backend", _MockBackend)
        backends = list_backends()
        assert backends == sorted(backends)

    def test_register_backend_overwrites(self):
        """Registering the same name twice keeps the latest."""
        register_backend("dup", _MockBackend)
        register_backend("dup", _MockBackend)
        assert _BACKEND_REGISTRY["dup"] is _MockBackend


# ---------------------------------------------------------------------------
# OllamaBackend protocol conformance
# ---------------------------------------------------------------------------

class TestOllamaBackendProtocol:
    """Verify OllamaBackend satisfies the LLMBackend protocol."""

    def test_isinstance_check(self):
        """OllamaBackend passes isinstance(obj, LLMBackend) at runtime."""
        with patch("rune_bench.backends.ollama.OllamaClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.base_url = "http://localhost:11434"
            mock_client_cls.return_value = mock_client

            from rune_bench.backends.ollama import OllamaBackend
            backend = OllamaBackend("http://localhost:11434")
            assert isinstance(backend, LLMBackend)

    def test_has_all_protocol_methods(self):
        """OllamaBackend defines every method in the LLMBackend protocol."""
        from rune_bench.backends.ollama import OllamaBackend
        required_methods = [
            "base_url",
            "get_model_capabilities",
            "list_models",
            "list_running_models",
            "normalize_model_name",
            "warmup",
        ]
        for method in required_methods:
            assert hasattr(OllamaBackend, method), f"Missing {method}"


# ---------------------------------------------------------------------------
# OllamaBackend method delegation
# ---------------------------------------------------------------------------

class TestOllamaBackendDelegation:
    """Verify OllamaBackend delegates to OllamaModelManager correctly."""

    @pytest.fixture()
    def backend(self):
        """Create an OllamaBackend with mocked HTTP layer."""
        with patch("rune_bench.backends.ollama.OllamaClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.base_url = "http://localhost:11434"
            mock_client_cls.return_value = mock_client
            from rune_bench.backends.ollama import OllamaBackend
            backend = OllamaBackend("http://localhost:11434")
            backend._mock_client = mock_client
            yield backend

    def test_base_url_property(self, backend):
        assert backend.base_url == "http://localhost:11434"

    def test_get_model_capabilities(self, backend):
        caps = ModelCapabilities(model_name="test-model", context_window=4096)
        backend._mock_client.get_model_capabilities.return_value = caps
        result = backend.get_model_capabilities("test-model")
        assert result == caps
        backend._mock_client.get_model_capabilities.assert_called_once_with("test-model")

    def test_list_models(self, backend):
        backend._mock_client.get_available_models.return_value = ["model-a", "model-b"]
        result = backend.list_models()
        assert result == ["model-a", "model-b"]

    def test_list_running_models(self, backend):
        backend._mock_client.get_running_models.return_value = {"model-a"}
        result = backend.list_running_models()
        assert result == ["model-a"]

    def test_normalize_model_name(self, backend):
        assert backend.normalize_model_name("ollama/tinyllama") == "tinyllama"
        assert backend.normalize_model_name("ollama_chat/tinyllama") == "tinyllama"
        assert backend.normalize_model_name("tinyllama") == "tinyllama"

    def test_warmup_delegates_to_manager(self, backend):
        backend._mock_client.get_running_models.return_value = set()
        backend._mock_client.load_model.return_value = None
        # After load, model appears in running
        backend._mock_client.get_running_models.side_effect = [
            set(),         # _unload_conflicting_models
            {"tinyllama"},  # poll check
        ]
        result = backend.warmup("tinyllama", timeout_seconds=5, poll_interval_seconds=0.1)
        assert result == "tinyllama"


# ---------------------------------------------------------------------------
# MockBackend protocol conformance
# ---------------------------------------------------------------------------

class TestMockBackendProtocol:
    """Verify _MockBackend satisfies the LLMBackend protocol."""

    def test_mock_isinstance_check(self):
        """_MockBackend passes isinstance(obj, LLMBackend) at runtime."""
        backend = _MockBackend("http://example.com")
        assert isinstance(backend, LLMBackend)
