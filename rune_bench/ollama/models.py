"""High-level model management for Ollama.

Provides operations for loading, unloading, and discovering models with smart defaults.
"""

import time
from dataclasses import dataclass

from .client import OllamaClient


@dataclass
class OllamaModelManager:
    """High-level manager for Ollama models on a specific server.
    
    Handles model discovery, warm-up with automatic conflict resolution,
    and cleanup operations.
    
    Attributes:
        client: OllamaClient instance for API communication
    """

    client: OllamaClient

    @classmethod
    def create(cls, base_url: str) -> "OllamaModelManager":
        """Create a new manager for an Ollama server at the given URL.
        
        Args:
            base_url: Ollama server URL (e.g., http://localhost:11434)
            
        Returns:
            OllamaModelManager instance
        """
        return cls(client=OllamaClient(base_url))

    def list_available_models(self) -> list[str]:
        """List all models available on the server.
        
        Returns:
            Sorted list of model names
        """
        return self.client.get_available_models()

    def list_running_models(self) -> list[str]:
        """List all models currently loaded in memory.
        
        Returns:
            Sorted list of running model names
        """
        return sorted(self.client.get_running_models())

    def warmup_model(
        self,
        model_name: str,
        *,
        timeout_seconds: int = 120,
        poll_interval_seconds: float = 2.0,
        keep_alive: str = "30m",
        unload_others: bool = True,
    ) -> str:
        """Load a model into memory and wait until it is ready.
        
        Automatically unloads any other running models before loading
        to ensure deterministic state.
        
        Args:
            model_name: Plain Ollama model name (without provider prefix)
            timeout_seconds: Maximum time to wait for model to load (default 120)
            poll_interval_seconds: Time between running status checks (default 2.0)
            keep_alive: Duration to keep model in memory (default "30m")
            unload_others: If True, unload any other running models first (default True)
            
        Returns:
            The normalized model name that was loaded
            
        Raises:
            RuntimeError: If model fails to load or times out
        """
        if unload_others:
            self._unload_conflicting_models(model_name)

        self.client.load_model(model_name, keep_alive=keep_alive)

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            running = self.client.get_running_models()
            if model_name in running:
                return model_name
            time.sleep(poll_interval_seconds)

        raise RuntimeError(
            f"Timed out waiting for Ollama model {model_name} to become ready at {self.client.base_url}"
        )

    def _unload_conflicting_models(self, target_model: str) -> None:
        """Unload all running models except the target model.
        
        Args:
            target_model: Model name to keep running (all others unloaded)
        """
        running = self.client.get_running_models()
        models_to_unload = sorted(name for name in running if name != target_model)
        for model in models_to_unload:
            self.client.unload_model(model)

    def normalize_model_name(self, model_name: str) -> str:
        """Convert provider-prefixed model identifiers to plain Ollama names.
        
        Strips prefixes like 'ollama/' or 'ollama_chat/' that are used for
        LiteLLM provider specification but not for Ollama API calls.
        
        Args:
            model_name: Potentially prefixed model name
            
        Returns:
            Plain model name without provider prefix
        """
        normalized = model_name.strip()
        for prefix in ("ollama/", "ollama_chat/"):
            if normalized.startswith(prefix):
                return normalized.removeprefix(prefix)
        return normalized
