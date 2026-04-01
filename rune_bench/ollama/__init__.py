"""Ollama integration for RUNE.

This package provides modular, reusable abstractions for interacting with Ollama servers.
"""

from .client import OllamaClient, OllamaModelCapabilities
from .models import OllamaModelManager

__all__ = ["OllamaClient", "OllamaModelCapabilities", "OllamaModelManager"]
