"""LLM backend protocol and implementations.

Backends define HOW to communicate with an LLM — handling auth, model
capability discovery, and inference calls — independent of WHERE the LLM
runs (resources/) or WHAT the agent does with it (agents/).
"""

from .base import BackendCredentials, LLMBackend, ModelCapabilities
from .ollama import OllamaClient, OllamaModelCapabilities, OllamaModelManager

__all__ = [
    "BackendCredentials",
    "LLMBackend",
    "ModelCapabilities",
    "OllamaClient",
    "OllamaModelCapabilities",
    "OllamaModelManager",
]
