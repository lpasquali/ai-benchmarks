# SPDX-License-Identifier: Apache-2.0
"""Block 2+3 — Model Selection and Disk Sizing.

Selects the largest LLM model that fits in the available VRAM and
calculates the required disk space with safety margins.
"""

import math
from dataclasses import dataclass
from typing import Dict, List


# Ordered largest → smallest so the first match is always the best fit.
MODELS: list[dict] = [
    {"name": "llama3.1:405b", "vram_mb": 260000},
    {"name": "mixtral:8x22b", "vram_mb": 95000},
    {"name": "command-r-plus:104b", "vram_mb": 75000},
    {"name": "qwen2.5-coder:72b", "vram_mb": 55000},
    {"name": "llama3.1:70b", "vram_mb": 50000},
    {"name": "mixtral:8x7b", "vram_mb": 32000},
    {"name": "command-r:35b", "vram_mb": 28000},
    {"name": "llama3.1:8b", "vram_mb": 8000},
]

# Disk headroom constants
_VRAM_OVERHEAD_FACTOR = 1.15  # 15 % buffer over the model weight size on disk
_BASE_DISK_GB = 32  # Fixed OS / container / Ollama daemon buffer


@dataclass
class SelectedModel:
    name: str
    vram_mb: int
    required_disk_gb: int


class ModelSelector:
    """Pick the best-fitting Ollama model for the available VRAM."""

    def __init__(self, models: list[dict] | None = None) -> None:
        self._models = models or MODELS

    def select(self, total_vram_mb: int) -> SelectedModel:
        """Return the largest model that fits in *total_vram_mb*.

        Raises:
            RuntimeError: if no model fits.
        """
        for m in self._models:
            if total_vram_mb >= m["vram_mb"]:
                disk_gb = self._calculate_disk(m["vram_mb"])
                return SelectedModel(
                    name=m["name"],
                    vram_mb=m["vram_mb"],
                    required_disk_gb=disk_gb,
                )
        raise RuntimeError(
            f"No configured model fits in {total_vram_mb} MB VRAM. "
            "Add a smaller model to the MODELS list or use a larger instance."
        )

    def list_models(self) -> list[SelectedModel]:
        """Return all configured Vast.ai candidate models with derived disk sizing."""
        return [
            SelectedModel(
                name=m["name"],
                vram_mb=m["vram_mb"],
                required_disk_gb=self._calculate_disk(m["vram_mb"]),
            )
            for m in self._models
        ]

    @staticmethod
    def _calculate_disk(vram_mb: int) -> int:
        """Disk = ceil(vram_gb * 1.15) + 32 GB base buffer."""
        return math.ceil((vram_mb / 1024) * _VRAM_OVERHEAD_FACTOR) + _BASE_DISK_GB


@dataclass(frozen=True)
class Modelinfo:
    name: str
    vram_mb: int = 0
    description: str = ""

# Popular official Ollama models for self-hosted installations
OLLAMA_DEFAULT_MODELS = [
    Modelinfo("llama3.1:8b", 5000, "Meta Llama 3.1 8B (General Purpose)"),
    Modelinfo("llama3.1:70b", 40000, "Meta Llama 3.1 70B (High Performance)"),
    Modelinfo("mistral:7b", 5000, "Mistral 7B v0.3"),
    Modelinfo("mixtral:8x7b", 26000, "Mixtral 8x7B MoE"),
    Modelinfo("phi3:mini", 2500, "Microsoft Phi-3 Mini"),
    Modelinfo("codellama:7b", 5000, "Code Llama 7B (Coding)"),
    Modelinfo("deepseek-coder:6.7b", 5000, "DeepSeek Coder 6.7B"),
    Modelinfo("gemma2:9b", 6000, "Google Gemma 2 9B"),
]

# Standard OpenAI models
OPENAI_DEFAULT_MODELS = [
    Modelinfo("gpt-4o", 0, "GPT-4o (Multimodal)"),
    Modelinfo("gpt-4o-mini", 0, "GPT-4o Mini"),
    Modelinfo("gpt-4-turbo", 0, "GPT-4 Turbo"),
    Modelinfo("gpt-3.5-turbo", 0, "GPT-3.5 Turbo"),
]

# Backend model mapping
BACKEND_MODEL_REGISTRY: Dict[str, List[Modelinfo]] = {
    "ollama": OLLAMA_DEFAULT_MODELS,
    "openai": OPENAI_DEFAULT_MODELS,
    "vastai": OLLAMA_DEFAULT_MODELS, # Vast.ai usually runs Ollama/vLLM images
}

def get_default_models(backend_type: str) -> List[str]:
    """Return a list of model names for the given backend."""
    models = BACKEND_MODEL_REGISTRY.get(backend_type.lower(), [])
    return [m.name for m in models]
