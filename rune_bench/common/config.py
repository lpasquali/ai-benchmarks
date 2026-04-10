# SPDX-License-Identifier: Apache-2.0
"""YAML configuration loader with profile support.

Precedence (highest to lowest):
  1. CLI flags (typer handles this automatically)
  2. Environment variables (typer envvar= handles this automatically)
  3. ./rune.yaml or ./rune.yml  (project-level, CWD)
  4. ~/.rune/config.yaml or ~/.rune/config.yml  (global user config)
  5. Built-in defaults (typer default= handles this automatically)

Steps 3-4 are handled here by injecting config values into os.environ
**only** when the corresponding env var is not already set by the user.
This means existing env vars always win, and explicit CLI flags still
override everything.

Secrets (API tokens, VAST_API_KEY, etc.) are intentionally excluded from
the YAML schema — they must remain in environment variables only.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

# Maps flat YAML config keys → corresponding RUNE_* env var names.
# Intentionally excludes secrets: RUNE_API_TOKEN, VAST_API_KEY.
_FIELD_ENV_MAP: dict[str, str] = {
    # Global / backend
    "backend": "RUNE_BACKEND",
    "api_base_url": "RUNE_API_BASE_URL",
    "api_tenant": "RUNE_API_TENANT",
    "debug": "RUNE_DEBUG",
    "insecure": "RUNE_INSECURE",
    # API server
    "api_host": "RUNE_API_HOST",
    "api_port": "RUNE_API_PORT",
    # Vast.ai
    "vastai": "RUNE_VASTAI",
    "template_hash": "RUNE_VASTAI_TEMPLATE",
    "max_dph": "RUNE_VASTAI_MAX_DPH",
    "min_dph": "RUNE_VASTAI_MIN_DPH",
    "reliability": "RUNE_VASTAI_RELIABILITY",
    "vastai_stop_instance": "RUNE_VASTAI_STOP_INSTANCE",
    # LLM Backend
    "backend_type": "RUNE_BACKEND_TYPE",
    "backend_url": "RUNE_BACKEND_URL",
    "backend_warmup": "RUNE_BACKEND_WARMUP",
    "backend_warmup_timeout": "RUNE_BACKEND_WARMUP_TIMEOUT",
    # Benchmark
    "question": "RUNE_QUESTION",
    "model": "RUNE_MODEL",
    "kubeconfig": "RUNE_KUBECONFIG",
}

# Nested database section (top-level and/or defaults/profiles).
_DATABASE_ENV_MAP: dict[str, str] = {
    "url": "RUNE_DB_URL",
}

# Maps nested attestation section keys → RUNE_ATTESTATION_* env vars.
_ATTESTATION_ENV_MAP: dict[str, str] = {
    "driver": "RUNE_ATTESTATION_DRIVER",
    "pcr_policy_path": "RUNE_ATTESTATION_PCR_POLICY_PATH",
}

# Search order: project-level (CWD) takes precedence over global user config.
# Within each tier, .yaml is tried before .yml.
_PROJECT_CANDIDATES: list[Path] = [Path("rune.yaml"), Path("rune.yml")]
_GLOBAL_CANDIDATES: list[Path] = [
    Path.home() / ".rune" / "config.yaml",
    Path.home() / ".rune" / "config.yml",
]

# Default YAML template used by `rune init`.
INIT_TEMPLATE = """\
# rune.yaml — RUNE project configuration
# https://github.com/lpasquali/rune
#
# Precedence: CLI flags > env vars > ./rune.yaml > ~/.rune/config.yaml > defaults
#
# IMPORTANT: Never store secrets here (API tokens, VAST_API_KEY).
#            Use environment variables for secrets.
#
# Activate a profile:
#   rune --profile production run-benchmark
#   RUNE_PROFILE=production rune run-benchmark

version: "1"

defaults:
  # Model and benchmark
  model: llama3.1:8b
  question: "What is unhealthy in this Kubernetes cluster?"
  kubeconfig: ~/.kube/config

  # Execution backend (local | http)
  backend: local

  # Database settings (leave unset to use the built-in local SQLite path).
  # database:
  #   url: postgresql://rune:change-me@postgres:5432/rune

  # LLM Backend settings
  backend_type: ollama
  backend_warmup: true
  backend_warmup_timeout: 300

  # Vast.ai settings
  vastai: false
  vastai_stop_instance: true
  template_hash: c166c11f035d3a97871a23bd32ca6aba

profiles:
  production:
    vastai: true
    min_dph: 2.3
    max_dph: 3.0
    reliability: 0.99
    backend_warmup: true

  staging:
    vastai: true
    min_dph: 1.0
    max_dph: 2.0
    reliability: 0.95
    backend_warmup: true

  local:
    vastai: false
    backend_url: http://localhost:11434
    backend_warmup: false

  ci:
    backend: http
    api_base_url: http://rune-api:8080
    vastai: false
    backend_warmup: false

  test:
    vastai: false
    backend_url: http://localhost:11434
    backend_warmup: false
    model: llama3.1:8b

# Top-level infrastructure fallback (can also live under defaults/profiles):
# database:
#   url: postgresql://rune:change-me@postgres:5432/rune

# Attestation settings (TPM 2.0 hardware PCR verification).
# Can be placed at the top level (infrastructure config) or under
# defaults/profiles (per-profile overrides take precedence).
# driver: noop   — always passes; safe for local/dev/CI (default)
# driver: tpm2   — calls tpm2_quote / tpm2_checkquote via subprocess;
#                  requires tpm2-tools and a provisioned AK context.
#
# Top-level example (applies to all profiles unless overridden):
# attestation:
#   driver: noop
#   pcr_policy_path: /etc/rune/pcr.policy
#
# Per-profile example (override in a specific profile):
# profiles:
#   production:
#     attestation:
#       driver: tpm2
#       pcr_policy_path: /etc/rune/pcr.policy
"""


def _find_config_file(candidates: list[Path]) -> Path | None:
    """Return the first existing file from a list of candidates."""
    return next((p for p in candidates if p.exists()), None)


def _parse_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML config file, raising ValueError on invalid content."""
    try:
        with path.open("r") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"rune config at {path} must be a YAML mapping, got {type(data).__name__}"
        )
    return data


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Shallow merge — override wins for any key that is explicitly set (not None)."""
    merged = dict(base)
    for key, value in override.items():
        if value is not None:
            merged[key] = value
    return merged


def _to_env_str(value: Any) -> str:
    """Convert a Python config value to its env-var string representation."""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def load_config(profile: str | None = None) -> dict[str, Any]:
    """Load YAML config files and inject values into os.environ.

    Only injects a value when the corresponding env var is **not** already set,
    preserving the precedence: CLI > env > yaml > built-in defaults.

    Args:
        profile: Profile name to activate (e.g. "production"). When ``None``
                 only the ``defaults`` section is applied.

    Returns:
        The merged effective config dict (used by ``rune config show``).

    Raises:
        ValueError: When the YAML is malformed or the requested profile does
                    not exist.
    """
    # Load global first, then project (project overrides global).
    global_file = _find_config_file(_GLOBAL_CANDIDATES)
    project_file = _find_config_file(_PROJECT_CANDIDATES)

    raw: dict[str, Any] = {}
    if global_file:
        raw = _merge(raw, _parse_yaml(global_file))
    if project_file:
        raw = _merge(raw, _parse_yaml(project_file))

    if not raw:
        return {}  # No config files found — nothing to do.

    defaults: dict[str, Any] = raw.get("defaults") or {}
    profiles: dict[str, Any] = raw.get("profiles") or {}

    effective = dict(defaults)

    if profile:
        if profile not in profiles:
            available = list(profiles.keys())
            raise ValueError(
                f"Profile '{profile}' not found in rune.yaml. "
                f"Available profiles: {available or ['(none defined)']}"
            )
        effective = _merge(effective, profiles[profile])

    # Inject into os.environ — only when env var is not already set by the user.
    for key, env_var in _FIELD_ENV_MAP.items():
        if key in effective and env_var not in os.environ:
            os.environ[env_var] = _to_env_str(effective[key])

    # Database config mirrors attestation: a top-level section acts as
    # infrastructure-wide fallback, while defaults/profiles can override it.
    database_cfg: dict[str, Any] = _merge(
        raw.get("database") or {},
        effective.get("database") or {},
    )
    for key, env_var in _DATABASE_ENV_MAP.items():
        if key in database_cfg and env_var not in os.environ:
            os.environ[env_var] = _to_env_str(database_cfg[key])

    # Handle nested attestation section.  The section may appear at the
    # top level of the YAML (as infrastructure config, separate from per-run
    # defaults) or under defaults/profiles.  Effective (defaults+profile)
    # takes precedence; top-level serves as fallback.
    attestation_cfg: dict[str, Any] = _merge(
        raw.get("attestation") or {},
        effective.get("attestation") or {},
    )
    for key, env_var in _ATTESTATION_ENV_MAP.items():
        if key in attestation_cfg and env_var not in os.environ:
            os.environ[env_var] = _to_env_str(attestation_cfg[key])

    return effective


def peek_profile_from_argv() -> str | None:
    """Peek at sys.argv for ``--profile`` before typer parses CLI args.

    This allows ``load_config()`` to be called at module import time with the
    correct profile, so module-level env-var reads in ``rune/__init__.py`` pick
    up the right values.
    """
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--profile" and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("--profile="):
            return arg.split("=", 1)[1]
    return os.environ.get("RUNE_PROFILE") or None


def get_loaded_config_files() -> list[Path]:
    """Return config files that currently exist (for display purposes)."""
    files = []
    if f := _find_config_file(_GLOBAL_CANDIDATES):
        files.append(f)
    if f := _find_config_file(_PROJECT_CANDIDATES):
        files.append(f)
    return files
