"""Tests for rune_bench.common.config — YAML configuration loader."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from rune_bench.common.config import (
    INIT_TEMPLATE,
    _FIELD_ENV_MAP,
    _to_env_str,
    get_loaded_config_files,
    load_config,
    peek_profile_from_argv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# ---------------------------------------------------------------------------
# load_config — no config files
# ---------------------------------------------------------------------------

class TestLoadConfigNoFiles:
    def test_returns_empty_dict_when_no_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = load_config()
        assert result == {}

    def test_no_env_injection_when_no_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("RUNE_MODEL", raising=False)
        load_config()
        assert "RUNE_MODEL" not in os.environ


# ---------------------------------------------------------------------------
# load_config — defaults only
# ---------------------------------------------------------------------------

class TestLoadConfigDefaults:
    def test_injects_defaults_into_env(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("RUNE_MODEL", raising=False)
        monkeypatch.delenv("RUNE_VASTAI", raising=False)
        _write_yaml(tmp_path / "rune.yaml", """\
version: "1"
defaults:
  model: mixtral:8x7b
  vastai: false
""")
        result = load_config()
        assert result["model"] == "mixtral:8x7b"
        assert os.environ["RUNE_MODEL"] == "mixtral:8x7b"
        assert os.environ["RUNE_VASTAI"] == "0"

    def test_bool_true_becomes_1(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("RUNE_OLLAMA_WARMUP", raising=False)
        _write_yaml(tmp_path / "rune.yaml", """\
version: "1"
defaults:
  ollama_warmup: true
""")
        load_config()
        assert os.environ["RUNE_OLLAMA_WARMUP"] == "1"

    def test_bool_false_becomes_0(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("RUNE_VASTAI", raising=False)
        _write_yaml(tmp_path / "rune.yaml", """\
version: "1"
defaults:
  vastai: false
""")
        load_config()
        assert os.environ["RUNE_VASTAI"] == "0"

    def test_numeric_value_is_stringified(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("RUNE_VASTAI_MAX_DPH", raising=False)
        _write_yaml(tmp_path / "rune.yaml", """\
version: "1"
defaults:
  max_dph: 3.5
""")
        load_config()
        assert os.environ["RUNE_VASTAI_MAX_DPH"] == "3.5"


# ---------------------------------------------------------------------------
# load_config — profile activation
# ---------------------------------------------------------------------------

class TestLoadConfigProfiles:
    _CONFIG = """\
version: "1"
defaults:
  model: llama3.1:8b
  vastai: false

profiles:
  production:
    vastai: true
    min_dph: 2.3
    max_dph: 3.0
    model: mixtral:8x22b
  test:
    ollama_url: http://localhost:11434
    ollama_warmup: false
"""

    def test_profile_overrides_defaults(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("RUNE_MODEL", raising=False)
        monkeypatch.delenv("RUNE_VASTAI", raising=False)
        monkeypatch.delenv("RUNE_VASTAI_MIN_DPH", raising=False)
        _write_yaml(tmp_path / "rune.yaml", self._CONFIG)
        result = load_config("production")
        assert result["model"] == "mixtral:8x22b"
        assert result["vastai"] is True
        assert os.environ["RUNE_MODEL"] == "mixtral:8x22b"
        assert os.environ["RUNE_VASTAI"] == "1"
        assert os.environ["RUNE_VASTAI_MIN_DPH"] == "2.3"

    def test_unknown_profile_raises_value_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_yaml(tmp_path / "rune.yaml", self._CONFIG)
        with pytest.raises(ValueError, match="Profile 'nonexistent' not found"):
            load_config("nonexistent")

    def test_no_profile_uses_defaults_only(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("RUNE_MODEL", raising=False)
        _write_yaml(tmp_path / "rune.yaml", self._CONFIG)
        result = load_config(None)
        assert result["model"] == "llama3.1:8b"
        assert os.environ["RUNE_MODEL"] == "llama3.1:8b"


# ---------------------------------------------------------------------------
# Precedence — env vars beat YAML
# ---------------------------------------------------------------------------

class TestPrecedence:
    def test_existing_env_var_wins_over_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("RUNE_MODEL", "llama3.1:405b")
        _write_yaml(tmp_path / "rune.yaml", """\
version: "1"
defaults:
  model: llama3.1:8b
""")
        load_config()
        # Env var was already set — yaml must not override it.
        assert os.environ["RUNE_MODEL"] == "llama3.1:405b"

    def test_profile_env_var_still_wins_over_yaml_profile(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("RUNE_VASTAI_MAX_DPH", "99.0")
        _write_yaml(tmp_path / "rune.yaml", """\
version: "1"
defaults:
  max_dph: 3.0
profiles:
  prod:
    max_dph: 5.0
""")
        load_config("prod")
        assert os.environ["RUNE_VASTAI_MAX_DPH"] == "99.0"


# ---------------------------------------------------------------------------
# Global vs project config file merging
# ---------------------------------------------------------------------------

class TestConfigFileMerging:
    def test_project_overrides_global(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("RUNE_MODEL", raising=False)

        global_dir = tmp_path / ".rune"
        global_cfg = global_dir / "config.yaml"
        _write_yaml(global_cfg, """\
version: "1"
defaults:
  model: llama3.1:70b
""")

        _write_yaml(tmp_path / "rune.yaml", """\
version: "1"
defaults:
  model: mixtral:8x7b
""")

        with patch("rune_bench.common.config._GLOBAL_CANDIDATES", [global_cfg]):
            result = load_config()

        assert result["model"] == "mixtral:8x7b"

    def test_global_only_config_is_used_when_no_project(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("RUNE_MODEL", raising=False)

        global_dir = tmp_path / ".rune"
        global_cfg = global_dir / "config.yaml"
        _write_yaml(global_cfg, """\
version: "1"
defaults:
  model: command-r:35b
""")

        with patch("rune_bench.common.config._GLOBAL_CANDIDATES", [global_cfg]):
            result = load_config()

        assert result["model"] == "command-r:35b"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_invalid_yaml_raises_value_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "rune.yaml").write_text("key: [unclosed")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_config()

    def test_non_mapping_yaml_raises_value_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "rune.yaml").write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            load_config()

    def test_empty_yaml_returns_empty_dict(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "rune.yaml").write_text("")
        result = load_config()
        assert result == {}

    def test_yaml_with_no_defaults_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_yaml(tmp_path / "rune.yaml", "version: '1'\n")
        result = load_config()
        assert result == {}


# ---------------------------------------------------------------------------
# peek_profile_from_argv
# ---------------------------------------------------------------------------

class TestPeekProfileFromArgv:
    def test_reads_profile_from_argv_space_separated(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["rune", "--profile", "production", "run-benchmark"])
        monkeypatch.delenv("RUNE_PROFILE", raising=False)
        assert peek_profile_from_argv() == "production"

    def test_reads_profile_from_argv_equals_separated(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["rune", "--profile=staging", "run-benchmark"])
        monkeypatch.delenv("RUNE_PROFILE", raising=False)
        assert peek_profile_from_argv() == "staging"

    def test_reads_profile_from_env_when_no_argv(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["rune", "run-benchmark"])
        monkeypatch.setenv("RUNE_PROFILE", "ci")
        assert peek_profile_from_argv() == "ci"

    def test_argv_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["rune", "--profile", "test", "run-benchmark"])
        monkeypatch.setenv("RUNE_PROFILE", "production")
        assert peek_profile_from_argv() == "test"

    def test_returns_none_when_no_profile(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["rune", "run-benchmark"])
        monkeypatch.delenv("RUNE_PROFILE", raising=False)
        assert peek_profile_from_argv() is None

    def test_handles_profile_at_end_of_argv(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["rune", "--profile"])
        monkeypatch.delenv("RUNE_PROFILE", raising=False)
        # --profile with no value — don't crash, return None
        assert peek_profile_from_argv() is None


# ---------------------------------------------------------------------------
# get_loaded_config_files
# ---------------------------------------------------------------------------

class TestGetLoadedConfigFiles:
    def test_returns_existing_project_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "rune.yaml").write_text("version: '1'\n")
        files = get_loaded_config_files()
        assert any(str(f).endswith("rune.yaml") for f in files)

    def test_returns_empty_when_no_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        files = get_loaded_config_files()
        assert files == []


# ---------------------------------------------------------------------------
# _to_env_str helper
# ---------------------------------------------------------------------------

class TestToEnvStr:
    def test_bool_true(self):
        assert _to_env_str(True) == "1"

    def test_bool_false(self):
        assert _to_env_str(False) == "0"

    def test_int(self):
        assert _to_env_str(300) == "300"

    def test_float(self):
        assert _to_env_str(2.3) == "2.3"

    def test_str(self):
        assert _to_env_str("production") == "production"


# ---------------------------------------------------------------------------
# INIT_TEMPLATE sanity
# ---------------------------------------------------------------------------

class TestInitTemplate:
    def test_template_is_valid_yaml(self):
        import yaml
        data = yaml.safe_load(INIT_TEMPLATE)
        assert isinstance(data, dict)
        assert "defaults" in data
        assert "profiles" in data

    def test_template_has_expected_profiles(self):
        import yaml
        data = yaml.safe_load(INIT_TEMPLATE)
        profiles = data["profiles"]
        for expected in ("production", "staging", "local", "ci", "test"):
            assert expected in profiles

    def test_field_env_map_covers_all_known_keys(self):
        known_keys = {
            "backend", "api_base_url", "api_tenant", "debug", "insecure",
            "api_host", "api_port",
            "vastai", "template_hash", "max_dph", "min_dph", "reliability", "vastai_stop_instance",
            "ollama_url", "ollama_warmup", "ollama_warmup_timeout",
            "question", "model", "kubeconfig",
        }
        assert set(_FIELD_ENV_MAP.keys()) == known_keys


# ---------------------------------------------------------------------------
# CLI commands: rune init / rune config
# ---------------------------------------------------------------------------

class TestCliInitConfig:
    """Integration tests for the rune init and rune config CLI commands."""

    def test_init_creates_rune_yaml(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner
        import rune
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(rune.app, ["init"])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "rune.yaml").exists()
        assert "Created" in result.output

    def test_init_refuses_to_overwrite_without_force(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner
        import rune
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)
        (tmp_path / "rune.yaml").write_text("# existing\n")
        result = runner.invoke(rune.app, ["init"])
        assert result.exit_code == 0
        assert "already exists" in result.output
        # original content unchanged
        assert (tmp_path / "rune.yaml").read_text() == "# existing\n"

    def test_init_force_overwrites(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner
        import rune
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)
        (tmp_path / "rune.yaml").write_text("# old\n")
        result = runner.invoke(rune.app, ["init", "--force"])
        assert result.exit_code == 0, result.output
        assert "Created" in result.output
        content = (tmp_path / "rune.yaml").read_text()
        assert "profiles:" in content

    def test_config_no_yaml_prints_defaults(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner
        import rune
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(rune.app, ["config"])
        assert result.exit_code == 0, result.output
        # Should show "No rune.yaml found" or config table
        assert result.output  # non-empty

    def test_config_with_yaml_shows_values(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner
        import rune
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)
        (tmp_path / "rune.yaml").write_text(
            "defaults:\n  model: llama3.1:70b\n  question: test-q\n"
        )
        with patch("rune_bench.common.config._GLOBAL_CANDIDATES", [tmp_path / "rune.yaml"]):
            result = runner.invoke(rune.app, ["config"])
        assert result.exit_code == 0, result.output
