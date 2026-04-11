# SPDX-License-Identifier: Apache-2.0
import pytest
import sys
import os
import asyncio
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch, AsyncMock
from rune import app

runner = CliRunner()

@pytest.fixture(autouse=True)
def clean_env():
    # Save original env
    original_env = os.environ.copy()
    yield
    # Restore original env
    os.environ.clear()
    os.environ.update(original_env)

def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "RUNE" in result.output

def test_cli_info_vastai_missing():
    with patch.dict("sys.modules", {"vastai": None}):
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "not installed" in result.output

def test_cli_info_holmes_missing():
    with patch.dict("sys.modules", {"holmes": None}):
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "not installed" in result.output

def test_cli_run_http_error_agent():
    payload = {"status": "success", "result": {}} 
    import rune
    with patch("rune._run_http_job_with_progress", return_value=payload):
        result = runner.invoke(app, ["--backend", "http", "run-agentic-agent", "--question", "q", "--backend-url", "http://u"])
        assert result.exit_code == 1
        assert "did not return an agent answer" in result.output

def test_cli_run_http_error_ollama():
    payload = {"status": "success", "result": {"mode": "unknown"}} 
    import rune
    with patch("rune._run_http_job_with_progress", return_value=payload):
        result = runner.invoke(app, ["--backend", "http", "run-llm-instance", "--backend-url", "http://u"])
        assert result.exit_code == 1
        assert "did not return an Ollama instance result" in result.output

def test_cli_print_error_and_exit_directly():
    from rune import _print_error_and_exit
    import typer
    with pytest.raises(typer.Exit) as exc:
        with patch("rune.console.print") as mock_print:
            _print_error_and_exit("err")
    assert exc.value.exit_code == 1

def test_vastai_models_print_coverage():
    from rune import _print_vastai_models
    mock_model = MagicMock()
    mock_model.name = "m1"
    mock_model.vram_mb = 100
    mock_model.required_disk_gb = 10
    
    with patch("rune.ModelSelector") as mock_sel:
        mock_sel.return_value.list_models.return_value = [mock_model]
        _print_vastai_models()
        assert mock_sel.return_value.list_models.called

def test_cli_vastai_list_models_local():
    result = runner.invoke(app, ["vastai-list-models"])
    assert result.exit_code == 0
    assert "Configured Vast.ai Models" in result.output

def test_cli_vastai_list_models_http():
    import rune
    payload = [{"name": "m1", "vram_mb": 100, "required_disk_gb": 10}]
    with patch("rune._http_client") as mock_client:
        mock_client.return_value.get_vastai_models.return_value = payload
        result = runner.invoke(app, ["--backend", "http", "vastai-list-models"])
        # If it returns 0, it means it worked.
        assert result.exit_code == 0

def test_apply_model_limits_with_env_mock():
    from rune import _apply_model_limits
    from rune_bench.backends.base import ModelCapabilities
    if "OVERRIDE_MAX_CONTENT_SIZE" in os.environ:
        del os.environ["OVERRIDE_MAX_CONTENT_SIZE"]
    
    _apply_model_limits(ModelCapabilities(model_name="m1", context_window=200, max_output_tokens=100))
    assert os.environ["OVERRIDE_MAX_CONTENT_SIZE"] == "200"
    assert os.environ["OVERRIDE_MAX_OUTPUT_TOKEN"] == "100"

def test_cli_run_agent_error():
    mock_runner = MagicMock()
    mock_runner.ask_structured = AsyncMock(side_effect=RuntimeError("oops"))
    with patch("rune.get_agent", return_value=mock_runner):
        result = runner.invoke(app, ["run-agentic-agent", "--question", "q"])
        assert result.exit_code == 1
        assert "Agent error" in result.output

def test_cli_run_benchmark_agent_error():
    mock_runner = MagicMock()
    mock_runner.ask_structured = AsyncMock(side_effect=FileNotFoundError("missing file"))
    from rune_bench.backends.base import ModelCapabilities
    with patch("rune.get_agent", return_value=mock_runner):
        with patch("rune._fetch_model_capabilities", return_value=ModelCapabilities(model_name="m", context_window=100)):
            with patch("rune._warmup_ollama_model"):
                result = runner.invoke(app, ["run-benchmark", "--question", "q", "--backend-url", "http://localhost:11434"])
                assert result.exit_code == 1
                assert "Agent error" in result.output

def test_cli_run_holmes_with_artifacts():
    mock_result = MagicMock()
    mock_result.answer = "ok"
    mock_result.result_type = "text"
    mock_result.artifacts = ["art1"]
    
    mock_runner = MagicMock()
    mock_runner.ask_structured = AsyncMock(return_value=mock_result)
    with patch("rune.get_agent", return_value=mock_runner):
        result = runner.invoke(app, ["run-agentic-agent", "--question", "q"])
        assert result.exit_code == 0
        assert "Artifacts" in result.output

def test_cli_run_benchmark_with_artifacts():
    payload = {
        "status": "success", 
        "result": {
            "answer": "ok",
            "result_type": "text",
            "artifacts": ["art1"]
        }
    }
    with patch("rune._run_http_job_with_progress", return_value=payload):
        result = runner.invoke(app, ["--backend", "http", "run-benchmark", "--question", "q", "--backend-url", "http://u"])
        assert result.exit_code == 0
        assert "Artifacts" in result.output

def test_cli_run_preflight_none():
    from rune_bench.backends.base import ModelCapabilities
    mock_result = MagicMock()
    mock_result.answer = "ok"
    mock_result.result_type = "text"
    mock_result.artifacts = []
    
    mock_runner = MagicMock()
    mock_runner.ask_structured = AsyncMock(return_value=mock_result)
    
    with patch("rune.run_preflight_cost_check", return_value=None):
        with patch("rune._fetch_model_capabilities", return_value=ModelCapabilities(model_name="m", context_window=100)):
            with patch("rune._warmup_ollama_model"):
                with patch("rune.get_agent", return_value=mock_runner):
                    result = runner.invoke(app, ["run-benchmark", "--question", "q", "--backend-url", "http://localhost:11434"])
                    assert result.exit_code == 0
