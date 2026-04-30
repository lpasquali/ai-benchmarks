# SPDX-License-Identifier: Apache-2.0
import os
import pytest
import typer
from unittest.mock import MagicMock, patch, AsyncMock
from typer.testing import CliRunner
import rune
from rune import (
    app,
    _is_containerized,
    _find_free_port,
    _resolve_serve_port,
    _enable_debug_if_requested,
)
from rune_bench.workflows import SpendGateAction

runner = CliRunner()


@pytest.fixture(autouse=True)
def clean_env():
    yield
    for key in ["OVERRIDE_MAX_CONTENT_SIZE", "OVERRIDE_MAX_OUTPUT_TOKEN"]:
        if key in os.environ:
            del os.environ[key]
    rune.BACKEND_MODE = "local"


def test_cli_info():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "RUNE" in result.stdout


def test_cli_info_no_config():
    with patch("rune.get_loaded_config_files", return_value=[]):
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0


def test_cli_init():
    with (
        patch("pathlib.Path.exists", return_value=False),
        patch("pathlib.Path.write_text") as mock_write,
    ):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert "Created" in result.stdout
        mock_write.assert_called_once()


def test_cli_init_exists():
    with patch("pathlib.Path.exists", return_value=True):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert "already exists" in result.stdout


def test_cli_config():
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert "Configuration" in result.stdout


def test_cli_config_no_config():
    with patch("rune.get_loaded_config_files", return_value=[]):
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        assert "No rune.yaml found" in result.stdout


def test_cli_config_with_effective():
    with patch("rune.load_config", return_value={"model": "test-model"}):
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        assert "test-model" in result.stdout


def test_cli_db_migrate_postgres():
    with patch(
        "rune_bench.storage.migrate_to_postgres.migrate_to_postgres"
    ) as mock_migrate:
        res1 = MagicMock(table="jobs", source_count=10, migrated_count=10)
        res2 = MagicMock(table="workflow_events", source_count=50, migrated_count=50)
        mock_migrate.return_value = [res1, res2]

        result = runner.invoke(
            app,
            ["db", "migrate-to-postgres", "--target", "postgresql://user:pass@host/db"],
        )
        assert result.exit_code == 0
        mock_migrate.assert_called_once()


def test_cli_db_migrate_postgres_dry_run():
    with patch(
        "rune_bench.storage.migrate_to_postgres.migrate_to_postgres"
    ) as mock_migrate:
        mock_migrate.return_value = []
        result = runner.invoke(
            app,
            [
                "db",
                "migrate-to-postgres",
                "--target",
                "postgresql://user:pass@host/db",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        assert "no data was written" in result.stdout


def test_cli_run_benchmark_vastai_confirm_no():
    with patch("rune._run_preflight_cost_check", side_effect=SystemExit(1)):
        result = runner.invoke(app, ["run-benchmark", "--vastai"])
        assert result.exit_code != 0


@patch("rune.get_agent")
def test_cli_run_agentic_agent_local(mock_get_agent):
    mock_agent = MagicMock()
    mock_get_agent.return_value = mock_agent

    mock_res = MagicMock()
    mock_res.answer = "Agent local answer"
    mock_res.result_type = "text"
    mock_res.artifacts = []
    mock_agent.ask_structured = AsyncMock(return_value=mock_res)

    with (
        patch("rune._warmup_ollama_model"),
        patch("rune.use_existing_backend_server") as mock_use,
    ):
        mock_use.return_value = MagicMock(url="http://local", model_name="m1")
        result = runner.invoke(
            app, ["run-agentic-agent", "--model", "m1", "--question", "q1"]
        )
        assert result.exit_code == 0


@patch("rune.RuneApiClient")
def test_cli_vastai_list_models_http(mock_client_cls):
    mock_client = mock_client_cls.return_value
    mock_client.get_vastai_models.return_value = [
        {"name": "UNIQUE_MODEL_X", "vram_mb": 1000, "required_disk_gb": 10}
    ]

    result = runner.invoke(app, ["--backend", "http", "vastai-list-models"])
    assert result.exit_code == 0
    assert "UNIQUE_MODEL_X" in result.stdout


@patch("rune.RuneApiClient")
def test_cli_vastai_list_models_http_error(mock_client_cls):
    mock_client = mock_client_cls.return_value
    mock_client.get_vastai_models.side_effect = RuntimeError("api error")
    result = runner.invoke(app, ["--backend", "http", "vastai-list-models"])
    assert result.exit_code != 0
    assert "api error" in result.stdout


@patch("rune.RuneApiClient")
def test_cli_ollama_list_models_http(mock_client_cls):
    mock_client = mock_client_cls.return_value
    mock_client.get_llm_models.return_value = {
        "backend_url": "http://remote",
        "models": ["m1_ollama"],
        "running_models": ["m1_ollama"],
    }

    result = runner.invoke(
        app,
        ["--backend", "http", "ollama-list-models", "--backend-url", "http://remote"],
    )
    assert result.exit_code == 0
    assert "m1_ollama" in result.stdout


@patch("rune.RuneApiClient")
def test_cli_ollama_list_models_http_error(mock_client_cls):
    mock_client = mock_client_cls.return_value
    mock_client.get_llm_models.side_effect = RuntimeError("api error")
    result = runner.invoke(
        app,
        ["--backend", "http", "ollama-list-models", "--backend-url", "http://remote"],
    )
    assert result.exit_code != 0
    assert "api error" in result.stdout


def test_cli_ollama_list_models_local():
    with (
        patch("rune.list_backend_models", return_value=["m1"]),
        patch("rune.list_running_backend_models", return_value=["m1"]),
    ):
        result = runner.invoke(
            app, ["ollama-list-models", "--backend-url", "http://local"]
        )
        assert result.exit_code == 0
        assert "m1" in result.stdout


def test_cli_ollama_list_models_local_error():
    with patch("rune.list_backend_models", side_effect=RuntimeError("list error")):
        result = runner.invoke(
            app, ["ollama-list-models", "--backend-url", "http://local"]
        )
        assert result.exit_code != 0
        assert "list error" in result.stdout


def test_vastai_sdk():
    from rune import _vastai_sdk

    with patch.dict(os.environ, {"VAST_API_KEY": "testkey"}):
        sdk = _vastai_sdk()
        assert sdk is not None


def test_resolve_backend_type():
    from rune import _resolve_backend_type

    assert _resolve_backend_type("ollama") == "ollama"
    with patch.dict(os.environ, {"RUNE_BACKEND_TYPE": "ollama"}):
        assert _resolve_backend_type(None) == "ollama"
    assert _resolve_backend_type(None) == "ollama"

    with pytest.raises(RuntimeError, match="Unsupported backend_type"):
        _resolve_backend_type("invalid")


def test_apply_model_limits():
    from rune import _apply_model_limits
    from rune_bench.backends.base import ModelCapabilities

    caps = ModelCapabilities(
        model_name="m1", context_window=1000, max_output_tokens=500
    )
    with patch.dict(os.environ, {}, clear=True):
        _apply_model_limits(caps)
        assert os.environ["OVERRIDE_MAX_CONTENT_SIZE"] == "1000"
        assert os.environ["OVERRIDE_MAX_OUTPUT_TOKEN"] == "500"


def test_confirm_instance_creation():
    from rune import _confirm_instance_creation

    with patch("rune.console.input", return_value="yes"):
        assert _confirm_instance_creation() is True
    with patch("rune.console.input", return_value="no"):
        assert _confirm_instance_creation() is False


def test_cli_run_benchmark_local():
    with (
        patch("rune.use_existing_backend_server") as mock_use,
        patch("rune.get_agent") as mock_get_agent,
        patch("rune._warmup_ollama_model"),
    ):
        mock_use.return_value = MagicMock(url="http://local", model_name="m1")
        mock_agent = MagicMock()
        mock_get_agent.return_value = mock_agent

        mock_res = MagicMock()
        mock_res.answer = "Benchmark local answer"
        mock_res.result_type = "text"
        mock_res.artifacts = []
        mock_agent.ask_structured = AsyncMock(return_value=mock_res)

        result = runner.invoke(
            app, ["run-benchmark", "--model", "m1", "--question", "q1"]
        )
        assert result.exit_code == 0


def test_warmup_ollama_model():
    from rune import _warmup_ollama_model

    with patch("rune.warmup_backend_model") as mock_warmup:
        _warmup_ollama_model(backend_url="u", model_name="m", timeout_seconds=10)
        mock_warmup.assert_called_once()


def test_warmup_ollama_model_error():
    from rune import _warmup_ollama_model

    with patch("rune.warmup_backend_model", side_effect=RuntimeError("warmup error")):
        with pytest.raises(typer.Exit):
            _warmup_ollama_model(backend_url="u", model_name="m", timeout_seconds=10)


def test_is_containerized():
    with patch.dict(os.environ, {"KUBERNETES_SERVICE_HOST": "1.2.3.4"}):
        assert _is_containerized() is True
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("pathlib.Path.exists", return_value=True),
    ):
        assert _is_containerized() is True
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("pathlib.Path.exists", return_value=False),
    ):
        assert _is_containerized() is False


def test_find_free_port():
    port = _find_free_port()
    assert isinstance(port, int)
    assert port > 0


def test_resolve_serve_port():
    with patch("rune._is_containerized", return_value=True):
        assert _resolve_serve_port() == 8080
    with (
        patch("rune._is_containerized", return_value=False),
        patch("rune._find_free_port", return_value=1234),
    ):
        assert _resolve_serve_port() == 1234


def test_enable_debug_if_requested():
    with patch("rune.set_debug") as mock_set:
        _enable_debug_if_requested(True)
        mock_set.assert_called_once_with(True)
        mock_set.reset_mock()
        _enable_debug_if_requested(False)
        mock_set.assert_not_called()


@patch("rune._run_http_job_with_progress")
@patch("rune._http_client")
def test_cli_run_benchmark_http(mock_http_client, mock_run_progress):
    mock_run_progress.return_value = {
        "result": {"answer": "HTTP answer", "result_type": "text", "artifacts": []}
    }
    result = runner.invoke(
        app, ["--backend", "http", "run-benchmark", "--model", "m1", "--question", "q1"]
    )
    assert result.exit_code == 0
    assert "HTTP answer" in result.stdout


@patch("rune._run_http_job_with_progress")
@patch("rune._http_client")
def test_cli_run_benchmark_http_error(mock_http_client, mock_run_progress):
    mock_run_progress.side_effect = RuntimeError("HTTP error")
    result = runner.invoke(
        app, ["--backend", "http", "run-benchmark", "--model", "m1", "--question", "q1"]
    )
    assert result.exit_code != 0
    assert "HTTP error" in result.stdout


@patch("rune._run_http_job_with_progress")
@patch("rune._http_client")
def test_cli_run_agentic_agent_http(mock_http_client, mock_run_progress):
    mock_run_progress.return_value = {
        "result": {
            "answer": "HTTP agent answer",
            "result_type": "text",
            "artifacts": [],
        }
    }
    result = runner.invoke(
        app,
        ["--backend", "http", "run-agentic-agent", "--model", "m1", "--question", "q1"],
    )
    assert result.exit_code == 0
    assert "HTTP agent answer" in result.stdout


@patch("rune._run_http_job_with_progress")
@patch("rune._http_client")
def test_cli_run_llm_instance_http_vastai(mock_http_client, mock_run_progress):
    mock_run_progress.return_value = {
        "result": {
            "mode": "vastai",
            "contract_id": "c1",
            "backend_url": "http://vast",
            "model_name": "m1",
        }
    }
    result = runner.invoke(app, ["--backend", "http", "run-llm-instance", "--vastai"])
    assert result.exit_code == 0
    assert "Provisioned contract: c1" in result.stdout


@patch("rune._run_http_job_with_progress")
@patch("rune._http_client")
def test_cli_run_llm_instance_http_existing(mock_http_client, mock_run_progress):
    mock_run_progress.return_value = {
        "result": {"mode": "existing", "backend_url": "http://existing"}
    }
    result = runner.invoke(
        app,
        ["--backend", "http", "run-llm-instance", "--backend-url", "http://existing"],
    )
    assert result.exit_code == 0
    assert "Existing server" in result.stdout


def test_print_existing_ollama_with_caps():
    from rune import _print_existing_ollama
    from rune_bench.workflows import ExistingOllamaServer
    from rune_bench.backends.base import ModelCapabilities

    server = ExistingOllamaServer(url="http://u", model_name="m")
    caps = ModelCapabilities(model_name="m", context_window=1000, max_output_tokens=500)
    _print_existing_ollama(server, caps)


def test_print_vastai_result_with_caps():
    from rune import _print_vastai_result
    from rune_bench.workflows import VastAIProvisioningResult
    from rune_bench.resources.vastai.instance import ConnectionDetails
    from rune_bench.backends.base import ModelCapabilities

    result = VastAIProvisioningResult(
        offer_id=1,
        total_vram_mb=8000,
        model_name="m1",
        model_vram_mb=4000,
        required_disk_gb=40,
        template_env="env",
        contract_id="c1",
        details=ConnectionDetails(
            contract_id="c1",
            status="running",
            ssh_host="h",
            ssh_port=22,
            machine_id="m1",
        ),
        backend_url="http://vast",
        pull_warning="warning",
    )
    caps = ModelCapabilities(
        model_name="m1", context_window=1000, max_output_tokens=500
    )
    _print_vastai_result(result, caps)


def test_print_metrics_summary():
    from rune import _print_metrics_summary

    mock_collector = MagicMock()
    mock_collector.summary_rows.return_value = [
        {
            "event": "event",
            "total": 1,
            "ok": 1,
            "error": 0,
            "avg_ms": 100.0,
            "min_ms": 100.0,
            "max_ms": 100.0,
        }
    ]
    _print_metrics_summary(mock_collector)

    mock_collector.summary_rows.return_value = []
    _print_metrics_summary(mock_collector)


@patch("rune._run_vastai_provisioning")
def test_cli_run_llm_instance_vastai(mock_run):
    from rune_bench.workflows import VastAIProvisioningResult
    from rune_bench.resources.vastai.instance import ConnectionDetails

    mock_run.return_value = VastAIProvisioningResult(
        offer_id=1,
        total_vram_mb=8000,
        model_name="m1",
        model_vram_mb=4000,
        required_disk_gb=40,
        template_env="env",
        contract_id="c1",
        details=ConnectionDetails(
            contract_id="c1",
            status="running",
            ssh_host="localhost",
            ssh_port=2222,
            machine_id="m1",
        ),
        backend_url="http://vast",
    )
    result = runner.invoke(app, ["run-llm-instance", "--vastai"])
    assert result.exit_code == 0


def test_cli_run_llm_instance_existing():
    with patch("rune.use_existing_backend_server") as mock_use:
        mock_use.return_value = MagicMock(url="http://existing", model_name="m1")
        result = runner.invoke(
            app, ["run-llm-instance", "--backend-url", "http://existing"]
        )
        assert result.exit_code == 0
        assert "Existing server" in result.stdout


@patch("rune._run_vastai_provisioning")
@patch("rune.get_agent")
@patch("rune.stop_vastai_instance")
@patch("rune._warmup_ollama_model")
def test_cli_run_benchmark_vastai_success(
    mock_warmup, mock_stop, mock_get_agent, mock_run_vast
):
    from rune_bench.workflows import VastAIProvisioningResult, TeardownResult
    from rune_bench.resources.vastai.instance import ConnectionDetails

    mock_run_vast.return_value = VastAIProvisioningResult(
        offer_id=1,
        total_vram_mb=8000,
        model_name="m1",
        model_vram_mb=4000,
        required_disk_gb=40,
        template_env="env",
        contract_id="c1",
        details=ConnectionDetails(
            contract_id="c1",
            status="running",
            ssh_host="h",
            ssh_port=22,
            machine_id="m1",
        ),
        backend_url="http://vast",
    )
    mock_agent = MagicMock()
    mock_get_agent.return_value = mock_agent
    mock_res = MagicMock()
    mock_res.answer = "Benchmark vast answer"
    mock_res.result_type = "text"
    mock_res.artifacts = ["art1"]
    mock_agent.ask_structured = AsyncMock(return_value=mock_res)

    mock_stop.return_value = TeardownResult(
        contract_id="c1", destroyed_instance=True, verification_ok=True
    )

    result = runner.invoke(app, ["run-benchmark", "--vastai", "--model", "m1"])
    assert result.exit_code == 0
    assert "Benchmark vast answer" in result.stdout
    assert "Artifacts (1)" in result.stdout
    mock_stop.assert_called_once()


@patch("rune._run_vastai_provisioning")
@patch("rune.stop_vastai_instance")
def test_cli_run_benchmark_vastai_teardown_error(mock_stop, mock_run_vast):
    from rune_bench.workflows import VastAIProvisioningResult
    from rune_bench.resources.vastai.instance import ConnectionDetails

    mock_run_vast.return_value = VastAIProvisioningResult(
        offer_id=1,
        total_vram_mb=8000,
        model_name="m1",
        model_vram_mb=4000,
        required_disk_gb=40,
        template_env="env",
        contract_id="c1",
        details=ConnectionDetails(
            contract_id="c1",
            status="running",
            ssh_host="h",
            ssh_port=22,
            machine_id="m1",
        ),
        backend_url=None,
    )
    mock_stop.side_effect = RuntimeError("stop error")
    result = runner.invoke(app, ["run-benchmark", "--vastai", "--model", "m1"])
    assert result.exit_code != 0
    assert "Failed to destroy Vast.ai contract" in result.stdout


@patch("rune._run_vastai_provisioning")
def test_cli_run_benchmark_vastai_no_url(mock_run_vast):
    from rune_bench.workflows import VastAIProvisioningResult
    from rune_bench.resources.vastai.instance import ConnectionDetails

    mock_run_vast.return_value = VastAIProvisioningResult(
        offer_id=1,
        total_vram_mb=8000,
        model_name="m1",
        model_vram_mb=4000,
        required_disk_gb=40,
        template_env="env",
        contract_id="c1",
        details=ConnectionDetails(
            contract_id="c1",
            status="running",
            ssh_host="h",
            ssh_port=22,
            machine_id="m1",
        ),
        backend_url=None,
    )
    with patch("rune.stop_vastai_instance"):
        result = runner.invoke(app, ["run-benchmark", "--vastai", "--model", "m1"])
        assert result.exit_code != 0
        assert "Could not determine Ollama URL" in result.stdout


def test_fetch_model_capabilities():
    from rune import _fetch_model_capabilities

    with patch("rune.get_backend") as mock_get_backend:
        mock_backend = MagicMock()
        mock_get_backend.return_value = mock_backend
        mock_backend.get_model_capabilities.return_value = MagicMock()

        res = _fetch_model_capabilities("http://u", "m")
        assert res is not None

        mock_backend.get_model_capabilities.side_effect = RuntimeError("caps error")
        res = _fetch_model_capabilities("http://u", "m")
        assert res is None


@patch("rune_bench.api_server.RuneApiApplication.from_env")
def test_cli_serve(mock_from_env):
    mock_app = mock_from_env.return_value
    result = runner.invoke(app, ["serve", "--port", "9999"])
    assert result.exit_code == 0
    mock_app.serve.assert_called_once_with(host="127.0.0.1", port=9999)


@patch("rune_bench.api_server.RuneApiApplication.from_env")
def test_cli_serve_error(mock_from_env):
    mock_app = mock_from_env.return_value
    mock_app.serve.side_effect = RuntimeError("serve error")
    result = runner.invoke(app, ["serve"])
    assert result.exit_code != 0
    assert "Server error: serve error" in result.stdout


def test_cli_backend_invalid():
    result = runner.invoke(app, ["--backend", "invalid", "info"])
    assert result.exit_code != 0


def test_normalize_backend_url_error():
    from rune_bench.common.backend_utils import normalize_backend_url

    with pytest.raises(RuntimeError, match="Missing Ollama URL"):
        normalize_backend_url(None)


@patch("rune.evaluate_spend_gate")
def test_run_preflight_cost_check_prompt(mock_gate):
    from rune import _run_preflight_cost_check

    mock_gate.return_value = SpendGateAction.PROMPT

    with (
        patch("sys.stdin", MagicMock(fileno=lambda: 0)),
        patch("os.isatty", return_value=True),
        patch("rune.console.input", return_value="no"),
    ):
        with pytest.raises(typer.Exit) as exc:
            import asyncio

            asyncio.run(
                _run_preflight_cost_check(
                    vastai=True, max_dph=1.0, min_dph=0.5, yes=False
                )
            )
        assert exc.value.exit_code == 1


def test_print_vastai_models():
    from rune import _print_vastai_models

    with patch("rune.ModelSelector") as mock_selector:
        mock_inst = mock_selector.return_value
        m1 = MagicMock()
        m1.name = "m1"
        m1.vram_mb = 1000
        m1.required_disk_gb = 10
        mock_inst.list_models.return_value = [m1]
        _print_vastai_models()


def test_run_preflight_cost_check_no_vastai():
    from rune import _run_preflight_cost_check
    import asyncio

    asyncio.run(
        _run_preflight_cost_check(vastai=False, max_dph=1.0, min_dph=0.5, yes=False)
    )


@patch("rune.evaluate_spend_gate")
def test_run_preflight_cost_check_allow(mock_gate):
    from rune import _run_preflight_cost_check

    mock_gate.return_value = SpendGateAction.ALLOW
    import asyncio

    asyncio.run(
        _run_preflight_cost_check(vastai=True, max_dph=1.0, min_dph=0.5, yes=False)
    )


@patch("rune.evaluate_spend_gate")
def test_run_preflight_cost_check_block(mock_gate):
    from rune import _run_preflight_cost_check

    mock_gate.return_value = SpendGateAction.BLOCK
    with pytest.raises(typer.Exit) as exc:
        import asyncio

        asyncio.run(
            _run_preflight_cost_check(vastai=True, max_dph=1.0, min_dph=0.5, yes=False)
        )
    assert exc.value.exit_code == 1


def test_run_preflight_cost_check_invalid_threshold():
    from rune import _run_preflight_cost_check

    with patch.dict(os.environ, {"RUNE_SPEND_WARNING_THRESHOLD": "invalid"}):
        import asyncio

        asyncio.run(
            _run_preflight_cost_check(vastai=True, max_dph=1.0, min_dph=0.5, yes=True)
        )


@patch("rune.provision_vastai_backend")
def test_run_vastai_provisioning_aborted(mock_prov):
    from rune import _run_vastai_provisioning
    from rune_bench.workflows import UserAbortedError

    mock_prov.side_effect = UserAbortedError()
    with pytest.raises(typer.Exit) as exc:
        _run_vastai_provisioning(
            template_hash="h", min_dph=0.1, max_dph=1.0, reliability=0.9, yes=False
        )
    assert exc.value.exit_code == 0


@patch("rune.provision_vastai_backend")
def test_run_vastai_provisioning_error(mock_prov):
    from rune import _run_vastai_provisioning

    mock_prov.side_effect = RuntimeError("prov error")
    with pytest.raises(typer.Exit) as exc:
        _run_vastai_provisioning(
            template_hash="h", min_dph=0.1, max_dph=1.0, reliability=0.9, yes=False
        )
    assert exc.value.exit_code == 1


@patch("rune.provision_vastai_backend")
def test_run_vastai_provisioning_confirm_yes(mock_prov):
    from rune import _run_vastai_provisioning

    with patch("builtins.input", return_value="yes"):
        _run_vastai_provisioning(
            template_hash="h", min_dph=0.1, max_dph=1.0, reliability=0.9, yes=False
        )
        confirm_callback = mock_prov.call_args[1]["confirm_create"]
        assert confirm_callback() is True


@patch("rune.provision_vastai_backend")
def test_run_vastai_provisioning_confirm_no(mock_prov):
    from rune import _run_vastai_provisioning

    with patch("builtins.input", return_value="no"):
        _run_vastai_provisioning(
            template_hash="h", min_dph=0.1, max_dph=1.0, reliability=0.9, yes=False
        )
        confirm_callback = mock_prov.call_args[1]["confirm_create"]
        assert confirm_callback() is False
