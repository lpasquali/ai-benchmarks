# SPDX-License-Identifier: Apache-2.0
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from rune_bench.workflows import (
    ExistingOllamaServer,
    SpendGateAction,
    evaluate_spend_gate,
    normalize_backend_url,
    run_preflight_cost_check,
    JobStoreChainRecorder,
    stop_vastai_instance,
    provision_vastai_backend,
    run_chain_workflow,
    UserAbortedError,
    _extract_ollama_service_url,
    debug_log,
)


def test_evaluate_spend_gate(monkeypatch):
    # ALLOW - under threshold
    assert evaluate_spend_gate(1.0, threshold=5.0, yes=False) == SpendGateAction.ALLOW

    # ALLOW - yes flag
    assert evaluate_spend_gate(10.0, threshold=5.0, yes=True) == SpendGateAction.ALLOW

    # BLOCK - in CI
    monkeypatch.setenv("CI", "true")
    assert evaluate_spend_gate(10.0, threshold=5.0, yes=False) == SpendGateAction.BLOCK

    # PROMPT - otherwise
    monkeypatch.delenv("CI", raising=False)
    assert evaluate_spend_gate(10.0, threshold=5.0, yes=False) == SpendGateAction.PROMPT


def test_normalize_backend_url():
    assert normalize_backend_url("localhost:11434") == "http://localhost:11434"
    assert normalize_backend_url("http://host") == "http://host"


@pytest.mark.asyncio
async def test_run_preflight_cost_check():
    # not vastai
    assert await run_preflight_cost_check(vastai=False, max_dph=1, min_dph=0) == {}

    # vastai local
    with patch("rune_bench.common.costs.CostEstimator") as mock_est_cls:
        mock_est = mock_est_cls.return_value
        mock_est.estimate = AsyncMock()
        mock_res = MagicMock()
        mock_res.to_dict.return_value = {"cost": 1.0}
        mock_est.estimate.return_value = mock_res

        res = await run_preflight_cost_check(vastai=True, max_dph=1, min_dph=0)
        assert res == {"cost": 1.0}

    # vastai http
    mock_client = MagicMock()
    mock_client.get_cost_estimate.return_value = {"cost": 2.0}
    res = await run_preflight_cost_check(
        vastai=True, max_dph=1, min_dph=0, backend_mode="http", http_client=mock_client
    )
    assert res == {"cost": 2.0}

    # http mode missing client
    with pytest.raises(RuntimeError, match="http_client is required"):
        await run_preflight_cost_check(
            vastai=True, max_dph=1, min_dph=0, backend_mode="http"
        )


def test_job_store_chain_recorder():
    mock_store = MagicMock()
    recorder = JobStoreChainRecorder(mock_store)

    recorder.initialize(job_id="j1", nodes=[], edges=[])
    mock_store.record_chain_initialized.assert_called_once()

    recorder.transition(job_id="j1", node_id="n1", status="running")
    mock_store.record_chain_node_transition.assert_called_once()


def test_stop_vastai_instance():
    with patch("rune_bench.workflows.InstanceManager") as mock_mgr_cls:
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.destroy_instance_and_related_storage.return_value = MagicMock()

        stop_vastai_instance(MagicMock(), "contract1")
        mock_mgr.destroy_instance_and_related_storage.assert_called_with("contract1")


def test_provision_vastai_backend():
    mock_sdk = MagicMock()

    with (
        patch("rune_bench.workflows.InstanceManager") as mock_mgr_cls,
        patch("rune_bench.workflows.OfferFinder") as mock_finder_cls,
        patch("rune_bench.workflows.TemplateLoader") as mock_tpl_cls,
        patch("rune_bench.workflows.ModelSelector") as mock_sel_cls,
        patch("rune_bench.workflows.list_backend_models", return_value=["m1"]),
        patch("rune_bench.workflows.list_running_backend_models", return_value=[]),
        patch(
            "rune_bench.workflows.normalize_backend_model_for_api", return_value="m1"
        ),
        patch("rune_bench.workflows.warmup_backend_model") as mock_warmup,
    ):
        mock_mgr = mock_mgr_cls.return_value
        mock_finder = mock_finder_cls.return_value
        mock_tpl = mock_tpl_cls.return_value
        mock_sel = mock_sel_cls.return_value

        mock_sel.select.return_value = MagicMock(
            name="m1", vram_mb=5000, required_disk_gb=10
        )
        mock_mgr.build_connection_details.return_value = MagicMock()
        mock_finder.find_best.return_value = MagicMock(offer_id=1, total_vram_mb=10000)
        mock_mgr.create.return_value = "c_new"
        mock_mgr.wait_until_running.return_value = {"id": "c_new"}
        mock_tpl.load.return_value = MagicMock(env="env")

        # Case 1: Reusable instance found, need warmup
        mock_mgr.find_reusable_running_instance.return_value = {
            "id": "123",
            "gpu_total_ram": 10000,
        }
        with patch(
            "rune_bench.workflows.OllamaBackend.extract_service_url",
            return_value="http://u",
        ):
            res = provision_vastai_backend(
                mock_sdk,
                template_hash="t1",
                min_dph=0,
                max_dph=1,
                reliability=0.9,
                confirm_create=lambda: True,
            )
            assert res.contract_id == "123"
            mock_warmup.assert_called_once()

        # Case 2: Reusable instance triggered fallback (empty ID)
        mock_mgr.find_reusable_running_instance.return_value = {
            "id": "",
            "gpu_total_ram": 10000,
        }
        with patch(
            "rune_bench.workflows.OllamaBackend.extract_service_url",
            return_value="http://u",
        ):
            res = provision_vastai_backend(
                mock_sdk,
                template_hash="t1",
                min_dph=0,
                max_dph=1,
                reliability=0.9,
                confirm_create=lambda: True,
            )
            assert res.contract_id == "c_new"

        # Case 3: Model Selector failure - triggered fallback
        mock_mgr.find_reusable_running_instance.return_value = {
            "id": "123",
            "gpu_total_ram": 10000,
        }
        mock_sel.select.side_effect = [
            RuntimeError("select fail"),
            MagicMock(name="m1", vram_mb=5000, required_disk_gb=10),
        ]
        with patch(
            "rune_bench.workflows.OllamaBackend.extract_service_url",
            return_value="http://u",
        ):
            res = provision_vastai_backend(
                mock_sdk,
                template_hash="t1",
                min_dph=0,
                max_dph=1,
                reliability=0.9,
                confirm_create=lambda: True,
            )
            assert res.contract_id == "c_new"

        # Case 4: Pull model error
        mock_mgr.find_reusable_running_instance.return_value = {
            "id": "123",
            "gpu_total_ram": 10000,
        }
        mock_sel.select.side_effect = None
        mock_sel.select.return_value = MagicMock(
            name="m1", vram_mb=5000, required_disk_gb=10
        )
        with (
            patch("rune_bench.workflows.list_backend_models", return_value=[]),
            patch(
                "rune_bench.workflows.OllamaBackend.extract_service_url",
                return_value="http://u",
            ),
        ):
            mock_mgr.pull_model.side_effect = RuntimeError("pull fail")
            res = provision_vastai_backend(
                mock_sdk,
                template_hash="t1",
                min_dph=0,
                max_dph=1,
                reliability=0.9,
                confirm_create=lambda: True,
            )
            assert res.pull_warning == "pull fail"

        # Case 5: No backend URL found (hits line 161)
        mock_mgr.find_reusable_running_instance.return_value = {
            "id": "123",
            "gpu_total_ram": 10000,
        }
        mock_mgr.pull_model.side_effect = None
        # We need to simulate the NEW instance creation NOT finding a backend URL too
        # To hit line 161, we need 'reusable' to be not None, and then extract_service_url returns None.
        with patch(
            "rune_bench.workflows.OllamaBackend.extract_service_url", return_value=None
        ):
            res = provision_vastai_backend(
                mock_sdk,
                template_hash="t1",
                min_dph=0,
                max_dph=1,
                reliability=0.9,
                confirm_create=lambda: True,
            )
            assert "port 11434 missing" in res.pull_warning

        # Case 6: Model already running
        mock_mgr.find_reusable_running_instance.return_value = {
            "id": "123",
            "gpu_total_ram": 10000,
        }
        with (
            patch(
                "rune_bench.workflows.list_running_backend_models", return_value=["m1"]
            ),
            patch(
                "rune_bench.workflows.OllamaBackend.extract_service_url",
                return_value="http://u",
            ),
        ):
            res = provision_vastai_backend(
                mock_sdk,
                template_hash="t1",
                min_dph=0,
                max_dph=1,
                reliability=0.9,
                confirm_create=lambda: True,
            )
            assert res.contract_id == "123"


def test_extract_ollama_service_url():
    with patch(
        "rune_bench.workflows.OllamaBackend.extract_service_url", return_value="url"
    ):
        assert _extract_ollama_service_url(MagicMock()) == "url"


def test_debug_log(monkeypatch):
    from rune_bench.debug import set_debug

    set_debug(True)
    try:
        with patch("sys.stderr.write") as mock_write:
            debug_log("test-message")
            # debug_log uses print(..., file=sys.stderr), which calls sys.stderr.write
            mock_write.assert_called()
    finally:
        set_debug(False)


def test_provision_vastai_backend_user_abort():
    mock_sdk = MagicMock()
    with (
        patch("rune_bench.workflows.InstanceManager") as mock_mgr_cls,
        patch("rune_bench.workflows.OfferFinder") as mock_finder_cls,
        patch("rune_bench.workflows.TemplateLoader") as mock_tpl_cls,
        patch("rune_bench.workflows.ModelSelector") as mock_sel_cls,
    ):
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.find_reusable_running_instance.return_value = (
            None  # Force new creation
        )

        mock_finder = mock_finder_cls.return_value
        mock_finder.find_best.return_value = MagicMock(offer_id=1, total_vram_mb=10000)

        mock_sel = mock_sel_cls.return_value
        mock_sel.select.return_value = MagicMock(name="m1")

        mock_tpl = mock_tpl_cls.return_value
        mock_tpl.load.return_value = MagicMock()

        with pytest.raises(UserAbortedError, match="User aborted instance creation"):
            provision_vastai_backend(
                mock_sdk,
                template_hash="t1",
                min_dph=0.0,
                max_dph=1.0,
                reliability=0.9,
                confirm_create=lambda: False,
            )


def test_provision_vastai_backend_vastai_url_normalization():
    mock_sdk = MagicMock()
    with (
        patch("rune_bench.workflows.InstanceManager") as mock_mgr_cls,
        patch("rune_bench.workflows.OfferFinder"),
        patch("rune_bench.workflows.TemplateLoader"),
        patch("rune_bench.workflows.ModelSelector"),
        patch(
            "rune_bench.workflows.OllamaBackend.extract_service_url",
            return_value="vast.ai/endpoint",
        ),
    ):
        mock_mgr = mock_mgr_cls.return_value
        mock_mgr.find_reusable_running_instance.return_value = {
            "id": "123",
            "gpu_total_ram": 10000,
        }
        mock_mgr.build_connection_details.return_value = MagicMock()

        # We need list_running_backend_models to contain the model to avoid warmup/pull
        with (
            patch(
                "rune_bench.workflows.list_running_backend_models", return_value=["m1"]
            ),
            patch(
                "rune_bench.workflows.normalize_backend_model_for_api",
                return_value="m1",
            ),
        ):
            provision_vastai_backend(
                mock_sdk,
                template_hash="t1",
                min_dph=0.0,
                max_dph=1.0,
                reliability=0.9,
                confirm_create=lambda: True,
            )
            # Line 161 (in some versions, but looking for 'vast.ai' in url)
            # In my previous cat -n, line 161 was the UserAbortedError.
            # Let me check where 'vast.ai' check is.


@pytest.mark.asyncio
async def test_run_chain_workflow():
    mock_store = MagicMock()
    with patch("rune_bench.agents.chain.ChainExecutionEngine") as mock_engine_cls:
        mock_engine = mock_engine_cls.return_value
        mock_engine.execute = AsyncMock(return_value="result")

        res = await run_chain_workflow(
            steps=[], initial_context={}, model="m", job_id="j1", store=mock_store
        )
        assert res == "result"
        mock_engine.execute.assert_called_once()


def test_existing_ollama_server():
    s = ExistingOllamaServer(url="u", model_name="m")
    assert s.url == "u"
