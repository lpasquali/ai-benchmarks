from unittest.mock import MagicMock

import pytest

from rune_bench.vastai.instance import InstanceManager
from rune_bench.vastai.offer import OfferFinder
from rune_bench.vastai.template import TemplateLoader


def test_offer_finder_find_best_success():
    sdk = MagicMock()
    sdk.search_offers.return_value = [{"id": 123, "gpu_total_ram": 24576, "dph": 2.5}]

    offer = OfferFinder(sdk).find_best(min_dph=2.0, max_dph=3.0, reliability=0.99)

    assert offer.offer_id == 123
    assert offer.total_vram_mb == 24576
    sdk.search_offers.assert_called_once()


def test_offer_finder_find_best_no_matches_raises():
    sdk = MagicMock()
    sdk.search_offers.return_value = []

    with pytest.raises(RuntimeError, match="No matching offers found"):
        OfferFinder(sdk).find_best(min_dph=2.0, max_dph=3.0, reliability=0.99)


def test_template_loader_load_success_appends_workspace_mount():
    sdk = MagicMock()
    sdk.show_templates.return_value = [
        {"id": "abc", "env": "-e FOO=bar", "image": "ollama:latest"}
    ]

    tpl = TemplateLoader(sdk).load("abc")

    assert tpl.image == "ollama:latest"
    assert "-e FOO=bar" in tpl.env
    assert "-v /workspace" in tpl.env


def test_template_loader_load_not_found_raises():
    sdk = MagicMock()
    sdk.show_templates.return_value = [{"id": "other"}]

    with pytest.raises(RuntimeError, match="Template 'missing' not found"):
        TemplateLoader(sdk).load("missing")


def test_instance_manager_reuse_selects_best_running_candidate():
    sdk = MagicMock()
    sdk.show_instances.return_value = [
        {
            "id": 1,
            "actual_status": "running",
            "gpu_total_ram": 24576,
            "dph_total": 2.8,
            "reliability": 0.995,
        },
        {
            "id": 2,
            "actual_status": "running",
            "gpu_total_ram": 16384,
            "dph_total": 2.4,
            "reliability": 0.999,
        },
        {
            "id": 3,
            "actual_status": "stopped",
            "gpu_total_ram": 65536,
            "dph_total": 2.6,
            "reliability": 1.0,
        },
    ]

    selected = InstanceManager(sdk).find_reusable_running_instance(
        min_dph=2.0,
        max_dph=3.0,
        reliability=0.99,
    )

    assert selected is not None
    assert selected["id"] == 1


def test_instance_manager_pull_model_requires_ollama_url():
    with pytest.raises(RuntimeError, match="missing Ollama URL"):
        InstanceManager(MagicMock()).pull_model(contract_id=123, model_name="foo:1", ollama_url=None)


def test_instance_manager_pull_model_wraps_ollama_errors(monkeypatch):
    manager = InstanceManager(MagicMock())

    class _FailingClient:
        def __init__(self, _url):
            pass

        def load_model(self, _name):
            raise RuntimeError("boom")

    monkeypatch.setattr("rune_bench.vastai.instance.OllamaClient", _FailingClient)

    with pytest.raises(RuntimeError, match="Model pull via Ollama API failed"):
        manager.pull_model(contract_id=999, model_name="foo:1", ollama_url="http://fake:11434")


def test_build_connection_details_extracts_direct_and_proxy_urls():
    info = {
        "actual_status": "running",
        "ssh_host": "1.2.3.4",
        "ssh_port": 22,
        "machine_id": "m-abc",
        "ports": {
            "ollama": [{"HostIp": "1.2.3.4", "HostPort": "11434"}],
            "web": [{"HostIp": "1.2.3.4", "HostPort": "8080"}],
        },
    }

    details = InstanceManager.build_connection_details(321, info)

    assert details.contract_id == 321
    assert details.status == "running"
    assert details.ssh_host == "1.2.3.4"
    assert len(details.service_urls) == 2
    assert details.service_urls[0]["direct"].startswith("http://1.2.3.4:")
    assert details.service_urls[0]["proxy"].startswith("https://server-m-abc.vast.ai:")


def test_destroy_instance_and_related_storage_is_fully_mockable(monkeypatch):
    manager = InstanceManager(MagicMock())

    monkeypatch.setattr(
        manager,
        "_fetch_instance",
        lambda _cid: {
            "volume_id": "vol-1",
            "volumes": [{"id": "vol-2"}],
        },
    )

    destroyed_instances = []
    destroyed_volumes = []

    monkeypatch.setattr(manager, "_destroy_instance", lambda cid: destroyed_instances.append(str(cid)))
    monkeypatch.setattr(manager, "_destroy_volume", lambda vid: destroyed_volumes.append(str(vid)))
    monkeypatch.setattr(manager, "_wait_until_instance_absent", lambda _cid: True)
    monkeypatch.setattr(manager, "_verify_volumes_deleted", lambda vids: (True, f"deleted: {', '.join(vids)}"))

    result = manager.destroy_instance_and_related_storage(contract_id=123)

    assert result.contract_id == 123
    assert result.destroyed_instance is True
    assert result.verification_ok is True
    assert destroyed_instances == ["123"]
    assert sorted(destroyed_volumes) == ["vol-1", "vol-2"]


def test_extract_related_volume_ids_collects_all_known_shapes():
    instance = {
        "volume_id": "vol-a",
        "volume": {"id": "vol-b"},
        "volumes": [{"id": "vol-c"}],
    }

    ids = InstanceManager._extract_related_volume_ids(instance)
    assert ids == ["vol-a", "vol-b", "vol-c"]
