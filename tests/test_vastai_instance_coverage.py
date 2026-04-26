# SPDX-License-Identifier: Apache-2.0
import pytest
from unittest.mock import MagicMock, patch
from rune_bench.resources.vastai.instance import InstanceManager
from rune_bench.resources.vastai.template import Template
from rune_bench.common.models import SelectedModel


@pytest.fixture
def manager():
    mock_sdk = MagicMock()
    return InstanceManager(mock_sdk)


def test_create_instance_error(manager):
    template = Template(env="", image="i", raw={})
    model = SelectedModel(name="m", vram_mb=100, required_disk_gb=10)
    manager._sdk.create_instance.side_effect = Exception("creation failed")
    with pytest.raises(RuntimeError, match="Instance creation failed"):
        manager.create(offer_id=1, model=model, template=template)


def test_create_instance_no_contract_id(manager):
    template = Template(env="", image="i", raw={})
    model = SelectedModel(name="m", vram_mb=100, required_disk_gb=10)
    manager._sdk.create_instance.return_value = {"something": "else"}
    with pytest.raises(RuntimeError, match="Could not parse contract id"):
        manager.create(offer_id=1, model=model, template=template)


def test_wait_until_running_timeout(manager):
    manager._sdk.show_instances.return_value = [{"id": 1, "state": "loading"}]
    with patch("time.sleep"):
        with patch("rune_bench.resources.vastai.instance._POLL_MAX_ATTEMPTS", 2):
            with pytest.raises(RuntimeError, match="did not reach 'running' state"):
                manager.wait_until_running(1)


def test_list_instances_error(manager):
    manager._sdk.show_instances.side_effect = Exception("list failed")
    with pytest.raises(RuntimeError, match="Failed to list Vast.ai instances"):
        manager.list_instances()


def test_list_instances_non_list(manager):
    manager._sdk.show_instances.return_value = "not a list"
    assert manager.list_instances() == []


def test_find_reusable_instance_filters(manager):
    instances = [
        {"id": 1, "dph_total": 2.0, "reliability2": 0.8},  # dph out of range
        {"id": 2, "dph_total": 0.5, "reliability2": 0.5},  # reliability low
    ]
    manager._sdk.show_instances.return_value = instances
    assert (
        manager.find_reusable_running_instance(
            min_dph=0.1, max_dph=1.0, reliability=0.9
        )
        is None
    )


def test_stop_instance(manager):
    manager.stop_instance(1)
    assert manager._sdk.destroy_instance.called


def test_destroy_instance_and_related_storage_volume_error(manager):
    # Mock _fetch_instance
    manager._sdk.show_instances.return_value = [{"id": 1, "volume_id": "v1"}]
    # Mock destroy_instance
    # Mock _destroy_volume to fail
    manager._sdk.destroy_volume.side_effect = RuntimeError("volume destroy failed")

    with patch("time.sleep"):
        # We need _wait_until_instance_absent to return True
        # First call to show_instances returns it, subsequent returns None
        manager._sdk.show_instances.side_effect = [
            [{"id": 1, "volume_id": "v1"}],  # _fetch_instance
            [],  # _wait_until_instance_absent
            [],  # _verify_volumes_deleted
        ]
        res = manager.destroy_instance_and_related_storage(1)
        assert res.destroyed_instance is True
        assert res.destroyed_volume_ids == []


def test_pull_model_missing_url(manager):
    with pytest.raises(RuntimeError, match="missing backend URL"):
        manager.pull_model(1, "m", backend_url=None)


def test_pull_model_fail(manager):
    with patch("rune_bench.resources.vastai.instance.get_backend") as mock_get:
        mock_backend = MagicMock()
        mock_backend.warmup.side_effect = RuntimeError("warmup fail")
        mock_get.return_value = mock_backend
        with pytest.raises(RuntimeError, match="Model pull via Ollama API failed"):
            manager.pull_model(1, "m", backend_url="http://u")


def test_wait_until_instance_absent_timeout(manager):
    manager._sdk.show_instances.return_value = [{"id": 1}]
    with patch("time.monotonic", side_effect=[0, 1000]):
        assert manager._wait_until_instance_absent(1) is False


def test_list_volumes_dict_response(manager):
    manager._sdk.show_volumes.return_value = {"volumes": [{"id": "v1"}]}
    assert manager._list_volumes_optional() == [{"id": "v1"}]


def test_list_volumes_error(manager):
    manager._sdk.show_volumes.side_effect = Exception("fail")
    assert manager._list_volumes_optional() is None


def test_verify_volumes_deleted_no_list(manager):
    manager._sdk.show_volumes.return_value = None
    res, msg = manager._verify_volumes_deleted(["v1"])
    assert res is False
    assert "could not verify" in msg


def test_verify_volumes_deleted_remaining(manager):
    manager._sdk.show_volumes.return_value = [{"id": "v1"}]
    res, msg = manager._verify_volumes_deleted(["v1"])
    assert res is False
    assert "remaining volumes" in msg


def test_extract_related_volume_ids_edge_cases():
    IM = InstanceManager
    assert IM._extract_related_volume_ids(None) == []
    assert IM._extract_related_volume_ids({"volume": {"id": "v1"}}) == ["v1"]
    assert IM._extract_related_volume_ids({"volumes": [{"id": "v2"}]}) == ["v2"]


def test_first_float_invalid(manager):
    assert manager._first_float({"a": "not-a-float"}, ("a",)) is None
