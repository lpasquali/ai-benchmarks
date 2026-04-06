import pytest
import httpx
from rune_bench.resources.vastai.sdk import VastAI

@pytest.fixture
def vastai():
    return VastAI("test-api-key")

def test_headers(vastai):
    headers = vastai._get_headers()
    assert headers["Authorization"] == "Bearer test-api-key"
    assert headers["Accept"] == "application/json"

def test_search_offers(vastai, respx_mock):
    respx_mock.get("https://console.vast.ai/api/v0/bundles/?q=%7B%22foo%22%3A%20%22bar%22%7D").respond(
        json={"offers": [{"id": 123}]}
    )
    offers = vastai.search_offers({"foo": "bar"})
    assert len(offers) == 1
    assert offers[0]["id"] == 123

def test_search_offers_invalid_query(vastai):
    with pytest.raises(ValueError):
        vastai.search_offers("invalid")

def test_show_templates(vastai, respx_mock):
    respx_mock.get("https://console.vast.ai/api/v0/users/current/templates/").respond(
        json=[{"id": 1}]
    )
    templates = vastai.show_templates()
    assert len(templates) == 1
    assert templates[0]["id"] == 1

def test_create_instance(vastai, respx_mock):
    respx_mock.put("https://console.vast.ai/api/v0/asks/456/").respond(
        json={"success": True}
    )
    res = vastai.create_instance(id="456", disk=10.0, env="-e FOO=bar", image="ubuntu:latest")
    assert res["success"] is True

def test_show_instances(vastai, respx_mock):
    respx_mock.get("https://console.vast.ai/api/v0/instances/").respond(
        json={"instances": [{"id": 456}]}
    )
    instances = vastai.show_instances()
    assert len(instances) == 1
    assert instances[0]["id"] == 456

def test_destroy_instance(vastai, respx_mock):
    respx_mock.delete("https://console.vast.ai/api/v0/instances/456/").respond(
        json={"success": True}
    )
    res = vastai.destroy_instance("456")
    assert res["success"] is True

def test_show_volumes(vastai, respx_mock):
    respx_mock.get("https://console.vast.ai/api/v0/volumes/").respond(
        json={"volumes": [{"id": "vol-1"}]}
    )
    vols = vastai.show_volumes()
    assert len(vols) == 1
    assert vols[0]["id"] == "vol-1"

def test_show_volumes_direct_list(vastai, respx_mock):
    respx_mock.get("https://console.vast.ai/api/v0/volumes/").respond(
        json=[{"id": "vol-2"}]
    )
    vols = vastai.show_volumes()
    assert len(vols) == 1
    assert vols[0]["id"] == "vol-2"

def test_destroy_volume(vastai, respx_mock):
    respx_mock.delete("https://console.vast.ai/api/v0/volumes/vol-1/").respond(
        json={"success": True}
    )
    res = vastai.destroy_volume("vol-1")
    assert res["success"] is True
