# SPDX-License-Identifier: Apache-2.0
import pytest
from unittest.mock import MagicMock, patch
from rune_bench.resources.vastai.provider import VastAIProvider
from rune_bench.resources.base import ProvisioningResult

@pytest.fixture
def provider():
    mock_sdk = MagicMock()
    return VastAIProvider(
        mock_sdk,
        template_hash="h",
        min_dph=0.1,
        max_dph=1.0,
        reliability=0.9,
        stop_on_teardown=True
    )

@pytest.mark.asyncio
async def test_vastai_provider_provision(provider):
    mock_result = MagicMock()
    mock_result.backend_url = "http://u"
    mock_result.model_name = "m"
    mock_result.contract_id = 123
    
    with patch("rune_bench.workflows.provision_vastai_backend", return_value=mock_result):
        res = await provider.provision()
        assert res.backend_url == "http://u"
        assert res.provider_handle == 123

@pytest.mark.asyncio
async def test_vastai_provider_teardown(provider):
    res = ProvisioningResult(backend_url="http://u", model="m", provider_handle=123)
    with patch("rune_bench.workflows.stop_vastai_instance") as mock_stop:
        await provider.teardown(res)
        assert mock_stop.called

@pytest.mark.asyncio
async def test_vastai_provider_teardown_no_stop(provider):
    provider._stop_on_teardown = False
    res = ProvisioningResult(backend_url="http://u", model="m", provider_handle=123)
    with patch("rune_bench.workflows.stop_vastai_instance") as mock_stop:
        await provider.teardown(res)
        assert not mock_stop.called
