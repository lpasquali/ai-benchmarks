# SPDX-License-Identifier: Apache-2.0
import pytest
from unittest.mock import MagicMock, patch
from rune_bench.backends.bedrock import BedrockBackend
from rune_bench.backends.base import BackendCredentials


@patch("boto3.client")
def test_bedrock_init_region_sources(mock_boto):
    # Source 1: base_url (first arg in factory)
    b1 = BedrockBackend(base_url="us-west-2")
    assert b1.base_url == "us-west-2"
    mock_boto.assert_any_call("bedrock", region_name="us-west-2")

    # Source 2: credentials.extra["region"]
    creds = BackendCredentials(extra={"region": "eu-central-1"})
    b2 = BedrockBackend(credentials=creds)
    assert b2.base_url == "eu-central-1"

    # Source 3: region kwarg
    b3 = BedrockBackend(region="ap-southeast-1")
    assert b3.base_url == "ap-southeast-1"


def test_bedrock_init_missing_region():
    with pytest.raises(ValueError, match="requires a region"):
        BedrockBackend()


def test_bedrock_normalize_model_name():
    b = BedrockBackend(base_url="us-east-1")
    assert (
        b.normalize_model_name("bedrock/anthropic.claude-v2") == "anthropic.claude-v2"
    )
    assert (
        b.normalize_model_name("aws/amazon.titan-text-express-v1")
        == "amazon.titan-text-express-v1"
    )
    assert b.normalize_model_name("meta.llama3-8b") == "meta.llama3-8b"


def test_bedrock_warmup():
    b = BedrockBackend(base_url="us-east-1")
    # Should return normalized name and be a no-op
    assert b.warmup("bedrock/m1") == "m1"


@patch("boto3.client")
def test_bedrock_get_model_capabilities_success(mock_boto):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    mock_client.get_foundation_model.return_value = {
        "modelDetails": {"modelId": "m1", "maxContextTokens": 8192}
    }

    b = BedrockBackend(base_url="us-east-1")
    caps = b.get_model_capabilities("m1")
    assert caps.model_name == "m1"
    assert caps.context_window == 8192


@patch("boto3.client")
def test_bedrock_get_model_capabilities_fail(mock_boto):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    mock_client.get_foundation_model.side_effect = Exception("aws down")

    b = BedrockBackend(base_url="us-east-1")
    caps = b.get_model_capabilities("m1")
    assert caps.model_name == "m1"
    assert caps.context_window is None


@patch("boto3.client")
def test_bedrock_list_models(mock_boto):
    mock_client = MagicMock()
    mock_boto.return_value = mock_client
    mock_client.list_foundation_models.return_value = {
        "modelSummaries": [{"modelId": "model-1"}, {"modelId": "model-2"}]
    }

    b = BedrockBackend(base_url="us-east-1")
    models = b.list_models()
    assert models == ["model-1", "model-2"]


def test_bedrock_list_running_models():
    b = BedrockBackend(base_url="us-east-1")
    assert b.list_running_models() == []


@patch("boto3.client")
def test_bedrock_invoke(mock_boto):
    mock_runtime = MagicMock()
    # Second call to boto3.client in __init__ is for bedrock-runtime
    mock_boto.side_effect = [MagicMock(), mock_runtime]

    mock_body = MagicMock()
    mock_body.read.return_value = b'{"output": "text"}'
    mock_runtime.invoke_model.return_value = {"body": mock_body}

    b = BedrockBackend(base_url="us-east-1")
    res = b.invoke("m1", {"prompt": "p"})
    assert res == {"output": "text"}
    mock_runtime.invoke_model.assert_called_once()
