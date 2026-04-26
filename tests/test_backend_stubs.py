# SPDX-License-Identifier: Apache-2.0
import pytest
from rune_bench.backends.base import BackendCredentials
from rune_bench.backends.openai import OpenAIBackend
from rune_bench.backends.bedrock import BedrockBackend


def test_openai_backend_stub():
    credentials = BackendCredentials(api_key="test-key")
    backend = OpenAIBackend(credentials=credentials)

    with pytest.raises(
        NotImplementedError, match="OpenAIBackend is not yet implemented"
    ):
        backend.get_model_capabilities("gpt-4o")


def test_bedrock_backend_requires_region():
    credentials = BackendCredentials(api_key=None, extra={})

    with pytest.raises(
        ValueError,
        match="BedrockBackend requires a region",
    ):
        BedrockBackend(credentials=credentials)


def test_bedrock_backend_active():
    credentials = BackendCredentials(api_key=None, extra={"region": "us-east-1"})
    # Mock boto3 to avoid AWS calls during init
    import unittest.mock
    with unittest.mock.patch("boto3.client"):
        backend = BedrockBackend(credentials=credentials)
        assert backend.base_url == "us-east-1"
