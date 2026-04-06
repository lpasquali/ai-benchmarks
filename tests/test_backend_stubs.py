import pytest
from rune_bench.backends.base import BackendCredentials
from rune_bench.backends.openai import OpenAIBackend
from rune_bench.backends.bedrock import BedrockBackend

def test_openai_backend_stub():
    credentials = BackendCredentials(api_key="test-key")
    backend = OpenAIBackend(credentials=credentials)
    
    with pytest.raises(NotImplementedError, match="OpenAIBackend is not yet implemented"):
        backend.get_model_capabilities("gpt-4o")

def test_bedrock_backend_requires_region():
    credentials = BackendCredentials(api_key=None, extra={})
    
    with pytest.raises(ValueError, match="BedrockBackend requires BackendCredentials.extra\\['region'\\] to be set."):
        BedrockBackend(credentials=credentials)

def test_bedrock_backend_stub():
    credentials = BackendCredentials(api_key=None, extra={"region": "us-east-1"})
    backend = BedrockBackend(credentials=credentials)
    
    with pytest.raises(NotImplementedError, match="BedrockBackend is not yet implemented"):
        backend.get_model_capabilities("anthropic.claude-3-5-sonnet")