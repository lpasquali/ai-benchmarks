# SPDX-License-Identifier: Apache-2.0
import pytest
from rune_bench.backends.k8s_inference import K8sInferenceBackend

def test_k8s_inference_backend_init():
    with pytest.raises(ValueError, match="requires a base_url"):
        K8sInferenceBackend()

    backend = K8sInferenceBackend(base_url="http://gateway.local")
    assert backend.base_url == "http://gateway.local"

def test_k8s_inference_backend_get_model_capabilities():
    backend = K8sInferenceBackend(base_url="http://gateway.local")
    caps = backend.get_model_capabilities("my-model")
    assert caps.model_name == "my-model"
    assert caps.context_window is None

def test_k8s_inference_backend_list_models():
    backend = K8sInferenceBackend(base_url="http://gateway.local")
    assert backend.list_models() == []

def test_k8s_inference_backend_list_running_models():
    backend = K8sInferenceBackend(base_url="http://gateway.local")
    assert backend.list_running_models() == []

def test_k8s_inference_backend_normalize_model_name():
    backend = K8sInferenceBackend(base_url="http://gateway.local")
    assert backend.normalize_model_name("k8s/vllm-mistral") == "vllm-mistral"
    assert backend.normalize_model_name("gateway/ollama-llama3") == "ollama-llama3"
    assert backend.normalize_model_name("plain-model") == "plain-model"
    assert backend.normalize_model_name("  k8s/spaces  ") == "spaces"

def test_k8s_inference_backend_warmup():
    backend = K8sInferenceBackend(base_url="http://gateway.local")
    result = backend.warmup("k8s/my-model")
    assert result == "my-model"
