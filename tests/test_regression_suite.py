# SPDX-License-Identifier: Apache-2.0
import pytest

from rune_bench.common.http_client import normalize_url


@pytest.mark.regression
def test_regression_normalize_url_keeps_https_scheme() -> None:
    assert (
        normalize_url("https://example.com:11434", "ollama")
        == "https://example.com:11434"
    )


@pytest.mark.regression
def test_regression_normalize_url_adds_http_scheme() -> None:
    assert normalize_url("localhost:11434", "ollama") == "http://localhost:11434"


@pytest.mark.regression
def test_regression_normalize_url_rejects_empty() -> None:
    with pytest.raises(RuntimeError):
        normalize_url("", "ollama")
