# SPDX-License-Identifier: Apache-2.0
"""Tests for enterprise drivers.

Each driver must:
1. Raise RuntimeError with the onboarding URL when the API key env var is unset.
2. Return correct metadata from ``_handle_info()`` with ``"status": "active"``.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Driver specs: (module_path, client_class, env_var, onboarding_url, driver_name)
# ---------------------------------------------------------------------------
_DRIVERS = [
    (
        "rune_bench.drivers.radiant",
        "RadiantDriverClient",
        "RUNE_RADIANT_API_KEY",
        "https://radiantsecurity.ai/",
        "radiant",
    ),
    (
        "rune_bench.drivers.xbow",
        "XbowDriverClient",
        "RUNE_XBOW_API_KEY",
        "https://xbow.com/",
        "xbow",
    ),
    (
        "rune_bench.drivers.harvey",
        "HarveyDriverClient",
        "RUNE_HARVEY_API_KEY",
        "https://www.harvey.ai/",
        "harvey",
    ),
    (
        "rune_bench.drivers.spellbook",
        "SpellbookDriverClient",
        "RUNE_SPELLBOOK_API_KEY",
        "https://www.spellbook.legal/",
        "spellbook",
    ),
    (
        "rune_bench.drivers.sierra",
        "SierraDriverClient",
        "RUNE_SIERRA_API_KEY",
        "https://sierra.ai/",
        "sierra",
    ),
    (
        "rune_bench.drivers.skillfortify",
        "SkillfortifyDriverClient",
        "RUNE_SKILLFORTIFY_API_KEY",
        "https://skillfortify.com/",
        "skillfortify",
    ),
    (
        "rune_bench.drivers.krea",
        "KreaDriverClient",
        "RUNE_KREA_API_KEY",
        "https://www.krea.ai/",
        "krea",
    ),
    (
        "rune_bench.drivers.midjourney",
        "MidjourneyDriverClient",
        "RUNE_MIDJOURNEY_API_KEY",
        "https://docs.midjourney.com/",
        "midjourney",
    ),
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure none of the enterprise API key env vars are set."""
    for _, _, env_var, _, _ in _DRIVERS:
        monkeypatch.delenv(env_var, raising=False)


# ---------------------------------------------------------------------------
# Client-level tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path, class_name, env_var, onboarding_url, driver_name",
    _DRIVERS,
    ids=[s[4] for s in _DRIVERS],
)
def test_ask_raises_without_api_key(
    module_path: str,
    class_name: str,
    env_var: str,
    onboarding_url: str,
    driver_name: str,
) -> None:
    """``ask()`` must raise RuntimeError when the API key env var is missing."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    client = cls(transport=MagicMock())

    with pytest.raises(RuntimeError, match=env_var):
        client.ask("test question", "test-model")


# ---------------------------------------------------------------------------
# __main__ handler-level tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path, class_name, env_var, onboarding_url, driver_name",
    _DRIVERS,
    ids=[s[4] for s in _DRIVERS],
)
def test_handle_info_returns_active_status(
    module_path: str,
    class_name: str,
    env_var: str,
    onboarding_url: str,
    driver_name: str,
) -> None:
    """``_handle_info()`` must return ``status: active`` and correct metadata."""
    main_mod = importlib.import_module(f"{module_path}.__main__")
    info = main_mod._handle_info({})

    assert info["name"] == driver_name
    assert info["status"] == "active"


@pytest.mark.parametrize(
    "module_path, class_name, env_var, onboarding_url, driver_name",
    _DRIVERS,
    ids=[s[4] for s in _DRIVERS],
)
def test_main_loop_success(
    module_path: str,
    class_name: str,
    env_var: str,
    onboarding_url: str,
    driver_name: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``main()`` must process JSON requests from stdin and write to stdout."""
    import json
    import io

    main_mod = importlib.import_module(f"{module_path}.__main__")

    # Test 'info' action
    input_data = json.dumps({"action": "info", "id": "123"}) + "\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(input_data))

    main_mod.main()

    out, _ = capsys.readouterr()
    resp = json.loads(out.strip())
    assert resp["status"] == "ok"
    assert resp["id"] == "123"
    assert resp["result"]["name"] == driver_name
