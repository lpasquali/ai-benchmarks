"""Tests for enterprise stub drivers.

Each stub must:
1. Raise RuntimeError with the onboarding URL when the API key env var is unset.
2. Return correct metadata from ``_handle_info()`` with ``"status": "enterprise_stub"``.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Driver specs: (module_path, client_class, env_var, onboarding_url, driver_name)
# ---------------------------------------------------------------------------
_STUBS = [
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
    for _, _, env_var, _, _ in _STUBS:
        monkeypatch.delenv(env_var, raising=False)


# ---------------------------------------------------------------------------
# Client-level tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path, class_name, env_var, onboarding_url, driver_name",
    _STUBS,
    ids=[s[4] for s in _STUBS],
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


@pytest.mark.parametrize(
    "module_path, class_name, env_var, onboarding_url, driver_name",
    _STUBS,
    ids=[s[4] for s in _STUBS],
)
def test_ask_error_contains_onboarding_url(
    module_path: str,
    class_name: str,
    env_var: str,
    onboarding_url: str,
    driver_name: str,
) -> None:
    """The RuntimeError message must contain the onboarding URL."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    client = cls(transport=MagicMock())

    with pytest.raises(RuntimeError, match=onboarding_url.replace(".", r"\.")):
        client.ask("test question", "test-model")


# ---------------------------------------------------------------------------
# __main__ handler-level tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path, class_name, env_var, onboarding_url, driver_name",
    _STUBS,
    ids=[s[4] for s in _STUBS],
)
def test_handle_info_returns_enterprise_stub(
    module_path: str,
    class_name: str,
    env_var: str,
    onboarding_url: str,
    driver_name: str,
) -> None:
    """``_handle_info()`` must return ``status: enterprise_stub`` and correct metadata."""
    main_mod = importlib.import_module(f"{module_path}.__main__")
    info = main_mod._handle_info({})

    assert info["name"] == driver_name
    assert info["version"] == "1"
    assert info["status"] == "enterprise_stub"
    assert info["onboarding_url"] == onboarding_url
    assert "ask" in info["actions"]
    assert "info" in info["actions"]


@pytest.mark.parametrize(
    "module_path, class_name, env_var, onboarding_url, driver_name",
    _STUBS,
    ids=[s[4] for s in _STUBS],
)
def test_handle_ask_raises_not_implemented_with_key(
    module_path: str,
    class_name: str,
    env_var: str,
    onboarding_url: str,
    driver_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_handle_ask()`` must raise NotImplementedError when the API key is present."""
    monkeypatch.setenv(env_var, "dummy-key")
    main_mod = importlib.import_module(f"{module_path}.__main__")

    with pytest.raises(NotImplementedError, match="enterprise stub"):
        main_mod._handle_ask({"question": "test", "model": "test"})


@pytest.mark.parametrize(
    "module_path, class_name, env_var, onboarding_url, driver_name",
    _STUBS,
    ids=[s[4] for s in _STUBS],
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


@pytest.mark.parametrize(
    "module_path, class_name, env_var, onboarding_url, driver_name",
    _STUBS,
    ids=[s[4] for s in _STUBS],
)
def test_main_loop_unknown_action(
    module_path: str,
    class_name: str,
    env_var: str,
    onboarding_url: str,
    driver_name: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``main()`` must handle unknown actions by returning an error JSON."""
    import json
    import io

    main_mod = importlib.import_module(f"{module_path}.__main__")
    
    input_data = json.dumps({"action": "invalid", "id": "456"}) + "\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(input_data))
    
    main_mod.main()
    
    out, _ = capsys.readouterr()
    resp = json.loads(out.strip())
    assert resp["status"] == "error"
    assert resp["id"] == "456"
    assert "Unknown action" in resp["error"]



@pytest.mark.parametrize(
    "module_path, class_name, env_var, onboarding_url, driver_name",
    _STUBS,
    ids=[s[4] for s in _STUBS],
)
def test_client_ask_with_api_key(
    module_path: str,
    class_name: str,
    env_var: str,
    onboarding_url: str,
    driver_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Client ask() delegates to transport when the API key is set."""
    monkeypatch.setenv(env_var, "dummy-key")
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)

    transport = MagicMock()
    transport.call.return_value = {"answer": "stub answer"}
    client = cls(transport=transport)

    result = client.ask("q", "m", ollama_url="http://localhost:11434")

    assert result == "stub answer"
    transport.call.assert_called_once()


@pytest.mark.parametrize(
    "module_path, class_name, env_var, onboarding_url, driver_name",
    _STUBS,
    ids=[s[4] for s in _STUBS],
)
def test_main_loop_empty_lines(
    module_path: str,
    class_name: str,
    env_var: str,
    onboarding_url: str,
    driver_name: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``main()`` must silently skip empty lines."""
    import io

    main_mod = importlib.import_module(f"{module_path}.__main__")
    monkeypatch.setattr("sys.stdin", io.StringIO("\n   \n"))
    main_mod.main()
    assert capsys.readouterr().out.strip() == ""


@pytest.mark.parametrize(
    "module_path, class_name, env_var, onboarding_url, driver_name",
    _STUBS,
    ids=[s[4] for s in _STUBS],
)
def test_main_loop_invalid_json(
    module_path: str,
    class_name: str,
    env_var: str,
    onboarding_url: str,
    driver_name: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``main()`` must handle invalid JSON gracefully."""
    import io
    import json as _json

    main_mod = importlib.import_module(f"{module_path}.__main__")
    monkeypatch.setattr("sys.stdin", io.StringIO("not-json\n"))
    main_mod.main()
    resp = _json.loads(capsys.readouterr().out.strip())
    assert resp["status"] == "error"
