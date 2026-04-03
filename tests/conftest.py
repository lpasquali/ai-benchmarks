"""Pytest configuration: stub optional dependencies unavailable outside the CI venv."""

import sys
import types


def _stub_holmes() -> None:
    """Inject a minimal ``holmes`` stub into sys.modules.

    ``rune_bench.agents.sre.holmes`` does a top-level ``import holmes`` which
    fails when holmesgpt is not installed.  Tests that exercise the real
    HolmesRunner use monkeypatch to replace the relevant symbols; the stub
    just satisfies the import so collection doesn't crash.
    """
    if "holmes" in sys.modules:
        return

    holmes_mod = types.ModuleType("holmes")
    core_mod = types.ModuleType("holmes.core")
    llm_mod = types.ModuleType("holmes.core.llm")
    main_mod = types.ModuleType("holmes.main")

    sys.modules["holmes"] = holmes_mod
    sys.modules["holmes.core"] = core_mod
    sys.modules["holmes.core.llm"] = llm_mod
    sys.modules["holmes.main"] = main_mod


_stub_holmes()
