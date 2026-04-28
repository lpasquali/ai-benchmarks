# SPDX-License-Identifier: Apache-2.0
import rune_bench.debug as debug


def test_set_debug_updates_flag_and_env(monkeypatch):
    monkeypatch.delenv("RUNE_DEBUG", raising=False)

    debug.set_debug(True)
    assert debug.is_debug_enabled() is True
    assert debug.os.environ["RUNE_DEBUG"] == "1"

    debug.set_debug(False)
    assert debug.is_debug_enabled() is False
    assert debug.os.environ["RUNE_DEBUG"] == "0"


def test_debug_log_only_when_enabled(capsys):
    debug.set_debug(False)
    debug.debug_log("hidden")
    out = capsys.readouterr()
    assert "hidden" not in out.err

    debug.set_debug(True)
    debug.debug_log("visible")
    out = capsys.readouterr()
    assert "[RUNE debug] visible" in out.err
