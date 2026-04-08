# SPDX-License-Identifier: Apache-2.0
"""Browser-use agentic runner — delegates to the browser-use driver.

Scope:      Ops/Misc  |  Rank 2  |  Rating 4.5
Capability: LLM-driven browser automation for complex web tasks.
Docs:       https://browser-use.com/
Ecosystem:  OSS Community / Playwright
"""

from __future__ import annotations

from rune_bench.drivers.browser_use import BrowserUseDriverClient

# Alias for registry resolution
BrowserUseRunner = BrowserUseDriverClient

__all__ = ["BrowserUseRunner", "BrowserUseDriverClient"]
