# SPDX-License-Identifier: Apache-2.0
"""Browser-Use agentic runner — delegates to the Browser-Use driver.

Scope:      Legal/Ops  |  Rank 4  |  Rating 4.0
Capability: AI-powered browser automation for web tasks.
Docs:       https://github.com/browser-use/browser-use
Ecosystem:  OSS Community
"""

from rune_bench.drivers.browseruse import BrowserUseDriverClient

# Alias for registry resolution
BrowserUseRunner = BrowserUseDriverClient

__all__ = ["BrowserUseRunner", "BrowserUseDriverClient"]
