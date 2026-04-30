# SPDX-License-Identifier: Apache-2.0
"""XBOW agentic runner — delegates to the XBOW driver.

Scope:      Cybersec  |  Rank 5  |  Rating 3.5
Capability: Autonomous web vulnerability discovery/exploit.
Docs:       https://xbow.com/
Ecosystem:  Sec Automation
"""

from rune_bench.drivers.xbow import XBOWDriverClient

# Backwards-compatible alias so existing imports of XBOWRunner keep working.
XBOWRunner = XBOWDriverClient

__all__ = ["XBOWRunner", "XBOWDriverClient"]
