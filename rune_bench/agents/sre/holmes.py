# SPDX-License-Identifier: Apache-2.0
"""HolmesGPT agent for RUNE — thin re-export kept for backward compatibility.

The implementation has moved to :mod:`rune_bench.drivers.holmes`.
``HolmesRunner`` is now an alias for :class:`~rune_bench.drivers.holmes.HolmesDriverClient`.
"""

from rune_bench.drivers.holmes import HolmesDriverClient

# Backward-compatible alias — existing call-sites require no changes.
HolmesRunner = HolmesDriverClient

__all__ = ["HolmesRunner", "HolmesDriverClient"]
