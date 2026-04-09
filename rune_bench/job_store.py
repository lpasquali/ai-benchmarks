# SPDX-License-Identifier: Apache-2.0
"""Backwards-compatibility shim. Use ``rune_bench.storage`` instead."""
from rune_bench.storage.sqlite import JobRecord  # noqa: F401
from rune_bench.storage.sqlite import SQLiteStorageAdapter  # noqa: F401
from rune_bench.storage.sqlite import SQLiteStorageAdapter as JobStore  # noqa: F401

__all__ = ["JobRecord", "JobStore", "SQLiteStorageAdapter"]
