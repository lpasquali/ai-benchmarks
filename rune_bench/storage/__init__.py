# SPDX-License-Identifier: Apache-2.0
"""Pluggable storage adapters for the RUNE job store.

This package exposes :class:`StoragePort` (the protocol every adapter must
satisfy) and :func:`make_storage` (a URL-based factory). Today only a
SQLite adapter ships; Postgres and other backends will plug in here without
touching the API server.
"""

from __future__ import annotations

from urllib.parse import urlparse

from rune_bench.storage.base import StoragePort
from rune_bench.storage.migrator import Migrator
from rune_bench.storage.sqlite import JobRecord, SQLiteStorageAdapter

__all__ = [
    "JobRecord",
    "Migrator",
    "SQLiteStorageAdapter",
    "StoragePort",
    "make_storage",
]


_SUPPORTED_SCHEMES = ("sqlite://",)


def make_storage(url: str) -> StoragePort:
    """Instantiate a :class:`StoragePort` from a URL.

    Supported schemes:

    * ``sqlite:///absolute/path/to/file.db`` — file-backed SQLite
    * ``sqlite:///:memory:`` — in-memory SQLite (test fixtures)

    Unknown schemes raise :class:`RuntimeError` with the list of supported
    schemes, so boot-time config errors surface with an actionable message.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme in ("sqlite", "sqlite+pysqlite"):
        # sqlite:///:memory:   -> parsed.path == "/:memory:"
        # sqlite:///abs/path   -> parsed.path == "/abs/path"
        path = parsed.path
        if path in ("/:memory:", ":memory:", ""):
            db_path = ":memory:"
        else:
            # urlparse strips the leading '//' so an absolute file path
            # arrives here as "/abs/path" — SQLite consumes that verbatim.
            db_path = path
        return SQLiteStorageAdapter(db_path)
    raise RuntimeError(
        f"unsupported storage URL scheme {parsed.scheme!r}; "
        f"supported: {', '.join(_SUPPORTED_SCHEMES)}"
    )
