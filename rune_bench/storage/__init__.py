# SPDX-License-Identifier: Apache-2.0
"""Pluggable storage adapters for the RUNE job store.

This package exposes :class:`StoragePort` (the protocol every adapter must
satisfy) and :func:`make_storage` (a URL-based factory). Today only a
SQLite adapter ships; Postgres and other backends will plug in here without
touching the API server.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from rune_bench.storage.base import StoragePort
from rune_bench.storage.migrator import Migrator
from rune_bench.storage.sqlite import JobRecord, SQLiteStorageAdapter

try:
    from rune_bench.storage.postgres import PostgresStorageAdapter
except ImportError:  # pragma: no cover - exercised when pg extras are absent
    PostgresStorageAdapter = None  # type: ignore[assignment]

__all__ = [
    "JobRecord",
    "Migrator",
    "PostgresStorageAdapter",
    "SQLiteStorageAdapter",
    "StoragePort",
    "default_storage_url",
    "make_storage",
    "resolve_storage_url",
]


_SUPPORTED_SCHEMES = ("sqlite://", "postgresql://")
_WINDOWS_ABS_PATH_RE = re.compile(r"^/[A-Za-z]:/")


def _default_sqlite_db_path() -> str:
    if sys.platform == "win32":
        base_dir = os.environ.get("LOCALAPPDATA") or str(
            Path.home() / "AppData" / "Local"
        )
    elif sys.platform == "darwin":
        base_dir = str(Path.home() / "Library" / "Application Support")
    else:
        base_dir = os.environ.get("XDG_DATA_HOME") or str(
            Path.home() / ".local" / "share"
        )
    return str(Path(base_dir) / "rune" / "jobs.db")


def _sqlite_path_to_url(db_path: str) -> str:
    normalized = db_path.strip()
    if normalized in ("", ":memory:"):
        return "sqlite:///:memory:"
    path = Path(normalized)
    if path.is_absolute():
        return f"sqlite:///{path.as_posix()}"
    return f"sqlite:///./{path.as_posix()}"


def default_storage_url() -> str:
    """Return the built-in SQLite storage URL used when nothing is configured."""
    return _sqlite_path_to_url(_default_sqlite_db_path())


def resolve_storage_url(
    db_url: str | None = None, *, legacy_db_path: str | None = None
) -> str:
    """Resolve storage config with back-compat for ``RUNE_API_DB_PATH``.

    Precedence:

    1. Explicit ``db_url`` argument
    2. ``RUNE_DB_URL``
    3. Explicit ``legacy_db_path`` argument
    4. ``RUNE_API_DB_PATH``
    5. built-in SQLite default under the platform data directory
    """
    explicit_url = (db_url or "").strip()
    if explicit_url:
        return explicit_url

    env_url = os.environ.get("RUNE_DB_URL", "").strip()
    if env_url:
        return env_url

    legacy_path = (legacy_db_path or "").strip()
    if not legacy_path:
        legacy_path = os.environ.get("RUNE_API_DB_PATH", "").strip()
    if legacy_path:
        return _sqlite_path_to_url(legacy_path)

    return default_storage_url()


def make_storage(url: str) -> StoragePort:
    """Instantiate a :class:`StoragePort` from a URL.

    Supported schemes:

    * ``sqlite:///absolute/path/to/file.db`` — file-backed SQLite
    * ``sqlite:///:memory:`` — in-memory SQLite (test fixtures)
    * ``postgresql://user:pass@host:5432/db`` — PostgreSQL via psycopg3

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
        elif path.startswith("/./") or path.startswith("/../"):
            db_path = path[1:]
        elif _WINDOWS_ABS_PATH_RE.match(path):
            db_path = path[1:]
        else:
            # urlparse strips the leading '//' so an absolute file path
            # arrives here as "/abs/path" — SQLite consumes that verbatim.
            db_path = path
        return SQLiteStorageAdapter(db_path)
    if scheme in ("postgres", "postgresql", "postgresql+psycopg", "postgres+psycopg"):
        if PostgresStorageAdapter is None:
            raise RuntimeError(
                "postgresql storage requires psycopg; install 'rune-bench[pg]'"
            )
        return PostgresStorageAdapter(url)
    raise RuntimeError(
        f"unsupported storage URL scheme {parsed.scheme!r}; "
        f"supported: {', '.join(_SUPPORTED_SCHEMES)}"
    )
