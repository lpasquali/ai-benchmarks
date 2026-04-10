# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`rune_bench.storage` factory and protocol conformance."""

from __future__ import annotations

import pytest

import rune_bench.storage as storage_module
from rune_bench.storage import (
    StoragePort,
    SQLiteStorageAdapter,
    default_storage_url,
    make_storage,
    resolve_storage_url,
)


def test_make_storage_sqlite_memory() -> None:
    store = make_storage("sqlite:///:memory:")

    assert isinstance(store, SQLiteStorageAdapter)
    # Sanity: the adapter is actually usable round-trip.
    job_id, created = store.create_job(
        tenant_id="tenant-a",
        kind="benchmark",
        request_payload={"question": "q"},
    )
    assert created is True
    fetched = store.get_job(job_id, tenant_id="tenant-a")
    assert fetched is not None
    assert fetched.job_id == job_id


def test_make_storage_sqlite_file_path(tmp_path) -> None:
    db_file = tmp_path / "nested" / "jobs.db"
    url = f"sqlite:///{db_file}"

    store = make_storage(url)

    assert isinstance(store, SQLiteStorageAdapter)
    assert db_file.parent.exists()
    job_id, _ = store.create_job(
        tenant_id="tenant-a",
        kind="benchmark",
        request_payload={"q": 1},
    )
    assert db_file.exists()
    assert store.get_job(job_id) is not None


def test_make_storage_sqlite_empty_path_defaults_to_memory() -> None:
    # ``sqlite://`` (no path component) is an edge case but must not
    # raise; it degrades to an in-memory database.
    store = make_storage("sqlite://")

    assert isinstance(store, SQLiteStorageAdapter)
    store.create_job(tenant_id="t", kind="benchmark", request_payload={})


def test_make_storage_sqlite_plus_pysqlite_alias(tmp_path) -> None:
    db_file = tmp_path / "jobs.db"
    store = make_storage(f"sqlite+pysqlite:///{db_file}")

    assert isinstance(store, SQLiteStorageAdapter)
    assert db_file.exists() or True  # created lazily on first write
    store.create_job(tenant_id="t", kind="benchmark", request_payload={})
    assert db_file.exists()


def test_make_storage_sqlite_relative_dot_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    store = make_storage("sqlite:///./nested/jobs.db")

    assert isinstance(store, SQLiteStorageAdapter)
    store.create_job(tenant_id="t", kind="benchmark", request_payload={})
    assert (tmp_path / "nested" / "jobs.db").exists()


def test_make_storage_postgresql_uses_postgres_adapter(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakePostgresAdapter:
        def __init__(self, url: str) -> None:
            captured["url"] = url

    monkeypatch.setattr(storage_module, "PostgresStorageAdapter", FakePostgresAdapter)

    store = make_storage("postgresql://user:pass@localhost:5432/rune")

    assert isinstance(store, FakePostgresAdapter)
    assert captured["url"] == "postgresql://user:pass@localhost:5432/rune"


def test_make_storage_postgresql_requires_pg_extra(monkeypatch) -> None:
    monkeypatch.setattr(storage_module, "PostgresStorageAdapter", None)

    with pytest.raises(RuntimeError, match="requires psycopg"):
        make_storage("postgresql://user:pass@localhost:5432/rune")


@pytest.mark.parametrize(
    "url",
    [
        "redis://localhost:6379/0",
        "mysql://localhost/db",
        "http://example.com/db",
    ],
)
def test_make_storage_unknown_scheme_raises(url: str) -> None:
    with pytest.raises(RuntimeError) as exc_info:
        make_storage(url)

    message = str(exc_info.value)
    assert "unsupported storage URL scheme" in message
    assert "sqlite://" in message  # lists supported schemes


def test_resolve_storage_url_prefers_explicit_url(monkeypatch) -> None:
    monkeypatch.setenv("RUNE_DB_URL", "postgresql://env")
    monkeypatch.setenv("RUNE_API_DB_PATH", ".rune-api/jobs.db")

    resolved = resolve_storage_url("postgresql://explicit", legacy_db_path="legacy.db")

    assert resolved == "postgresql://explicit"


def test_resolve_storage_url_uses_env_url(monkeypatch) -> None:
    monkeypatch.setenv("RUNE_DB_URL", "postgresql://env")
    monkeypatch.delenv("RUNE_API_DB_PATH", raising=False)

    assert resolve_storage_url() == "postgresql://env"


def test_resolve_storage_url_legacy_memory_path(monkeypatch) -> None:
    monkeypatch.delenv("RUNE_DB_URL", raising=False)
    monkeypatch.delenv("RUNE_API_DB_PATH", raising=False)

    assert resolve_storage_url(None, legacy_db_path=":memory:") == "sqlite:///:memory:"


def test_resolve_storage_url_legacy_relative_path(monkeypatch) -> None:
    monkeypatch.delenv("RUNE_DB_URL", raising=False)
    monkeypatch.setenv("RUNE_API_DB_PATH", ".rune-api/jobs.db")

    resolved = resolve_storage_url()

    assert resolved == "sqlite:///./.rune-api/jobs.db"


def test_resolve_storage_url_falls_back_to_default(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("RUNE_DB_URL", raising=False)
    monkeypatch.delenv("RUNE_API_DB_PATH", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    assert resolve_storage_url() == f"sqlite:///{tmp_path.as_posix()}/rune/jobs.db"


def test_default_storage_url_uses_xdg_data_home(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    assert default_storage_url() == f"sqlite:///{tmp_path.as_posix()}/rune/jobs.db"


def test_default_storage_url_uses_localappdata_on_windows(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(storage_module.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    assert default_storage_url() == f"sqlite:///{tmp_path.as_posix()}/rune/jobs.db"


def test_default_storage_url_uses_application_support_on_darwin(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(storage_module.sys, "platform", "darwin")
    fake_home = tmp_path
    monkeypatch.setattr(storage_module.Path, "home", lambda: fake_home)

    expected = fake_home / "Library" / "Application Support" / "rune" / "jobs.db"
    assert default_storage_url() == f"sqlite:///{expected.as_posix()}"


def test_sqlite_connection_yields_raw_connection() -> None:
    store = make_storage("sqlite:///:memory:")
    with store.connection() as conn:
        row = conn.execute("SELECT 1 AS n").fetchone()
        assert row["n"] == 1


def test_sqlite_list_jobs_for_finops_includes_succeeded_benchmarks() -> None:
    store = make_storage("sqlite:///:memory:")
    job_id, _ = store.create_job(
        tenant_id="tenant-a",
        kind="benchmark",
        request_payload={"model": "m"},
    )
    store.update_job(job_id, status="succeeded", result_payload={"tokens": 1})
    rows = store.list_jobs_for_finops(tenant_id="tenant-a", limit=10)
    assert len(rows) == 1
    assert rows[0]["kind"] == "benchmark"
    assert rows[0]["request_payload"] == {"model": "m"}
    assert rows[0]["result_payload"] == {"tokens": 1}
    assert rows[0]["duration_seconds"] >= 1e-3


def test_make_storage_windows_absolute_path_drops_leading_slash(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakeSQLiteAdapter:
        def __init__(self, db_path: str) -> None:
            captured["db_path"] = db_path

    monkeypatch.setattr(storage_module, "SQLiteStorageAdapter", FakeSQLiteAdapter)

    store = make_storage("sqlite:///C:/temp/jobs.db")

    assert isinstance(store, FakeSQLiteAdapter)
    assert captured["db_path"] == "C:/temp/jobs.db"


def test_storage_port_protocol_matches_sqlite_adapter(tmp_path) -> None:
    store = SQLiteStorageAdapter(tmp_path / "jobs.db")

    # runtime_checkable Protocol — structural subtype check.
    assert isinstance(store, StoragePort)


def test_storage_port_protocol_rejects_non_conforming_object() -> None:
    class NotAStore:
        def create_job(self) -> None:  # wrong signature, missing other methods
            pass

    assert not isinstance(NotAStore(), StoragePort)
