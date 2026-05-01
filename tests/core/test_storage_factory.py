# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`rune_bench.storage` factory and protocol conformance."""

from __future__ import annotations

import pytest

from rune_bench.storage import StoragePort, SQLiteStorageAdapter, make_storage


def test_make_storage_sqlite_memory() -> None:
    store = make_storage("sqlite:///:memory:")
    try:
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
    finally:
        store.close()


def test_make_storage_sqlite_file_path(tmp_path) -> None:
    db_file = tmp_path / "nested" / "jobs.db"
    url = f"sqlite:///{db_file}"

    store = make_storage(url)
    try:
        assert isinstance(store, SQLiteStorageAdapter)
        assert db_file.parent.exists()
        job_id, _ = store.create_job(
            tenant_id="tenant-a",
            kind="benchmark",
            request_payload={"q": 1},
        )
        assert db_file.exists()
        assert store.get_job(job_id) is not None
    finally:
        store.close()


def test_make_storage_sqlite_empty_path_defaults_to_memory() -> None:
    # ``sqlite://`` (no path component) is an edge case but must not
    # raise; it degrades to an in-memory database.
    store = make_storage("sqlite://")
    try:
        assert isinstance(store, SQLiteStorageAdapter)
        store.create_job(tenant_id="t", kind="benchmark", request_payload={})
    finally:
        store.close()




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


def test_storage_port_protocol_matches_sqlite_adapter(tmp_path) -> None:
    store = SQLiteStorageAdapter(tmp_path / "jobs.db")
    try:
        # runtime_checkable Protocol — structural subtype check.
        assert isinstance(store, StoragePort)
    finally:
        store.close()


def test_storage_port_protocol_rejects_non_conforming_object() -> None:
    class NotAStore:
        def create_job(self) -> None:  # wrong signature, missing other methods
            pass

    assert not isinstance(NotAStore(), StoragePort)
