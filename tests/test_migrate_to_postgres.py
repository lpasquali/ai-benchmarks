# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
try:
    import psycopg
except ImportError:
    pytest.skip("psycopg not installed", allow_module_level=True)

from contextlib import contextmanager
import rune_bench.storage.migrate_to_postgres as migration_mod


class _Result:
    def __init__(self, *, one=None, many=None):
        self._one = one
        self._many = many or []

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._many


class _FakeSourceConnection:
    def __init__(self, tables: dict[str, list[dict]]):
        self.tables = tables

    def execute(self, query: str, params=None):
        normalized = " ".join(query.split())
        if normalized.startswith("SELECT COUNT(*) AS total FROM "):
            table_name = normalized.rsplit(" ", 1)[-1]
            return _Result(one={"total": len(self.tables[table_name])})
        if normalized.startswith("SELECT ") and " LIMIT ? OFFSET ?" in normalized:
            table_name = normalized.split(" FROM ", 1)[1].split(" ORDER BY", 1)[0]
            batch_size, offset = params
            rows = self.tables[table_name][offset : offset + batch_size]
            return _Result(many=[dict(row) for row in rows])
        raise AssertionError(f"unexpected source query: {query}")

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _FakeTargetConnection:
    def __init__(self) -> None:
        self.tables: dict[str, list[dict]] = {}
        self._keys: dict[str, set[tuple]] = {}
        self.commits = 0
        self.rollbacks = 0
        self.sequence_reset = False
        self.fail_on_table: str | None = None

    def execute(self, query: str, params=None):
        normalized = " ".join(query.split())
        if normalized.startswith("INSERT INTO "):
            table_name = normalized.split("INSERT INTO ", 1)[1].split(" (", 1)[0]
            if self.fail_on_table == table_name:
                raise RuntimeError("insert failed")
            columns_text = normalized.split(f"INSERT INTO {table_name} (", 1)[1].split(
                ") VALUES", 1
            )[0]
            conflict_text = normalized.split("ON CONFLICT (", 1)[1].split(
                ") DO NOTHING", 1
            )[0]
            columns = [item.strip() for item in columns_text.split(",")]
            conflict_columns = [item.strip() for item in conflict_text.split(",")]
            row = dict(zip(columns, params, strict=True))
            row_key = tuple(row[column] for column in conflict_columns)
            seen = self._keys.setdefault(table_name, set())
            if row_key not in seen:
                seen.add(row_key)
                self.tables.setdefault(table_name, []).append(row)
            return _Result()
        if "setval(" in normalized:
            self.sequence_reset = True
            return _Result()
        raise AssertionError(f"unexpected target query: {query}")

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class _FakeSQLiteAdapter:
    def __init__(self, tables: dict[str, list[dict]]):
        self._conn = _FakeSourceConnection(tables)

    @contextmanager
    def connection(self):
        yield self._conn


class _FakePostgresAdapter:
    def __init__(self):
        self._conn = _FakeTargetConnection()

    @contextmanager
    def connection(self):
        yield self._conn


def _patch_storage(monkeypatch, source, target) -> None:
    storages = [source, target]
    monkeypatch.setattr(migration_mod, "SQLiteStorageAdapter", _FakeSQLiteAdapter)
    monkeypatch.setattr(migration_mod, "PostgresStorageAdapter", _FakePostgresAdapter)
    monkeypatch.setattr(migration_mod, "make_storage", lambda _url: storages.pop(0))


def test_migrate_to_postgres_copies_rows_in_batches(monkeypatch) -> None:
    source = _FakeSQLiteAdapter(
        {
            "jobs": [
                {
                    "job_id": "job-1",
                    "tenant_id": "tenant-a",
                    "kind": "benchmark",
                    "status": "queued",
                    "request_json": "{}",
                    "result_json": None,
                    "error": None,
                    "message": "accepted",
                    "created_at": 1.0,
                    "updated_at": 1.0,
                }
            ],
            "idempotency_keys": [
                {
                    "tenant_id": "tenant-a",
                    "operation": "benchmark",
                    "idempotency_key": "idem-1",
                    "job_id": "job-1",
                    "created_at": 1.0,
                }
            ],
            "workflow_events": [
                {
                    "id": 7,
                    "job_id": "job-1",
                    "event": "phase.a",
                    "status": "ok",
                    "duration_ms": 12.3,
                    "error_type": None,
                    "labels_json": "{}",
                    "recorded_at": 2.0,
                }
            ],
            "chain_state": [
                {
                    "job_id": "job-1",
                    "state_json": '{"nodes": [], "edges": []}',
                    "overall_status": "pending",
                    "updated_at": 2.0,
                }
            ],
            "audit_artifact": [
                {
                    "artifact_id": "artifact-1",
                    "job_id": "job-1",
                    "kind": "sbom",
                    "name": "sbom.json",
                    "size_bytes": 2,
                    "sha256": "ab" * 32,
                    "content": b"{}",
                    "created_at": 3.0,
                }
            ],
        }
    )
    target = _FakePostgresAdapter()
    _patch_storage(monkeypatch, source, target)
    progress: list[tuple[str, int, int, bool]] = []

    results = migration_mod.migrate_to_postgres(
        source_url="sqlite:///:memory:",
        target_url="postgresql://localhost/rune",
        batch_size=1,
        reporter=lambda table, copied, total, dry_run: progress.append(
            (table, copied, total, dry_run)
        ),
    )

    assert [result.table for result in results] == [
        "jobs",
        "idempotency_keys",
        "workflow_events",
        "chain_state",
        "audit_artifact",
    ]
    assert target._conn.tables["jobs"][0]["job_id"] == "job-1"
    assert target._conn.tables["workflow_events"][0]["id"] == 7
    assert target._conn.sequence_reset is True
    assert ("jobs", 1, 1, False) in progress

    # Re-running the migration is a no-op on the target because inserts use
    # ON CONFLICT DO NOTHING on each table's primary key.
    _patch_storage(monkeypatch, source, target)
    migration_mod.migrate_to_postgres(
        source_url="sqlite:///:memory:",
        target_url="postgresql://localhost/rune",
        batch_size=2,
    )
    assert len(target._conn.tables["jobs"]) == 1
    assert len(target._conn.tables["workflow_events"]) == 1


def test_migrate_to_postgres_dry_run_does_not_write(monkeypatch) -> None:
    source = _FakeSQLiteAdapter(
        {
            "jobs": [{"job_id": "job-1"}],
            "idempotency_keys": [],
            "workflow_events": [],
            "chain_state": [],
            "audit_artifact": [],
        }
    )
    target = _FakePostgresAdapter()
    _patch_storage(monkeypatch, source, target)

    results = migration_mod.migrate_to_postgres(
        source_url="sqlite:///:memory:",
        target_url="postgresql://localhost/rune",
        dry_run=True,
    )

    assert results[0].table == "jobs"
    assert results[0].source_count == 1
    assert results[0].migrated_count == 0
    assert results[0].dry_run is True
    assert target._conn.tables == {}


def test_migrate_to_postgres_rejects_invalid_batch_size(monkeypatch) -> None:
    source = _FakeSQLiteAdapter(
        {
            "jobs": [],
            "idempotency_keys": [],
            "workflow_events": [],
            "chain_state": [],
            "audit_artifact": [],
        }
    )
    target = _FakePostgresAdapter()
    _patch_storage(monkeypatch, source, target)

    with pytest.raises(RuntimeError, match="batch_size must be >= 1"):
        migration_mod.migrate_to_postgres(
            source_url="sqlite:///:memory:",
            target_url="postgresql://localhost/rune",
            batch_size=0,
        )


def test_migrate_to_postgres_requires_sqlite_source(monkeypatch) -> None:
    target = _FakePostgresAdapter()
    storages = [object(), target]
    monkeypatch.setattr(migration_mod, "SQLiteStorageAdapter", _FakeSQLiteAdapter)
    monkeypatch.setattr(migration_mod, "PostgresStorageAdapter", _FakePostgresAdapter)
    monkeypatch.setattr(migration_mod, "make_storage", lambda _url: storages.pop(0))

    with pytest.raises(RuntimeError, match="source must be a sqlite:// URL"):
        migration_mod.migrate_to_postgres(
            source_url="sqlite:///:memory:",
            target_url="postgresql://localhost/rune",
        )


def test_migrate_to_postgres_requires_postgres_target(monkeypatch, tmp_path) -> None:
    from rune_bench.storage.sqlite import SQLiteStorageAdapter

    db_a = tmp_path / "a.db"
    db_b = tmp_path / "b.db"
    adapters = [SQLiteStorageAdapter(db_a), SQLiteStorageAdapter(db_b)]
    monkeypatch.setattr(migration_mod, "make_storage", lambda _url: adapters.pop(0))

    with pytest.raises(RuntimeError, match="target must be a postgresql:// URL"):
        migration_mod.migrate_to_postgres(
            source_url=f"sqlite:///{db_a.as_posix()}",
            target_url="postgresql://localhost/rune",
        )


def test_insert_batch_no_rows_is_noop() -> None:
    target = _FakePostgresAdapter()
    spec = migration_mod._TABLE_SPECS[0]
    migration_mod._insert_batch(target, spec, [])
    assert target._conn.tables == {}


def test_migrate_to_postgres_rolls_back_failing_batch(monkeypatch) -> None:
    source = _FakeSQLiteAdapter(
        {
            "jobs": [
                {
                    "job_id": "job-1",
                    "tenant_id": "tenant-a",
                    "kind": "benchmark",
                    "status": "queued",
                    "request_json": "{}",
                    "result_json": None,
                    "error": None,
                    "message": "accepted",
                    "created_at": 1.0,
                    "updated_at": 1.0,
                }
            ],
            "idempotency_keys": [],
            "workflow_events": [],
            "chain_state": [],
            "audit_artifact": [],
        }
    )
    target = _FakePostgresAdapter()
    target._conn.fail_on_table = "jobs"
    _patch_storage(monkeypatch, source, target)

    with pytest.raises(RuntimeError, match="failed migrating table jobs row"):
        migration_mod.migrate_to_postgres(
            source_url="sqlite:///:memory:",
            target_url="postgresql://localhost/rune",
        )

    assert target._conn.rollbacks == 1
