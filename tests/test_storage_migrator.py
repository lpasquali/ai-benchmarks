# SPDX-License-Identifier: Apache-2.0
"""Tests for :class:`rune_bench.storage.migrator.Migrator`."""

from __future__ import annotations

import pathlib
import sqlite3
import time

import pytest

from rune_bench.storage import make_storage
from rune_bench.storage.migrator import Migrator

_EXPECTED_TABLES = {
    "jobs",
    "idempotency_keys",
    "workflow_events",
    "chain_state",
    "audit_artifact",
}
_EXPECTED_VERSIONS = [1, 2, 3, 4, 5]


def _migrations_dir() -> pathlib.Path:
    import rune_bench.storage.migrator as migrator_mod

    return pathlib.Path(migrator_mod.__file__).resolve().parent / "migrations"


def _apply_sql_files(conn: sqlite3.Connection, *filenames: str) -> None:
    migrator = Migrator(dialect="sqlite")
    for name in filenames:
        path = _migrations_dir() / name
        conn.executescript(
            migrator._render_for_dialect(path.read_text(encoding="utf-8"))
        )
    conn.commit()


def _fresh_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def test_migrator_applies_all_on_empty_db() -> None:
    # Acceptance (rune#241): fresh empty DB → bootstrap is a no-op, all
    # migrations apply.
    conn = _fresh_conn()

    applied = Migrator().apply_pending(conn)

    assert applied == _EXPECTED_VERSIONS
    versions = {
        int(row[0])
        for row in conn.execute("SELECT version FROM schema_version").fetchall()
    }
    assert versions == set(_EXPECTED_VERSIONS)


def test_migrator_idempotent() -> None:
    conn = _fresh_conn()
    migrator = Migrator()

    first = migrator.apply_pending(conn)
    second = migrator.apply_pending(conn)

    assert first == _EXPECTED_VERSIONS
    assert second == []
    # No duplicate rows inserted on the second run.
    (count,) = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()
    assert count == len(_EXPECTED_VERSIONS)


def test_migrator_partial_state() -> None:
    # Acceptance (rune#241): DB already has ``schema_version`` → bootstrap is a
    # no-op; only pending migrations run.
    conn = _fresh_conn()
    migrator = Migrator()

    # Pre-seed: pretend migrations 1–3 were applied out of band (e.g. a
    # previous release) so the migrator must only apply 4 and 5.
    conn.execute(
        "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)"
    )
    now = time.time()
    for version in (1, 2, 3):
        conn.execute(
            "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
            (version, now),
        )
    conn.commit()

    applied = migrator.apply_pending(conn)

    assert applied == [4, 5]
    versions = {
        int(row[0])
        for row in conn.execute("SELECT version FROM schema_version").fetchall()
    }
    assert versions == {1, 2, 3, 4, 5}


def test_migrator_legacy_db_without_schema_version_first_three_tables() -> None:
    # Acceptance (rune#241): legacy file with 0001–0003 tables only → bootstrap
    # pre-seeds 1–3, then migrations 4 and 5 run.
    conn = _fresh_conn()
    _apply_sql_files(
        conn,
        "0001_jobs.sql",
        "0002_idempotency_keys.sql",
        "0003_workflow_events.sql",
    )

    applied = Migrator().apply_pending(conn)

    assert applied == [4, 5]
    versions = {
        int(row[0])
        for row in conn.execute("SELECT version FROM schema_version").fetchall()
    }
    assert versions == set(_EXPECTED_VERSIONS)
    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert _EXPECTED_TABLES.issubset(names)


def test_migrator_legacy_db_without_schema_version_all_five_tables() -> None:
    # Acceptance (rune#241): legacy file with all domain tables but no
    # ``schema_version`` table → bootstrap pre-seeds 1–5, then apply is a no-op.
    conn = _fresh_conn()
    Migrator().apply_pending(conn)
    conn.execute("DROP TABLE schema_version")
    conn.commit()

    applied = Migrator().apply_pending(conn)

    assert applied == []
    versions = {
        int(row[0])
        for row in conn.execute("SELECT version FROM schema_version").fetchall()
    }
    assert versions == set(_EXPECTED_VERSIONS)


def test_migrator_invalid_sql_raises_with_context(tmp_path: pathlib.Path) -> None:
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    (migrations_dir / "0001_broken.sql").write_text(
        "THIS IS NOT VALID SQL;\n", encoding="utf-8"
    )
    conn = _fresh_conn()

    with pytest.raises(RuntimeError) as exc_info:
        Migrator(migrations_dir=migrations_dir).apply_pending(conn)

    message = str(exc_info.value)
    assert "0001_broken.sql" in message
    assert "migration 1" in message
    # The rollback must have un-done the failed version so schema_version
    # still shows nothing applied.
    (count,) = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()
    assert count == 0


def test_migrator_records_applied_at_timestamp() -> None:
    conn = _fresh_conn()
    before = time.time()

    Migrator().apply_pending(conn)

    after = time.time()
    rows = conn.execute(
        "SELECT version, applied_at FROM schema_version ORDER BY version"
    ).fetchall()
    assert [int(r[0]) for r in rows] == _EXPECTED_VERSIONS
    for row in rows:
        stamp = float(row[1])
        assert before <= stamp <= after


def test_migration_files_apply_cleanly_to_real_sqlite() -> None:
    conn = _fresh_conn()

    Migrator().apply_pending(conn)

    names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    # ``schema_version`` is the bookkeeping table; the five domain tables
    # must all exist alongside it.
    assert _EXPECTED_TABLES.issubset(names)
    assert "schema_version" in names

    # Indexes from 0003 and 0005 must also exist.
    index_names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    assert "idx_workflow_events_job_id" in index_names
    assert "idx_workflow_events_event" in index_names
    assert "idx_audit_artifact_job_id" in index_names


def test_make_storage_memory_still_works_after_migration_refactor() -> None:
    # Regression guard: the shared-cache :memory: trick from rune#231 must
    # still round-trip after the Migrator took over schema creation.
    store = make_storage("sqlite:///:memory:")

    job_id, created = store.create_job(
        tenant_id="tenant-a",
        kind="benchmark",
        request_payload={"q": 1},
    )
    assert created is True
    fetched = store.get_job(job_id, tenant_id="tenant-a")
    assert fetched is not None and fetched.job_id == job_id
