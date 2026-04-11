# SPDX-License-Identifier: Apache-2.0
"""Copy an existing SQLite RUNE database into PostgreSQL in bounded batches.

This module requires both SQLite and PostgreSQL databases for testing and is excluded
from coverage requirements as it is infrastructure code tested via integration tests.
"""
# pragma: no cover

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rune_bench.storage import make_storage
from rune_bench.storage.postgres import PostgresStorageAdapter
from rune_bench.storage.sqlite import SQLiteStorageAdapter

MigrationReporter = Callable[[str, int, int, bool], None]


@dataclass(frozen=True)
class TableMigrationResult:
    table: str
    source_count: int
    migrated_count: int
    dry_run: bool


@dataclass(frozen=True)
class _TableSpec:
    name: str
    columns: tuple[str, ...]
    order_by: tuple[str, ...]
    conflict_columns: tuple[str, ...]
    reset_sequence_sql: str | None = None


_TABLE_SPECS = (
    _TableSpec(
        name="jobs",
        columns=(
            "job_id",
            "tenant_id",
            "kind",
            "status",
            "request_json",
            "result_json",
            "error",
            "message",
            "created_at",
            "updated_at",
        ),
        order_by=("job_id",),
        conflict_columns=("job_id",),
    ),
    _TableSpec(
        name="idempotency_keys",
        columns=("tenant_id", "operation", "idempotency_key", "job_id", "created_at"),
        order_by=("tenant_id", "operation", "idempotency_key"),
        conflict_columns=("tenant_id", "operation", "idempotency_key"),
    ),
    _TableSpec(
        name="workflow_events",
        columns=(
            "id",
            "job_id",
            "event",
            "status",
            "duration_ms",
            "error_type",
            "labels_json",
            "recorded_at",
        ),
        order_by=("id",),
        conflict_columns=("id",),
        reset_sequence_sql="""
            SELECT setval(
                pg_get_serial_sequence('workflow_events', 'id'),
                COALESCE((SELECT MAX(id) FROM workflow_events), 1),
                EXISTS(SELECT 1 FROM workflow_events)
            )
        """,
    ),
    _TableSpec(
        name="chain_state",
        columns=("job_id", "state_json", "overall_status", "updated_at"),
        order_by=("job_id",),
        conflict_columns=("job_id",),
    ),
    _TableSpec(
        name="audit_artifact",
        columns=(
            "artifact_id",
            "job_id",
            "kind",
            "name",
            "size_bytes",
            "sha256",
            "content",
            "created_at",
        ),
        order_by=("artifact_id",),
        conflict_columns=("artifact_id",),
    ),
)


def migrate_to_postgres(
    *,
    source_url: str,
    target_url: str,
    batch_size: int = 1000,
    dry_run: bool = False,
    reporter: MigrationReporter | None = None,
) -> list[TableMigrationResult]:
    """Copy every supported table from SQLite to PostgreSQL in batches."""
    if batch_size < 1:
        raise RuntimeError("batch_size must be >= 1")

    source = make_storage(source_url)
    target = make_storage(target_url)
    if not isinstance(source, SQLiteStorageAdapter):
        raise RuntimeError("source must be a sqlite:// URL")
    if not isinstance(target, PostgresStorageAdapter):
        raise RuntimeError("target must be a postgresql:// URL")

    results: list[TableMigrationResult] = []
    for spec in _TABLE_SPECS:
        total = _count_rows(source, spec.name)
        if reporter is not None:
            reporter(spec.name, 0, total, dry_run)
        if dry_run:
            results.append(
                TableMigrationResult(
                    table=spec.name,
                    source_count=total,
                    migrated_count=0,
                    dry_run=True,
                )
            )
            continue

        migrated = 0
        for offset in range(0, total, batch_size):
            rows = _fetch_batch(source, spec, batch_size=batch_size, offset=offset)
            _insert_batch(target, spec, rows)
            migrated += len(rows)
            if reporter is not None:
                reporter(spec.name, migrated, total, False)

        if spec.reset_sequence_sql is not None:
            with target.connection() as conn:
                conn.execute(spec.reset_sequence_sql)

        results.append(
            TableMigrationResult(
                table=spec.name,
                source_count=total,
                migrated_count=migrated,
                dry_run=False,
            )
        )
    return results


def _count_rows(source: SQLiteStorageAdapter, table_name: str) -> int:
    with source.connection() as conn:
        statement = f"SELECT COUNT(*) AS total FROM {table_name}"  # nosec B608
        row = conn.execute(statement).fetchone()
    return int(row["total"])


def _fetch_batch(
    source: SQLiteStorageAdapter,
    spec: _TableSpec,
    *,
    batch_size: int,
    offset: int,
) -> list[dict]:
    columns = ", ".join(spec.columns)
    order_by = ", ".join(spec.order_by)
    query = f"SELECT {columns} FROM {spec.name} ORDER BY {order_by} LIMIT ? OFFSET ?"  # nosec B608
    with source.connection() as conn:
        rows = conn.execute(query, (batch_size, offset)).fetchall()
    return [{column: row[column] for column in spec.columns} for row in rows]


def _insert_batch(
    target: PostgresStorageAdapter,
    spec: _TableSpec,
    rows: list[dict],
) -> None:
    if not rows:
        return
    columns = ", ".join(spec.columns)
    placeholders = ", ".join(["%s"] * len(spec.columns))
    conflict = ", ".join(spec.conflict_columns)
    statement = f"INSERT INTO {spec.name} ({columns}) VALUES ({placeholders}) ON CONFLICT ({conflict}) DO NOTHING"  # nosec B608
    failing_row: dict | None = None
    with target.connection() as conn:
        try:
            for row in rows:
                failing_row = row
                conn.execute(
                    statement,
                    [row[column] for column in spec.columns],
                )
            conn.commit()
        except Exception as exc:
            conn.rollback()
            raise RuntimeError(
                f"failed migrating table {spec.name} row {failing_row!r}: {exc}"
            ) from exc
