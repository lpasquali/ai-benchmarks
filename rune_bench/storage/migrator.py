# SPDX-License-Identifier: Apache-2.0
"""Hand-rolled schema migrator for RUNE storage adapters.

Applies versioned ``NNNN_*.sql`` files from
``rune_bench/storage/migrations/`` in lexicographic order inside explicit
``BEGIN ... COMMIT`` transactions and records every applied version in a
``schema_version`` bookkeeping table so re-runs are no-ops.

Why hand-rolled? alembic would pull SQLAlchemy (~80 transitive deps) for a
project with a handful of tables; yoyo and dbmate add comparable weight for
marginal benefit at this scale. ~50 LOC gives us everything we actually need
(idempotent, transactional, ordered, auditable) without the footprint.
"""

from __future__ import annotations

import pathlib
import re
import sqlite3
import time

_FILENAME_RE = re.compile(r"^(\d{4})_.*\.sql$")

# Tables created by migrations 0001–0005 before ``schema_version`` existed
# (legacy on-disk SQLite). Used to pre-seed ``schema_version`` so unconditional
# ``CREATE TABLE`` migrations do not fail on upgrade. See rune#241.
_PREEXISTING_TABLE_TO_MIGRATION: dict[str, int] = {
    "jobs": 1,
    "idempotency_keys": 2,
    "workflow_events": 3,
    "chain_state": 4,
    "audit_artifact": 5,
}


class Migrator:
    """Apply pending SQL migrations to a SQLite connection.

    The migrations directory contains ``NNNN_<slug>.sql`` files. Each file is
    executed in a single transaction and, on success, the version is recorded
    in ``schema_version(version, applied_at)``. Re-running against an
    up-to-date database is a no-op.
    """

    def __init__(self, *, migrations_dir: pathlib.Path | None = None) -> None:
        self._migrations_dir = (
            migrations_dir
            if migrations_dir is not None
            else pathlib.Path(__file__).parent / "migrations"
        )

    def apply_pending(self, conn: sqlite3.Connection) -> list[int]:
        """Apply every migration newer than the current ``schema_version``.

        Returns the list of newly-applied version numbers (empty if the
        database is already up to date). Each migration runs inside an
        explicit ``BEGIN ... COMMIT``; a failure rolls back that migration
        and re-raises :class:`RuntimeError` with the offending version and
        filename so the underlying driver error is never swallowed.
        """
        self._bootstrap_legacy_schema(conn)
        self._ensure_bookkeeping_table(conn)
        applied = self._already_applied(conn)
        newly_applied: list[int] = []

        for version, path in self._discover():
            if version in applied:
                continue
            sql_text = path.read_text(encoding="utf-8")
            # SQLite's ``executescript()`` issues an implicit COMMIT before
            # running, which would break our explicit BEGIN…COMMIT envelope.
            # Split on ';' and execute each non-empty statement individually
            # so the whole migration lives inside one transaction and we can
            # ROLLBACK cleanly on failure.
            statements = [s.strip() for s in sql_text.split(";") if s.strip()]
            conn.execute("BEGIN")
            try:
                for statement in statements:
                    conn.execute(statement)
                conn.execute(
                    "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
                    (version, time.time()),
                )
                conn.commit()
            except Exception as exc:
                conn.rollback()
                raise RuntimeError(
                    f"migration {version} ({path.name}) failed: {exc}"
                ) from exc
            newly_applied.append(version)

        return newly_applied

    def _bootstrap_legacy_schema(self, conn: sqlite3.Connection) -> None:
        """Pre-seed ``schema_version`` when upgrading a pre-migration SQLite file.

        Legacy databases created before the migrator landed already contain domain
        tables but no ``schema_version`` row. :meth:`_ensure_bookkeeping_table`
        would create an empty bookkeeping table and the first migration would
        fail with "table already exists". Detect that case via ``sqlite_master``
        and record versions for tables that are already present.
        """
        has_schema_version = (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'"
            ).fetchone()
            is not None
        )
        if has_schema_version:
            return

        existing = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        pre_existing_versions = sorted(
            v
            for tbl, v in _PREEXISTING_TABLE_TO_MIGRATION.items()
            if tbl in existing
        )
        if not pre_existing_versions:
            return

        conn.execute(
            """
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY,
                applied_at REAL NOT NULL
            )
            """
        )
        now = time.time()
        for version in pre_existing_versions:
            conn.execute(
                "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
                (version, now),
            )
        conn.commit()

    def _ensure_bookkeeping_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at REAL NOT NULL
            )
            """
        )
        conn.commit()

    def _already_applied(self, conn: sqlite3.Connection) -> set[int]:
        rows = conn.execute("SELECT version FROM schema_version").fetchall()
        return {int(row[0]) for row in rows}

    def _discover(self) -> list[tuple[int, pathlib.Path]]:
        discovered: list[tuple[int, pathlib.Path]] = []
        for path in sorted(self._migrations_dir.glob("[0-9][0-9][0-9][0-9]_*.sql")):
            match = _FILENAME_RE.match(path.name)
            if match is None:  # pragma: no cover - glob guarantees the shape
                continue
            discovered.append((int(match.group(1)), path))
        return discovered
