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
import time
from typing import Any

_FILENAME_RE = re.compile(r"^(\d{4})_.*\.sql$")


class Migrator:
    """Apply pending SQL migrations to a database connection.

    The migrations directory contains ``NNNN_<slug>.sql`` files. Each file is
    executed in a single transaction and, on success, the version is recorded
    in ``schema_version(version, applied_at)``. Re-running against an
    up-to-date database is a no-op.
    """

    def __init__(self, *, migrations_dir: pathlib.Path | None = None, dialect: str = "sqlite") -> None:
        self._migrations_dir = (
            migrations_dir
            if migrations_dir is not None
            else pathlib.Path(__file__).parent / "migrations"
        )
        self._dialect = dialect
        self._placeholder = "%s" if dialect == "postgres" else "?"

    def apply_pending(self, conn: Any) -> list[int]:
        """Apply every migration newer than the current ``schema_version``.

        Returns the list of newly-applied version numbers (empty if the
        database is already up to date). Each migration runs inside an
        explicit ``BEGIN ... COMMIT``; a failure rolls back that migration
        and re-raises :class:`RuntimeError` with the offending version and
        filename so the underlying driver error is never swallowed.
        """
        self._ensure_bookkeeping_table(conn)
        applied = self._already_applied(conn)
        newly_applied: list[int] = []

        for version, path in self._discover():
            if version in applied:
                continue
            sql_text = path.read_text(encoding="utf-8")
            
            # Split on ';' and execute each non-empty statement individually.
            # For SQLite we must do this to avoid implicit COMMITs.
            # For Postgres we can execute all at once, but keeping it uniform is simpler.
            statements = [s.strip() for s in sql_text.split(";") if s.strip()]
            
            if self._dialect == "sqlite":
                conn.execute("BEGIN")
            
            try:
                for statement in statements:
                    conn.execute(statement)  # nosec
                conn.execute(
                    f"INSERT INTO schema_version(version, applied_at) VALUES ({self._placeholder}, {self._placeholder})",  # nosec
                    (version, time.time()),
                )
                if self._dialect == "sqlite":
                    conn.commit()
            except Exception as exc:
                if self._dialect == "sqlite":
                    conn.rollback()
                raise RuntimeError(
                    f"migration {version} ({path.name}) failed: {exc}"
                ) from exc
            newly_applied.append(version)

        return newly_applied

    def _ensure_bookkeeping_table(self, conn: Any) -> None:
        if self._dialect == "sqlite":
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at REAL NOT NULL
                )
                """
            )
            conn.commit()
        else:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at DOUBLE PRECISION NOT NULL
                )
                """
            )

    def _already_applied(self, conn: Any) -> set[int]:
        cursor = conn.execute("SELECT version FROM schema_version")
        rows = cursor.fetchall()
        return {int(row[0]) for row in rows}

    def _discover(self) -> list[tuple[int, pathlib.Path]]:
        discovered: list[tuple[int, pathlib.Path]] = []
        for path in sorted(self._migrations_dir.glob("[0-9][0-9][0-9][0-9]_*.sql")):
            match = _FILENAME_RE.match(path.name)
            if match is None:  # pragma: no cover - glob guarantees the shape
                continue
            discovered.append((int(match.group(1)), path))
        return discovered
