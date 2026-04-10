# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typer.testing import CliRunner

import rune
import rune_bench.storage.migrate_to_postgres as migration_mod
from rune_bench.storage.migrate_to_postgres import TableMigrationResult


def test_db_migrate_to_postgres_cli_dry_run(monkeypatch) -> None:
    runner = CliRunner()

    monkeypatch.setattr(
        migration_mod,
        "migrate_to_postgres",
        lambda **_kwargs: [
            TableMigrationResult(
                table="jobs",
                source_count=3,
                migrated_count=0,
                dry_run=True,
            )
        ],
    )

    result = runner.invoke(
        rune.app,
        [
            "db",
            "migrate-to-postgres",
            "--source",
            "sqlite:///:memory:",
            "--target",
            "postgresql://localhost/rune",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Database Migration Plan" in result.output
    assert "Dry run complete" in result.output
    assert "jobs" in result.output


def test_db_migrate_to_postgres_cli_reports_failure(monkeypatch) -> None:
    runner = CliRunner()

    monkeypatch.setattr(
        migration_mod,
        "migrate_to_postgres",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = runner.invoke(
        rune.app,
        [
            "db",
            "migrate-to-postgres",
            "--source",
            "sqlite:///:memory:",
            "--target",
            "postgresql://localhost/rune",
        ],
    )

    assert result.exit_code == 1
    assert "boom" in result.output
