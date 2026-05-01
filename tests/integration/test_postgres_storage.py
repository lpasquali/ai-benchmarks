# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

try:
    import psycopg  # noqa: F401
except ImportError:
    pytest.skip("psycopg not installed", allow_module_level=True)

import os

import psycopg


@pytest.fixture
def pg_url():
    url = os.environ.get("RUNE_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("RUNE_TEST_POSTGRES_URL not set")
    return url


@pytest.mark.integration_postgres
def test_postgres_service_connectivity(pg_url):
    """Verify the CI Postgres service accepts connections (no RUNE schema init).

    ``PostgresStorageAdapter`` applies SQLite-derived migration SQL that is not
    yet valid on PostgreSQL; keep this job as a connectivity gate only.
    """
    with psycopg.connect(pg_url) as conn:
        conn.execute("SELECT 1")
# Trigger CI
