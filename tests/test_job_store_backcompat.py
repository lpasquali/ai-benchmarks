# SPDX-License-Identifier: Apache-2.0
"""Tests that the legacy :mod:`rune_bench.job_store` imports keep working.

This protects downstream consumers (tests, metrics, vastai instance manager)
that still `from rune_bench.job_store import JobStore` / `JobRecord`.
"""

from __future__ import annotations


def test_old_import_path_still_works(tmp_path) -> None:
    from rune_bench.job_store import JobRecord, JobStore
    from rune_bench.storage.sqlite import JobRecord as NewJobRecord
    from rune_bench.storage.sqlite import SQLiteStorageAdapter

    # The legacy names must alias the new implementation exactly.
    assert JobStore is SQLiteStorageAdapter
    assert JobRecord is NewJobRecord

    # And the aliased class must still be instantiable + functional.
    store = JobStore(tmp_path / "jobs.db")
    job_id, created = store.create_job(
        tenant_id="tenant-a",
        kind="benchmark",
        request_payload={"question": "q"},
    )
    assert created is True
    fetched = store.get_job(job_id, tenant_id="tenant-a")
    assert isinstance(fetched, JobRecord)
    assert fetched.job_id == job_id


def test_shim_exports_all_symbols() -> None:
    import rune_bench.job_store as shim

    assert set(shim.__all__) == {"JobRecord", "JobStore", "SQLiteStorageAdapter"}
    assert shim.JobStore is shim.SQLiteStorageAdapter
