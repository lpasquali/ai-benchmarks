# SPDX-License-Identifier: Apache-2.0
"""Tests for the rune_bench.metrics module."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from rune_bench.metrics import (
    InMemoryCollector,
    MetricsEvent,
    NullCollector,
    SQLiteMetricsCollector,
    clear_collector,
    get_collector,
    set_collector,
    set_job_id,
    span,
)


# ---------------------------------------------------------------------------
# NullCollector
# ---------------------------------------------------------------------------


def test_null_collector_accepts_any_event():
    null = NullCollector()
    ev = MetricsEvent(
        event="test.event",
        status="ok",
        duration_ms=1.0,
        labels={},
        recorded_at=time.time(),
    )
    null.record(ev)  # must not raise


# ---------------------------------------------------------------------------
# InMemoryCollector
# ---------------------------------------------------------------------------


def test_in_memory_collector_records_events():
    coll = InMemoryCollector()
    ev = MetricsEvent(
        event="test.event",
        status="ok",
        duration_ms=42.5,
        labels={"model": "llama3"},
        recorded_at=time.time(),
    )
    coll.record(ev)
    assert len(coll.all_events()) == 1
    assert coll.all_events()[0].event == "test.event"


def test_in_memory_collector_summary_rows_aggregates():
    coll = InMemoryCollector()
    for _ in range(3):
        coll.record(MetricsEvent("a.event", "ok", 100.0, {}, time.time()))
    coll.record(MetricsEvent("a.event", "error", 200.0, {}, time.time()))
    coll.record(MetricsEvent("b.event", "ok", 50.0, {}, time.time()))

    rows = {r["event"]: r for r in coll.summary_rows()}
    assert rows["a.event"]["total"] == 4
    assert rows["a.event"]["ok"] == 3
    assert rows["a.event"]["error"] == 1
    assert rows["a.event"]["min_ms"] == 100.0
    assert rows["a.event"]["max_ms"] == 200.0
    assert rows["b.event"]["total"] == 1


def test_in_memory_collector_empty_summary():
    coll = InMemoryCollector()
    assert coll.summary_rows() == []


def test_in_memory_collector_thread_safe():
    coll = InMemoryCollector()
    results = []

    def worker():
        for _ in range(50):
            coll.record(MetricsEvent("e", "ok", 1.0, {}, time.time()))
        results.append(True)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 4
    assert len(coll.all_events()) == 200


# ---------------------------------------------------------------------------
# Thread-local collector registry
# ---------------------------------------------------------------------------


def test_get_collector_returns_null_by_default():
    clear_collector()
    assert isinstance(get_collector(), NullCollector)


def test_set_and_get_collector():
    coll = InMemoryCollector()
    set_collector(coll)
    assert get_collector() is coll
    clear_collector()


def test_set_collector_is_thread_local():
    coll_main = InMemoryCollector()
    coll_thread = InMemoryCollector()
    set_collector(coll_main)

    results = []

    def worker():
        set_collector(coll_thread)
        results.append(get_collector() is coll_thread)

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert results == [True]
    assert get_collector() is coll_main
    clear_collector()


# ---------------------------------------------------------------------------
# span() context manager
# ---------------------------------------------------------------------------


def test_span_records_ok_event():
    coll = InMemoryCollector()
    set_collector(coll)
    try:
        with span("my.operation", model="llama3"):
            time.sleep(0.001)
    finally:
        clear_collector()

    events = coll.all_events()
    assert len(events) == 1
    ev = events[0]
    assert ev.event == "my.operation"
    assert ev.status == "ok"
    assert ev.duration_ms >= 0
    assert ev.labels == {"model": "llama3"}
    assert ev.error_type is None


def test_span_records_error_event_and_reraises():
    coll = InMemoryCollector()
    set_collector(coll)
    try:
        with pytest.raises(ValueError, match="boom"):
            with span("failing.op"):
                raise ValueError("boom")
    finally:
        clear_collector()

    events = coll.all_events()
    assert len(events) == 1
    ev = events[0]
    assert ev.status == "error"
    assert ev.error_type == "ValueError"


def test_span_captures_job_id():
    coll = InMemoryCollector()
    set_collector(coll)
    set_job_id("job-123")
    try:
        with span("op"):
            pass
    finally:
        set_job_id(None)
        clear_collector()

    assert coll.all_events()[0].job_id == "job-123"


def test_span_with_null_collector_does_not_raise():
    clear_collector()
    with span("noop.op"):
        pass  # NullCollector, no error expected


def test_span_metrics_error_does_not_propagate():
    """If the collector itself raises, the span must not propagate that error."""

    class BrokenCollector:
        def record(self, event: MetricsEvent) -> None:
            raise RuntimeError("storage down")

    set_collector(BrokenCollector())  # type: ignore[arg-type]
    try:
        with span("op"):
            pass  # should not raise despite broken collector
    finally:
        clear_collector()


# ---------------------------------------------------------------------------
# SQLiteMetricsCollector + JobStore integration
# ---------------------------------------------------------------------------


def test_sqlite_metrics_collector_persists_events(tmp_path: Path):
    from rune_bench.job_store import JobStore

    store = JobStore(tmp_path / "jobs.db")
    coll = SQLiteMetricsCollector(store)

    ev = MetricsEvent(
        event="vastai.offer_search",
        status="ok",
        duration_ms=1234.5,
        labels={"min_dph": 2.3, "max_dph": 3.0},
        recorded_at=time.time(),
        job_id="job-abc",
    )
    coll.record(ev)

    rows = store.get_events_for_job("job-abc")
    assert len(rows) == 1
    assert rows[0]["event"] == "vastai.offer_search"
    assert rows[0]["status"] == "ok"
    assert abs(rows[0]["duration_ms"] - 1234.5) < 0.01
    assert rows[0]["labels"]["min_dph"] == 2.3


def test_job_store_events_summary_aggregates(tmp_path: Path):
    from rune_bench.job_store import JobStore

    store = JobStore(tmp_path / "jobs.db")
    coll = SQLiteMetricsCollector(store)

    for i in range(3):
        coll.record(MetricsEvent("phase.a", "ok", float(100 + i * 10), {}, time.time(), job_id="j1"))
    coll.record(MetricsEvent("phase.a", "error", 500.0, {}, time.time(), job_id="j1", error_type="RuntimeError"))
    coll.record(MetricsEvent("phase.b", "ok", 50.0, {}, time.time(), job_id="j1"))

    summary = {r["event"]: r for r in store.get_events_summary()}
    assert summary["phase.a"]["total"] == 4
    assert summary["phase.a"]["ok"] == 3
    assert summary["phase.a"]["error"] == 1
    assert summary["phase.b"]["total"] == 1

    # Filter by job_id
    summary_j1 = {r["event"]: r for r in store.get_events_summary(job_id="j1")}
    assert "phase.a" in summary_j1

    summary_other = store.get_events_summary(job_id="other-job")
    assert summary_other == []


def test_sqlite_collector_survives_storage_error(tmp_path: Path):
    """Record must not raise even if the DB is inaccessible."""
    from unittest.mock import MagicMock

    broken_store = MagicMock()
    broken_store.record_workflow_event.side_effect = RuntimeError("disk full")

    coll = SQLiteMetricsCollector(broken_store)
    ev = MetricsEvent("e", "ok", 1.0, {}, time.time())
    coll.record(ev)  # must not raise


def test_span_with_sqlite_collector_end_to_end(tmp_path: Path):
    from rune_bench.job_store import JobStore

    store = JobStore(tmp_path / "jobs.db")
    set_collector(SQLiteMetricsCollector(store))
    set_job_id("job-xyz")
    try:
        with span("workflow.provision", backend="vastai"):
            time.sleep(0.001)
        with pytest.raises(RuntimeError):
            with span("agent.ask", model="llama3"):
                raise RuntimeError("timeout")
    finally:
        set_job_id(None)
        clear_collector()

    rows = store.get_events_for_job("job-xyz")
    assert len(rows) == 2
    events_by_name = {r["event"]: r for r in rows}
    assert events_by_name["workflow.provision"]["status"] == "ok"
    assert events_by_name["agent.ask"]["status"] == "error"
    assert events_by_name["agent.ask"]["error_type"] == "RuntimeError"
