"""Workflow lifecycle metrics for RUNE.

A lightweight, thread-safe metrics layer that counts and times workflow phases.
Use the ``span()`` context manager around any block of work to record its
duration and outcome.  Attach a collector to the current thread with
``set_collector()`` before calling instrumented code.

Three collector implementations are provided:

* ``NullCollector``           — the default, zero-overhead no-op.
* ``InMemoryCollector``       — accumulates events for CLI-mode summary printing.
* ``SQLiteMetricsCollector``  — persists events to the job store for cross-run comparison.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from rune_bench.job_store import JobStore


@dataclass
class MetricsEvent:
    event: str
    status: str           # "ok" | "error"
    duration_ms: float
    labels: dict          # extra dimensions: model, kind, backend, …
    recorded_at: float    # Unix timestamp
    job_id: str | None = None
    error_type: str | None = None


class MetricsCollector(Protocol):
    def record(self, event: MetricsEvent) -> None: ...


class NullCollector:
    """No-op collector — default when none is configured."""

    def record(self, event: MetricsEvent) -> None:
        pass


class InMemoryCollector:
    """Accumulates events in memory; suitable for CLI mode.

    After the workflow completes, call :meth:`summary_rows` to get
    per-event aggregates for display.
    """

    def __init__(self) -> None:
        self._events: list[MetricsEvent] = []
        self._lock = threading.Lock()

    def record(self, event: MetricsEvent) -> None:
        with self._lock:
            self._events.append(event)

    def all_events(self) -> list[MetricsEvent]:
        with self._lock:
            return list(self._events)

    def summary_rows(self) -> list[dict]:
        """Return per-event aggregate rows sorted by event name."""
        buckets: dict[str, list[MetricsEvent]] = defaultdict(list)
        for ev in self.all_events():
            buckets[ev.event].append(ev)

        rows = []
        for event_name in sorted(buckets):
            evs = buckets[event_name]
            ok = sum(1 for e in evs if e.status == "ok")
            err = sum(1 for e in evs if e.status == "error")
            durations = [e.duration_ms for e in evs]
            avg_ms = sum(durations) / len(durations) if durations else 0.0
            rows.append({
                "event": event_name,
                "total": len(evs),
                "ok": ok,
                "error": err,
                "avg_ms": round(avg_ms, 1),
                "min_ms": round(min(durations), 1) if durations else 0.0,
                "max_ms": round(max(durations), 1) if durations else 0.0,
            })
        return rows


class SQLiteMetricsCollector:
    """Persists MetricsEvents to the workflow_events table in the job store."""

    def __init__(self, store: "JobStore") -> None:
        self._store = store

    def record(self, event: MetricsEvent) -> None:
        try:
            self._store.record_workflow_event(event)
        except Exception:
            pass  # never let storage errors affect the workflow


# ---------------------------------------------------------------------------
# Thread-local collector registry
# ---------------------------------------------------------------------------

_tls = threading.local()
_null = NullCollector()


def set_collector(collector: MetricsCollector) -> None:
    """Bind *collector* to the calling thread."""
    _tls.collector = collector


def clear_collector() -> None:
    """Remove any collector bound to the calling thread (reverts to NullCollector)."""
    _tls.collector = _null


def set_job_id(job_id: str | None) -> None:
    """Attach a job_id to all spans emitted from the calling thread."""
    _tls.job_id = job_id


def get_collector() -> MetricsCollector:
    return getattr(_tls, "collector", _null)


class _SpanContext:
    """Class-based context manager for span() — avoids PEP-479 issues with generators."""

    def __init__(self, event: str, **labels: object) -> None:
        self._event = event
        self._labels = labels
        self._start: float = 0.0
        self._exc: BaseException | None = None

    def __enter__(self) -> "_SpanContext":
        self._start = time.monotonic()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> Literal[False]:
        self._exc = exc_val
        duration_ms = (time.monotonic() - self._start) * 1000
        status = "error" if exc_val is not None else "ok"
        error_type = type(exc_val).__name__ if exc_val is not None else None
        job_id: str | None = getattr(_tls, "job_id", None)
        ev = MetricsEvent(
            event=self._event,
            status=status,
            duration_ms=duration_ms,
            labels=dict(self._labels),
            recorded_at=time.time(),
            job_id=job_id,
            error_type=error_type,
        )
        try:
            get_collector().record(ev)
        except Exception:
            pass  # never let metrics errors propagate into the caller
        return False  # never suppress exceptions


def span(event: str, **labels: object) -> _SpanContext:
    """Time a block and record its outcome to the current thread's collector.

    On success the event status is ``"ok"``; on any exception it is ``"error"``
    and the exception class name is captured as ``error_type``.  The exception
    is always re-raised so the caller's control flow is unaffected.

    Usage::

        with span("vastai.offer_search", model="llama3.1:8b"):
            offer = OfferFinder(sdk).find_best(...)
    """
    return _SpanContext(event, **labels)
