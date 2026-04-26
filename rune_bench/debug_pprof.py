# SPDX-License-Identifier: Apache-2.0
"""Optional diagnostics HTTP server for RUNE Python processes (API server).

Mirrors the *separate bind address* pattern used for Go ``pprof`` on the
operator: main traffic on one port, profiling on another. Set
``RUNE_PPROF_BIND_ADDRESS`` to a TCP address (e.g. ``127.0.0.1:6060``). Use
``0`` or leave unset to disable.

Endpoints are *pprof-inspired* (index layout similar to ``net/http/pprof``).
Heap output is ``tracemalloc`` text, not protobuf pprof; use
``go tool pprof`` only for Go endpoints on the operator.

Security: bind to loopback unless you intentionally expose diagnostics inside
a secured network namespace (e.g. Kubernetes + NetworkPolicy).
"""

from __future__ import annotations

import html
import logging
import os
import sys
import threading
import traceback
import tracemalloc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

_logger = logging.getLogger(__name__)

_started = False
_start_lock = threading.Lock()
# Populated after a successful bind (for tests and introspection only).
diag_server: ThreadingHTTPServer | None = None


def _heap_text(limit: int = 40) -> str:
    if not tracemalloc.is_tracing():
        return "tracemalloc is not active\n"
    snap = tracemalloc.take_snapshot()
    lines = [f"Top {limit} allocations by traceback:\n"]
    for stat in snap.statistics("traceback")[:limit]:
        lines.append(str(stat))
        lines.append("")
    return "\n".join(lines)


def _threads_text() -> str:
    out: list[str] = []
    for ident, frame in sys._current_frames().items():
        out.append(f"Thread {ident}:")
        out.extend(traceback.format_stack(frame))
        out.append("")
    return "\n".join(out)


class _PprofHandler(BaseHTTPRequestHandler):
    server_version = "RuneDebugPprof/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        _logger.debug(fmt, *args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        qs = parse_qs(parsed.query)

        if path == "/debug/pprof":
            body = """<html><head><title>/debug/pprof/</title></head><body>
/debug/pprof/<br>
<br>
Types of profiles available:<br>
<table>
<tr><td align=right><a href="/debug/pprof/cmdline">cmdline</a>:</td><td>Process label</td></tr>
<tr><td align=right><a href="/debug/pprof/heap">heap</a>:</td><td>tracemalloc allocation summary (text)</td></tr>
<tr><td align=right><a href="/debug/pprof/goroutine?debug=1">goroutine</a>:</td><td>Python thread stacks (debug=1)</td></tr>
</table>
<p><small>Enable with RUNE_PPROF_BIND_ADDRESS. Not binary-compatible with go tool pprof.</small></p>
</body></html>
"""
            self._write(200, body.encode(), "text/html; charset=utf-8")
            return

        if path == "/debug/pprof/cmdline":
            label = os.environ.get("RUNE_PPROF_CMDLINE", "rune python")
            self._write(200, (label + "\n").encode(), "text/plain; charset=utf-8")
            return

        if path == "/debug/pprof/heap":
            data = _heap_text().encode()
            self._write(200, data, "text/plain; charset=utf-8")
            return

        if path == "/debug/pprof/goroutine":
            if qs.get("debug", ["0"])[0] == "1":
                data = _threads_text().encode()
                self._write(200, data, "text/plain; charset=utf-8")
                return
            self._write(
                200,
                b"goroutine profile: pass ?debug=1 for Python thread stacks\n",
                "text/plain; charset=utf-8",
            )
            return

        msg = f"unknown path: {html.escape(self.path)}"
        self._write(404, msg.encode(), "text/plain; charset=utf-8")

    def _write(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _parse_bind_addr(raw: str) -> tuple[str, int]:
    s = raw.strip()
    if s.startswith(":"):
        return "", int(s[1:])
    host, _, port_s = s.rpartition(":")
    if not host or not port_s:
        raise ValueError(
            f"invalid RUNE_PPROF_BIND_ADDRESS (use host:port or :port): {raw!r}"
        )
    return host, int(port_s)


def start_background_server_if_configured() -> None:
    """Start a daemon ThreadingHTTPServer if RUNE_PPROF_BIND_ADDRESS is set."""
    global _started, diag_server
    raw = os.environ.get("RUNE_PPROF_BIND_ADDRESS", "").strip()
    if not raw or raw == "0":
        return
    with _start_lock:
        if _started:
            return
        try:
            host, port = _parse_bind_addr(raw)
        except ValueError as exc:
            _logger.warning("RUNE_PPROF_BIND_ADDRESS ignored: %s", exc)
            return

        if not tracemalloc.is_tracing():
            tracemalloc.start(25)

        try:
            httpd = ThreadingHTTPServer((host, port), _PprofHandler)
        except OSError as exc:
            _logger.warning(
                "RUNE pprof diagnostics server failed to bind %r: %s", raw, exc
            )
            return

        diag_server = httpd
        host_out, port_out = httpd.server_address[:2]
        _logger.warning(
            "RUNE pprof diagnostics listening on %s:%s (RUNE_PPROF_BIND_ADDRESS); do not expose publicly",
            host_out,
            port_out,
        )

        def _run() -> None:
            httpd.serve_forever()

        threading.Thread(target=_run, name="rune-pprof", daemon=True).start()
        _started = True


def reset_for_tests() -> None:
    """Test hook: stop diagnostics server and allow a fresh ``start_*`` call."""
    global _started, diag_server
    with _start_lock:
        if diag_server is not None:
            try:
                diag_server.shutdown()
            except OSError:
                pass
            diag_server = None
        _started = False
