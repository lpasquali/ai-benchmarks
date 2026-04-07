# SPDX-License-Identifier: Apache-2.0
"""Driver transport protocol — the base contract for all driver transports.

A ``DriverTransport`` is anything that can send an action + params to an
external driver process and return its result dict.  Two concrete
implementations exist:

* :class:`~rune_bench.drivers.stdio.StdioTransport` — spawns a subprocess,
  sends one JSON line on stdin, reads one JSON line from stdout.
* :class:`~rune_bench.drivers.http.HttpTransport` — submits a job to an HTTP
  driver server and polls until completion.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DriverTransport(Protocol):
    """Structural protocol satisfied by :class:`StdioTransport` and :class:`HttpTransport`.

    Any object with a matching ``call`` signature is a valid transport.
    """

    def call(self, action: str, params: dict) -> dict:
        """Call a driver action and return the result dict.

        Args:
            action: Driver-specific action name (e.g. ``"ask"``).
            params: Free-form parameter mapping for the action.

        Returns:
            Result dict returned by the driver (e.g. ``{"answer": "..."}``).

        Raises:
            RuntimeError: if the driver process fails or returns an error status.
        """
        ...
