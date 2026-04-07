# SPDX-License-Identifier: Apache-2.0
"""Dagger driver client — delegates CI/CD pipeline queries to the dagger driver process.

The driver process is launched via :func:`~rune_bench.drivers.make_driver_transport`
(stdio subprocess by default, HTTP server in production).  The driver itself
(``rune_bench.drivers.dagger.__main__``) uses the Dagger Python SDK (async)
and therefore only requires ``dagger-io`` to be installed in the *subprocess*
environment — not in the rune core process.
"""

from __future__ import annotations

from rune_bench.debug import debug_log
from rune_bench.drivers import DriverTransport, make_driver_transport


class DaggerDriverClient:
    """Orchestrate CI/CD pipelines by delegating to the dagger driver process.

    Unlike the Holmes driver, Dagger does not require a kubeconfig — it
    connects to a local or remote Dagger engine automatically.
    """

    def __init__(
        self,
        *,
        transport: DriverTransport | None = None,
    ) -> None:
        self._transport: DriverTransport = transport or make_driver_transport("dagger")

    def ask(self, question: str, model: str, backend_url: str | None = None, backend_type: str = "ollama") -> str:
        """Dispatch a pipeline objective to the dagger driver and return the result.

        Args:
            question: Pipeline command or objective to execute.
            model: Model identifier (injected as env var in container steps).
            backend_url: Base URL of the Ollama server (optional).

        Returns:
            Pipeline stdout/result as a string.
        """
        params: dict = {
            "question": question,
            "model": model,
        }
        if backend_url:
            params["backend_url"] = backend_url

        debug_log(
            f"DaggerDriverClient.ask: question={question!r} model={model!r} "
            f"backend_url={backend_url or '<none>'}"
        )
        result = self._transport.call("ask", params)

        if "answer" not in result:
            raise RuntimeError("Dagger driver response did not include an answer.")

        answer = result["answer"]
        if answer is None:
            raise RuntimeError("Dagger driver returned an empty answer.")

        answer_text = str(answer)
        if not answer_text:
            raise RuntimeError("Dagger driver returned an empty answer.")

        return answer_text
