# SPDX-License-Identifier: Apache-2.0
"""BurpGPT driver client -- delegates web vulnerability scanning to the burpgpt driver process.

The driver process (``rune_bench.drivers.burpgpt.__main__``) talks to the
Burp Suite REST API to launch scans and retrieve findings.  Burp Suite Pro
must be running locally (or remotely) with the REST API enabled.

Configuration:
    RUNE_BURPGPT_BURP_API_URL  Base URL of the Burp REST API
                                (default: http://localhost:1337)

Security notice: only scan targets you own or have explicit written
authorization to test.
"""

from __future__ import annotations

from rune_bench.debug import debug_log
from rune_bench.drivers import DriverTransport, make_driver_transport


class BurpGPTDriverClient:
    """Run web vulnerability scans by delegating to the burpgpt driver.

    Unlike Holmes, BurpGPT does **not** require a kubeconfig -- it operates
    against the Burp Suite REST API.
    """

    def __init__(
        self,
        *,
        transport: DriverTransport | None = None,
    ) -> None:
        self._transport: DriverTransport = transport or make_driver_transport("burpgpt")

    def ask(self, question: str, model: str, backend_url: str | None = None, backend_type: str = "ollama") -> str:
        """Dispatch a scan request to the burpgpt driver and return findings.

        Args:
            question: Target URL or scan objective.
            model: Ollama model identifier (currently unused by BurpGPT but
                   kept for interface consistency).
            backend_url: Base URL of the Ollama server (currently unused).

        Returns:
            Formatted vulnerability findings from the Burp scan.
        """
        params: dict = {
            "question": question,
            "model": model.strip(),
        }
        if backend_url:
            params["backend_url"] = backend_url

        debug_log(
            f"BurpGPTDriverClient.ask: question={question!r} model={model!r} "
            f"backend_url={backend_url or '<none>'}"
        )
        result = self._transport.call("ask", params)

        if "answer" not in result:
            raise RuntimeError("BurpGPT driver response did not include an answer.")

        answer = result["answer"]
        if answer is None:
            raise RuntimeError("BurpGPT driver returned an empty answer.")

        answer_text = str(answer)
        if not answer_text:
            raise RuntimeError("BurpGPT driver returned an empty answer.")

        return answer_text
