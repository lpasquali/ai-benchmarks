"""Block 10 — HolmesGPT Runner.

Runs HolmesGPT via its Python SDK against a Kubernetes cluster,
using the provisioned model as the LLM backend.
"""

from pathlib import Path
import os
import subprocess
import sys

import holmes  # type: ignore

from rune_bench.debug import debug_log
from rune_bench.backends.ollama import OllamaClient, OllamaModelManager


class HolmesRunner:
    """Investigate a Kubernetes cluster using the HolmesGPT SDK."""

    def __init__(self, kubeconfig: Path) -> None:
        if not kubeconfig.exists():
            raise FileNotFoundError(f"kubeconfig not found: {kubeconfig}")
        self._kubeconfig = kubeconfig

    def ask(self, question: str, model: str, ollama_url: str | None = None) -> str:
        """Run a HolmesGPT query and return the answer as a string.

        Tries module-level ``holmes.ask(...)`` first, then falls back to
        the class-based ``holmes.Holmes(...).ask(...)`` pattern.

        Raises:
            RuntimeError: if no supported SDK entry-point is found.
        """
        answer = None
        resolved_model = self._resolve_model(model)
        self._configure_ollama_model_limits(model=resolved_model, ollama_url=ollama_url)
        debug_log(
            f"Holmes request: question={question!r} model={resolved_model!r} "
            f"ollama_url={ollama_url or '<none>'} kubeconfig={self._kubeconfig}"
        )

        if hasattr(holmes, "ask"):
            try:
                answer = holmes.ask(
                    question=question,
                    model=resolved_model,
                    kubeconfig=str(self._kubeconfig),
                )
            except TypeError:
                pass

        if answer is None and hasattr(holmes, "Holmes"):
            try:
                client = holmes.Holmes(
                    model=resolved_model,
                    kubeconfig=str(self._kubeconfig),
                )
                answer = client.ask(question=question)
            except TypeError:
                client = holmes.Holmes(kubeconfig=str(self._kubeconfig))
                answer = client.ask(question=question, model=resolved_model)

        if answer is None:
            answer = self._ask_via_cli(question=question, model=resolved_model, ollama_url=ollama_url)

        if answer is None:
            raise RuntimeError(
                "Unsupported HolmesGPT SDK API shape. "
                "Expected holmes.ask(...) or holmes.Holmes(...).ask(...)."
            )

        return str(answer)

    def _resolve_model(self, model: str) -> str:
        return model.strip()

    def _configure_ollama_model_limits(self, *, model: str, ollama_url: str | None) -> None:
        """Set Holmes/LiteLLM override env vars from Ollama model metadata."""
        if not ollama_url:
            return

        normalized_model = OllamaModelManager.create(ollama_url).normalize_model_name(model)
        try:
            capabilities = OllamaClient(ollama_url).get_model_capabilities(normalized_model)
        except RuntimeError as exc:
            debug_log(f"Skipping Holmes model-limit auto-config for {normalized_model}: {exc}")
            return

        self._set_model_limit_override(
            env_name="OVERRIDE_MAX_CONTENT_SIZE",
            value=capabilities.context_window,
        )
        self._set_model_limit_override(
            env_name="OVERRIDE_MAX_OUTPUT_TOKEN",
            value=capabilities.max_output_tokens,
        )

    def _set_model_limit_override(self, *, env_name: str, value: int | None) -> None:
        if value is None or value <= 0:
            return

        existing = os.environ.get(env_name)
        if existing:
            debug_log(f"Preserving existing {env_name}={existing}")
            return

        os.environ[env_name] = str(value)
        debug_log(f"Set {env_name}={value}")

        try:
            import holmes.core.llm as holmes_llm  # type: ignore

            setattr(holmes_llm, env_name, value)
        except Exception as exc:
            debug_log(f"Could not update holmes.core.llm.{env_name}: {exc}")

    def _ask_via_cli(self, question: str, model: str, ollama_url: str | None = None) -> str | None:
        """Fallback to Holmes CLI entrypoint for packages that expose no SDK symbols.

        Streams Holmes output live to the terminal while also collecting it so the
        caller can still render the final answer after completion.
        """
        cmd = [
            sys.executable,
            "-m",
            "holmes.main",
            "ask",
            question,
            "--model",
            model,
            "--no-interactive",
        ]
        env = os.environ.copy()
        env["KUBECONFIG"] = str(self._kubeconfig)
        env.setdefault("DISABLE_PROMETHEUS_TOOLSET", "true")
        if ollama_url:
            env["OLLAMA_API_BASE"] = ollama_url
            env["OPENAI_API_BASE"] = ollama_url
        for name in ("OVERRIDE_MAX_CONTENT_SIZE", "OVERRIDE_MAX_OUTPUT_TOKEN"):
            value = os.environ.get(name)
            if value:
                env[name] = value

        try:
            debug_log(
                f"Holmes CLI command: {' '.join(cmd)} env_overrides="
                f"KUBECONFIG={env['KUBECONFIG']} "
                f"OLLAMA_API_BASE={env.get('OLLAMA_API_BASE', '<none>')}"
            )
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise RuntimeError(f"Failed to execute Holmes CLI fallback: {exc}") from exc

        output_chunks: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            output_chunks.append(line)
            print(line, end="", flush=True)

        proc.wait()

        stdout = "".join(output_chunks).strip()
        stderr = ""
        debug_log(
            f"Holmes CLI result: returncode={proc.returncode} stdout={stdout!r} stderr={stderr!r}"
        )

        if proc.returncode != 0:
            detail = stderr or stdout or f"exit code {proc.returncode}"
            raise RuntimeError(f"Holmes CLI fallback failed: {detail}")

        return stdout or stderr or None
