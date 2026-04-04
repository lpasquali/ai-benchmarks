"""TPM 2.0 attestation driver using the tpm2-tools CLI.

Invokes ``tpm2_quote`` to obtain a PCR quote from the local TPM chip and,
when a PCR policy file is supplied, runs ``tpm2_checkquote`` to validate it.

Requirements:
    tpm2-tools >= 5.x must be installed (``apt install tpm2-tools`` or
    equivalent).  The Attestation Key context file ``ak.ctx`` must already
    exist in the working directory (created once via ``tpm2_createak``).

Note on replay protection:
    This scaffold passes the *target* string (e.g. kubeconfig path) as the
    TPM qualifying data.  This binds the quote to that specific target but
    does **not** provide freshness — a static target can be replayed.  For
    production use, replace ``target`` with a per-request server-generated
    nonce (challenge) and validate it inside ``tpm2_checkquote`` to prevent
    quote replay attacks.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile

from .interface import AttestationDriver, AttestationResult

# PCR indices used for OS boot-integrity measurement.
_DEFAULT_PCR_LIST = "sha256:0,1,2,3,4,7"
_QUOTE_TIMEOUT = 30  # seconds


class TPM2Driver(AttestationDriver):
    """Attestation driver that calls tpm2-tools subprocesses.

    Args:
        pcr_policy_path: Optional path to a PCR policy file passed to
            ``tpm2_checkquote``.  When ``None``, quote verification is
            skipped and only the raw PCR measurement is returned.
    """

    def __init__(self, pcr_policy_path: str | None = None) -> None:
        self.pcr_policy_path = pcr_policy_path
        self._check_tpm2_tools()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_tpm2_tools(self) -> None:
        required = ["tpm2_quote"]
        if self.pcr_policy_path is not None:
            required.append("tpm2_checkquote")
        missing = [cmd for cmd in required if shutil.which(cmd) is None]
        if missing:
            raise RuntimeError(
                f"tpm2-tools binaries not found: {missing}. "
                "Install tpm2-tools to use the TPM2 attestation driver "
                "(e.g. `apt install tpm2-tools`)."
            )

    # ------------------------------------------------------------------
    # AttestationDriver interface
    # ------------------------------------------------------------------

    def verify(self, target: str) -> AttestationResult:
        """Run tpm2_quote and optionally tpm2_checkquote for *target*.

        The *target* string is encoded as the qualifying data fed to
        ``tpm2_quote``, binding the quote to this scheduling target.

        .. note::
            ``target`` is typically a stable value (e.g. kubeconfig path) and
            does **not** provide freshness — it cannot prevent quote replay.
            For replay protection, pass a server-generated per-request nonce
            as ``target`` and validate it in ``tpm2_checkquote``.

        Each call writes TPM artefacts to an isolated temporary directory that
        is cleaned up before this method returns, so concurrent calls cannot
        cross-contaminate each other.
        """
        qualifying_data = target.encode("utf-8", errors="replace").hex()

        with tempfile.TemporaryDirectory(prefix="rune-attest-") as workdir:
            sig_path = f"{workdir}/quote.sig"
            pcrs_path = f"{workdir}/quote.pcrs"
            msg_path = f"{workdir}/quote.msg"

            cmd = [
                "tpm2_quote",
                "--key-context", "ak.ctx",
                "--pcr-list", _DEFAULT_PCR_LIST,
                "--qualifying-data", qualifying_data,
                "--signature", sig_path,
                "--pcrs_output", pcrs_path,
                "--message", msg_path,
            ]

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=_QUOTE_TIMEOUT,
                )
            except subprocess.TimeoutExpired:
                return AttestationResult(
                    passed=False,
                    pcr_digest=None,
                    message=f"tpm2_quote timed out after {_QUOTE_TIMEOUT}s for target {target!r}",
                )
            except FileNotFoundError:
                raise RuntimeError(
                    "tpm2_quote binary disappeared after driver initialisation. "
                    "Ensure tpm2-tools remains installed throughout the process lifetime."
                )

            if proc.returncode != 0:
                return AttestationResult(
                    passed=False,
                    pcr_digest=None,
                    message=f"tpm2_quote failed (rc={proc.returncode}): {proc.stderr.strip()}",
                )

            pcr_digest = proc.stdout.strip() or None

            if self.pcr_policy_path:
                check_cmd = [
                    "tpm2_checkquote",
                    "--public", "ak.pub",
                    "--message", msg_path,
                    "--signature", sig_path,
                    "--pcrs_input", pcrs_path,
                    "--pcr-list", _DEFAULT_PCR_LIST,
                    "--qualification", qualifying_data,
                    "--policy", self.pcr_policy_path,
                ]
                try:
                    check_proc = subprocess.run(
                        check_cmd,
                        capture_output=True,
                        text=True,
                        timeout=_QUOTE_TIMEOUT,
                    )
                except subprocess.TimeoutExpired:
                    return AttestationResult(
                        passed=False,
                        pcr_digest=pcr_digest,
                        message=f"tpm2_checkquote timed out after {_QUOTE_TIMEOUT}s",
                    )
                except FileNotFoundError:
                    raise RuntimeError(
                        "tpm2_checkquote binary disappeared after driver initialisation."
                    )

                if check_proc.returncode != 0:
                    return AttestationResult(
                        passed=False,
                        pcr_digest=pcr_digest,
                        message=(
                            f"tpm2_checkquote policy mismatch (rc={check_proc.returncode}): "
                            f"{check_proc.stderr.strip()}"
                        ),
                    )

        return AttestationResult(
            passed=True,
            pcr_digest=pcr_digest,
            message=f"TPM2 PCR attestation passed for target {target!r}",
        )
