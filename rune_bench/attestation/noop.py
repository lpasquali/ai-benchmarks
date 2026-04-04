"""NoOp attestation driver for local/dev environments.

Always passes — use only when hardware TPM is unavailable or unnecessary.
"""

from __future__ import annotations

from .interface import AttestationDriver, AttestationResult


class NoOpDriver(AttestationDriver):
    """Attestation driver that unconditionally passes verification.

    Intended for local development and CI environments where a physical
    TPM 2.0 chip is absent.  Should *not* be used in production.
    """

    def verify(self, target: str) -> AttestationResult:
        return AttestationResult(
            passed=True,
            pcr_digest=None,
            message=(
                f"NoOp attestation: PCR verification skipped for {target!r} "
                "(dev/local mode — not suitable for production)"
            ),
        )
