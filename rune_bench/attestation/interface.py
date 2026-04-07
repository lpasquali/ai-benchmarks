# SPDX-License-Identifier: Apache-2.0
"""Attestation driver interface for TPM 2.0 PCR verification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class AttestationResult:
    passed: bool
    pcr_digest: str | None
    message: str


class AttestationDriver(ABC):
    @abstractmethod
    def verify(self, target: str) -> AttestationResult:
        """Verify PCR attestation for the given scheduling target.

        Args:
            target: Identifier for the scheduling target (e.g. kubeconfig path
                    or cluster endpoint) used as the attestation nonce.

        Returns:
            AttestationResult with ``passed=True`` when attestation succeeds.
        """
