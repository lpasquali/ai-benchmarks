"""TPM 2.0 remote attestation scaffold for benchmark job scheduling targets."""

from .factory import get_driver
from .interface import AttestationDriver, AttestationResult

__all__ = ["AttestationDriver", "AttestationResult", "get_driver"]
