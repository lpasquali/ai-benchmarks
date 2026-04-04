"""Factory for attestation drivers.

Resolves the driver name from the supplied *config* dict, falling back to
the ``RUNE_ATTESTATION_DRIVER`` environment variable (default: ``noop``).
"""

from __future__ import annotations

import os

from .interface import AttestationDriver


def get_driver(config: dict | None = None) -> AttestationDriver:
    """Return the AttestationDriver configured by *config* or environment.

    Args:
        config: Attestation section from ``rune.yaml``, e.g.::

                {"driver": "noop"}
                {"driver": "tpm2", "pcr_policy_path": "/etc/rune/pcr.policy"}

            When ``None`` or empty, falls back to the
            ``RUNE_ATTESTATION_DRIVER`` env var (default: ``"noop"``).

    Returns:
        A concrete :class:`AttestationDriver` instance.

    Raises:
        ValueError: When the driver name is not ``"noop"`` or ``"tpm2"``.
        RuntimeError: When ``driver="tpm2"`` and tpm2-tools is not installed.
    """
    cfg = config or {}
    driver_name = cfg.get("driver") or os.environ.get("RUNE_ATTESTATION_DRIVER", "noop")
    pcr_policy_path = cfg.get("pcr_policy_path") or os.environ.get(
        "RUNE_ATTESTATION_PCR_POLICY_PATH"
    )

    if driver_name == "noop":
        from .noop import NoOpDriver

        return NoOpDriver()

    if driver_name == "tpm2":
        from .tpm2 import TPM2Driver

        return TPM2Driver(pcr_policy_path=pcr_policy_path)

    raise ValueError(
        f"Unknown attestation driver: {driver_name!r}. "
        "Expected 'noop' or 'tpm2'. "
        "Set attestation.driver in rune.yaml or RUNE_ATTESTATION_DRIVER env var."
    )
