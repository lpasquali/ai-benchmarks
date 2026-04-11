# SPDX-License-Identifier: Apache-2.0
"""Protocol and data types for LLM resource provisioning."""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ProvisioningResult:
    """Returned by any LLMResourceProvider after provisioning.

    ``provider_handle`` is an opaque value passed back to :meth:`LLMResourceProvider.teardown`
    so the provider can identify which resource to release (e.g., a Vast.ai contract ID).
    It is excluded from equality and hash comparisons deliberately.
    """

    backend_url: str | None
    model: str | None = None
    backend_type: str = "ollama"
    provider_handle: Any = field(default=None, hash=False, compare=False)


class LLMResourceProvider(Protocol):
    """Protocol for LLM resource providers.

    Implement this protocol to add a new provisioning backend (e.g., Lambda Labs, Modal)
    alongside the existing Vast.ai and existing-Ollama providers.
    """

    async def provision(self) -> ProvisioningResult:
        """Provision an LLM resource and return connection details."""
        ...

    async def teardown(self, result: ProvisioningResult) -> None:
        """Release the provisioned resource. May be a no-op."""
        ...
