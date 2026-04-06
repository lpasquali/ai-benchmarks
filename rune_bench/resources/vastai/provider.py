"""Vast.ai LLM resource provider."""

from rune_bench.resources.vastai.sdk import VastAI

from rune_bench.resources.base import ProvisioningResult


class VastAIProvider:
    """LLMResourceProvider backed by Vast.ai GPU instances.

    Wraps :func:`rune_bench.workflows.provision_vastai_ollama` and
    :func:`rune_bench.workflows.stop_vastai_instance`.
    """

    def __init__(
        self,
        sdk: VastAI,
        *,
        template_hash: str,
        min_dph: float,
        max_dph: float,
        reliability: float,
        stop_on_teardown: bool = False,
    ) -> None:
        self._sdk = sdk
        self._template_hash = template_hash
        self._min_dph = min_dph
        self._max_dph = max_dph
        self._reliability = reliability
        self._stop_on_teardown = stop_on_teardown

    def provision(self) -> ProvisioningResult:
        from rune_bench.workflows import provision_vastai_ollama
        result = provision_vastai_ollama(
            self._sdk,
            template_hash=self._template_hash,
            min_dph=self._min_dph,
            max_dph=self._max_dph,
            reliability=self._reliability,
            confirm_create=lambda: True,
        )
        return ProvisioningResult(
            backend_url=result.backend_url,
            model=result.model_name,
            provider_handle=result.contract_id,
        )

    def teardown(self, result: ProvisioningResult) -> None:
        if self._stop_on_teardown and result.provider_handle is not None:
            from rune_bench.workflows import stop_vastai_instance
            stop_vastai_instance(self._sdk, result.provider_handle)
