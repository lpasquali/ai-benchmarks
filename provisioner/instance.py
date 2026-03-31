"""Block 6+7+8+9 — Instance lifecycle.

Handles instance creation, polling until running, model pull via
Ollama, and surfacing external connection details.
"""

import time
from dataclasses import dataclass, field

from vastai import VastAI

from .models import SelectedModel
from .template import Template

_POLL_INTERVAL_S = 10
_POLL_MAX_ATTEMPTS = 36  # ~6 minutes


@dataclass
class ConnectionDetails:
    contract_id: int | str
    status: str
    ssh_host: str | None
    ssh_port: int | None
    machine_id: str | None
    service_urls: list[dict] = field(default_factory=list)
    # Each entry: {"name": str, "direct": str, "proxy": str | None}


class InstanceManager:
    """Create, monitor, and interact with a Vast.ai instance."""

    def __init__(self, sdk: VastAI) -> None:
        self._sdk = sdk

    # ------------------------------------------------------------------ #
    # Block 6 — Create instance                                           #
    # ------------------------------------------------------------------ #

    def create(
        self,
        offer_id: int,
        model: SelectedModel,
        template: Template,
    ) -> int | str:
        """Create an instance and return the new contract id.

        Raises:
            RuntimeError: if creation or contract-id parsing fails.
        """
        kwargs: dict = {
            "id": str(offer_id),
            "disk": float(model.required_disk_gb),
            "env": template.env,
            "raw": True,
        }
        if template.image:
            kwargs["image"] = template.image

        try:
            result = self._sdk.create_instance(**kwargs)
        except Exception as exc:
            raise RuntimeError(f"Instance creation failed: {exc}") from exc

        contract_id = None
        if isinstance(result, dict):
            contract_id = result.get("new_contract") or result.get("id")

        if not contract_id:
            raise RuntimeError(
                f"Could not parse contract id from create response: {result}"
            )
        return contract_id

    # ------------------------------------------------------------------ #
    # Block 7 — Poll until running                                        #
    # ------------------------------------------------------------------ #

    def wait_until_running(
        self,
        contract_id: int | str,
        on_poll: callable | None = None,
    ) -> dict:
        """Block until the instance reaches 'running' state.

        Args:
            contract_id: The contract/instance id to monitor.
            on_poll: Optional callback(status: str) called after each poll.

        Returns:
            The full instance info dict from the Vast.ai API.

        Raises:
            RuntimeError: if the timeout expires before the instance is running.
        """
        for _ in range(_POLL_MAX_ATTEMPTS):
            instance = self._fetch_instance(contract_id)
            if instance:
                status = instance.get("actual_status", instance.get("state", "unknown"))
                if on_poll:
                    on_poll(str(status))
                if str(status).lower() == "running":
                    return instance
            time.sleep(_POLL_INTERVAL_S)

        raise RuntimeError(
            f"Instance {contract_id} did not reach 'running' state "
            f"within {_POLL_MAX_ATTEMPTS * _POLL_INTERVAL_S}s."
        )

    # ------------------------------------------------------------------ #
    # Block 8 — Pull model via Ollama                                     #
    # ------------------------------------------------------------------ #

    def pull_model(self, contract_id: int | str, model_name: str) -> None:
        """Run `ollama pull <model>` inside the remote instance.

        Raises:
            RuntimeError: if the execute call fails.
        """
        command = f"ollama pull '{model_name}' && echo 'Model pulled successfully'"
        try:
            try:
                self._sdk.execute(id=str(contract_id), command=command, raw=True)
            except TypeError:
                self._sdk.execute(id=str(contract_id), COMMAND=command, raw=True)
        except Exception as exc:
            raise RuntimeError(f"Model pull execution failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Block 9 — Build connection details                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_connection_details(
        contract_id: int | str,
        instance_info: dict,
    ) -> ConnectionDetails:
        """Extract SSH, direct, and Vast proxy URLs from an instance info dict."""
        ports = instance_info.get("ports", {}) or {}
        machine_id = instance_info.get("machine_id")
        service_urls: list[dict] = []

        if isinstance(ports, dict):
            for svc, mappings in ports.items():
                if not mappings or not isinstance(mappings, list):
                    continue
                first = mappings[0]
                host_ip = first.get("HostIp")
                host_port = first.get("HostPort")
                if not host_ip or not host_port:
                    continue
                entry: dict = {
                    "name": svc,
                    "direct": f"http://{host_ip}:{host_port}",
                    "proxy": (
                        f"https://server-{machine_id}.vast.ai:{host_port}"
                        if machine_id
                        else None
                    ),
                }
                service_urls.append(entry)

        return ConnectionDetails(
            contract_id=contract_id,
            status=instance_info.get(
                "actual_status", instance_info.get("state", "unknown")
            ),
            ssh_host=instance_info.get("ssh_host"),
            ssh_port=instance_info.get("ssh_port"),
            machine_id=machine_id,
            service_urls=service_urls,
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _fetch_instance(self, contract_id: int | str) -> dict | None:
        try:
            instances = self._sdk.show_instances(raw=True)
        except Exception:
            return None
        if not isinstance(instances, list):
            return None
        return next(
            (i for i in instances if str(i.get("id")) == str(contract_id)),
            None,
        )
