"""Block 6+7+8+9 — Instance lifecycle.

Handles instance creation, polling until running, model pull via
Ollama, and surfacing external connection details.
"""

import time
from dataclasses import dataclass, field
from typing import Callable

from vastai import VastAI  # type: ignore[import-untyped, import-not-found]  # Reason: vastai SDK does not provide type hints

from rune_bench.common import SelectedModel
from rune_bench.debug import debug_log
from rune_bench.backends.ollama import OllamaClient

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


@dataclass
class TeardownResult:
    contract_id: int | str
    destroyed_instance: bool
    destroyed_volume_ids: list[str] = field(default_factory=list)
    verification_ok: bool = False
    verification_message: str = ""


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
            debug_log(f"Vast.ai API call: create_instance kwargs={kwargs}")
            result = self._sdk.create_instance(**kwargs)
        except Exception as exc:
            raise RuntimeError(f"Instance creation failed: {exc}") from exc

        debug_log(f"Vast.ai API result: create_instance result={result}")

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
        on_poll: Callable[[str], None] | None = None,
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
                debug_log(f"Vast.ai polling: contract_id={contract_id} status={status}")
                if on_poll:
                    on_poll(str(status))
                if str(status).lower() == "running":
                    return instance
            time.sleep(_POLL_INTERVAL_S)

        raise RuntimeError(
            f"Instance {contract_id} did not reach 'running' state "
            f"within {_POLL_MAX_ATTEMPTS * _POLL_INTERVAL_S}s."
        )

    def list_instances(self) -> list[dict]:
        """Return raw instance dictionaries from Vast.ai."""
        try:
            debug_log("Vast.ai API call: show_instances raw=True")
            instances = self._sdk.show_instances(raw=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to list Vast.ai instances: {exc}") from exc

        debug_log(
            f"Vast.ai API result: show_instances count={len(instances) if isinstance(instances, list) else '<non-list>'}"
        )

        if not isinstance(instances, list):
            return []
        return [i for i in instances if isinstance(i, dict)]

    def find_reusable_running_instance(
        self,
        *,
        min_dph: float,
        max_dph: float,
        reliability: float,
    ) -> dict | None:
        """Return a best-effort running instance matching constraints, if any."""
        candidates: list[dict] = []
        for inst in self.list_instances():
            status = str(inst.get("actual_status", inst.get("state", inst.get("status", "")))).lower()
            if status != "running":
                continue

            dph = self._first_float(inst, ("dph_total", "dph", "price"))
            if dph is not None and not (min_dph <= dph <= max_dph):
                continue

            rel = self._first_float(inst, ("reliability2", "reliability", "machine_reliability"))
            if rel is not None and rel < reliability:
                continue

            candidates.append(inst)

        if not candidates:
            return None

        # Prefer bigger VRAM, then lower price.
        def _score(item: dict) -> tuple[int, float]:
            vram = int(self._first_float(item, ("gpu_total_ram", "gpu_ram")) or 0)
            dph = float(self._first_float(item, ("dph_total", "dph", "price")) or 9999.0)
            return (vram, -dph)

        candidates.sort(key=_score, reverse=True)
        debug_log(
            f"Vast.ai reusable instance selected: contract_id={candidates[0].get('id')} "
            f"gpu_total_ram={candidates[0].get('gpu_total_ram')}"
        )
        return candidates[0]

    def stop_instance(self, contract_id: int | str) -> None:
        """Destroy only the instance contract."""
        self._destroy_instance(contract_id)

    def destroy_instance_and_related_storage(self, contract_id: int | str) -> TeardownResult:
        """Destroy instance and related volumes, then verify spend stopped.

        This attempts to remove the contract and any attached volume IDs discoverable
        from the instance payload, then polls the API to ensure the contract no longer
        appears among active instances.
        """
        snapshot = self._fetch_instance(contract_id)
        volume_ids = self._extract_related_volume_ids(snapshot or {})

        self._destroy_instance(contract_id)

        deleted_volumes: list[str] = []
        for volume_id in volume_ids:
            try:
                self._destroy_volume(volume_id)
                deleted_volumes.append(volume_id)
            except RuntimeError:
                # Keep going; verification below will report any leftovers.
                continue

        instance_gone = self._wait_until_instance_absent(contract_id)
        volumes_gone, volume_msg = self._verify_volumes_deleted(volume_ids)
        verification_ok = instance_gone and volumes_gone

        parts = []
        parts.append(
            "contract no longer active" if instance_gone else "contract still present in show_instances"
        )
        parts.append(volume_msg)

        return TeardownResult(
            contract_id=contract_id,
            destroyed_instance=True,
            destroyed_volume_ids=deleted_volumes,
            verification_ok=verification_ok,
            verification_message="; ".join(parts),
        )

    def _destroy_instance(self, contract_id: int | str) -> None:
        """Destroy an instance so billing stops."""
        cid = str(contract_id)
        try:
            debug_log(f"Vast.ai API call: destroy_instance id={cid} raw=True")
            self._sdk.destroy_instance(id=cid, raw=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to destroy Vast.ai instance {cid}: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Block 8 — Pull model via Ollama                                     #
    # ------------------------------------------------------------------ #

    def pull_model(self, contract_id: int | str, model_name: str, ollama_url: str | None = None) -> None:
        """Load a model into Ollama on the remote instance via Ollama API.

        Args:
            contract_id: The Vast.ai contract/instance id (for error context).
            model_name: The model name to load (e.g., "mistral:latest").
            ollama_url: Ollama API URL (e.g., "http://ip:11434").

        Raises:
            RuntimeError: if no Ollama URL is provided or API load fails.
        """
        if not ollama_url:
            raise RuntimeError(
                f"Cannot pull model for contract {contract_id}: missing Ollama URL"
            )

        try:
            debug_log(
                f"Vast.ai Ollama model load: contract_id={contract_id} ollama_url={ollama_url} model={model_name}"
            )
            client = OllamaClient(ollama_url)
            client.load_model(model_name)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Model pull via Ollama API failed for contract {contract_id}: {exc}"
            ) from exc

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
            debug_log("Vast.ai API call: show_instances raw=True")
            instances = self._sdk.show_instances(raw=True)
        except Exception:
            return None
        if not isinstance(instances, list):
            return None
        return next(
            (i for i in instances if str(i.get("id")) == str(contract_id)),
            None,
        )

    def _wait_until_instance_absent(self, contract_id: int | str, timeout_seconds: int = 120) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self._fetch_instance(contract_id) is None:
                return True
            time.sleep(5)
        return self._fetch_instance(contract_id) is None

    def _list_volumes_optional(self) -> list[dict] | None:
        try:
            debug_log("Vast.ai API call: show_volumes raw=True")
            result = self._sdk.show_volumes(raw=True)
        except Exception:
            return None

        if isinstance(result, list):
            return [v for v in result if isinstance(v, dict)]
        if isinstance(result, dict):
            items = result.get("volumes")
            if isinstance(items, list):
                return [v for v in items if isinstance(v, dict)]
        return None

    def _destroy_volume(self, volume_id: str) -> None:
        try:
            debug_log(f"Vast.ai API call: destroy_volume id={volume_id} raw=True")
            self._sdk.destroy_volume(id=volume_id, raw=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to destroy volume {volume_id}: {exc}") from exc

    def _verify_volumes_deleted(self, volume_ids: list[str]) -> tuple[bool, str]:
        if not volume_ids:
            return True, "no related volumes detected"

        volumes = self._list_volumes_optional()
        if volumes is None:
            return False, "could not verify volumes (SDK has no volume-list method)"

        existing_ids = {
            str(v.get("id"))
            for v in volumes
            if v.get("id") is not None
        }
        remaining = [vid for vid in volume_ids if str(vid) in existing_ids]
        if remaining:
            return False, f"remaining volumes: {', '.join(remaining)}"
        return True, "related volumes deleted"

    @staticmethod
    def _extract_related_volume_ids(instance: dict) -> list[str]:
        ids: set[str] = set()

        if not isinstance(instance, dict):
            return []

        keys = (
            "volume_id",
            "volumeId",
            "vol_id",
            "storage_id",
            "disk_id",
        )
        for key in keys:
            value = instance.get(key)
            if value is not None and str(value).strip():
                ids.add(str(value))

        volume = instance.get("volume")
        if isinstance(volume, dict) and volume.get("id") is not None:
            ids.add(str(volume.get("id")))

        volumes = instance.get("volumes")
        if isinstance(volumes, list):
            for item in volumes:
                if isinstance(item, dict) and item.get("id") is not None:
                    ids.add(str(item.get("id")))

        return sorted(ids)

    @staticmethod
    def _first_float(data: dict, fields: tuple[str, ...]) -> float | None:
        for field_name in fields:
            value = data.get(field_name)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None
