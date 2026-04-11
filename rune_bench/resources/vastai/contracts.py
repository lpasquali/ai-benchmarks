# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

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
