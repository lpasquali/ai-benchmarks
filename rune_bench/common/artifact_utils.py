# SPDX-License-Identifier: Apache-2.0
"""Utility to process and normalize agent artifacts."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from rune_bench.storage.base import StoragePort

log = logging.getLogger(__name__)

def process_agent_artifacts(
    job_id: str,
    artifacts: list[Any] | dict[str, Any] | None,
    storage: StoragePort,
) -> list[Any] | dict[str, Any] | None:
    """Detect local file paths in artifacts, upload them, and return proxy URLs."""
    if not artifacts:
        return artifacts

    if isinstance(artifacts, list):
        return [process_agent_artifacts(job_id, a, storage) for a in artifacts]

    if isinstance(artifacts, dict):
        new_artifacts = {}
        for k, v in artifacts.items():
            if isinstance(v, (list, dict)):
                new_artifacts[k] = process_agent_artifacts(job_id, v, storage)
            elif isinstance(v, str) and (v.startswith("/") or (":" in v and "\\" in v)):
                # Heuristic: looks like an absolute path
                p = Path(v)
                if p.exists() and p.is_file():
                    try:
                        with p.open("rb") as f:
                            content = f.read()
                        
                        kind = _guess_artifact_kind(p.name)
                        artifact_id = storage.record_audit_artifact(
                            job_id=job_id,
                            kind=kind,
                            name=p.name,
                            content=content
                        )
                        # Return the proxy URL instead of the local path
                        new_artifacts[k] = f"/v1/runs/{job_id}/artifacts/{artifact_id}"
                        log.info("Uploaded artifact %s -> %s", v, new_artifacts[k])
                    except Exception as exc:
                        log.warning("Failed to upload artifact %s: %s", v, exc)
                        new_artifacts[k] = v
                else:
                    new_artifacts[k] = v
            else:
                new_artifacts[k] = v
        return new_artifacts

    return artifacts

def _guess_artifact_kind(filename: str) -> str:
    """Map filename extension to an audit_artifact kind."""
    ext = filename.split(".")[-1].lower()
    if ext in ("png", "jpg", "jpeg", "gif", "webp"):
        return "screenshot" # We use screenshot as a generic image kind for now
    if ext == "json":
        return "sbom" # Heuristic, could be others
    if ext in ("txt", "log"):
        return "log"
    return "rekor_entry" # fallback to a generic kind supported by schema
