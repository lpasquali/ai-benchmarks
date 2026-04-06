"""Lightweight Vast.ai API client to replace vastai-sdk and avoid CVEs."""

import json
import urllib.parse
from typing import Any

import httpx



class VastAI:
    """Minimal wrapper around the Vast.ai REST API."""

    def __init__(self, api_key: str, raw: bool = True) -> None:
        self.api_key = api_key
        self.base_url = "https://console.vast.ai/api/v0"

    def _get_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def search_offers(
        self, query: dict[str, Any], order: str = "score-", disable_bundling: bool = True, raw: bool = True
    ) -> list[dict[str, Any]]:
        if not isinstance(query, dict):
            raise ValueError("query must be a dict")
        query_str = json.dumps(query)
        query_encoded = urllib.parse.quote(query_str)
        url = f"{self.base_url}/bundles/?q={query_encoded}"
        
        with httpx.Client() as client:
            resp = client.get(url, headers=self._get_headers(), timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            return data.get("offers", [])

    def show_templates(self, raw: bool = True) -> list[dict[str, Any]]:
        url = f"{self.base_url}/users/current/templates/"
        with httpx.Client() as client:
            resp = client.get(url, headers=self._get_headers(), timeout=30.0)
            resp.raise_for_status()
            return resp.json()

    def create_instance(self, id: str, disk: float, env: str, image: str | None = None, raw: bool = True) -> dict[str, Any]:
        url = f"{self.base_url}/asks/{id}/"
        payload = {
            "client_id": "me",
            "disk": disk,
            "env": env,
            "runtype": "ssh_direct",
        }
        if image:
            payload["image"] = image

        with httpx.Client() as client:
            resp = client.put(url, json=payload, headers=self._get_headers(), timeout=30.0)
            resp.raise_for_status()
            return resp.json()

    def show_instances(self, raw: bool = True) -> list[dict[str, Any]]:
        url = f"{self.base_url}/instances/"
        with httpx.Client() as client:
            resp = client.get(url, headers=self._get_headers(), timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            return data.get("instances", [])

    def destroy_instance(self, id: str, raw: bool = True) -> dict[str, Any]:
        url = f"{self.base_url}/instances/{id}/"
        with httpx.Client() as client:
            resp = client.delete(url, headers=self._get_headers(), timeout=30.0)
            resp.raise_for_status()
            return resp.json()

    def show_volumes(self, raw: bool = True) -> list[dict[str, Any]]:
        url = f"{self.base_url}/volumes/"
        with httpx.Client() as client:
            resp = client.get(url, headers=self._get_headers(), timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            # API returns a list directly or {"volumes": [...]}
            if isinstance(data, dict):
                return data.get("volumes", [])
            return data

    def destroy_volume(self, id: str, raw: bool = True) -> dict[str, Any]:
        url = f"{self.base_url}/volumes/{id}/"
        with httpx.Client() as client:
            resp = client.delete(url, headers=self._get_headers(), timeout=30.0)
            resp.raise_for_status()
            return resp.json()
