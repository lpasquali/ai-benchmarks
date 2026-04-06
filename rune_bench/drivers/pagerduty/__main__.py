"""PagerDuty driver entry point -- receives JSON actions on stdin, writes results to stdout.

Run as::

    python -m rune_bench.drivers.pagerduty

Wire protocol (v1):
    stdin  line: {"action": "ACTION", "params": {...}, "id": "UUID"}
    stdout line: {"status": "ok"|"error", "result": {...}, "error": "...", "id": "UUID"}

Supported actions
-----------------
ask
    params: question (str), model (str, optional),
            backend_url (str, optional)
    result: {"answer": str, "incidents": list}

    .. note:: The ``triage_actions`` field is not yet included in the result.
       A future version will add recommended actions alongside the triage summary.

info
    params: (none)
    result: {"name": "pagerduty", "version": "1", "actions": [...]}
"""

from __future__ import annotations

import json
import os
import sys

from rune_bench.common.http_client import make_http_request, normalize_url

_PAGERDUTY_API_BASE = "https://api.pagerduty.com"


def _pd_request(path: str, api_key: str, *, action: str | None = None) -> dict:
    """Make an authenticated GET request to the PagerDuty REST v2 API."""
    if action is None:
        # Derive a human-readable action from the endpoint path.
        segment = path.lstrip("/").split("?")[0].replace("/", " ")
        action = f"fetch PagerDuty {segment}"
    url = f"{_PAGERDUTY_API_BASE}{path}"
    return make_http_request(
        url,
        method="GET",
        action=action,
        headers={
            "Authorization": f"Token token={api_key}",
            "Accept": "application/vnd.pagerduty+json;version=2",
        },
    )


def _fetch_open_incidents(api_key: str) -> list[dict]:
    """Fetch triggered and acknowledged incidents from PagerDuty.

    Paginates through all result pages using the ``more`` flag and ``offset``
    parameter returned by the PagerDuty REST v2 API.
    """
    all_incidents: list[dict] = []
    limit = 25
    offset = 0
    while True:
        data = _pd_request(
            f"/incidents?statuses[]=triggered&statuses[]=acknowledged&limit={limit}&offset={offset}",
            api_key,
            action="fetch PagerDuty incidents",
        )
        all_incidents.extend(data.get("incidents", []))
        if not data.get("more", False):
            break
        offset += limit
    return all_incidents


def _fetch_alerts_for_incident(incident_id: str, api_key: str) -> list[dict]:
    """Fetch alerts associated with a specific incident."""
    data = _pd_request(f"/incidents/{incident_id}/alerts", api_key)
    return data.get("alerts", [])


def _format_incident_data(incidents: list[dict], alerts_by_incident: dict[str, list[dict]]) -> str:
    """Format incident and alert data as structured text for LLM consumption."""
    if not incidents:
        return "No open incidents found."

    lines: list[str] = []
    for inc in incidents:
        inc_id = inc.get("id", "unknown")
        lines.append(f"Incident: {inc.get('title', 'N/A')} (ID: {inc_id})")
        lines.append(f"  Status: {inc.get('status', 'N/A')}")
        lines.append(f"  Urgency: {inc.get('urgency', 'N/A')}")
        lines.append(f"  Service: {inc.get('service', {}).get('summary', 'N/A')}")
        lines.append(f"  Created: {inc.get('created_at', 'N/A')}")

        alerts = alerts_by_incident.get(inc_id, [])
        if alerts:
            lines.append(f"  Alerts ({len(alerts)}):")
            for alert in alerts:
                lines.append(f"    - {alert.get('summary', 'N/A')} "
                             f"(severity: {alert.get('severity', 'N/A')})")
        lines.append("")

    return "\n".join(lines)


def _call_ollama(prompt: str, model: str, backend_url: str) -> str:
    """Call the Ollama /api/generate endpoint for triage synthesis."""
    base = normalize_url(backend_url, "Ollama")
    url = f"{base.rstrip('/')}/api/generate"
    result = make_http_request(
        url,
        method="POST",
        payload={"model": model, "prompt": prompt, "stream": False},
        action="synthesize via Ollama",
    )
    return result.get("response", "")


def _handle_ask(params: dict) -> dict:
    question: str = params["question"]
    model: str = params.get("model", "")
    backend_url: str | None = params.get("backend_url")

    api_key = os.environ.get("RUNE_PAGERDUTY_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "RUNE_PAGERDUTY_API_KEY environment variable is not set. "
            "A PagerDuty REST API v2 token is required."
        )

    incidents = _fetch_open_incidents(api_key)

    # Short-circuit: skip alert fetching and LLM synthesis when there are no incidents.
    if not incidents:
        return {"answer": "No open incidents found.", "incidents": []}

    alerts_by_incident: dict[str, list[dict]] = {}
    for inc in incidents:
        inc_id = inc.get("id", "")
        if inc_id:
            alerts_by_incident[inc_id] = _fetch_alerts_for_incident(inc_id, api_key)

    formatted_data = _format_incident_data(incidents, alerts_by_incident)

    if model and backend_url:
        prompt = (
            "You are an SRE triage agent. Given these PagerDuty incidents, provide: "
            "1) Prioritized triage summary, 2) Correlation between related incidents, "
            "3) Recommended immediate actions.\n\n"
            f"Question: {question}\n\n"
            f"Incidents:\n{formatted_data}"
        )
        answer = _call_ollama(prompt, model, backend_url)
    else:
        answer = formatted_data

    # Build a minimal incident summary list for structured consumers.
    incidents_list = [
        {
            "id": inc.get("id", ""),
            "title": inc.get("title", ""),
            "status": inc.get("status", ""),
            "urgency": inc.get("urgency", ""),
        }
        for inc in incidents
    ]

    return {"answer": answer, "incidents": incidents_list}


def _handle_info(_params: dict) -> dict:
    return {"name": "pagerduty", "version": "1", "actions": ["ask", "info"]}


_HANDLERS: dict = {
    "ask": "_handle_ask",
    "info": "_handle_info",
}


def main() -> None:
    """Read JSON requests from stdin and write JSON responses to stdout."""
    current_module = sys.modules[__name__]
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        req_id = ""
        try:
            request = json.loads(line)
            req_id = str(request.get("id", ""))
            action = str(request.get("action", ""))
            params = request.get("params") or {}

            handler_name = _HANDLERS.get(action)
            if handler_name is None:
                raise RuntimeError(f"Unknown action: {action!r}")
            handler = getattr(current_module, handler_name)

            result = handler(params)
            print(json.dumps({"status": "ok", "result": result, "id": req_id}), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(
                json.dumps({"status": "error", "error": str(exc), "id": req_id}),
                flush=True,
            )


if __name__ == "__main__":
    main()
