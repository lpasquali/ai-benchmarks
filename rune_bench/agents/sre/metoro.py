"""Metoro agentic runner -- delegates to the Metoro driver.

Scope:      SRE  |  Rank 4  |  Rating 4.0
Capability: Uses eBPF for autonomous service mapping and debugging.
Docs:       https://metoro.io/docs
            https://metoro.io/docs/api
Ecosystem:  CNCF / eBPF

Implementation notes:
- Auth:     RUNE_METORO_API_KEY env var
- SDK:      REST API (no official Python SDK as of writing)
            Base URL: https://app.metoro.io/api  (or self-hosted endpoint)
- Key endpoints:
    GET  /services                    # list detected services
    GET  /traces?service=<s>          # eBPF-captured traces
    POST /ai/explain                  # LLM-powered incident explanation
            body: { question, service, time_range }
- The `question` maps to the explanation query.
- `model` / `backend_url` may be passed to the self-hosted Metoro AI endpoint.
"""

from rune_bench.drivers.metoro import MetoroDriverClient

MetoroRunner = MetoroDriverClient
