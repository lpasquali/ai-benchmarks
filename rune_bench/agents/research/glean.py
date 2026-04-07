# SPDX-License-Identifier: Apache-2.0
"""Glean agentic runner -- delegates to the Glean driver.

Scope:      Research  |  Rank 2  |  Rating 4.8
Capability: Autonomous internal knowledge discovery for enterprises.
Docs:       https://developers.glean.com/
            https://developers.glean.com/docs/search_api/
Ecosystem:  Enterprise Search

Implementation notes:
- Auth:     RUNE_GLEAN_API_TOKEN + RUNE_GLEAN_INSTANCE (subdomain) env vars
- SDK:      REST API; no official Python SDK
            Base URL: https://<instance>-be.glean.com/api/v1
- Key endpoints:
    POST /search          # full-text + semantic search
    POST /chat            # agentic chat with internal knowledge
            body: { messages: [{role, content}], stream: false }
- The `question` maps to the chat message content.
- `model` and `backend_url` are not used (Glean uses its own hosted model).
"""

from rune_bench.drivers.glean import GleanDriverClient

GleanRunner = GleanDriverClient
