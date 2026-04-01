from __future__ import annotations

import os

from rune_bench.api_server import RuneApiApplication


def main() -> None:
    host = os.environ.get("RUNE_API_HOST", "0.0.0.0")
    port = int(os.environ.get("RUNE_API_PORT", "8080"))
    app = RuneApiApplication.from_env()
    app.serve(host=host, port=port)


if __name__ == "__main__":
    main()
