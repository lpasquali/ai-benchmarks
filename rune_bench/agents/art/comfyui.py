"""ComfyUI agentic runner stub.

Scope:      Art/Creative  |  Rank 2  |  Rating 4.5
Capability: Node-based autonomous art pipeline orchestration.
Docs:       https://github.com/comfy-org/ComfyUI
            https://github.com/comfy-org/ComfyUI/wiki/API
            https://docs.comfy.org/
Ecosystem:  OSS Community

Implementation notes:
- Install:  ComfyUI runs as a local server (Python); no pip install needed.
            git clone https://github.com/comfy-org/ComfyUI && python main.py
- Auth:     No auth by default (local); COMFYUI_BASE_URL env var (default: http://127.0.0.1:8188)
- SDK:      REST + WebSocket API
    POST /prompt                body: { prompt: <workflow_json>, client_id: str }
    GET  /history/{prompt_id}   poll until outputs appear
    GET  /view?filename=<f>     download generated image
- Approach:
    1. Load a workflow JSON template.
    2. Inject the text prompt into the KSampler / CLIPTextEncode node.
    3. POST to /prompt, poll /history, retrieve output image.
- The `question` maps to the positive text prompt.
- `model` can specify the checkpoint model (injected into CheckpointLoaderSimple node).
- `ollama_url` is not used.
"""


class ComfyUIRunner:
    """Art/Creative agent: autonomous art pipeline orchestration via ComfyUI node graph."""

    def __init__(self, base_url: str = "http://127.0.0.1:8188") -> None:
        self._base_url = base_url.rstrip("/")

    def ask(self, question: str, model: str, ollama_url: str | None = None) -> str:
        """Run a ComfyUI workflow with the given prompt and return output image path/URL."""
        raise NotImplementedError(
            "ComfyUIRunner is not yet implemented. "
            "See https://github.com/comfy-org/ComfyUI/wiki/API for REST+WebSocket API details."
        )
