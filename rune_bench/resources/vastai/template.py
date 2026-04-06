"""Block 4 — Template Loading.

Fetches Vast.ai templates and resolves a template by its hash,
extracting the env flags and Docker image for instance creation.
"""

from dataclasses import dataclass

from rune_bench.resources.vastai.sdk import VastAI

from rune_bench.debug import debug_log

_HASH_FIELDS = ("id", "hash_id", "hash", "template_hash")


@dataclass
class Template:
    env: str          # raw env string from the template (e.g. "-e VAR=val -p 11434:11434")
    image: str | None # docker image, if set in the template
    raw: dict


class TemplateLoader:
    """Resolve a Vast.ai template by hash and extract its configuration."""

    def __init__(self, sdk: VastAI) -> None:
        self._sdk = sdk

    def load(self, template_hash: str) -> Template:
        """Fetch all templates and return the one matching *template_hash*.

        The /workspace volume mount flag is appended to the template env so
        model weights are stored in the correct persistent path.

        Raises:
            RuntimeError: if the template is not found.
        """
        try:
            debug_log("Vast.ai API call: show_templates raw=True")
            templates = self._sdk.show_templates(raw=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch Vast.ai templates: {exc}") from exc

        debug_log(
            f"Vast.ai API result: show_templates count={len(templates) if isinstance(templates, list) else '<non-list>'}"
        )

        if not isinstance(templates, list):
            templates = []

        match = self._find(templates, template_hash)
        if match is None:
            raise RuntimeError(
                f"Template '{template_hash}' not found. "
                "Check the hash or run: python -m vastai show templates"
            )

        debug_log(f"Vast.ai template selected: hash={template_hash} image={match.get('image') or match.get('docker_image')}")

        raw_env = str(match.get("env", "")).strip()
        final_env = f"{raw_env} -v /workspace".strip()
        image = match.get("image") or match.get("docker_image")

        return Template(env=final_env, image=image, raw=match)

    @staticmethod
    def _find(templates: list[dict], template_hash: str) -> dict | None:
        for t in templates:
            for field in _HASH_FIELDS:
                if str(t.get(field, "")) == template_hash:
                    return t
        return None
