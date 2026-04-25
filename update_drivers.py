#!/usr/bin/env python3
import os

DRIVERS = {
    "midjourney": ("art.midjourney", "MidjourneyRunner", "image"),
    "krea": ("art.krea", "KreaRunner", "image"),
    "comfyui": ("art.comfyui", "ComfyUIRunner", "image"),
    "sierra": ("ops.sierra", "SierraRunner", "text"),
    "multion": ("ops.multion", "MultiOnRunner", "text"),
    "xbow": ("cyber.xbow", "XBOWRunner", "text"),
    "radiant": ("cybersec.radiant", "RadiantSecurityRunner", "text"),
    "cleric": ("sre.cleric", "ClericRunner", "text"),
    "spellbook": ("legal.spellbook", "SpellbookRunner", "text"),
    "harvey": ("legal.harvey", "HarveyAIRunner", "text"),
    "browseruse": ("ops.browser_use", "BrowserUseRunner", "text"),
    "skillfortify": ("ops.skillfortify", "SkillFortifyRunner", "text"),
}

TEMPLATE = """# SPDX-License-Identifier: Apache-2.0
\"\"\"Actual implementation for {name} driver.\"\"\"

from __future__ import annotations

import json
import os
import sys

from rune_bench.agents.{module} import {class_name}


def _handle_ask(params: dict) -> dict:
    api_key = os.getenv(\"RUNE_{upper_name}_API_KEY\")
    if not api_key:
        # Re-verify driver-specific env var for tests that expect it
        raise RuntimeError(f\"RUNE_{upper_name}_API_KEY not set\")
    
    api_base = os.getenv(\"RUNE_{upper_name}_API_BASE\")
    
    question = params.get(\"question\", \"\")
    model = params.get(\"model\", \"\")
    
    # Instantiate runner (names vary slightly but we pass what we have)
    try:
        runner = {class_name}(api_key=api_key)
    except TypeError:
        # Some might take base_url instead or as well
        runner = {class_name}(api_key=api_key, api_base=api_base)
    
    answer = runner.ask(question, model=model)
    
    return {{
        \"answer\": answer,
        \"result_type\": \"{result_type}\",
    }}


def _handle_info(_params: dict) -> dict:
    return {{
        \"name\": \"{name}\",
        \"version\": \"1\",
        \"actions\": [\"ask\", \"info\"],
        \"status\": \"active\",
    }}


_HANDLERS: dict = {{
    \"ask\": \"_handle_ask\",
    \"info\": \"_handle_info\",
}}


def main() -> None:
    \"\"\"Read JSON requests from stdin and write JSON responses to stdout.\"\"\"
    current_module = sys.modules[__name__]
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        req_id = \"\"
        try:
            request = json.loads(line)
            req_id = str(request.get(\"id\", \"\"))
            action = str(request.get(\"action\", \"\"))
            params = request.get(\"params\") or {{}}
            handler_name = _HANDLERS.get(action)
            if handler_name is None:
                raise RuntimeError(f\"Unknown action: {{action!r}}\")
            handler = getattr(current_module, handler_name)
            result = handler(params)
            print(
                json.dumps({{\"status\": \"ok\", \"result\": result, \"id\": req_id}}), flush=True
            )
        except Exception as exc:  # noqa: BLE001
            print(
                json.dumps({{\"status\": \"error\", \"error\": str(exc), \"id\": req_id}}),
                flush=True,
            )


if __name__ == \"__main__\":
    main()
"""

def update_drivers():
    base_path = "rune_bench/drivers"
    for name, (module, class_name, res_type) in DRIVERS.items():
        dir_path = os.path.join(base_path, name)
        os.makedirs(dir_path, exist_ok=True)
        
        main_path = os.path.join(dir_path, "__main__.py")
        content = TEMPLATE.format(
            name=name,
            upper_name=name.upper(),
            module=module,
            class_name=class_name,
            result_type=res_type
        )
        with open(main_path, "w") as f:
            f.write(content)
        
        init_path = os.path.join(dir_path, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w") as f:
                f.write(f"# SPDX-License-Identifier: Apache-2.0\nfrom rune_bench.agents.{module} import {class_name} as {class_name}Client\n")

if __name__ == "__main__":
    update_drivers()
