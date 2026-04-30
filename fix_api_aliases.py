import os
from pathlib import Path

RUNE_DIR = Path("/home/luca/Devel/rune")

def replace_deprecated_api(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    new_content = content
    # Replace method names in client calls
    new_content = new_content.replace("get_llm_models", "get_llm_models")
    new_content = new_content.replace("submit_llm_instance_job", "submit_llm_instance_job")
    
    # Replace endpoint strings in tests/server
    new_content = new_content.replace("/v1/llm/models", "/v1/llm/models")
    new_content = new_content.replace("/v1/jobs/llm-instance", "/v1/jobs/llm-instance")
    new_content = new_content.replace("llm-instance", "llm-instance")

    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        return True
    return False

for root, dirs, files in os.walk(RUNE_DIR):
    for file in files:
        if file.endswith('.py'):
            file_path = Path(root) / file
            if replace_deprecated_api(file_path):
                print(f"Updated {file_path}")
