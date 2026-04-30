import os
import re
from pathlib import Path

TESTS_DIR = Path("/home/luca/Devel/rune/tests")

def finalize_test_fix(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    new_content = content
    
    # 1. Replace ExistingOllamaProvider with ExistingBackendProvider
    new_content = new_content.replace("ExistingOllamaProvider", "ExistingBackendProvider")
    new_content = new_content.replace("from rune_bench.resources.existing_ollama_provider import ExistingBackendProvider", 
                                      "from rune_bench.resources.existing_backend_provider import ExistingBackendProvider")
    
    # 2. Replace job_store imports
    new_content = new_content.replace("from rune_bench.job_store import JobStore", "from rune_bench.storage.sqlite import SQLiteStorageAdapter as JobStore")
    new_content = new_content.replace("from rune_bench.job_store import", "from rune_bench.storage.sqlite import")
    new_content = new_content.replace("rune_bench.job_store", "rune_bench.storage.sqlite")
    
    # 3. Replace OllamaModelCapabilities with ModelCapabilities
    new_content = new_content.replace("OllamaModelCapabilities", "ModelCapabilities")
    
    # 4. Specific fix for new_agents_coverage (ComfyuiDriverClient vs ComfyUIDriverClient)
    new_content = new_content.replace("ComfyuiDriverClient", "ComfyUIDriverClient")

    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        return True
    return False

for root, dirs, files in os.walk(TESTS_DIR):
    for file in files:
        if file.endswith('.py'):
            file_path = Path(root) / file
            if finalize_test_fix(file_path):
                print(f"Finalized {file_path}")
