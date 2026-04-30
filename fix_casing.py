import os
import re
from pathlib import Path

TESTS_DIR = Path("/home/luca/Devel/rune/tests")

def fix_casing_in_tests(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    new_content = content
    
    # Standardize DriverClient casing
    replacements = {
        "InvokeaiDriverClient": "InvokeAIDriverClient",
        "BrowseruseDriverClient": "BrowserUseDriverClient",
        "PentestgptDriverClient": "PentestGPTDriverClient",
        "CrewaiDriverClient": "CrewAIDriverClient",
        "LanggraphDriverClient": "LangGraphDriverClient",
        "XbowDriverClient": "XbowDriverClient", # Correct as per __init__.py read
        "MultionDriverClient": "MultiOnDriverClient",
    }
    
    for old, new in replacements.items():
        new_content = new_content.replace(old, new)

    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        return True
    return False

for root, dirs, files in os.walk(TESTS_DIR):
    for file in files:
        if file.endswith('.py'):
            file_path = Path(root) / file
            if fix_casing_in_tests(file_path):
                print(f"Fixed casing in {file_path}")
