import os
import re
from pathlib import Path

TESTS_DIR = Path("/home/luca/Devel/rune/tests")

def fix_test_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # 1. Update imports from agents to drivers
    # Match: from rune_bench.agents.sre.holmes import HolmesRunner
    # Replace: from rune_bench.drivers.holmes import HolmesDriverClient
    
    # Simple map of agent names to their driver modules
    agent_to_driver = {
        "holmes": "holmes",
        "k8sgpt": "k8sgpt",
        "metoro": "metoro",
        "pagerduty": "pagerduty",
        "perplexity": "perplexity",
        "glean": "glean",
        "elicit": "elicit",
        "langgraph": "langgraph",
        "consensus": "consensus",
        "pentestgpt": "pentestgpt",
        "radiant": "radiant",
        "mindgard": "mindgard",
        "burpgpt": "burpgpt",
        "xbow": "xbow",
        "harvey": "harvey",
        "spellbook": "spellbook",
        "dagger": "dagger",
        "crewai": "crewai",
        "browseruse": "browseruse",
        "multion": "multion",
        "sierra": "sierra",
        "cleric": "cleric",
        "skillfortify": "skillfortify",
        "midjourney": "midjourney",
        "invokeai": "invokeai",
        "comfyui": "comfyui",
        "krea": "krea",
    }
    
    new_content = content
    
    # Replace imports
    for agent, driver in agent_to_driver.items():
        # Match from rune_bench.agents.<scope>.<agent> import <Agent>Runner
        # Since we don't know the scope easily, we use a wildcard
        import_pattern = re.compile(rf'from rune_bench\.agents\.[a-z.]+\.{agent} import ([A-Za-z0-9]+Runner|[A-Za-z0-9]+AIRunner|[A-Za-z0-9]+SecurityRunner)')
        
        def import_replacer(match):
            return f'from rune_bench.drivers.{driver} import {agent.capitalize()}DriverClient'
        
        # Special case for HarveyAIRunner, RadiantSecurityRunner, K8sGPTRunner
        if agent == "harvey":
            import_pattern = re.compile(r'from rune_bench\.agents\.legal\.harvey import HarveyAIRunner')
            new_content = import_pattern.sub(f'from rune_bench.drivers.harvey import HarveyDriverClient', new_content)
        elif agent == "radiant":
            import_pattern = re.compile(r'from rune_bench\.agents\.cybersec\.radiant import RadiantSecurityRunner')
            new_content = import_pattern.sub(f'from rune_bench.drivers.radiant import RadiantSecurityDriverClient', new_content)
        elif agent == "k8sgpt":
             import_pattern = re.compile(r'from rune_bench\.agents\.sre\.k8sgpt import K8sGPTRunner')
             new_content = import_pattern.sub(f'from rune_bench.drivers.k8sgpt import K8sGPTDriverClient', new_content)
        else:
            new_content = import_pattern.sub(import_replacer, new_content)

    # 2. Update usages in code
    # Replace <Agent>Runner with <Agent>DriverClient
    for agent in agent_to_driver:
        if agent == "harvey":
            new_content = new_content.replace("HarveyAIRunner", "HarveyDriverClient")
        elif agent == "radiant":
            new_content = new_content.replace("RadiantSecurityRunner", "RadiantSecurityDriverClient")
        elif agent == "k8sgpt":
            new_content = new_content.replace("K8sGPTRunner", "K8sGPTDriverClient")
        elif agent == "pagerduty":
            new_content = new_content.replace("PagerDutyAIRunner", "PagerDutyDriverClient")
        else:
            new_content = new_content.replace(f"{agent.capitalize()}Runner", f"{agent.capitalize()}DriverClient")

    # 3. Remove alias tests
    # Match: def test_.*_alias.*?:.*?(\n\n|\Z)
    test_pattern = re.compile(r'def test_.*_alias.*?\n(    .*?\n)*', re.MULTILINE)
    new_content = test_pattern.sub('', new_content)
    
    # 4. Remove comments about aliases
    new_content = re.sub(r'# .*alias.*?\n', '', new_content, flags=re.IGNORECASE)

    # 5. Fix specific failed imports like from rune_bench.drivers.glean import GleanDriverClient, GleanRunner
    new_content = re.sub(r'from rune_bench\.drivers\.([a-z0-9]+) import ([A-Za-z0-9]+DriverClient), [A-Za-z0-9]+Runner', r'from rune_bench.drivers.\1 import \2', new_content)

    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        return True
    return False

for root, dirs, files in os.walk(TESTS_DIR):
    for file in files:
        if file.endswith('.py'):
            file_path = Path(root) / file
            if fix_test_file(file_path):
                print(f"Updated {file_path}")
