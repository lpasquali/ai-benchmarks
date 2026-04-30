import os
import re
from pathlib import Path

DRIVERS_DIR = Path("/home/luca/Devel/rune/rune_bench/drivers")

def update_driver_init(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # 1. Update __init__ to accept **kwargs (simplest way to support all registry needs)
    # Match: def __init__(self, ..., transport: DriverTransport | None = None) -> None:
    # Or variations
    
    # Let's find the __init__ and add **kwargs
    init_pattern = re.compile(r'def __init__\s*\((.*?)\)\s*->\s*None:', re.DOTALL)
    
    def init_replacer(match):
        args = match.group(1).strip()
        if '**kwargs' in args:
            return match.group(0)
        
        # Add **kwargs to the end of arguments
        if args.endswith(','):
            new_args = args + ' **kwargs'
        else:
            new_args = args + ', **kwargs'
        
        return f'def __init__({new_args}) -> None:'

    new_content = init_pattern.sub(init_replacer, content)
    
    # 2. Remove Runner aliases
    # Match: name_pattern = re.compile(r'^[A-Za-z0-9]+Runner = [A-Za-z0-9]+DriverClient$', re.MULTILINE)
    # Or variations
    alias_patterns = [
        r'^[A-Za-z0-9]+Runner = [A-Za-z0-9]+DriverClient.*$',
        r'^[A-Za-z0-9]+AIRunner = [A-Za-z0-9]+DriverClient.*$',
        r'^[A-Za-z0-9]+SecurityRunner = [A-Za-z0-9]+DriverClient.*$',
    ]
    
    for p in alias_patterns:
        new_content = re.sub(p, '', new_content, flags=re.MULTILINE)

    # 3. Remove from __all__
    # This is trickier, but let's see if we can just remove the runner names from __all__ = [...]
    all_pattern = re.compile(r'__all__ = \[(.*?)\]', re.DOTALL)
    
    def all_replacer(match):
        items = match.group(1)
        # Remove anything ending in Runner
        new_items = re.sub(r'"[A-Za-z0-9]+Runner",?\s*', '', items)
        new_items = re.sub(r"'[A-Za-z0-9]+Runner',?\s*", '', new_items)
        # Cleanup trailing commas
        new_items = new_items.strip()
        if new_items.endswith(','):
            new_items = new_items[:-1].strip()
        return f'__all__ = [{new_items}]'

    new_content = all_pattern.sub(all_replacer, new_content)

    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        return True
    return False

for root, dirs, files in os.walk(DRIVERS_DIR):
    if '__init__.py' in files:
        file_path = Path(root) / '__init__.py'
        if update_driver_init(file_path):
            print(f"Updated {file_path}")
