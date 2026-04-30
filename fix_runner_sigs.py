import os
import re
from pathlib import Path

DRIVERS_DIR = Path("/home/luca/Devel/rune/rune_bench/drivers")

def fix_runner_signatures(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # 1. Update __init__ to accept **kwargs
    init_pattern = re.compile(r'def __init__\s*\((.*?)\)\s*->\s*None:', re.DOTALL)
    
    def init_replacer(match):
        args = match.group(1).strip()
        if '**kwargs' in args:
            return match.group(0)
        if args.endswith(','):
            new_args = args + ' **kwargs'
        else:
            new_args = args + ', **kwargs'
        return f'def __init__({new_args}) -> None:'

    new_content = init_pattern.sub(init_replacer, content)
    
    # 2. Update ask to accept **kwargs
    ask_pattern = re.compile(r'def ask\s*\((.*?)\)\s*->\s*str:', re.DOTALL)
    
    def ask_replacer(match):
        args = match.group(1).strip()
        if '**kwargs' in args:
            return match.group(0)
        if args.endswith(','):
            new_args = args + ' **kwargs'
        else:
            new_args = args + ', **kwargs'
        return f'def ask({new_args}) -> str:'

    new_content = ask_pattern.sub(ask_replacer, new_content)

    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        return True
    return False

for root, dirs, files in os.walk(DRIVERS_DIR):
    if 'runner.py' in files:
        file_path = Path(root) / 'runner.py'
        if fix_runner_signatures(file_path):
            print(f"Fixed signatures in {file_path}")
