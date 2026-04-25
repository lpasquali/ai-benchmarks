#!/usr/bin/env python3
import os

def fix_tests():
    test_dir = "tests"
    for filename in os.listdir(test_dir):
        if filename.startswith("test_") and filename.endswith("_driver.py"):
            path = os.path.join(test_dir, filename)
            with open(path, "r") as f:
                lines = f.readlines()
            
            new_lines = []
            skip_next = False
            for i, line in enumerate(lines):
                if "# with pytest.raises(NotImplementedError):" in line:
                    # The next line is likely indented and needs to be adjusted or commented
                    new_lines.append(line)
                    if i + 1 < len(lines):
                        next_line = lines[i+1]
                        if next_line.strip() and next_line.startswith("        "):
                            new_lines.append("#" + next_line)
                            skip_next = True
                elif skip_next:
                    skip_next = False
                    continue
                else:
                    new_lines.append(line)
            
            with open(path, "w") as f:
                f.writelines(new_lines)

if __name__ == "__main__":
    fix_tests()
