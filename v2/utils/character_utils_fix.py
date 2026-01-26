# Temporary fix - let me read the actual file content around line 427
import sys
with open('v2/utils/character_utils.py', 'r') as f:
    lines = f.readlines()
    # Show lines 424-432
    for i in range(423, min(432, len(lines))):
        line = lines[i]
        spaces = len(line) - len(line.lstrip())
        print(f"Line {i+1}: {spaces} spaces | {repr(line[:80])}")

