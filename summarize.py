"""
summarize.py — CLI for abstraction layer (Phase 14).

Usage:
  python summarize.py [debugger_output.json]
  python summarize.py [debugger_output.json] --mode brief
  python summarize.py [debugger_output.json] --mode verbose
  python summarize.py [debugger_output.json] --top-k 3

Modes: verbose | standard | brief
"""

import json
import sys

from abstraction import abstract


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = sys.argv[1:]

    input_path = args[0] if args else "debugger_output.json"
    mode = "standard"
    top_k = 2

    for i, flag in enumerate(flags):
        if flag == "--mode" and i + 1 < len(flags):
            mode = flags[i + 1]
        if flag == "--top-k" and i + 1 < len(flags):
            top_k = int(flags[i + 1])

    with open(input_path) as f:
        debugger_output = json.load(f)

    result = abstract(debugger_output, top_k=top_k, mode=mode)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
