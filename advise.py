"""
advise.py — CLI for decision support (Phase 15 + Phase 20).

Usage:
  python advise.py [debugger_output.json]
  python advise.py [debugger_output.json] --plan-only
  python advise.py [debugger_output.json] --with-abstraction
  python advise.py [debugger_output.json] --with-learning
"""

import json
import sys

from decision_support import decide


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    input_path = args[0] if args else "debugger_output.json"
    plan_only = "--plan-only" in flags
    with_abstraction = "--with-abstraction" in flags
    with_learning = "--with-learning" in flags

    with open(input_path) as f:
        debugger_output = json.load(f)

    abstraction_output = None
    if with_abstraction:
        from abstraction import abstract
        abstraction_output = abstract(debugger_output)

    policies = None
    if with_learning:
        from policy_loader import load_policies
        policies = load_policies()

    result = decide(debugger_output, abstraction_output, policies)

    if plan_only:
        print(json.dumps(result["action_plan"], indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
