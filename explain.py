"""
explain.py — CLI for explanation generation (Phase 10).

Usage:
  python explain.py [debugger_output.json]                  # LLM hybrid mode
  python explain.py --deterministic [debugger_output.json]  # No LLM, draft only
  python explain.py --enhanced [debugger_output.json]       # Enhanced draft (no LLM)
  python explain.py --dry-run [debugger_output.json]        # Show prompt, don't call LLM

Modes:
  default          Hybrid: deterministic draft + LLM smoothing (requires API key)
  --deterministic  Deterministic draft only (no API key needed)
  --enhanced       Enhanced draft with context, interpretation, risk (no API key needed)
  --dry-run        Show the prompt that would be sent to the LLM
"""

import json
import sys

from explainer import (
    build_explanation_package,
    build_prompt,
    render_draft,
    explain,
)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    input_path = args[0] if args else "debugger_output.json"
    deterministic = "--deterministic" in flags
    enhanced = "--enhanced" in flags
    dry_run = "--dry-run" in flags

    with open(input_path) as f:
        debugger_output = json.load(f)

    if dry_run:
        package = build_explanation_package(debugger_output)
        draft = render_draft(package)
        system, user = build_prompt(package, draft)
        output = {
            "mode": "dry_run",
            "explanation_package": package,
            "draft": draft,
            "system_prompt": system,
            "user_prompt": user,
        }
        print(json.dumps(output, indent=2))
    elif deterministic or enhanced:
        result = explain(debugger_output, use_llm=False, enhanced=enhanced)
        print(json.dumps(result, indent=2))
    else:
        result = explain(debugger_output)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()