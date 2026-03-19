"""
autofix.py

Phase 16: Deterministic autofix generation.

Pipeline:
  decision_support output → fix selection → patch generation

Selection rules:
  - Only high priority failures
  - Not suppressed by conflict resolution
  - safety != low (low = too risky for autofix)
"""

import json
import sys

from fix_templates import AUTOFIX_MAP
from decision_support import decide


def _select_fix_candidates(decision_output: dict,
                           top_k: int = 1) -> list[dict]:
    """
    Select safe, high-value fixes.
    High-priority fills first, medium only if slots remain.
    """
    suppressed = set()
    for c in decision_output.get("conflict_resolutions", []):
        for s in c.get("deprioritized", []):
            suppressed.add(s)

    high_candidates = []
    medium_candidates = []

    for action in decision_output.get("recommended_actions", []):
        fid = action["target_failure"]

        if fid not in AUTOFIX_MAP:
            continue
        if fid in suppressed:
            continue

        template = AUTOFIX_MAP[fid]
        if template["safety"] == "low":
            continue

        entry = {
            "failure": fid,
            "priority_score": action["priority_score"],
            "template": template,
        }

        if action["priority"] == "high":
            high_candidates.append(entry)
        elif action["priority"] == "medium":
            medium_candidates.append(entry)

    high_candidates.sort(key=lambda x: x["priority_score"], reverse=True)
    medium_candidates.sort(key=lambda x: x["priority_score"], reverse=True)

    # High fills first, medium only if slots remain
    selected = high_candidates[:top_k]
    if len(selected) < top_k:
        selected += medium_candidates[:(top_k - len(selected))]

    return selected


def _build_patch(candidate: dict) -> dict:
    template = candidate["template"]
    return {
        "target_failure": candidate["failure"],
        "fix_type": template["fix_type"],
        "target": template["target"],
        "patch": template["patch"],
        "safety": template["safety"],
        "review_required": template["safety"] != "high",
    }


def generate_autofix(decision_output: dict, top_k: int = 1) -> dict:
    """Generate autofix patches from decision support output."""
    candidates = _select_fix_candidates(decision_output, top_k=top_k)

    patches = [_build_patch(c) for c in candidates]

    return {
        "recommended_fixes": patches,
        "patch_plan": {
            "high_safety": [p for p in patches if p["safety"] == "high"],
            "needs_review": [p for p in patches if p["safety"] != "high"],
        },
        "review_notes": [
            "High-priority fixes are always preferred and fill first.",
            "Medium-priority fixes are included only when top-k allows and high slots remain.",
            "Suppressed competing causes are excluded.",
            "Medium-safety patches require human confirmation.",
        ],
    }


def main():
    """CLI: debugger_output.json → decision → autofix"""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = sys.argv[1:]

    input_path = args[0] if args else "debugger_output.json"
    top_k = 1
    for i, flag in enumerate(flags):
        if flag == "--top-k" and i + 1 < len(flags):
            top_k = int(flags[i + 1])

    with open(input_path) as f:
        debugger_output = json.load(f)

    decision_output = decide(debugger_output)
    autofix_output = generate_autofix(decision_output, top_k=top_k)

    print(json.dumps(autofix_output, indent=2))


if __name__ == "__main__":
    main()
