"""
apply_fix.py — Dry-run display of autofix patches.

Usage:
  python apply_fix.py [autofix_output.json]
"""

import json
import sys


def format_patch(patch: dict) -> str:
    lines = []
    lines.append(f"  Target: {patch['target_failure']}")
    lines.append(f"  Type:   {patch['fix_type']}")
    lines.append(f"  Dest:   {patch['target']}")
    lines.append(f"  Safety: {patch['safety']}")
    if patch.get("review_required"):
        lines.append("  ⚠ REVIEW REQUIRED")
    lines.append(f"  Patch:  {json.dumps(patch['patch'], indent=4)}")
    return "\n".join(lines)


def dry_run(autofix_output: dict):
    patches = autofix_output.get("recommended_fixes", [])
    if not patches:
        print("No patches to display.")
        return

    print(f"\n=== DRY RUN: {len(patches)} patch(es) ===\n")
    for i, p in enumerate(patches, 1):
        print(f"[{i}]")
        print(format_patch(p))
        print()

    plan = autofix_output.get("patch_plan", {})
    high = plan.get("high_safety", [])
    review = plan.get("needs_review", [])
    print(f"High safety (auto-safe): {len(high)}")
    print(f"Needs review:            {len(review)}")


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "autofix_output.json"

    with open(input_path) as f:
        autofix_output = json.load(f)

    dry_run(autofix_output)


if __name__ == "__main__":
    main()
