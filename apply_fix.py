"""
apply_fix.py — Dry-run display of autofix patches.

Usage:
  python apply_fix.py [autofix_output.json]

Or full pipeline from debugger output:
  python autofix.py debugger_output.json > autofix.json
  python apply_fix.py autofix.json
"""

import json
import sys


def format_patch(patch: dict) -> str:
    lines = [
        f"=== FIX: {patch['target_failure']} ===",
        f"Type: {patch['fix_type']}",
        f"Target: {patch['target']}",
        f"Safety: {patch['safety']}",
        f"Review Required: {patch['review_required']}",
        "",
        "Patch:",
        json.dumps(patch["patch"], indent=2),
    ]
    return "\n".join(lines)


def dry_run(autofix_output: dict):
    print("\n=== AUTO-FIX DRY RUN ===\n")

    fixes = autofix_output.get("recommended_fixes", [])
    if not fixes:
        print("No fixes recommended.")
        return

    for patch in fixes:
        print(format_patch(patch))
        print()

    plan = autofix_output.get("patch_plan", {})
    print("=== SUMMARY ===")
    print(f"High safety:  {len(plan.get('high_safety', []))}")
    print(f"Needs review: {len(plan.get('needs_review', []))}")

    notes = autofix_output.get("review_notes", [])
    if notes:
        print("\nNotes:")
        for note in notes:
            print(f"  - {note}")


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "autofix.json"

    with open(input_path) as f:
        data = json.load(f)

    dry_run(data)


if __name__ == "__main__":
    main()
