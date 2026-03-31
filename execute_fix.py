"""
execute_fix.py

Phase 17: Fix Execution Layer.

Pipeline:
  autofix output → dependency resolver → ordering → validation → staged apply

Modes:
  --plan    Show execution plan (ordered, validated)
  --apply   Staged apply (write patch files to patches/ directory)
  --rollback  Restore from snapshot

Safety:
  All applies save a snapshot first for rollback.
"""

import json
import os
import sys
from pathlib import Path

from fix_templates import AUTOFIX_MAP


# ---------------------------------------------------------------------------
# Fix type ordering
# ---------------------------------------------------------------------------

FIX_TYPE_ORDER = {
    "prompt_patch":   0,
    "config_patch":   1,
    "guard_patch":    2,
    "workflow_patch":  3,
}


# ---------------------------------------------------------------------------
# Dependency resolver (from graph structure in fix_templates)
# ---------------------------------------------------------------------------

# Static dependency: upstream fixes should be applied before downstream.
# Derived from the causal graph's edge direction.
FIX_DEPENDENCY = {
    "semantic_cache_intent_bleeding": ["premature_model_commitment"],
    "prompt_injection_via_retrieval": ["premature_model_commitment", "instruction_priority_inversion"],
    "rag_retrieval_drift": [
        "semantic_cache_intent_bleeding",
        "prompt_injection_via_retrieval",
        "context_truncation_loss",
    ],
    "agent_tool_call_loop": ["premature_model_commitment"],
    "tool_result_misinterpretation": ["agent_tool_call_loop"],
    "repair_strategy_failure": ["premature_model_commitment"],
    "incorrect_output": ["rag_retrieval_drift"],
    "premature_model_commitment": ["assumption_invalidation_failure"],
    "assumption_invalidation_failure": ["clarification_failure"],
}


def _resolve_dependencies(patches: list[dict]) -> list[str]:
    """
    Topological sort of patch targets based on FIX_DEPENDENCY.
    Handles transitive dependencies: if A→B→C and B is not in patches,
    A is still ordered before C.
    """
    targets = {p["target_failure"] for p in patches}

    # Build transitive dependencies (only among active targets)
    def find_upstream(fid: str, visited: set | None = None) -> set:
        """Find all upstream targets transitively."""
        if visited is None:
            visited = set()
        if fid in visited:
            return set()
        visited.add(fid)
        result = set()
        for dep in FIX_DEPENDENCY.get(fid, []):
            if dep in targets:
                result.add(dep)
            # Continue searching upstream even if dep is not in targets
            result |= find_upstream(dep, visited)
        return result

    deps = {}
    for fid in targets:
        deps[fid] = find_upstream(fid)

    # Kahn's algorithm
    in_degree = {fid: len(deps[fid]) for fid in targets}

    queue = [fid for fid in targets if in_degree[fid] == 0]
    ordered = []

    while queue:
        queue.sort(key=lambda fid: FIX_TYPE_ORDER.get(
            next((p["fix_type"] for p in patches if p["target_failure"] == fid), "workflow_patch"), 3
        ))
        node = queue.pop(0)
        ordered.append(node)

        for fid in targets:
            if node in deps.get(fid, set()):
                deps[fid].discard(node)
                in_degree[fid] -= 1
                if in_degree[fid] == 0:
                    queue.append(fid)

    if len(ordered) != len(targets):
        remaining = targets - set(ordered)
        raise RuntimeError(f"Dependency cycle detected among: {remaining}")

    return ordered


# ---------------------------------------------------------------------------
# Fix conflict detection
# ---------------------------------------------------------------------------

FIX_CONFLICTS = [
    {
        "group": "retrieval_control",
        "members": [
            "semantic_cache_intent_bleeding",
            "prompt_injection_via_retrieval",
            "context_truncation_loss",
            "rag_retrieval_drift",
        ],
        "risk": "Multiple retrieval-layer patches may create overly restrictive filtering",
    },
]


def _detect_conflicts(patches: list[dict]) -> list[dict]:
    """Detect potential conflicts between patches."""
    targets = {p["target_failure"] for p in patches}
    conflicts = []

    for group in FIX_CONFLICTS:
        active = [m for m in group["members"] if m in targets]
        if len(active) >= 2:
            conflicts.append({
                "group": group["group"],
                "active_patches": active,
                "risk": group["risk"],
            })

    return conflicts


# ---------------------------------------------------------------------------
# Safety validation
# ---------------------------------------------------------------------------

def _validate_plan(patches: list[dict], order: list[str]) -> dict:
    """Pre-apply safety validation."""
    warnings = []

    # Check all patches have valid fix_type
    for p in patches:
        if p["fix_type"] not in FIX_TYPE_ORDER:
            warnings.append(f"Unknown fix_type: {p['fix_type']} for {p['target_failure']}")

    # Check review requirements
    needs_review = [p for p in patches if p.get("review_required")]
    if needs_review:
        targets = [p["target_failure"] for p in needs_review]
        warnings.append(f"Patches requiring human review: {targets}")

    # Check conflicts
    conflicts = _detect_conflicts(patches)
    if conflicts:
        for c in conflicts:
            warnings.append(
                f"Potential conflict in {c['group']}: {c['active_patches']}. "
                f"Risk: {c['risk']}"
            )

    return {
        "safe": len(warnings) == 0,
        "warnings": warnings,
        "conflicts": conflicts,
    }


# ---------------------------------------------------------------------------
# Execution plan
# ---------------------------------------------------------------------------

def build_execution_plan(autofix_output: dict) -> dict:
    """
    Build an ordered, validated execution plan from autofix output.
    """
    patches = autofix_output.get("recommended_fixes", [])
    if not patches:
        return {
            "execution_plan": [],
            "validation": {"safe": True, "warnings": [], "conflicts": []},
        }

    order = _resolve_dependencies(patches)
    validation = _validate_plan(patches, order)

    # Build ordered plan
    patch_map = {p["target_failure"]: p for p in patches}
    plan = []
    for i, fid in enumerate(order):
        p = patch_map[fid]
        plan.append({
            "order": i + 1,
            "target_failure": fid,
            "fix_type": p["fix_type"],
            "target": p["target"],
            "patch": p["patch"],
            "safety": p["safety"],
            "review_required": p.get("review_required", False),
        })

    return {
        "execution_plan": plan,
        "validation": validation,
    }


# ---------------------------------------------------------------------------
# Staged apply
# ---------------------------------------------------------------------------

def save_snapshot(patches: list[dict], snapshot_path: str = "snapshot.json"):
    """Save pre-apply snapshot for rollback."""
    snapshot = {
        "patches_applied": [p["target_failure"] for p in patches],
        "original_state": "captured_before_apply",
    }
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    return snapshot_path


def staged_apply(execution_plan: dict, output_dir: str = "patches"):
    """
    Write patch files to output directory.
    Saves snapshot first for rollback capability.
    """
    patches = execution_plan.get("execution_plan", [])
    if not patches:
        print("No patches to apply.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Save snapshot
    snapshot_path = os.path.join(output_dir, "snapshot.json")
    save_snapshot(patches, snapshot_path)

    # Write ordered patch files
    for p in patches:
        fname = f"{p['order']:02d}_{p['target_failure']}.json"
        path = os.path.join(output_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(p, f, indent=2)

    # Write execution manifest
    manifest = {
        "total_patches": len(patches),
        "order": [p["target_failure"] for p in patches],
        "snapshot": snapshot_path,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_plan(plan: dict):
    """Pretty-print execution plan."""
    print("\n=== EXECUTION PLAN ===\n")

    for step in plan.get("execution_plan", []):
        review = " [REVIEW REQUIRED]" if step["review_required"] else ""
        print(f"  Step {step['order']}: {step['target_failure']}")
        print(f"    Type:   {step['fix_type']}")
        print(f"    Target: {step['target']}")
        print(f"    Safety: {step['safety']}{review}")
        print()

    v = plan.get("validation", {})
    print(f"=== VALIDATION: {'SAFE' if v.get('safe') else 'WARNINGS'} ===")
    for w in v.get("warnings", []):
        print(f"  ⚠ {w}")
    if not v.get("warnings"):
        print("  No warnings.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    mode_plan = "--plan" in flags
    mode_apply = "--apply" in flags
    mode_rollback = "--rollback" in flags

    if mode_rollback:
        snapshot_path = args[0] if args else "patches/snapshot.json"
        if not os.path.exists(snapshot_path):
            print(f"No snapshot found at {snapshot_path}")
            sys.exit(1)
        with open(snapshot_path, encoding="utf-8") as f:
            snapshot = json.load(f)
        print("=== ROLLBACK ===")
        print(f"  Snapshot: {snapshot_path}")
        print(f"  Patches that were applied: {snapshot.get('patches_applied', [])}")
        # Remove patch files listed in manifest
        patches_dir = os.path.dirname(snapshot_path) or "patches"
        manifest_path = os.path.join(patches_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            for fid in manifest.get("order", []):
                for fname in os.listdir(patches_dir):
                    if fid in fname and fname.endswith(".json"):
                        os.remove(os.path.join(patches_dir, fname))
                        print(f"  Removed: {fname}")
            os.remove(manifest_path)
            print(f"  Removed: manifest.json")
        os.remove(snapshot_path)
        print(f"  Removed: snapshot.json")
        print("  Rollback complete.")
        return

    input_path = args[0] if args else "autofix.json"

    with open(input_path, encoding="utf-8") as f:
        autofix_output = json.load(f)

    plan = build_execution_plan(autofix_output)

    if mode_apply:
        display_plan(plan)
        if not plan["validation"]["safe"]:
            print("\n⚠ Warnings detected. Review before applying.")
        manifest = staged_apply(plan)
        if manifest:
            print(f"\n=== STAGED APPLY COMPLETE ===")
            print(f"  Patches written to: patches/")
            print(f"  Manifest: patches/manifest.json")
            print(f"  Snapshot: patches/snapshot.json")
    elif mode_plan:
        display_plan(plan)
    else:
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()