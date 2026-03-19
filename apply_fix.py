"""
auto_apply.py

Phase 21: Auto-Apply with Confidence Gating.

Pipeline:
  autofix_output → execution_plan → gate → apply/review/proposal
  → evaluate_fix → keep/rollback → update_policy

Gate rules:
  score >= 0.85  → auto_apply
  0.65 <= score < 0.85 → staged_review
  score < 0.65  → proposal_only

Hard blockers (override score):
  - review_required == true
  - safety != "high"
  - execution plan has conflicts
  - execution plan validation.safe == false

Usage:
  python auto_apply.py debugger_output.json autofix.json
  python auto_apply.py debugger_output.json autofix.json --apply
  python auto_apply.py debugger_output.json autofix.json --json-only
"""

import json
import sys
from pathlib import Path

from execute_fix import build_execution_plan, staged_apply
from evaluate_fix import (
    simulate_after_state, compute_delta, detect_regressions,
    decide_keep_or_rollback,
)
from graph_loader import load_graph


DEBUGGER_DIR = Path(__file__).parent
GRAPH_PATH = DEBUGGER_DIR / "failure_graph.yaml"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_auto_apply_score(fix: dict, debugger_output: dict,
                             policies: dict | None = None) -> float:
    """
    auto_apply_score =
      0.35 * priority_score
    + 0.30 * effectiveness_score (scaled)
    + 0.20 * root_confidence
    + 0.15 * regression_safety

    All inputs are clamped to [0, 1].
    """
    priority_score = min(1.0, fix.get("priority_score", 0.0))

    # Effectiveness: use scaled value (Phase 20 fix ①)
    eff_raw = fix.get("effectiveness_score", 0.0)
    try:
        from policy_loader import scale_effectiveness
        eff_scaled = scale_effectiveness(eff_raw)
    except ImportError:
        eff_scaled = max(0.0, min(1.0, (eff_raw - 0.5) * 2))

    # Root confidence: top root from debugger_output
    root_ranking = debugger_output.get("root_ranking", [])
    root_confidence = root_ranking[0]["score"] if root_ranking else 0.0

    # Regression safety: high=1.0, else 0.5
    safety = fix.get("safety", "medium")
    regression_safety = 1.0 if safety == "high" else 0.5

    score = (0.35 * priority_score
             + 0.30 * eff_scaled
             + 0.20 * root_confidence
             + 0.15 * regression_safety)

    return round(score, 4)


# ---------------------------------------------------------------------------
# Hard blockers
# ---------------------------------------------------------------------------

def check_hard_blockers(fix: dict, execution_plan: dict) -> list[str]:
    """
    Return list of reasons this fix cannot be auto-applied.
    Empty list = no blockers.
    """
    blockers = []

    if fix.get("review_required", False):
        blockers.append("review_required is true")

    if fix.get("safety", "medium") != "high":
        blockers.append(f"safety is {fix.get('safety', 'unknown')} (must be high)")

    # Fix type restriction
    allowed_types = {"config_patch", "guard_patch", "prompt_patch"}
    if fix.get("fix_type", "") not in allowed_types:
        blockers.append(f"fix_type {fix.get('fix_type')} not in auto-apply allowed set")

    # Execution plan checks
    validation = execution_plan.get("validation", {})
    if not validation.get("safe", True):
        blockers.append("execution plan validation failed")

    conflicts = validation.get("conflicts", [])
    if conflicts:
        groups = [c["group"] for c in conflicts]
        blockers.append(f"execution plan has conflicts: {groups}")

    return blockers


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

def _mode_from_score(score: float) -> str:
    if score >= 0.85:
        return "auto_apply"
    elif score >= 0.65:
        return "staged_review"
    return "proposal_only"


def _score_reasons(score: float, fix: dict, root_confidence: float) -> list[str]:
    """Generate human-readable reasons for the gate decision."""
    reasons = []
    if fix.get("safety") == "high":
        reasons.append("high safety")
    eff = fix.get("effectiveness_score", 0.0)
    if eff >= 0.8:
        reasons.append("high effectiveness")
    elif eff >= 0.5:
        reasons.append("good effectiveness")
    if root_confidence >= 0.8:
        reasons.append("high root confidence")
    elif root_confidence >= 0.6:
        reasons.append("medium root confidence")
    ps = fix.get("priority_score", 0.0)
    if ps >= 0.8:
        reasons.append("high priority")
    return reasons


def gate_autofix(debugger_output: dict, autofix_output: dict,
                 execution_plan: dict,
                 policies: dict | None = None) -> dict:
    """
    Gate each recommended fix. Returns per-fix gate results + overall mode.

    Overall mode is the MOST restrictive across all fixes.
    """
    fixes = autofix_output.get("recommended_fixes", [])
    root_ranking = debugger_output.get("root_ranking", [])
    root_confidence = root_ranking[0]["score"] if root_ranking else 0.0

    fix_gates = []

    for fix in fixes:
        score = compute_auto_apply_score(fix, debugger_output, policies)
        blockers = check_hard_blockers(fix, execution_plan)

        if blockers:
            mode = "proposal_only"
        else:
            mode = _mode_from_score(score)

        reasons = _score_reasons(score, fix, root_confidence)

        fix_gates.append({
            "target_failure": fix["target_failure"],
            "mode": mode,
            "score": score,
            "reasons": reasons,
            "blocked_by": blockers,
        })

    # Overall mode: most restrictive
    modes = [g["mode"] for g in fix_gates]
    if "proposal_only" in modes:
        overall_mode = "proposal_only"
    elif "staged_review" in modes:
        overall_mode = "staged_review"
    else:
        overall_mode = "auto_apply"

    # Overall score: minimum across auto-applicable fixes
    auto_scores = [g["score"] for g in fix_gates if not g["blocked_by"]]
    overall_score = min(auto_scores) if auto_scores else 0.0

    # Aggregate blockers
    all_blockers = []
    for g in fix_gates:
        all_blockers.extend(g["blocked_by"])

    return {
        "gate": {
            "mode": overall_mode,
            "score": overall_score,
            "reasons": list(dict.fromkeys(
                r for g in fix_gates for r in g["reasons"]
            )),
            "blocked_by": list(dict.fromkeys(all_blockers)),
        },
        "fix_gates": fix_gates,
        "execution": {
            "action": overall_mode.replace("_", " "),
            "target_fixes": [f["target_failure"] for f in fixes],
        },
    }


# ---------------------------------------------------------------------------
# Rollback policy (Phase 21 refinement of evaluate_fix decisions)
# ---------------------------------------------------------------------------

# Phase 21 fix ①: type-based regression classification
# Decouples from evaluate_fix's severity labels
HARD_REGRESSION_TYPES = {
    "new_failure_introduced",
    "failure_count_increase",
    "root_not_mitigated",
}

SOFT_REGRESSION_TYPES = {
    "no_effect",
    "path_length_increase",
    "conflict_increase",
}


def _classify_regressions(regressions: list[dict]) -> dict:
    """
    Classify regressions into hard (immediate rollback)
    and soft (mark for review).

    Uses regression type rather than severity label for robustness.
    """
    hard = [r for r in regressions if r["type"] in HARD_REGRESSION_TYPES]
    soft = [r for r in regressions if r["type"] in SOFT_REGRESSION_TYPES]
    # Any unknown type is treated as soft (conservative)
    unknown = [r for r in regressions
               if r["type"] not in HARD_REGRESSION_TYPES
               and r["type"] not in SOFT_REGRESSION_TYPES]
    soft.extend(unknown)
    return {"hard": hard, "soft": soft}


def _rollback_decision(regressions: list[dict]) -> str:
    """
    Phase 21 rollback policy:
      hard regression → rollback
      soft regression → review
      no regression → keep
    """
    classified = _classify_regressions(regressions)
    if classified["hard"]:
        return "rollback"
    elif classified["soft"]:
        return "review"
    return "keep"


# ---------------------------------------------------------------------------
# Maybe apply
# ---------------------------------------------------------------------------

def maybe_apply(gate_result: dict, debugger_output: dict,
                autofix_output: dict, execution_plan: dict,
                graph: dict) -> dict:
    """
    If gate mode is auto_apply:
      1. staged_apply (write patches)
      2. simulate + evaluate
      3. decide keep/rollback
      4. rollback if needed

    Returns full result with post_apply data.
    """
    mode = gate_result["gate"]["mode"]

    if mode != "auto_apply":
        gate_result["post_apply"] = None
        return gate_result

    # Step 1: staged apply
    manifest = staged_apply(execution_plan)

    # Step 2: evaluate
    after = simulate_after_state(debugger_output, autofix_output, graph)
    delta = compute_delta(debugger_output, after)
    regressions = detect_regressions(debugger_output, after, delta)

    # Step 3: decide
    decision = _rollback_decision(regressions)
    rollback_executed = False

    # Step 4: rollback if needed
    if decision == "rollback":
        import os
        patches_dir = "patches"
        snapshot_path = os.path.join(patches_dir, "snapshot.json")
        manifest_path = os.path.join(patches_dir, "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                m = json.load(f)
            for fid in m.get("order", []):
                for fname in os.listdir(patches_dir):
                    if fid in fname and fname.endswith(".json"):
                        os.remove(os.path.join(patches_dir, fname))
            if os.path.exists(manifest_path):
                os.remove(manifest_path)
        if os.path.exists(snapshot_path):
            os.remove(snapshot_path)
        rollback_executed = True

    gate_result["post_apply"] = {
        "evaluation_decision": decision,
        "rollback_executed": rollback_executed,
        "delta": delta,
        "regressions": regressions,
    }

    return gate_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    if len(args) < 2:
        print("Usage: python auto_apply.py debugger_output.json autofix.json [--apply] [--json-only]")
        sys.exit(1)

    with open(args[0]) as f:
        debugger_output = json.load(f)
    with open(args[1]) as f:
        autofix_output = json.load(f)

    # Load policies if available
    policies = None
    try:
        from policy_loader import load_policies
        policies = load_policies()
    except ImportError:
        pass

    # Build execution plan
    execution_plan = build_execution_plan(autofix_output)

    # Gate
    gate_result = gate_autofix(
        debugger_output, autofix_output, execution_plan, policies
    )

    # Apply if requested and gate allows
    if "--apply" in flags:
        graph = load_graph(str(GRAPH_PATH))
        result = maybe_apply(
            gate_result, debugger_output, autofix_output,
            execution_plan, graph
        )
    else:
        result = gate_result
        result["post_apply"] = None

    if "--json-only" in flags:
        print(json.dumps(result, indent=2))
    else:
        _display(result)


def _display(result: dict):
    """Human-readable output."""
    g = result["gate"]
    print(f"\n=== AUTO-APPLY GATE ===\n")
    print(f"  Mode:  {g['mode']}")
    print(f"  Score: {g['score']}")
    if g["reasons"]:
        print(f"  Reasons: {', '.join(g['reasons'])}")
    if g["blocked_by"]:
        print(f"  Blocked: {', '.join(g['blocked_by'])}")

    # Per-fix breakdown
    for fg in result.get("fix_gates", []):
        marker = "✅" if fg["mode"] == "auto_apply" else "⚠" if fg["mode"] == "staged_review" else "❌"
        print(f"\n  {marker} {fg['target_failure']}")
        print(f"    mode={fg['mode']}  score={fg['score']}")
        if fg["blocked_by"]:
            print(f"    blocked: {fg['blocked_by']}")

    e = result.get("execution", {})
    print(f"\n  Action: {e.get('action', 'unknown')}")
    print(f"  Targets: {e.get('target_fixes', [])}")

    pa = result.get("post_apply")
    if pa:
        print(f"\n=== POST-APPLY ===")
        print(f"  Decision: {pa['evaluation_decision']}")
        print(f"  Rollback: {'yes' if pa['rollback_executed'] else 'no'}")
    elif result["gate"]["mode"] == "auto_apply":
        print(f"\n  (Use --apply to execute)")


if __name__ == "__main__":
    main()
