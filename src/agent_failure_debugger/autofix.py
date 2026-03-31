"""
autofix.py

Phase 16: Deterministic autofix generation.
Phase 20: Learning-aware fix selection + safety promotion.

Pipeline:
  decision_support output → fix selection → patch generation

Selection rules:
  - Only high priority failures
  - Not suppressed by conflict resolution
  - safety != low (low = too risky for autofix)

Phase 20 additions:
  - final_fix_score = 0.6 * priority_score + 0.4 * effectiveness_score
  - Safety promotion: effectiveness >= 0.9 and rollback == 0 → safety override to high
  - --use-learning CLI flag
"""

import json
import sys

from agent_failure_debugger.fix_templates import AUTOFIX_MAP
from agent_failure_debugger.decision_support import decide


def _select_fix_candidates(decision_output: dict,
                           top_k: int = 1,
                           policies: dict | None = None) -> list[dict]:
    """
    Select safe, high-value fixes.
    High-priority fills first, medium only if slots remain.

    Phase 20: when policies provided, ranking uses final_fix_score
    and safety can be promoted based on effectiveness track record.
    """
    suppressed = set()
    for c in decision_output.get("conflict_resolutions", []):
        for s in c.get("deprioritized", []):
            suppressed.add(s)

    fix_eff = (policies or {}).get("fix_effectiveness", {})
    _scale_eff = None
    if fix_eff:
        from policy_loader import scale_effectiveness
        _scale_eff = scale_effectiveness

    high_candidates = []
    medium_candidates = []

    for action in decision_output.get("recommended_actions", []):
        fid = action["target_failure"]

        if fid not in AUTOFIX_MAP:
            continue
        if fid in suppressed:
            continue

        template = AUTOFIX_MAP[fid]
        effective_safety = template["safety"]

        # Phase 20: safety promotion
        if fix_eff and effective_safety != "low":
            record = fix_eff.get(fid, {}).get(template["fix_type"])
            if record:
                eff_score = record.get("effectiveness_score", 0.0)
                rollbacks = record.get("rollback", 0)
                if eff_score >= 0.9 and rollbacks == 0:
                    effective_safety = "high"

        if effective_safety == "low":
            continue

        # Phase 20: final_fix_score with learning
        priority_score = action["priority_score"]
        if fix_eff:
            eff_entries = fix_eff.get(fid, {})
            best_eff_raw = 0.0
            if eff_entries:
                best_eff_raw = max(
                    e.get("effectiveness_score", 0.0)
                    for e in eff_entries.values()
                )
            # Phase 20 ①: scale for ranking spread
            best_eff_scaled = _scale_eff(best_eff_raw) if _scale_eff else best_eff_raw
            final_score = 0.6 * priority_score + 0.4 * best_eff_scaled
        else:
            final_score = priority_score
            best_eff_raw = 0.0

        entry = {
            "failure": fid,
            "priority_score": priority_score,
            "final_fix_score": round(final_score, 4),
            "effectiveness_score": best_eff_raw,
            "template": template,
            "effective_safety": effective_safety,
            "safety_promoted": effective_safety != template["safety"],
        }

        if action["priority"] == "high":
            high_candidates.append(entry)
        elif action["priority"] == "medium":
            medium_candidates.append(entry)

    # Sort by final_fix_score (learning-aware)
    sort_key = "final_fix_score" if fix_eff else "priority_score"
    high_candidates.sort(key=lambda x: x[sort_key], reverse=True)
    medium_candidates.sort(key=lambda x: x[sort_key], reverse=True)

    # High fills first, medium only if slots remain
    selected = high_candidates[:top_k]
    if len(selected) < top_k:
        selected += medium_candidates[:(top_k - len(selected))]

    return selected


def _build_patch(candidate: dict, use_learning: bool = False) -> dict:
    template = candidate["template"]
    effective_safety = candidate.get("effective_safety", template["safety"])

    patch = {
        "target_failure": candidate["failure"],
        "fix_type": template["fix_type"],
        "target": template["target"],
        "patch": template["patch"],
        "safety": effective_safety,
        "review_required": effective_safety != "high",
        "priority_score": candidate.get("priority_score", 0.0),
    }

    # Phase 20: annotate learning-derived fields
    if use_learning:
        patch["final_fix_score"] = candidate.get("final_fix_score", 0.0)
        patch["effectiveness_score"] = candidate.get("effectiveness_score", 0.0)
        if candidate.get("safety_promoted"):
            patch["safety_promoted_from"] = template["safety"]

    return patch


def generate_autofix(decision_output: dict, top_k: int = 1,
                     policies: dict | None = None) -> dict:
    """Generate autofix patches from decision support output."""
    use_learning = policies is not None and bool(
        policies.get("fix_effectiveness")
    )

    candidates = _select_fix_candidates(
        decision_output, top_k=top_k, policies=policies
    )

    patches = [_build_patch(c, use_learning=use_learning) for c in candidates]

    result = {
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

    if use_learning:
        result["review_notes"].append(
            "Learning-aware: fix ranking uses final_fix_score "
            "(0.6 * priority + 0.4 * effectiveness)."
        )
        # Report promotions
        promotions = [p for p in patches if p.get("safety_promoted_from")]
        if promotions:
            result["safety_promotions"] = [
                {
                    "failure": p["target_failure"],
                    "fix_type": p["fix_type"],
                    "from": p["safety_promoted_from"],
                    "to": "high",
                }
                for p in promotions
            ]

    return result


def main():
    """CLI: debugger_output.json → decision → autofix"""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = sys.argv[1:]

    input_path = args[0] if args else "debugger_output.json"
    top_k = 1
    use_learning = "--use-learning" in flags

    for i, flag in enumerate(flags):
        if flag == "--top-k" and i + 1 < len(flags):
            top_k = int(flags[i + 1])

    with open(input_path, encoding="utf-8") as f:
        debugger_output = json.load(f)

    policies = None
    if use_learning:
        from policy_loader import load_policies
        policies = load_policies()

    decision_output = decide(debugger_output, policies=policies)
    autofix_output = generate_autofix(
        decision_output, top_k=top_k, policies=policies
    )

    print(json.dumps(autofix_output, indent=2))


if __name__ == "__main__":
    main()