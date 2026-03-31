"""
pipeline_summary.py

Summary generation for pipeline output.

Extracted from pipeline.py for responsibility separation.
Logic is identical — no behavioral changes.
"""


def build_pipeline_summary(diagnosis: dict, fix_result: dict) -> dict:
    """
    Build the human-readable summary dict from diagnosis and fix results.
    """
    root = diagnosis["root_ranking"][0] if diagnosis.get("root_ranking") else {}
    gate = fix_result["gate"].get("gate", {})
    fixes = fix_result["autofix"].get("recommended_fixes", [])
    post_apply = fix_result["gate"].get("post_apply")

    summary = {
        "root_cause": root.get("id", "unknown"),
        "root_confidence": root.get("score", 0.0),
        "failure_count": len(diagnosis.get("failures", [])),
        "fix_count": len(fixes),
        "gate_mode": gate.get("mode", "unknown"),
        "gate_score": gate.get("score", 0.0),
    }

    if post_apply:
        summary["applied"] = True
        summary["evaluation_decision"] = post_apply.get("evaluation_decision")
        summary["rollback"] = post_apply.get("rollback_executed", False)
    else:
        summary["applied"] = False

    return summary