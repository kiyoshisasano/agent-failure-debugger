"""
pipeline_summary.py

Summary generation for pipeline output.

Extracted from pipeline.py for responsibility separation.

v0.2.0: Added execution_quality block (status, termination, indicators).
"""

from agent_failure_debugger.execution_quality import classify_execution_quality


def build_pipeline_summary(
    diagnosis: dict,
    fix_result: dict,
    telemetry: dict | None = None,
    diagnosis_context: dict | None = None,
) -> dict:
    """
    Build the human-readable summary dict from diagnosis and fix results.

    Args:
        diagnosis: Diagnosis output from run_diagnosis().
        fix_result: Fix pipeline output from run_fix().
        telemetry: Optional raw telemetry dict (adapter output).
            Enables richer execution quality assessment.
        diagnosis_context: Optional context from diagnose().
            Provides observation coverage for quality assessment.

    Returns:
        Summary dict with root_cause, fix, gate, and execution_quality.
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

    # Execution quality assessment
    summary["execution_quality"] = classify_execution_quality(
        diagnosis,
        telemetry=telemetry,
        diagnosis_context=diagnosis_context,
    )

    return summary