"""
pipeline_post_apply.py

Post-apply processing: evaluation runner dispatch and
built-in counterfactual simulation.

Extracted from pipeline.py for responsibility separation.
Logic is identical — no behavioral changes.
"""

from graph_loader import load_graph
from auto_apply import maybe_apply


def _decision_from_runner_result(runner_result: dict) -> str:
    """Map runner result to evaluation decision."""
    if runner_result.get("has_hard_regression"):
        return "rollback"
    if runner_result.get("success"):
        return "keep"
    return "review"


def run_post_apply(diagnosis: dict, fix_result: dict,
                   auto_apply: bool = False,
                   evaluation_runner=None,
                   graph_path: str = None) -> dict:
    """
    Handle post-apply evaluation if auto_apply gate passes.

    Modifies fix_result in place (adds post_apply to gate).
    Returns fix_result for convenience.
    """
    if not auto_apply:
        return fix_result

    if fix_result["gate"]["gate"]["mode"] != "auto_apply":
        return fix_result

    if evaluation_runner is not None:
        # External evaluation: delegate to user's environment
        runner_input = {
            "autofix": fix_result["autofix"],
            "execution_plan": fix_result["execution_plan"],
            "diagnosis_summary": {
                "root_cause": (diagnosis["root_ranking"][0]["id"]
                               if diagnosis.get("root_ranking") else "unknown"),
                "failure_count": len(diagnosis.get("failures", [])),
            },
        }
        try:
            runner_result = evaluation_runner(runner_input)
        except Exception as e:
            fix_result["gate"]["post_apply"] = {
                "evaluation_mode": "runner",
                "runner_result": None,
                "evaluation_decision": "review",
                "rollback_executed": False,
                "runner_error": str(e),
            }
            runner_result = None

        if runner_result is not None:
            fix_result["gate"]["post_apply"] = {
                "evaluation_mode": "runner",
                "runner_result": runner_result,
                "evaluation_decision": _decision_from_runner_result(runner_result),
                "rollback_executed": runner_result.get("has_hard_regression", False),
            }
    else:
        # Built-in counterfactual evaluation
        graph = load_graph(graph_path) if graph_path else load_graph("failure_graph.yaml")
        apply_result = maybe_apply(
            fix_result["gate"], diagnosis, fix_result["autofix"],
            fix_result["execution_plan"], graph
        )
        fix_result["gate"] = apply_result

    return fix_result