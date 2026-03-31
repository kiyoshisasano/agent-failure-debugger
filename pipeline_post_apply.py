"""
pipeline_post_apply.py

Post-apply processing: evaluation runner dispatch and
built-in counterfactual simulation.

Extracted from pipeline.py for responsibility separation.
Logic is identical — no behavioral changes.
"""

from graph_loader import load_graph
from auto_apply import maybe_apply

try:
    from config import GRAPH_PATH
except ImportError:
    from pathlib import Path
    GRAPH_PATH = Path(__file__).parent.parent / "llm-failure-atlas" / "failure_graph.yaml"


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

    Return structure (fix_result):
        {
            "decision": {...},
            "autofix": {"recommended_fixes": [...]},
            "execution_plan": {...},
            "gate": {
                "gate": {"mode": str, "score": float, ...},
                "fix_gates": [...],
                "post_apply": {              # added by this function
                    "evaluation_mode": str,  # "runner" or "builtin"
                    "runner_result": dict | None,
                    "evaluation_decision": str,  # "keep" | "review" | "rollback"
                    "rollback_executed": bool,
                } | None
            }
        }
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
        graph = load_graph(graph_path) if graph_path else load_graph(str(GRAPH_PATH))
        apply_result = maybe_apply(
            fix_result["gate"], diagnosis, fix_result["autofix"],
            fix_result["execution_plan"], graph
        )
        fix_result["gate"] = apply_result

    return fix_result