"""
pipeline.py

API wrapper for the full agent-failure-debugger pipeline.

Provides a single entry point for external systems (APIs, LLMs, CI/CD)
to run the diagnosis-to-fix pipeline without CLI knowledge.

Usage:
  from pipeline import run_pipeline, run_diagnosis, run_fix

  # Full pipeline
  result = run_pipeline(matcher_output, use_learning=True)

  # Diagnosis only
  diag = run_diagnosis(matcher_output)

  # Fix only (from existing diagnosis)
  fix = run_fix(debugger_output, use_learning=True, auto_apply=False)
"""

import json
import sys
from pathlib import Path

from graph_loader import load_graph
from causal_resolver import resolve
from formatter import format_output
from decision_support import decide
from autofix import generate_autofix
from execute_fix import build_execution_plan
from auto_apply import gate_autofix, maybe_apply
from abstraction import abstract
from evaluate_fix import (
    simulate_after_state, compute_delta, detect_regressions,
    decide_keep_or_rollback,
)


# ---------------------------------------------------------------------------
# Paths (use config if available, fallback to local)
# ---------------------------------------------------------------------------

try:
    from config import GRAPH_PATH, LEARNING_DIR
except ImportError:
    GRAPH_PATH = Path(__file__).parent / "failure_graph.yaml"
    LEARNING_DIR = Path(__file__).parent.parent / "llm-failure-atlas" / "learning"


def _load_policies():
    """Load learning policies if available."""
    try:
        from policy_loader import load_policies
        return load_policies()
    except (ImportError, Exception):
        return None


# ---------------------------------------------------------------------------
# Input validation (fail fast)
# ---------------------------------------------------------------------------

def _validate_matcher_output(matcher_output):
    """Validate matcher_output before entering the pipeline.

    Checks structure only — does not validate values or enforce
    strict schema. Raises ValueError/TypeError on invalid input.
    """
    if not isinstance(matcher_output, list):
        raise TypeError(
            f"matcher_output must be a list, got {type(matcher_output).__name__}"
        )

    for i, entry in enumerate(matcher_output):
        if not isinstance(entry, dict):
            raise TypeError(
                f"matcher_output[{i}] must be a dict, got {type(entry).__name__}"
            )
        for field in ("failure_id", "diagnosed", "confidence"):
            if field not in entry:
                raise ValueError(
                    f"matcher_output[{i}] missing required field '{field}'"
                )


def _validate_debugger_output(debugger_output):
    """Validate debugger_output before entering the fix pipeline."""
    if not isinstance(debugger_output, dict):
        raise TypeError(
            f"debugger_output must be a dict, got {type(debugger_output).__name__}"
        )
    for field in ("failures", "root_ranking"):
        if field not in debugger_output:
            raise ValueError(
                f"debugger_output missing required field '{field}'"
            )


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

def run_diagnosis(matcher_output: list[dict]) -> dict:
    """
    Run diagnosis pipeline: matcher_output → debugger_output.

    Args:
        matcher_output: List of diagnosed failure dicts from matcher.py.
            Each dict must have: failure_id, diagnosed, confidence, signals.

    Returns:
        debugger_output dict with:
          root_candidates, root_ranking, failures,
          causal_paths, primary_path, conflicts, evidence
    """
    _validate_matcher_output(matcher_output)
    graph = load_graph(str(GRAPH_PATH))
    resolved = resolve(graph, matcher_output)
    output = format_output(resolved)
    return output


# ---------------------------------------------------------------------------
# Fix
# ---------------------------------------------------------------------------

def run_fix(debugger_output: dict,
            use_learning: bool = False,
            top_k: int = 1,
            auto_apply: bool = False) -> dict:
    """
    Run fix pipeline: debugger_output → decision → autofix → gate.

    Args:
        debugger_output: Output from run_diagnosis().
        use_learning: Use learning policies for scoring.
        top_k: Number of fixes to generate.
        auto_apply: If True and gate passes, execute fixes.

    Returns:
        Dict with: decision, autofix, gate, post_apply (if auto_apply).
    """
    _validate_debugger_output(debugger_output)
    policies = _load_policies() if use_learning else None

    # Decision support
    decision_output = decide(debugger_output, policies=policies)

    # Autofix
    autofix_output = generate_autofix(
        decision_output, top_k=top_k, policies=policies
    )

    # Execution plan
    execution_plan = build_execution_plan(autofix_output)

    # Gate
    gate_result = gate_autofix(
        debugger_output, autofix_output, execution_plan, policies
    )

    result = {
        "decision": decision_output,
        "autofix": autofix_output,
        "execution_plan": execution_plan,
        "gate": gate_result,
    }

    # Auto-apply if requested
    if auto_apply and gate_result["gate"]["mode"] == "auto_apply":
        graph = load_graph(str(GRAPH_PATH))
        apply_result = maybe_apply(
            gate_result, debugger_output, autofix_output,
            execution_plan, graph
        )
        result["gate"] = apply_result

    return result


# ---------------------------------------------------------------------------
# Evaluation runner support (Phase 25-lite)
# ---------------------------------------------------------------------------

def _decision_from_runner_result(result: dict) -> str:
    """
    Convert external evaluation runner result to keep/review/rollback decision.
    """
    if result.get("has_hard_regression"):
        return "rollback"
    if not result.get("success", False):
        return "review"
    return "keep"


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(matcher_output: list[dict],
                 use_learning: bool = False,
                 top_k: int = 1,
                 auto_apply: bool = False,
                 include_abstraction: bool = False,
                 evaluation_runner=None) -> dict:
    """
    Run the complete pipeline: diagnosis → fix → evaluation.

    This is the primary entry point for external systems.

    Args:
        matcher_output: List of diagnosed failure dicts from matcher.py.
        use_learning: Use learning policies for scoring and fix selection.
        top_k: Number of fixes to generate.
        auto_apply: If True and gate passes, execute fixes automatically.
        include_abstraction: Include abstraction layer output.
        evaluation_runner: Optional callback for external evaluation.
            If provided and auto_apply gate passes, this function is called
            instead of the built-in counterfactual simulation.
            Signature: evaluation_runner(patch_bundle: dict) -> dict
            Must return: {success, failure_count, root, has_hard_regression, notes}

    Returns:
        Dict with: diagnosis, abstraction (optional), fix, summary.

    Example:
        >>> from pipeline import run_pipeline
        >>> import json
        >>> with open("matcher_output.json") as f:
        ...     matcher_output = json.load(f)
        >>> result = run_pipeline(matcher_output, use_learning=True)
        >>> print(result["summary"]["root_cause"])
        'premature_model_commitment'
        >>> print(result["summary"]["gate_mode"])
        'auto_apply'

    Example with evaluation_runner:
        >>> def my_runner(bundle):
        ...     # Run fixes in your staging environment
        ...     return {"success": True, "failure_count": 0,
        ...             "root": None, "has_hard_regression": False,
        ...             "notes": "passed staging tests"}
        >>> result = run_pipeline(matcher_output, auto_apply=True,
        ...                       evaluation_runner=my_runner)
    """
    # Step 1: Diagnosis
    diagnosis = run_diagnosis(matcher_output)

    # Step 2: Abstraction (optional)
    abstraction_output = None
    if include_abstraction:
        abstraction_output = abstract(diagnosis)

    # Step 3: Fix pipeline
    fix_result = run_fix(
        diagnosis,
        use_learning=use_learning,
        top_k=top_k,
        auto_apply=False,  # handle auto_apply here with runner support
    )

    # Step 3b: Auto-apply with evaluation_runner support (Phase 25-lite)
    if auto_apply and fix_result["gate"]["gate"]["mode"] == "auto_apply":
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
                # External evaluation failed — fall back to staged_review
                # deterministically rather than crashing the pipeline.
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
            graph = load_graph(str(GRAPH_PATH))
            apply_result = maybe_apply(
                fix_result["gate"], diagnosis, fix_result["autofix"],
                fix_result["execution_plan"], graph
            )
            fix_result["gate"] = apply_result

    # Step 4: Summary
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

    result = {
        "diagnosis": diagnosis,
        "fix": fix_result,
        "summary": summary,
    }

    if abstraction_output:
        result["abstraction"] = abstraction_output

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """CLI: run full pipeline from matcher_output.json."""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    if not args:
        print("Usage: python pipeline.py matcher_output.json [--use-learning] [--top-k N] [--auto-apply]")
        sys.exit(1)

    with open(args[0], encoding="utf-8") as f:
        matcher_output = json.load(f)

    top_k = 1
    for i, a in enumerate(sys.argv[1:]):
        if a == "--top-k" and i + 1 < len(sys.argv) - 1:
            top_k = int(sys.argv[i + 2])

    result = run_pipeline(
        matcher_output,
        use_learning="--use-learning" in flags,
        top_k=top_k,
        auto_apply="--auto-apply" in flags,
        include_abstraction="--with-abstraction" in flags,
    )

    if "--json-only" in flags:
        print(json.dumps(result, indent=2))
    else:
        s = result["summary"]
        print(f"\n=== PIPELINE RESULT ===")
        print(f"  Root cause:  {s['root_cause']} (confidence: {s['root_confidence']})")
        print(f"  Failures:    {s['failure_count']}")
        print(f"  Fixes:       {s['fix_count']}")
        print(f"  Gate:        {s['gate_mode']} (score: {s['gate_score']})")
        if s["applied"]:
            print(f"  Applied:     yes → {s['evaluation_decision']}")
            if s["rollback"]:
                print(f"  Rollback:    yes")
        else:
            print(f"  Applied:     no")


if __name__ == "__main__":
    main()