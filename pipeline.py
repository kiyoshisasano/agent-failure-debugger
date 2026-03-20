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
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(matcher_output: list[dict],
                 use_learning: bool = False,
                 top_k: int = 1,
                 auto_apply: bool = False,
                 include_abstraction: bool = False) -> dict:
    """
    Run the complete pipeline: diagnosis → fix → evaluation.

    This is the primary entry point for external systems.

    Args:
        matcher_output: List of diagnosed failure dicts from matcher.py.
        use_learning: Use learning policies for scoring and fix selection.
        top_k: Number of fixes to generate.
        auto_apply: If True and gate passes, execute fixes automatically.
        include_abstraction: Include abstraction layer output.

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
        auto_apply=auto_apply,
    )

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

    with open(args[0]) as f:
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
