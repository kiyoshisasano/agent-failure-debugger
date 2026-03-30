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
from auto_apply import gate_autofix
from abstraction import abstract
from explainer import explain as run_explanation
from pipeline_post_apply import run_post_apply
from pipeline_summary import build_pipeline_summary


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
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(matcher_output: list[dict],
                 use_learning: bool = False,
                 top_k: int = 1,
                 auto_apply: bool = False,
                 include_abstraction: bool = False,
                 include_explanation: bool = False,
                 evaluation_runner=None,
                 diagnosis_context: dict | None = None) -> dict:
    """
    Run the complete pipeline: diagnosis → fix → evaluation.

    This is the primary entry point for external systems.

    Args:
        matcher_output: List of diagnosed failure dicts from matcher.py.
        use_learning: Use learning policies for scoring and fix selection.
        top_k: Number of fixes to generate.
        auto_apply: If True and gate passes, execute fixes automatically.
        include_abstraction: Include abstraction layer output.
        include_explanation: Include enhanced explanation in result.
        evaluation_runner: Optional callback for external evaluation.
        diagnosis_context: Optional context dict from diagnose().
            If provided, observation summary is taken from context
            instead of re-deriving from matcher_output.
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

    # Step 3b: Post-apply evaluation
    fix_result = run_post_apply(
        diagnosis, fix_result,
        auto_apply=auto_apply,
        evaluation_runner=evaluation_runner,
        graph_path=str(GRAPH_PATH),
    )

    # Step 4: Summary
    summary = build_pipeline_summary(diagnosis, fix_result)

    result = {
        "diagnosis": diagnosis,
        "fix": fix_result,
        "summary": summary,
    }

    if abstraction_output:
        result["abstraction"] = abstraction_output

    # Step 5: Explanation (optional)
    if include_explanation:
        result["explanation"] = _build_explanation_block(
            diagnosis, matcher_output, diagnosis_context
        )

    return result


def _build_explanation_block(diagnosis: dict, matcher_output: list,
                             diagnosis_context: dict | None = None) -> dict:
    """Build explanation with observation summary.

    Uses diagnosis_context if available, otherwise falls back to
    deriving observation quality from matcher_output.
    """
    expl = run_explanation(
        diagnosis, use_llm=False, enhanced=True,
        diagnosis_context=diagnosis_context,
    )
    explanation = expl["response"]

    # Fallback: if explainer didn't include observation
    if "observation" not in explanation:
        observed = []
        missing = []
        for entry in matcher_output:
            oq = entry.get("observation_quality", {})
            for sig_name, info in oq.items():
                if info.get("observed"):
                    if sig_name not in observed:
                        observed.append(sig_name)
                elif info.get("missing"):
                    if sig_name not in missing:
                        missing.append(sig_name)
        total = len(observed) + len(missing)
        if total > 0:
            ratio = len(observed) / total
            if ratio >= 0.8:
                coverage = "high"
            elif ratio >= 0.5:
                coverage = "medium"
            else:
                coverage = "low"
        else:
            coverage = "unknown"
        explanation["observation"] = {
            "observed_signals": sorted(observed),
            "missing_signals": sorted(missing),
            "coverage": coverage,
        }

    return explanation


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