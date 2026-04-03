"""
agent_failure_debugger — Diagnose agent execution behavior.

Deterministic causal analysis with fix generation.

Single-run:
    from agent_failure_debugger import diagnose
    result = diagnose(raw_log, adapter="langchain")
    print(result["summary"]["execution_quality"]["status"])  # healthy/degraded/failed

    from agent_failure_debugger import watch
    graph = watch(workflow.compile(), auto_diagnose=True)

Multi-run:
    from agent_failure_debugger import compare_runs, diff_runs

    # Stability analysis: is the agent consistent?
    stability = compare_runs(run_results)
    print(stability["stability"]["root_cause_agreement"])

    # Differential diagnosis: what separates success from failure?
    diff = diff_runs(success_runs, failure_runs)
    print(diff["hypothesis"])
"""

__version__ = "0.2.1"


# ---------------------------------------------------------------------------
# Single-run API
# ---------------------------------------------------------------------------

def diagnose(raw_log, adapter="langchain", **kwargs):
    """Diagnose failures from a raw agent log.

    This is the primary entry point for single-run analysis.

    Args:
        raw_log: Raw log/response from the agent or service.
        adapter: Adapter name ("langchain", "langsmith", "crewai", "redis_help_demo").
        **kwargs: Passed to run_pipeline (e.g. use_learning, top_k).

    Returns:
        Dict with: diagnosis, fix, summary (including execution_quality),
        explanation, telemetry, matcher_output.
    """
    from agent_failure_debugger.diagnose import diagnose as _diagnose
    return _diagnose(raw_log, adapter=adapter, **kwargs)


def watch(compiled_graph, **kwargs):
    """Wrap a LangGraph agent for live failure detection.

    Requires langchain-core: pip install agent-failure-debugger[langchain]

    Args:
        compiled_graph: A compiled LangGraph graph.
        **kwargs: auto_diagnose (bool), auto_pipeline (bool), verbose (bool).

    Returns:
        A wrapped graph with Atlas diagnosis injected.
    """
    try:
        from llm_failure_atlas.adapters.callback_handler import watch as _watch
    except ImportError:
        raise ImportError(
            "watch() requires langchain-core. "
            "Install with: pip install agent-failure-debugger[langchain]"
        )
    return _watch(compiled_graph, **kwargs)


# ---------------------------------------------------------------------------
# Multi-run API
# ---------------------------------------------------------------------------

def compare_runs(runs, task_id=None):
    """Analyze detection stability across multiple runs of the same task.

    Measures how consistently the debugger diagnoses the same failures
    across runs. Variation reflects agent behavior variation, not
    matcher instability (the matcher is deterministic).

    Args:
        runs: List of run_pipeline() outputs (at least 2).
        task_id: Optional task identifier for consistency validation.

    Returns:
        Dict with: run_count, stability (root_cause_agreement,
        failure_set_jaccard, stable/intermittent failures,
        confidence_cv), interpretation.
    """
    from agent_failure_debugger.reliability import compare_runs as _compare
    return _compare(runs, task_id=task_id)


def diff_runs(success_runs, failure_runs, task_id=None):
    """Identify structural differences between successful and failed runs.

    While compare_runs() measures stability across homogeneous runs,
    diff_runs() identifies what separates success from failure.

    Typical workflow:
        1. compare_runs(all_runs) → detect instability
        2. diff_runs(success_runs, failure_runs) → identify cause

    Args:
        success_runs: List of run_pipeline() outputs for successful runs.
        failure_runs: List of run_pipeline() outputs for failed runs.
        task_id: Optional task identifier for consistency validation.

    Returns:
        Dict with: failure_set_diff, root_cause_diff, signal_diff,
        confidence_diff, causal_path_diff, hypothesis.
    """
    from agent_failure_debugger.reliability import diff_runs as _diff
    return _diff(success_runs, failure_runs, task_id=task_id)