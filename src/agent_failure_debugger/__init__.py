"""
agent_failure_debugger — Diagnose why your LLM agent failed.

Deterministic causal analysis with fix generation.

Primary API:
    from agent_failure_debugger import diagnose
    result = diagnose(raw_log, adapter="langchain")

    from agent_failure_debugger import watch
    graph = watch(workflow.compile(), auto_diagnose=True)
"""

__version__ = "0.1.0"


def diagnose(raw_log, adapter="langchain", **kwargs):
    """Diagnose failures from a raw agent log.

    This is the primary entry point for the tool.

    Args:
        raw_log: Raw log/response from the agent or service.
        adapter: Adapter name ("langchain", "langsmith", "crewai", "redis_help_demo").
        **kwargs: Passed to run_pipeline (e.g. use_learning, top_k).

    Returns:
        Dict with: diagnosis, fix, summary, explanation, telemetry, matcher_output.
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
