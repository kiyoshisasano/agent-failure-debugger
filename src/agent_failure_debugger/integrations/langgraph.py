"""
langgraph.py — Self-healing health check node for LangGraph.

Adds automatic failure detection and informed retry to any LangGraph agent.
On retry, the diagnosis is injected into State as a SystemMessage so the
LLM can adjust its behavior (not a blind retry).

Usage:
    from agent_failure_debugger.integrations.langgraph import (
        create_health_check,
    )

    health_check, route = create_health_check(max_retries=2)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("health_check", health_check)
    workflow.add_edge("tools", "agent")
    workflow.add_conditional_edges("agent", should_continue,
                                   {"tools": "tools", "check": "health_check"})
    workflow.add_conditional_edges("health_check", route,
                                   {"retry": "agent", "end": END})

Requires:
    pip install agent-failure-debugger[langchain]
    (langchain-core is needed for message types)

Design notes:
  - Not all failure patterns benefit from retry. Only patterns where
    LLM non-determinism or transient errors might produce a different
    result are retried. Structural failures (bad prompts, truncation)
    are reported immediately.
  - On retry, a SystemMessage with the diagnosis is appended to state
    so the LLM can adjust its approach (informed retry, not blind).
  - State key for messages defaults to "messages" (MessagesState).
    For custom State schemas, pass inject_feedback to override.
"""

from __future__ import annotations

import json
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Retryable pattern classification
# ---------------------------------------------------------------------------

# Patterns where retry may help due to LLM non-determinism or transient errors
RETRYABLE_PATTERNS = frozenset({
    "agent_tool_call_loop",       # transient API errors, rate limits
    "failed_termination",         # transient execution errors
    "premature_termination",      # temperature-dependent completion path
    "premature_model_commitment", # may explore different hypothesis
    "incorrect_output",           # non-determinism may improve result
    "tool_result_misinterpretation",  # LLM may parse differently
})

# Patterns where retry won't help — structural/config issues
STRUCTURAL_PATTERNS = frozenset({
    "context_truncation_loss",
    "instruction_priority_inversion",
    "clarification_failure",
    "assumption_invalidation_failure",
    "rag_retrieval_drift",
    "semantic_cache_intent_bleeding",
    "prompt_injection_via_retrieval",
    "repair_strategy_failure",
})

# Meta patterns — not actionable for retry
META_PATTERNS = frozenset({
    "unmodeled_failure",
    "insufficient_observability",
    "conflicting_signals",
})


# ---------------------------------------------------------------------------
# State → raw_log conversion
# ---------------------------------------------------------------------------

def _messages_to_raw_log(messages: list) -> dict:
    """Convert LangGraph MessagesState messages to langchain_adapter format.

    The langchain_adapter expects:
        {
            "inputs": {"query": "..."},
            "outputs": {"response": "..."},
            "steps": [{"type": "llm"|"tool"|"retriever", "name": ..., ...}],
        }

    We walk the messages list and reconstruct this structure.
    """
    query = ""
    response = ""
    steps = []

    for msg in messages:
        msg_type = getattr(msg, "type", None)

        if msg_type == "human":
            # First human message is the query; later ones are follow-ups
            if not query:
                query = msg.content if isinstance(msg.content, str) else str(msg.content)

        elif msg_type == "ai":
            content = msg.content if isinstance(msg.content, str) else str(msg.content)

            # Record as LLM step
            step = {
                "type": "llm",
                "name": "llm",
                "inputs": {},
                "outputs": {"text": content},
                "metadata": {},
                "error": None,
            }
            # Extract model info from response_metadata if available
            meta = getattr(msg, "response_metadata", None) or {}
            if meta.get("model_name"):
                step["metadata"]["model"] = meta["model_name"]

            steps.append(step)

            # Track tool calls as pending (will be resolved by ToolMessages)
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                steps.append({
                    "type": "tool",
                    "name": tc.get("name", "unknown"),
                    "inputs": tc.get("args", {}),
                    "outputs": {},  # filled by ToolMessage
                    "metadata": {},
                    "error": None,
                    "_tool_call_id": tc.get("id"),
                })

            # Last AI message with content is the response
            if content:
                response = content

        elif msg_type == "tool":
            # Match to pending tool step by tool_call_id
            tool_call_id = getattr(msg, "tool_call_id", None)
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            matched = False
            for step in steps:
                if (step["type"] == "tool"
                        and step.get("_tool_call_id") == tool_call_id
                        and not step["outputs"]):
                    step["outputs"] = {"result": content}
                    # Check for error status
                    status = getattr(msg, "status", None)
                    if status == "error":
                        step["error"] = content
                    matched = True
                    break
            if not matched:
                # Orphan tool message — create step anyway
                steps.append({
                    "type": "tool",
                    "name": getattr(msg, "name", None) or "unknown",
                    "inputs": {},
                    "outputs": {"result": content},
                    "metadata": {},
                    "error": None,
                })

        elif msg_type == "system":
            # System messages don't create steps
            pass

    # Clean up internal tracking fields
    for step in steps:
        step.pop("_tool_call_id", None)

    return {
        "inputs": {"query": query},
        "outputs": {"response": response},
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Feedback message generation
# ---------------------------------------------------------------------------

def _build_feedback_message(
    diagnosis_result: dict,
    retry_count: int,
    max_retries: int,
) -> str:
    """Build a feedback message from diagnosis results for the LLM.

    This message is injected into the conversation as a SystemMessage
    so the LLM understands why its previous attempt failed and can
    adjust its approach.
    """
    summary = diagnosis_result.get("summary", {})
    eq = summary.get("execution_quality", {})
    status = eq.get("status", "unknown")
    indicators = eq.get("indicators", [])

    # Identify diagnosed failures
    diagnosed = diagnosis_result.get("matcher_output", [])
    failure_ids = [d["failure_id"] for d in diagnosed]

    # Get fix suggestions
    fix_result = diagnosis_result.get("fix", {})
    autofix = fix_result.get("autofix", {}) if isinstance(fix_result, dict) else {}
    fixes = autofix.get("recommended_fixes", [])

    parts = [
        f"[Health Check] Previous attempt status: {status} "
        f"(retry {retry_count}/{max_retries}).",
    ]

    if failure_ids:
        parts.append(f"Detected issues: {', '.join(failure_ids)}.")

    # Add specific actionable guidance from indicators
    for ind in indicators[:2]:
        parts.append(f"Concern: {ind['concern']}.")

    # Add fix suggestions if available
    for fix in fixes[:1]:
        patch = fix.get("patch", {})
        content = patch.get("content")
        if content:
            parts.append(f"Suggestion: {content}")

    # Pattern-specific guidance
    if "agent_tool_call_loop" in failure_ids:
        parts.append(
            "You are repeating the same tool calls without progress. "
            "Try a different approach or use a different tool."
        )
    if "premature_model_commitment" in failure_ids:
        parts.append(
            "You committed to a single interpretation too early. "
            "Consider alternative hypotheses before answering."
        )
    if "tool_result_misinterpretation" in failure_ids:
        parts.append(
            "The tool output may have been misread. "
            "Re-examine the tool results carefully."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Default feedback injection
# ---------------------------------------------------------------------------

def _default_inject_feedback(state: dict, feedback_text: str) -> dict:
    """Inject feedback into MessagesState as a SystemMessage.

    Default implementation for the standard MessagesState schema
    where messages are stored under the "messages" key.
    """
    try:
        from langchain_core.messages import SystemMessage
    except ImportError:
        raise ImportError(
            "langchain-core is required for the LangGraph integration. "
            "Install with: pip install agent-failure-debugger[langchain]"
        )

    return {"messages": [SystemMessage(content=feedback_text)]}


# ---------------------------------------------------------------------------
# Health check node factory
# ---------------------------------------------------------------------------

_HEALTH_CHECK_STATE_KEY = "__health_check_retries"


def create_health_check(
    *,
    max_retries: int = 2,
    retry_on_degraded: bool = False,
    inject_feedback: Callable[[dict, str], dict] | None = None,
    on_diagnosis: Callable[[dict], None] | None = None,
    verbose: bool = True,
) -> tuple:
    """Create a health check node and routing function for LangGraph.

    Args:
        max_retries: Maximum retry attempts (default 2).
        retry_on_degraded: Whether to retry on degraded status (default False).
            Degraded means output was produced but quality is low. Retrying
            may not improve it for most patterns.
        inject_feedback: Custom function(state, feedback_text) -> state_update
            for non-standard State schemas. Default: appends SystemMessage
            to state["messages"].
        on_diagnosis: Optional callback receiving the full diagnosis result
            on every health check. Useful for logging/monitoring.
        verbose: Print health check results to stdout.

    Returns:
        Tuple of (health_check_node, route_on_health) functions.

    Example:
        health_check, route = create_health_check(max_retries=2)
        workflow.add_node("health_check", health_check)
        workflow.add_conditional_edges("health_check", route,
                                       {"retry": "agent", "end": END})
    """
    feedback_fn = inject_feedback or _default_inject_feedback

    def health_check_node(state: dict) -> dict:
        """Diagnose agent output and decide whether to retry.

        Reads the messages from state, converts to a raw log,
        runs diagnose(), and either passes through or injects
        feedback for retry.
        """
        from agent_failure_debugger.diagnose import diagnose

        messages = state.get("messages", [])
        if not messages:
            return state

        # Convert state messages to raw_log format
        raw_log = _messages_to_raw_log(messages)

        # Run diagnosis
        try:
            result = diagnose(raw_log, adapter="langchain")
        except Exception as e:
            if verbose:
                print(f"[Health Check] Diagnosis error: {e}")
            # On diagnosis failure, pass through (don't block the agent)
            return {"messages": [], "__health_check": {
                "status": "pass",
                "reason": f"diagnosis_error: {e}",
            }}

        # Notify callback
        if on_diagnosis:
            try:
                on_diagnosis(result)
            except Exception:
                pass

        # Extract execution quality
        summary = result.get("summary", {})
        eq = summary.get("execution_quality", {})
        status = eq.get("status", "healthy")
        indicators = eq.get("indicators", [])

        # Identify failure patterns
        diagnosed = result.get("matcher_output", [])
        failure_ids = {d["failure_id"] for d in diagnosed}
        retryable = failure_ids & RETRYABLE_PATTERNS
        structural = failure_ids & STRUCTURAL_PATTERNS

        # Track retry count
        retry_count = state.get(_HEALTH_CHECK_STATE_KEY, 0)

        if verbose:
            _print_health_check(status, failure_ids, indicators,
                                retry_count, max_retries)

        # Decision logic
        should_retry = False

        if status == "healthy":
            should_retry = False

        elif status == "failed":
            if retryable and retry_count < max_retries:
                should_retry = True

        elif status == "degraded":
            if retry_on_degraded and retryable and retry_count < max_retries:
                should_retry = True

        # Build state update
        if should_retry:
            feedback_text = _build_feedback_message(
                result, retry_count + 1, max_retries,
            )
            state_update = feedback_fn(state, feedback_text)

            # Increment retry counter
            state_update[_HEALTH_CHECK_STATE_KEY] = retry_count + 1
            state_update["__health_check"] = {
                "status": "retry",
                "attempt": retry_count + 1,
                "failures": sorted(failure_ids),
                "retryable": sorted(retryable),
            }

            if verbose:
                print(f"[Health Check] → Retrying (attempt {retry_count + 1}/{max_retries})")

            return state_update
        else:
            # Pass through — attach diagnosis as metadata
            check_info = {
                "status": "pass" if status == "healthy" else "warn",
                "execution_status": status,
                "failures": sorted(failure_ids),
            }
            if indicators:
                check_info["warnings"] = [
                    ind["concern"] for ind in indicators[:3]
                ]

            if verbose and status != "healthy":
                reason = "max retries" if retry_count >= max_retries else "structural"
                print(f"[Health Check] → Passing with warnings ({reason})")

            return {"messages": [], "__health_check": check_info}

    def route_on_health(state: dict) -> str:
        """Route based on health check result.

        Returns "retry" to send back to the agent node,
        or "end" to proceed to output.
        """
        check = state.get("__health_check", {})
        if check.get("status") == "retry":
            return "retry"
        return "end"

    return health_check_node, route_on_health


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _print_health_check(
    status: str,
    failure_ids: set,
    indicators: list,
    retry_count: int,
    max_retries: int,
) -> None:
    """Print health check results to stdout."""
    icons = {"healthy": "✅", "degraded": "⚠️", "failed": "❌"}
    icon = icons.get(status, "❓")

    print(f"\n{'='*50}")
    print(f"  {icon} Health Check: {status.upper()}")
    print(f"{'='*50}")

    if failure_ids:
        retryable = failure_ids & RETRYABLE_PATTERNS
        structural = failure_ids & STRUCTURAL_PATTERNS
        for fid in sorted(failure_ids):
            tag = " [retryable]" if fid in RETRYABLE_PATTERNS else " [structural]"
            print(f"  • {fid}{tag}")

    if indicators:
        for ind in indicators[:3]:
            print(f"  ⚑ {ind['concern']}")

    if retry_count > 0:
        print(f"  Retry: {retry_count}/{max_retries}")