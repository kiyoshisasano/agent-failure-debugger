"""Test health_check_node routing logic without LLM calls."""
import sys
sys.path.insert(0, "/home/claude/repos/debugger-packaged/src")

from unittest.mock import patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agent_failure_debugger.integrations.langgraph import (
    create_health_check,
    _HEALTH_CHECK_STATE_KEY,
)


def _make_diagnosis(status, failure_ids=None):
    """Create a mock diagnosis result."""
    return {
        "summary": {
            "root_cause": failure_ids[0] if failure_ids else "unknown",
            "root_confidence": 0.8,
            "failure_count": len(failure_ids or []),
            "fix_count": 0,
            "gate_mode": "manual",
            "gate_score": 0.0,
            "applied": False,
            "execution_quality": {
                "status": status,
                "termination": {"mode": "normal", "reasons": []},
                "indicators": [],
                "summary": f"Status: {status}",
            },
        },
        "matcher_output": [
            {"failure_id": fid, "diagnosed": True, "confidence": 0.8}
            for fid in (failure_ids or [])
        ],
        "fix": {"autofix": {"recommended_fixes": []}},
        "telemetry": {},
    }


def test_healthy_passes_through():
    """Healthy status → pass, no retry."""
    health_check, route = create_health_check(verbose=False)

    state = {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ],
    }

    with patch("agent_failure_debugger.diagnose.diagnose") as mock_diag:
        mock_diag.return_value = _make_diagnosis("healthy")
        result = health_check(state)

    assert result["__health_check"]["status"] == "pass"
    assert route(result) == "end"
    print("✅ test_healthy_passes_through")


def test_failed_retryable_retries():
    """Failed + retryable pattern → retry with feedback."""
    health_check, route = create_health_check(verbose=False)

    state = {
        "messages": [
            HumanMessage(content="Get data"),
            AIMessage(content="Error occurred"),
        ],
    }

    with patch("agent_failure_debugger.diagnose.diagnose") as mock_diag:
        mock_diag.return_value = _make_diagnosis("failed", ["agent_tool_call_loop"])
        result = health_check(state)

    assert result["__health_check"]["status"] == "retry"
    assert result[_HEALTH_CHECK_STATE_KEY] == 1
    assert route(result) == "retry"

    # Feedback should be injected as SystemMessage
    feedback_msgs = result.get("messages", [])
    assert len(feedback_msgs) == 1
    assert isinstance(feedback_msgs[0], SystemMessage)
    assert "agent_tool_call_loop" in feedback_msgs[0].content
    print("✅ test_failed_retryable_retries")


def test_failed_structural_no_retry():
    """Failed + structural pattern → pass with warnings, no retry."""
    health_check, route = create_health_check(verbose=False)

    state = {
        "messages": [
            HumanMessage(content="Get data"),
            AIMessage(content="Wrong answer"),
        ],
    }

    with patch("agent_failure_debugger.diagnose.diagnose") as mock_diag:
        mock_diag.return_value = _make_diagnosis("failed", ["context_truncation_loss"])
        result = health_check(state)

    assert result["__health_check"]["status"] == "warn"
    assert route(result) == "end"
    print("✅ test_failed_structural_no_retry")


def test_max_retries_respected():
    """Retry count at max → pass, no more retries."""
    health_check, route = create_health_check(max_retries=2, verbose=False)

    state = {
        "messages": [
            HumanMessage(content="Get data"),
            AIMessage(content="Still failing"),
        ],
        _HEALTH_CHECK_STATE_KEY: 2,  # Already retried twice
    }

    with patch("agent_failure_debugger.diagnose.diagnose") as mock_diag:
        mock_diag.return_value = _make_diagnosis("failed", ["agent_tool_call_loop"])
        result = health_check(state)

    assert result["__health_check"]["status"] == "warn"
    assert route(result) == "end"
    print("✅ test_max_retries_respected")


def test_degraded_default_no_retry():
    """Degraded with retryable pattern but retry_on_degraded=False → no retry."""
    health_check, route = create_health_check(retry_on_degraded=False, verbose=False)

    state = {
        "messages": [
            HumanMessage(content="Compare data"),
            AIMessage(content="Here is the comparison"),
        ],
    }

    with patch("agent_failure_debugger.diagnose.diagnose") as mock_diag:
        mock_diag.return_value = _make_diagnosis("degraded", ["incorrect_output"])
        result = health_check(state)

    assert result["__health_check"]["status"] == "warn"
    assert route(result) == "end"
    print("✅ test_degraded_default_no_retry")


def test_degraded_opt_in_retry():
    """Degraded with retry_on_degraded=True + retryable → retry."""
    health_check, route = create_health_check(
        retry_on_degraded=True, verbose=False,
    )

    state = {
        "messages": [
            HumanMessage(content="Compare data"),
            AIMessage(content="Here is the comparison"),
        ],
    }

    with patch("agent_failure_debugger.diagnose.diagnose") as mock_diag:
        mock_diag.return_value = _make_diagnosis("degraded", ["incorrect_output"])
        result = health_check(state)

    assert result["__health_check"]["status"] == "retry"
    assert route(result) == "retry"
    print("✅ test_degraded_opt_in_retry")


def test_custom_inject_feedback():
    """Custom inject_feedback function is used."""
    def my_inject(state, text):
        return {"chat_log": [{"role": "system", "text": text}]}

    health_check, route = create_health_check(
        inject_feedback=my_inject, verbose=False,
    )

    state = {
        "messages": [
            HumanMessage(content="Get data"),
            AIMessage(content="Error"),
        ],
    }

    with patch("agent_failure_debugger.diagnose.diagnose") as mock_diag:
        mock_diag.return_value = _make_diagnosis("failed", ["failed_termination"])
        result = health_check(state)

    assert "chat_log" in result
    assert result["chat_log"][0]["role"] == "system"
    print("✅ test_custom_inject_feedback")


def test_on_diagnosis_callback():
    """on_diagnosis callback is called."""
    captured = []

    health_check, route = create_health_check(
        on_diagnosis=lambda d: captured.append(d),
        verbose=False,
    )

    state = {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi"),
        ],
    }

    with patch("agent_failure_debugger.diagnose.diagnose") as mock_diag:
        mock_diag.return_value = _make_diagnosis("healthy")
        health_check(state)

    assert len(captured) == 1
    print("✅ test_on_diagnosis_callback")


def test_mixed_retryable_and_structural():
    """Both retryable and structural patterns → retry (retryable takes priority)."""
    health_check, route = create_health_check(verbose=False)

    state = {
        "messages": [
            HumanMessage(content="Complex query"),
            AIMessage(content="Bad output"),
        ],
    }

    with patch("agent_failure_debugger.diagnose.diagnose") as mock_diag:
        mock_diag.return_value = _make_diagnosis(
            "failed", ["agent_tool_call_loop", "context_truncation_loss"]
        )
        result = health_check(state)

    # Has retryable pattern, so should retry
    assert result["__health_check"]["status"] == "retry"
    assert route(result) == "retry"
    print("✅ test_mixed_retryable_and_structural")


if __name__ == "__main__":
    test_healthy_passes_through()
    test_failed_retryable_retries()
    test_failed_structural_no_retry()
    test_max_retries_respected()
    test_degraded_default_no_retry()
    test_degraded_opt_in_retry()
    test_custom_inject_feedback()
    test_on_diagnosis_callback()
    test_mixed_retryable_and_structural()
    print("\n✅ All routing tests passed")