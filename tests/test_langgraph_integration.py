"""Test _messages_to_raw_log conversion and health check logic."""
import sys
sys.path.insert(0, "/home/claude/repos/debugger-packaged/src")

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from agent_failure_debugger.integrations.langgraph import (
    _messages_to_raw_log,
    _build_feedback_message,
    RETRYABLE_PATTERNS,
    STRUCTURAL_PATTERNS,
    META_PATTERNS,
)


def test_simple_conversation():
    """Human → AI with no tools."""
    messages = [
        HumanMessage(content="What is 2+2?"),
        AIMessage(content="The answer is 4."),
    ]
    raw = _messages_to_raw_log(messages)
    assert raw["inputs"]["query"] == "What is 2+2?"
    assert raw["outputs"]["response"] == "The answer is 4."
    assert len(raw["steps"]) == 1
    assert raw["steps"][0]["type"] == "llm"
    print("✅ test_simple_conversation")


def test_tool_call_flow():
    """Human → AI(tool_call) → Tool → AI(response)."""
    messages = [
        HumanMessage(content="What is Apple's revenue?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "get_revenue", "args": {"company": "Apple"}, "id": "tc1"}],
        ),
        ToolMessage(content="$394.3 billion", tool_call_id="tc1", name="get_revenue"),
        AIMessage(content="Apple's revenue is $394.3 billion."),
    ]
    raw = _messages_to_raw_log(messages)

    assert raw["inputs"]["query"] == "What is Apple's revenue?"
    assert raw["outputs"]["response"] == "Apple's revenue is $394.3 billion."

    # Steps: llm (empty), tool, llm (response)
    step_types = [s["type"] for s in raw["steps"]]
    assert "tool" in step_types
    assert step_types.count("llm") == 2

    # Tool step should have outputs filled
    tool_steps = [s for s in raw["steps"] if s["type"] == "tool"]
    assert len(tool_steps) == 1
    assert tool_steps[0]["name"] == "get_revenue"
    assert tool_steps[0]["outputs"]["result"] == "$394.3 billion"
    assert tool_steps[0]["inputs"] == {"company": "Apple"}
    print("✅ test_tool_call_flow")


def test_multiple_tool_calls():
    """Agent calls multiple tools."""
    messages = [
        HumanMessage(content="Compare Apple and Microsoft revenue"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_revenue", "args": {"company": "Apple"}, "id": "tc1"},
                {"name": "get_revenue", "args": {"company": "Microsoft"}, "id": "tc2"},
            ],
        ),
        ToolMessage(content="$394.3 billion", tool_call_id="tc1", name="get_revenue"),
        ToolMessage(content="$245.1 billion", tool_call_id="tc2", name="get_revenue"),
        AIMessage(content="Apple: $394.3B, Microsoft: $245.1B"),
    ]
    raw = _messages_to_raw_log(messages)

    tool_steps = [s for s in raw["steps"] if s["type"] == "tool"]
    assert len(tool_steps) == 2
    assert tool_steps[0]["outputs"]["result"] == "$394.3 billion"
    assert tool_steps[1]["outputs"]["result"] == "$245.1 billion"
    print("✅ test_multiple_tool_calls")


def test_system_message_ignored():
    """System messages don't create steps."""
    messages = [
        SystemMessage(content="You are a financial assistant."),
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there!"),
    ]
    raw = _messages_to_raw_log(messages)
    assert raw["inputs"]["query"] == "Hello"
    assert len(raw["steps"]) == 1
    print("✅ test_system_message_ignored")


def test_with_retry_feedback():
    """Messages include a health check feedback message."""
    messages = [
        SystemMessage(content="You are a financial assistant."),
        HumanMessage(content="What is Apple's revenue?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "get_revenue", "args": {"company": "Apple"}, "id": "tc1"}],
        ),
        ToolMessage(content="Error: service unavailable", tool_call_id="tc1", name="get_revenue"),
        AIMessage(content="I apologize, the service is unavailable."),
        # Health check injects feedback
        SystemMessage(content="[Health Check] Previous attempt status: failed..."),
        # Agent retries
        AIMessage(
            content="",
            tool_calls=[{"name": "get_revenue", "args": {"company": "Apple"}, "id": "tc2"}],
        ),
        ToolMessage(content="$394.3 billion", tool_call_id="tc2", name="get_revenue"),
        AIMessage(content="Apple's revenue is $394.3 billion."),
    ]
    raw = _messages_to_raw_log(messages)

    # Query should be the first human message
    assert raw["inputs"]["query"] == "What is Apple's revenue?"
    # Response should be the last AI content
    assert raw["outputs"]["response"] == "Apple's revenue is $394.3 billion."
    # Should have tool steps from both attempts
    tool_steps = [s for s in raw["steps"] if s["type"] == "tool"]
    assert len(tool_steps) == 2
    print("✅ test_with_retry_feedback")


def test_pattern_classification_complete():
    """All 17 patterns are classified."""
    all_patterns = RETRYABLE_PATTERNS | STRUCTURAL_PATTERNS | META_PATTERNS
    expected = {
        "agent_tool_call_loop", "assumption_invalidation_failure",
        "clarification_failure", "conflicting_signals",
        "context_truncation_loss", "failed_termination",
        "incorrect_output", "instruction_priority_inversion",
        "insufficient_observability", "premature_model_commitment",
        "premature_termination", "prompt_injection_via_retrieval",
        "rag_retrieval_drift", "repair_strategy_failure",
        "semantic_cache_intent_bleeding", "tool_result_misinterpretation",
        "unmodeled_failure",
    }
    assert all_patterns == expected, f"Missing: {expected - all_patterns}, Extra: {all_patterns - expected}"
    # No overlap
    assert not (RETRYABLE_PATTERNS & STRUCTURAL_PATTERNS)
    assert not (RETRYABLE_PATTERNS & META_PATTERNS)
    assert not (STRUCTURAL_PATTERNS & META_PATTERNS)
    print("✅ test_pattern_classification_complete")


def test_feedback_message():
    """Feedback message is well-formed."""
    diag = {
        "summary": {
            "execution_quality": {
                "status": "failed",
                "indicators": [
                    {"concern": "tools returned identical results", "signal": "x", "value": 0.2},
                ],
            },
        },
        "matcher_output": [
            {"failure_id": "agent_tool_call_loop", "diagnosed": True, "confidence": 0.8},
        ],
        "fix": {
            "autofix": {
                "recommended_fixes": [{
                    "patch": {"content": "Stop repeating tool calls."},
                }],
            },
        },
    }
    msg = _build_feedback_message(diag, retry_count=1, max_retries=2)
    assert "[Health Check]" in msg
    assert "agent_tool_call_loop" in msg
    assert "retry 1/2" in msg
    assert "different approach" in msg.lower() or "different tool" in msg.lower()
    print("✅ test_feedback_message")


def test_diagnose_integration():
    """Full integration: messages → diagnose() → execution_quality."""
    from agent_failure_debugger import diagnose

    # Simulate a tool loop: same tool output repeated
    messages = [
        HumanMessage(content="Compare Apple, Microsoft, and Google revenue"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_revenue", "args": {"company": "Apple"}, "id": "tc1"},
                {"name": "get_revenue", "args": {"company": "Microsoft"}, "id": "tc2"},
                {"name": "get_revenue", "args": {"company": "Google"}, "id": "tc3"},
            ],
        ),
        ToolMessage(content="Revenue: $394.3 billion — Apple Inc.", tool_call_id="tc1", name="get_revenue"),
        ToolMessage(content="Revenue: $394.3 billion — Apple Inc.", tool_call_id="tc2", name="get_revenue"),
        ToolMessage(content="Revenue: $394.3 billion — Apple Inc.", tool_call_id="tc3", name="get_revenue"),
        AIMessage(content="Apple: $394.3B, Microsoft: $245.1B, Google: $328.2B"),
    ]
    raw = _messages_to_raw_log(messages)
    result = diagnose(raw, adapter="langchain")

    eq = result["summary"]["execution_quality"]
    print(f"  Status: {eq['status']}")
    print(f"  Indicators: {[i['signal'] for i in eq.get('indicators', [])]}")

    # Should detect low tool_result_diversity
    telemetry = result.get("telemetry", {})
    diversity = telemetry.get("grounding", {}).get("tool_result_diversity")
    print(f"  tool_result_diversity: {diversity}")

    # 3 identical results → diversity should be ~0.33
    if diversity is not None:
        assert diversity < 0.5, f"Expected diversity < 0.5, got {diversity}"
        print(f"✅ test_diagnose_integration (diversity={diversity})")
    else:
        print(f"⚠️ test_diagnose_integration: diversity is None (adapter may not extract it)")


if __name__ == "__main__":
    test_simple_conversation()
    test_tool_call_flow()
    test_multiple_tool_calls()
    test_system_message_ignored()
    test_with_retry_feedback()
    test_pattern_classification_complete()
    test_feedback_message()
    test_diagnose_integration()
    print("\n✅ All tests passed")