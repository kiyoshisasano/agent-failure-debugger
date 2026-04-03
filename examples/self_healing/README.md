# Self-Healing Agent Example

Adds automatic failure detection and **informed retry** to a LangGraph agent.

When the health check detects a retryable failure, it injects the diagnosis
into the conversation as a SystemMessage — the LLM reads *why* it failed
and adjusts its approach. This is not a blind retry.

## Quick Start

```bash
pip install agent-failure-debugger[langchain] langgraph langchain-openai
export OPENAI_API_KEY=...

python self_healing_demo.py
```

## What It Does

```
User Input → Agent → Health Check → healthy?  → END (output)
                         │
                         ├── failed (retryable) → inject feedback → Agent (retry)
                         ├── failed (structural) → END + diagnosis report
                         └── degraded → END + warnings
```

On retry, the health check appends a SystemMessage like:

> [Health Check] Previous attempt status: failed (retry 1/2).
> Detected issues: agent_tool_call_loop.
> You are repeating the same tool calls without progress.
> Try a different approach or use a different tool.

The LLM reads this and can change its strategy.

## Scenarios

| Scenario | What happens | Expected |
|---|---|---|
| `healthy` | Tool works normally | healthy, no retry |
| `tool_flaky` | Tool fails first call, works on retry | retry → healthy |
| `tool_loop` | Tool returns identical data for all queries | degraded (tool_result_diversity) |

## Which Failures Are Retryable?

Not all failures benefit from retry. The integration classifies each pattern:

**Retryable** (LLM non-determinism or transient errors may resolve):
- `agent_tool_call_loop` — transient API errors, rate limits
- `failed_termination` — transient execution errors
- `premature_termination` — temperature-dependent completion
- `premature_model_commitment` — may explore different hypothesis
- `incorrect_output` — non-determinism may improve
- `tool_result_misinterpretation` — LLM may parse differently

**Structural** (retry won't help — fix the prompt/config):
- `context_truncation_loss` — input too long
- `instruction_priority_inversion` — prompt structure issue
- `clarification_failure` — missing clarification logic
- `assumption_invalidation_failure` — reasoning structure
- `rag_retrieval_drift` — retrieval config issue
- `semantic_cache_intent_bleeding` — cache config issue
- `prompt_injection_via_retrieval` — missing guard
- `repair_strategy_failure` — repair design issue

## Usage in Your Agent

```python
from agent_failure_debugger.integrations.langgraph import create_health_check
from langgraph.graph import StateGraph, MessagesState, START, END

# Create health check node and routing function
health_check, route = create_health_check(max_retries=2)

# Build your graph as usual
workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("health_check", health_check)

# Wire it up
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue,
                               {"tools": "tools", "check": "health_check"})
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges("health_check", route,
                               {"retry": "agent", "end": END})

graph = workflow.compile()
```

## Configuration

```python
health_check, route = create_health_check(
    max_retries=2,           # Maximum retry attempts
    retry_on_degraded=False, # Retry on degraded status (default: only on failed)
    inject_feedback=None,    # Custom feedback injection for non-standard State
    on_diagnosis=callback,   # Called with full diagnosis on every check
    verbose=True,            # Print results to stdout
)
```

### Custom State Schema

If your State doesn't use `messages` as the key:

```python
def my_inject(state, feedback_text):
    from langchain_core.messages import SystemMessage
    return {"chat_history": [SystemMessage(content=feedback_text)]}

health_check, route = create_health_check(inject_feedback=my_inject)
```

## Running With Multiple Models

```bash
# Test with Claude
python self_healing_demo.py --model claude

# Test with Gemini
python self_healing_demo.py --model gemini

# Test all three
python self_healing_demo.py --model all
```