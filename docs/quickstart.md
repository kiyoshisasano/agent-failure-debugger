# Quick Start Guide

## Install

```bash
pip install agent-failure-debugger
```

This installs both `agent-failure-debugger` and `llm-failure-atlas` (its dependency). No other setup needed.

---

## Recommended for most users: `diagnose()`

The simplest way to use the tool. One function handles everything: adapt → detect → diagnose → explain → assess execution quality.

```python
from agent_failure_debugger import diagnose

raw_log = {
    "inputs": {"query": "Change my flight to tomorrow morning"},
    "outputs": {"response": "I've found several hotels near the airport for you."},
    "steps": [
        {
            "type": "llm",
            "name": "ChatOpenAI",
            "inputs": {"prompt": "User wants to change flight..."},
            "outputs": {"text": "Let me check available flights."},
            "metadata": {"model": "gpt-4"}
        },
        {
            "type": "tool",
            "name": "search_flights",
            "inputs": {"date": "2025-03-20"},
            "outputs": {"flights": []},
            "error": None
        },
        {
            "type": "tool",
            "name": "search_flights",
            "inputs": {"date": "2025-03-20"},
            "outputs": {"flights": []},
            "error": None
        },
        {
            "type": "tool",
            "name": "search_flights",
            "inputs": {"date": "2025-03-20"},
            "outputs": {"flights": []},
            "error": None
        },
        {
            "type": "llm",
            "name": "ChatOpenAI",
            "inputs": {"prompt": "Based on the documents..."},
            "outputs": {"text": "I've found several hotels near the airport for you."},
            "metadata": {"model": "gpt-4"}
        }
    ],
    "feedback": {
        "user_correction": "I asked about flights, not hotels.",
        "user_rating": 1
    },
    "latency_ms": 4500
}

result = diagnose(raw_log, adapter="langchain")

s = result.get("summary", {})
print(f"Root cause:  {s.get('root_cause', 'none')}")
print(f"Confidence:  {s.get('root_confidence', 0)}")
print(f"Failures:    {s.get('failure_count', 0)}")
print(f"Fixes:       {s.get('fix_count', 0)}")

# Execution quality
eq = s.get("execution_quality", {})
print(f"Status:      {eq.get('status', 'unknown')}")   # healthy / degraded / failed
print(f"Termination: {eq.get('termination', {}).get('mode', 'unknown')}")
```

**Available adapters:** `langchain`, `langsmith`, `crewai`, `redis_help_demo`. See [Adapter Formats](https://github.com/kiyoshisasano/llm-failure-atlas/blob/main/docs/adapter_formats.md) for the input shape each adapter expects.

**Minimal input requirement (langchain adapter):**
- `inputs.query` — the user's question
- `outputs.response` — the agent's answer
- `steps` with `type` set to `"llm"` or `"tool"`

Without these, detection will be limited or return zero failures. The system does not raise errors for incomplete input — it silently returns zero failures.

---

## Execution quality

Every `diagnose()` result includes an execution quality assessment in the summary:

```python
eq = result["summary"]["execution_quality"]
```

| Status | Meaning |
|---|---|
| `healthy` | No significant issues detected |
| `degraded` | Output was produced but quality indicators are weak (low alignment, weak grounding) |
| `failed` | Execution did not produce usable output (silent exit, error, or task incomplete — tools failed and agent gave up) |

Termination mode classifies how the agent ended:

| Mode | Meaning |
|---|---|
| `normal` | Agent completed and produced output |
| `silent_exit` | Agent stopped without output or error |
| `error_exit` | Agent stopped due to an execution error |
| `partial_exit` | Agent produced output despite an error |
| `unknown` | Insufficient telemetry to determine |

Execution quality uses existing telemetry and diagnosis results. No new matcher patterns are added — this assessment sits in the summary layer.

---

## Use for live systems: `watch()`

Wraps a LangGraph agent for real-time failure detection. Requires `pip install langchain-core`.

```python
from llm_failure_atlas.adapters.callback_handler import watch

graph = watch(workflow.compile(), auto_diagnose=True)
result = graph.invoke({"messages": [...]})
# → failures printed on completion
```

Add `auto_pipeline=True` to also run the full debugger pipeline (root cause + fix proposal) on completion.

---

## Self-healing agent (LangGraph)

Add automatic failure detection and informed retry to any LangGraph agent. Requires `pip install agent-failure-debugger[langchain] langgraph`.

```python
from agent_failure_debugger import create_health_check
from langgraph.graph import StateGraph, MessagesState, START, END

health_check, route = create_health_check(max_retries=2)

workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("health_check", health_check)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue,
                               {"tools": "tools", "check": "health_check"})
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges("health_check", route,
                               {"retry": "agent", "end": END})
```

On retry, the diagnosis is injected into the conversation as a HumanMessage — the LLM reads *why* it failed and adjusts its approach (informed retry, not blind).

Not all failures benefit from retry. The 17 Atlas patterns are classified as retryable (transient errors, LLM non-determinism) or structural (bad prompts, config issues). Structural failures are reported immediately without wasting retries. See [examples/self_healing/](../examples/self_healing/) for a working demo.

---

## CI integration (pytest)

Use [pytest-agent-health](https://github.com/kiyoshisasano/pytest-agent-health) to catch silent agent failures and regressions in CI:

```bash
pip install pytest-agent-health
```

```python
def test_agent_completes_task(agent_health):
    raw_log = my_agent.run("What was Q3 revenue?")
    agent_health.check(raw_log, adapter="langchain")
    # → FAIL if execution failed or regression detected
    # → WARN if quality concerns detected
    # → PASS if healthy
```

```bash
# Establish baselines
pytest --agent-health --agent-health-update-baseline

# Detect regressions against baseline
pytest --agent-health

# Strict mode: degraded + risk = FAIL
pytest --agent-health --agent-health-strict

# Force FAIL on specific patterns
pytest --agent-health --agent-health-fail-on=premature_termination
```

The plugin automatically compares against previous CI runs. New failure patterns, status degradation, and new risk indicators trigger FAIL. Commit `.agent-health/` to git to share baselines across CI runs.

CI detects failures and regressions. Production heals them with `create_health_check()`.

---

## Multi-run analysis

When the same task is run multiple times (common with non-deterministic LLM agents), use the two-step workflow:

### Step 1: Is the agent stable?

```python
from agent_failure_debugger import compare_runs

stability = compare_runs(all_run_results)
print(stability["stability"]["root_cause_agreement"])  # 1.0 = fully stable
print(stability["interpretation"])
```

`compare_runs()` takes a list of `run_pipeline()` outputs (at least 2) and measures how consistently the debugger diagnoses the same failures across runs.

### Step 2: What separates success from failure?

```python
from agent_failure_debugger import diff_runs

diff = diff_runs(success_runs, failure_runs)
print(diff["hypothesis"])
print(diff["failure_set_diff"]["failure_only"])
```

`diff_runs()` compares two groups of runs and identifies structural differences: patterns that only appear in failures, root cause shifts, signal-level divergence, and causal path differences.

### Separating runs by execution quality

Use `execution_quality.status` to automatically split runs into success and failure groups:

```python
from agent_failure_debugger import diagnose, compare_runs, diff_runs

# Run the same task multiple times
results = [diagnose(log, adapter="langchain") for log in run_logs]

# Step 1: Check stability
stability = compare_runs(results)
if stability["stability"]["root_cause_agreement"] < 1.0:
    # Step 2: Identify divergence
    success = [r for r in results if r["summary"]["execution_quality"]["status"] == "healthy"]
    failure = [r for r in results if r["summary"]["execution_quality"]["status"] != "healthy"]
    if success and failure:
        diff = diff_runs(success, failure)
        print(diff["hypothesis"])
```

For runnable examples, see [examples/multi_run_stability](../examples/multi_run_stability/) and [examples/termination_divergence](../examples/termination_divergence/).

---

## Advanced / debugging: direct telemetry

Bypass adapters entirely by constructing the telemetry dict yourself. Use this when testing detection behavior, building custom integrations, or debugging why a pattern doesn't fire.

```python
from llm_failure_atlas.matcher import run
from llm_failure_atlas.resource_loader import get_patterns_dir
from agent_failure_debugger.pipeline import run_pipeline
from pathlib import Path
import json, tempfile, os

telemetry = {
    "input": {"ambiguity_score": 0.9},
    "interaction": {"clarification_triggered": False, "user_correction_detected": False},
    "reasoning": {"replanned": False, "hypothesis_count": 1},
    "cache": {"hit": False, "similarity": 0.0, "query_intent_similarity": 1.0},
    "retrieval": {"skipped": False},
    "response": {"alignment_score": 0.4},
    "tools": {"call_count": 0, "repeat_count": 0, "soft_error_count": 0},
    "state": {"progress_made": True, "tool_progress": {}, "any_tool_looping": False,
              "output_produced": True, "chain_error_occurred": False},
    "grounding": {"tool_provided_data": False, "uncertainty_acknowledged": False,
                  "response_length": 500, "source_data_length": 0, "expansion_ratio": 0.0},
}

tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
json.dump(telemetry, tmp)
tmp.close()

diagnosed = []
for pf in sorted(Path(get_patterns_dir()).glob("*.yaml")):
    r = run(str(pf), tmp.name)
    if r.get("diagnosed"):
        diagnosed.append(r)
        print(f"  Detected: {r['failure_id']}  conf={r['confidence']}")

os.unlink(tmp.name)

if diagnosed:
    result = run_pipeline(diagnosed)
    s = result["summary"]
    print(f"\nRoot cause: {s['root_cause']}, Fixes: {s['fix_count']}")
    print(f"Execution:  {s['execution_quality']['status']}")
```

---

## Common Mistakes

**⚠ No error is raised for wrong inputs.** The system will silently return zero failures if the adapter cannot extract signals. A result of "0 failures detected" may mean the input was correct and no failure occurred, or it may mean the input was malformed and nothing could be analyzed. Check observation coverage and input format to confirm.

**"No failures detected" on a clearly bad log:**
The adapter needs enough data to extract signals. A minimal log with just `{"steps": [{"type": "llm", "output": "..."}]}` won't trigger most patterns because the adapter can't compute tool loops, cache misuse, or grounding signals. Provide complete traces with tool calls, retriever results, and input/output pairs.

**Wrong adapter:**
Each adapter expects a specific input shape. Using `langchain` adapter with a LangSmith run-tree export (or vice versa) produces empty telemetry and detects nothing. No error is raised — it silently returns zero failures. See [Adapter Formats](https://github.com/kiyoshisasano/llm-failure-atlas/blob/main/docs/adapter_formats.md).

**"0 failures" doesn't mean your agent is fine:**
It means no detectable pattern matched with the available signals. If your adapter doesn't produce `state` or `grounding` fields (e.g., `crewai` adapter), some patterns can't fire. See [Adapter Coverage](limitations_faq.md#adapter-coverage).

**"degraded" doesn't mean broken:**
`execution_quality.status = "degraded"` means the agent produced output but quality indicators suggest concerns (low alignment, weak grounding, high expansion ratio). It is a signal to review, not a definitive failure judgment. The thresholds are intentionally conservative.