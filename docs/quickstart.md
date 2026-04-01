# Quick Start Guide

## Install

```bash
pip install agent-failure-debugger
```

This installs both `agent-failure-debugger` and `llm-failure-atlas` (its dependency). No other setup needed.

---

## Recommended for most users: `diagnose()`

The simplest way to use the tool. One function handles everything: adapt → detect → diagnose → explain.

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
```

**Available adapters:** `langchain`, `langsmith`, `crewai`, `redis_help_demo`. See [Adapter Formats](adapter_formats.md) for the input shape each adapter expects.

**Minimal input requirement (langchain adapter):**
- `inputs.query` — the user's question
- `outputs.response` — the agent's answer
- `steps` with `type` set to `"llm"` or `"tool"`

Without these, detection will be limited or return zero failures. The system does not raise errors for incomplete input — it silently returns zero failures.

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
```

---

## Common Mistakes

**⚠ No error is raised for wrong inputs.** The system will silently return zero failures if the adapter cannot extract signals. A result of "0 failures detected" may mean the input was correct and no failure occurred, or it may mean the input was malformed and nothing could be analyzed. Check observation coverage and input format to confirm.

**"No failures detected" on a clearly bad log:**
The adapter needs enough data to extract signals. A minimal log with just `{"steps": [{"type": "llm", "output": "..."}]}` won't trigger most patterns because the adapter can't compute tool loops, cache misuse, or grounding signals. Provide complete traces with tool calls, retriever results, and input/output pairs.

**Wrong adapter:**
Each adapter expects a specific input shape. Using `langchain` adapter with a LangSmith run-tree export (or vice versa) produces empty telemetry and detects nothing. No error is raised — it silently returns zero failures. See [Adapter Formats](adapter_formats.md).

**"0 failures" doesn't mean your agent is fine:**
It means no detectable pattern matched with the available signals. If your adapter doesn't produce `state` or `grounding` fields (e.g., `crewai` adapter), some patterns can't fire. See [Adapter Coverage](limitations_faq.md#adapter-coverage).