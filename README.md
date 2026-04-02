# agent-failure-debugger

[![PyPI version](https://badge.fury.io/py/agent-failure-debugger.svg)](https://pypi.org/project/agent-failure-debugger/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://pypi.org/project/agent-failure-debugger/)

Diagnoses agent execution behavior — not just *what* failed, but *why*, and whether execution quality is healthy, degraded, or failed. Deterministic causal analysis with fix generation.

```bash
pip install agent-failure-debugger
```

```python
from agent_failure_debugger import diagnose

result = diagnose(raw_log, adapter="langchain")
print(result["summary"]["execution_quality"]["status"])  # healthy / degraded / failed
print(result["explanation"]["context_summary"])
```

---

## Use the Debugger

Use this when:
- An agent gives confident answers without data
- Tools return empty results or errors
- Behavior changes between runs and you need to understand why

Choose your entry point:

- **During development** — use Atlas [`watch()`](https://github.com/kiyoshisasano/llm-failure-atlas) to observe live executions and diagnose behavior as it happens
- **After failures** — use `diagnose()` to analyze a raw log or exported trace after the fact

Atlas detects failures; the debugger explains why they happened and proposes fixes. You can use Atlas alone for detection, but diagnosis requires the debugger.

### From a raw log (simplest)

```python
from agent_failure_debugger import diagnose

# Example: LangChain agent trace (no tool data)
raw_log = {
    "steps": [
        {"type": "llm", "output": "The Q4 revenue was $4.2M, up 31% year-over-year."}
    ],
    "tool_calls": [],
}

result = diagnose(raw_log, adapter="langchain")

print(result["summary"])
# → {'root_cause': '...', 'failure_count': ..., 'gate_mode': '...', ...}

print(result["explanation"]["context_summary"])
# → describes what happened and why
```

`raw_log` is a loosely structured dict — its format depends on the source. The adapter normalizes it into the telemetry format Atlas expects. The more structured and complete the log (especially tool calls and outputs), the more accurate the diagnosis. Minimal logs may result in incomplete or degraded analysis.

One function: adapt → detect (via Atlas) → diagnose → explain. Atlas is installed automatically as a dependency. Output quality depends entirely on the input log — incomplete telemetry will silently degrade detection and diagnosis.



**Which adapter to use:**

Adapters normalize raw logs from different sources into Atlas's telemetry format.

| Adapter | Use for |
|---|---|
| `langchain` | LangChain / LangGraph traces |
| `langsmith` | LangSmith run-tree exports |
| `crewai` | CrewAI crew execution logs |
| `redis_help_demo` | [Redis workshop](https://github.com/bhavana-giri/movie-recommender-rag-semantic-cache-workshop) Help Center |

If unsure: use `"langchain"` for agent traces, `"redis_help_demo"` for the Redis workshop demo. For the JSON format each adapter expects, see [Adapter Formats](https://github.com/kiyoshisasano/llm-failure-atlas/blob/main/docs/adapter_formats.md).

Note: `crewai` and `redis_help_demo` adapters do not yet produce `state` or `grounding` telemetry. Some failure patterns (e.g., `agent_tool_call_loop`) may not fire through these adapters. See the [Atlas adapter verification status](https://github.com/kiyoshisasano/llm-failure-atlas#tested-with-real-agents) for details.

**CLI:**

```bash
# From a raw log (full pipeline)
python -m agent_failure_debugger.diagnose log.json --adapter langchain

# From matcher output (diagnosis only)
python -m agent_failure_debugger.main matcher_output.json
```

### From matcher output (direct)

```python
from agent_failure_debugger.pipeline import run_pipeline

result = run_pipeline(
    matcher_output,
    use_learning=True,
    include_explanation=True,
)

print(result["summary"]["root_cause"])
print(result["explanation"]["interpretation"])
print(result["explanation"]["risk"]["level"])
```

Use this when you already have matcher output, or when building a custom adapter.

### From a live agent (via Atlas watch)

Atlas's `watch()` wraps a LangGraph agent and runs the debugger pipeline on completion. It is a separate entry point from `diagnose()` — both produce the same pipeline output but from different starting points: `watch()` captures telemetry from a live execution, while `diagnose()` accepts a raw log after the fact.

If you use [llm-failure-atlas](https://github.com/kiyoshisasano/llm-failure-atlas) for detection, `watch()` runs the debugger automatically:

```python
from llm_failure_atlas.adapters.callback_handler import watch

graph = watch(workflow.compile(), auto_diagnose=True, auto_pipeline=True)
result = graph.invoke({"messages": [...]})
# → detection + debugger pipeline + explanation printed automatically
```

For a copy-paste example without an API key, see [Reproducible Examples](#reproducible-examples) below.

---

## Quick Start

```bash
pip install agent-failure-debugger
```

### From Python (copy-paste-run)

```python
from agent_failure_debugger import diagnose

raw_log = {
    "inputs": {"query": "Change my flight to tomorrow morning"},
    "outputs": {"response": "I've found several hotels near the airport for you."},
    "steps": [
        {"type": "llm", "outputs": {"text": "Let me check available flights."}},
        {"type": "tool", "name": "search_flights", "inputs": {"date": "2025-03-20"},
         "outputs": {"flights": []}, "error": None},
        {"type": "tool", "name": "search_flights", "inputs": {"date": "2025-03-20"},
         "outputs": {"flights": []}, "error": None},
        {"type": "tool", "name": "search_flights", "inputs": {"date": "2025-03-20"},
         "outputs": {"flights": []}, "error": None},
        {"type": "llm", "outputs": {"text": "I've found several hotels near the airport."}}
    ],
    "feedback": {"user_correction": "I asked about flights, not hotels."}
}

result = diagnose(raw_log, adapter="langchain")
print(result["summary"]["root_cause"])
```

### From matcher output (advanced)

If you already have matcher output (e.g., from a custom integration):

```python
from agent_failure_debugger.pipeline import run_pipeline

result = run_pipeline(matcher_output, use_learning=True)
print(result["summary"])
```

See [Quick Start Guide](docs/quickstart.md) for more usage patterns including `watch()` and direct telemetry.

## Common Mistakes

| Problem | Cause | Fix |
|---|---|---|
| "0 failures detected" | Adapter got insufficient data | Provide complete trace with tool calls |
| Wrong results | Input format doesn't match adapter | See [Adapter Formats](https://github.com/kiyoshisasano/llm-failure-atlas/blob/main/docs/adapter_formats.md) |
| Pattern doesn't fire | Adapter doesn't produce required fields | Check [Adapter Coverage](docs/limitations_faq.md#adapter-coverage) |

**⚠ No error is raised for wrong inputs.** The system silently returns zero failures if the adapter cannot extract signals.

## This Tool Cannot

- Verify factual correctness of agent responses
- Detect semantic mismatch (requires embeddings)
- Analyze multi-agent system coordination

See [Limitations & FAQ](docs/limitations_faq.md) for details.

---

## API Details

### Execution quality

Every `diagnose()` and `run_pipeline()` result now includes execution quality assessment in the summary:

```python
eq = result["summary"]["execution_quality"]
print(eq["status"])              # "healthy" | "degraded" | "failed"
print(eq["termination"]["mode"]) # "normal" | "silent_exit" | "error_exit" | "partial_exit" | "unknown"
print(eq["indicators"])          # list of degradation concerns (empty if healthy)
print(eq["summary"])             # one-line human-readable assessment
```

- **healthy** — no significant issues detected
- **degraded** — output may have been produced but quality indicators are weak (low alignment, weak grounding, unmodeled failures)
- **failed** — execution did not produce usable output (silent exit or error)

Execution quality uses existing telemetry and diagnosis results. No new matcher patterns are added.

### Multi-run analysis

```python
from agent_failure_debugger import compare_runs, diff_runs

# Step 1: Is the agent stable across runs?
stability = compare_runs(all_run_results)
print(stability["stability"]["root_cause_agreement"])  # 1.0 = fully stable
print(stability["interpretation"])

# Step 2: What separates success from failure?
diff = diff_runs(success_runs, failure_runs)
print(diff["hypothesis"])
print(diff["failure_set_diff"]["failure_only"])  # patterns only in failures
print(diff["causal_path_diff"])                  # where paths diverge
```

`compare_runs()` measures stability — whether the same task produces consistent diagnoses across runs. `diff_runs()` identifies divergence — what structural differences separate successful runs from failed ones.

### Enhanced explanation

```python
expl = result["explanation"]
print(expl["context_summary"])     # what happened
print(expl["interpretation"])      # why it happened
print(expl["risk"]["level"])       # HIGH / MEDIUM / LOW
print(expl["recommendation"])      # what to do
print(expl["observation"])         # signal coverage info
```

When observation coverage is low (many signals were not observed), the risk level is automatically raised and the interpretation notes that the diagnosis may be incomplete.

CLI: `python -m agent_failure_debugger.explain --enhanced debugger_output.json`

### Individual steps

```python
from agent_failure_debugger.pipeline import run_diagnosis, run_fix

diag = run_diagnosis(matcher_output)
fix_result = run_fix(diag, use_learning=True, top_k=2)
```

### External evaluation

```python
def my_staging_test(bundle):
    fixes = bundle["autofix"]["recommended_fixes"]
    # apply fixes in your staging env
    return {
        "success": True,
        "failure_count": 0,
        "root": None,
        "has_hard_regression": False,
        "notes": "passed staging tests",
    }

result = run_pipeline(
    matcher_output,
    auto_apply=True,
    evaluation_runner=my_staging_test,
)
```

If `evaluation_runner` is not provided, the built-in counterfactual simulation is used. If the runner raises an exception, the pipeline falls back to `staged_review` deterministically.

For real-world interpretation examples — including before/after fix effects — see [Applied Debugging Examples](https://github.com/kiyoshisasano/llm-failure-atlas/blob/main/docs/applied_debugging_examples.md) and [Operational Playbook](https://github.com/kiyoshisasano/llm-failure-atlas/blob/main/docs/operational_playbook.md) in the Atlas repository.

---

## Input Format

A JSON array of failure results from the matcher. Each entry needs `failure_id`, `diagnosed`, and `confidence`:

```json
[
  {
    "failure_id": "premature_model_commitment",
    "diagnosed": true,
    "confidence": 0.7,
    "signals": {
      "ambiguity_without_clarification": true,
      "assumption_persistence_after_correction": true
    }
  }
]
```

The pipeline validates input at entry and rejects malformed data with clear error messages.

---

## Output Format

```json
{
  "root_candidates": ["premature_model_commitment"],
  "root_ranking": [{"id": "premature_model_commitment", "score": 0.85}],
  "failures": [
    {"id": "premature_model_commitment", "confidence": 0.7},
    {"id": "semantic_cache_intent_bleeding", "confidence": 0.7,
     "caused_by": ["premature_model_commitment"]}
  ],
  "causal_paths": [
    ["premature_model_commitment", "semantic_cache_intent_bleeding", "rag_retrieval_drift"]
  ]
}
```

---

## Auto-Apply Gate

| Score | Mode | Behavior |
|---|---|---|
| >= 0.85 | `auto_apply` | Apply, evaluate, keep or rollback |
| 0.65-0.85 | `staged_review` | Write to patches/, await human approval |
| < 0.65 | `proposal_only` | Present fix proposal only |

Hard blockers (force proposal_only regardless of score):
- `safety != "high"`
- `review_required == true`
- `fix_type == "workflow_patch"`
- Execution plan has conflicts or failed validation
- `grounding_gap_not_acknowledged` signal active

## Fix Safety

Fixes are generated from predefined templates, not learned behavior. They are deterministic and reproducible, but not guaranteed to be correct — some fixes may introduce regressions in complex workflows.

Safety mechanisms: the confidence gate prevents low-evidence fixes from auto-apply, hard blockers prevent unsafe categories of changes, the evaluation runner validates fixes before acceptance, and rollback is triggered automatically if evaluation fails.

Always review or evaluate fixes before applying in production environments.

## Automation Guidance

| Environment | Recommended mode | Notes |
|---|---|---|
| Development | `auto_apply` | Iterate quickly, evaluate fixes automatically |
| Staging | `staged_review` | Use evaluation_runner to validate before applying |
| Production | `proposal_only` | Human approval required, avoid auto_apply |

The debugger is designed for assisted decision-making, not fully autonomous system modification.

---

## Pipeline Steps

```
matcher_output.json
  → pipeline.py (orchestrator)
    ├ main.py               causal resolution + root ranking
    ├ abstraction.py        top-k path selection (optional)
    ├ decision_support.py   priority scoring + action plan
    ├ autofix.py            fix selection + patch generation
    ├ auto_apply.py         confidence gate + reason_code
    ├ pipeline_post_apply.py  evaluation runner or counterfactual
    ├ pipeline_summary.py     summary + execution quality assessment
    ├ execution_quality.py    healthy/degraded/failed classification
    └ explainer.py          explanation (context + risk + observation)
```

---

## File Structure

| File | Role |
|---|---|
| `diagnose.py` | Single entry point: raw log → full diagnosis |
| `pipeline.py` | Pipeline orchestrator (from matcher output) |
| `pipeline_post_apply.py` | Post-apply evaluation (runner + counterfactual) |
| `pipeline_summary.py` | Summary generation |
| `main.py` | CLI entry point for diagnosis only (from matcher output) |
| `config.py` | Paths, weights, thresholds |
| `graph_loader.py` | Load failure_graph.yaml |
| `causal_resolver.py` | Normalize, find roots, build paths, rank |
| `formatter.py` | Path scoring + conflict resolution |
| `labels.py` | SIGNAL_MAP (34) + FAILURE_MAP (17) |
| `explainer.py` | Deterministic + optional LLM explanation |
| `explain.py` | CLI for explanation generation (`--enhanced`, `--deterministic`) |
| `decision_support.py` | Failure to action mapping |
| `autofix.py` | Fix selection + patch generation |
| `fix_templates.py` | 17 fix definitions (14 domain + 3 meta) |
| `auto_apply.py` | Confidence gate + auto-apply |
| `execute_fix.py` | Dependency ordering + staged apply |
| `evaluate_fix.py` | Counterfactual simulation |
| `policy_loader.py` | Read-only learning store access |
| `reliability.py` | Cross-run stability and differential analysis |
| `execution_quality.py` | Single-run execution behavior assessment |

---

## Graph Source

The canonical `failure_graph.yaml` is bundled in the `llm-failure-atlas` package. The debugger loads the graph automatically via the Atlas package.

```python
from agent_failure_debugger.config import GRAPH_PATH
print(GRAPH_PATH)  # shows which graph is loaded
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_FAILURE_ATLAS_GRAPH_PATH` | Bundled in package | Override graph location |
| `LLM_FAILURE_ATLAS_PATTERNS_DIR` | Bundled in package | Override patterns directory |
| `LLM_FAILURE_ATLAS_LEARNING_DIR` | Bundled in package | Override learning store |

All scoring weights and gate thresholds are in `config.py`.

---

## Design Principles

- **Deterministic** — same matcher output, same root cause, same fix, same gate decision
- **Graph is for interpretation only** — not used during detection
- **Signal names are contracts** — no redefinition allowed
- **Learning is suggestion-only** — structure is never auto-modified
- **Fail fast on invalid input** — pipeline validates at entry
- **Enhanced explanations** — `include_explanation=True` adds context, interpretation, risk, and recommendation

---

## Related Repositories

| Repository | Role |
|---|---|
| [llm-failure-atlas](https://github.com/kiyoshisasano/llm-failure-atlas) | Failure patterns, causal graph, matcher, adapters |
| [agent-pld-metrics](https://github.com/kiyoshisasano/agent-pld-metrics) | Behavioral stability framework (PLD) |

---

## Reproducible Examples

**Try without an API key** (copy-paste-run):

```bash
pip install agent-failure-debugger[langchain] langgraph
```

```python
from langchain_core.language_models import FakeListLLM
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from llm_failure_atlas.adapters.callback_handler import watch

llm = FakeListLLM(responses=[
    "The revenue was $4.2M in Q3 2024, representing 31% year-over-year "
    "growth. The Asia-Pacific segment contributed 45% of total revenue. "
    "Operating margins expanded to 19.3% across all regions."
])

def agent(state: MessagesState):
    return {"messages": [AIMessage(content=llm.invoke(state["messages"]))]}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent)
workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

graph = watch(workflow.compile(), auto_diagnose=True)
graph.invoke({"messages": [HumanMessage(content="What was Q3 revenue?")]})
```

**Regression test examples:**

10 examples in [llm-failure-atlas](https://github.com/kiyoshisasano/llm-failure-atlas) under `examples/`. Each contains `log.json`, `matcher_output.json`, and `expected_debugger_output.json`.

```bash
python -m agent_failure_debugger.main matcher_output.json
```

---

## Internals

**Root ranking formula:**

```
score = 0.5 * confidence + 0.3 * normalized_downstream + 0.2 * (1 - normalized_depth)
```

More downstream impact ranks higher, even with lower confidence. This reflects causal priority, not detection confidence alone.

This tool implements a single control step within the [PLD](https://github.com/kiyoshisasano/agent-pld-metrics) loop: post-incident causal analysis and intervention decision.

---

## License

MIT License. See [LICENSE](LICENSE).