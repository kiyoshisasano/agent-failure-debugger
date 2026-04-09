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

Call `diagnose()` after every agent run. It returns execution quality (healthy, degraded, or failed), root cause analysis when failures are detected, and fix proposals.

```python
result = diagnose(raw_log, adapter="langchain")
status = result["summary"]["execution_quality"]["status"]

# In CI/CD or automated pipelines:
assert status != "failed", f"Agent execution failed: {result['summary']['root_cause']}"
```

When the agent runs normally, you get `healthy` with confidence scores and grounding state. When something goes wrong, you get the root cause, causal path, and a fix proposal — without changing how you call the tool.

**Three ways to use it:**

- **Failure diagnosis** — an agent broke, you need to know why. `diagnose()` returns root cause, causal path, explanation, and a fix proposal. This is the core use case.
- **Health check** — call `diagnose()` after every run and check `execution_quality.status`. Healthy runs return `healthy`; degraded quality (weak grounding, redundant tool results, low alignment) is surfaced before it becomes a failure. Track degraded frequency over time to catch regressions early.
- **Run comparison** — same prompt produces different results across runs. `compare_runs()` measures stability; `diff_runs()` identifies what structurally separates successful runs from failed ones.

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

### Self-healing agent (LangGraph)

Add automatic failure detection and informed retry to any LangGraph agent. When the health check detects a retryable failure, it injects the diagnosis into the conversation — the LLM reads *why* it failed and adjusts its approach. This is not a blind retry.

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

On retry, the health check appends a message like: *"Previous attempt status: failed. The tool may have experienced a transient error that has since resolved. Please call the tool again."* — the LLM reads this and retries the tool.

Not all failures benefit from retry. The integration classifies all 17 Atlas patterns as either retryable (transient errors, LLM non-determinism) or structural (bad prompts, config issues). Structural failures are reported immediately without wasting retries. See [examples/self_healing/](examples/self_healing/) for a working demo validated across GPT, Claude, and Gemini.

**CI integration:** Use [pytest-agent-health](https://github.com/kiyoshisasano/pytest-agent-health) to catch failures and regressions in CI, and `create_health_check()` to recover in production. The pytest plugin automatically compares against previous CI runs to detect new failure patterns and status degradation.

### Behavioral Audit

The diagnostic pipeline can produce a formal audit report: run controlled scenarios across multiple LLM providers, diagnose each trace, and generate a PDF with pass/fail verdict, priority-ranked findings, and remediation owners.

Sample findings from a customer service agent audit (GPT-4o-mini, Claude Haiku 4.5, Gemini 2.5 Flash × 6 scenarios):

| Priority | Finding | Owner |
|---|---|---|
| P0 — CRITICAL | Agent fabricates refund information when backend service is down (`tool_provided_data = False`, all 3 models) | Backend / Integration |
| P0 — CRITICAL | Cancellation request ignored — agent upsells instead (`incorrect_output`, conf 0.7, 100% cross-model agreement) | Prompt Design |
| P1 — HIGH | Wrong product category forwarded to customer without notice (not detected — known coverage gap) | Retrieval / Adapter |
| P2 — MEDIUM | Tool retry loop without strategy change (models self-limited before detection threshold) | Agent Infrastructure |

Audit verdict: **FAIL** — critical findings in user-facing flows, adjusted healthy rate 44%.

→ [Full sample report (PDF)](examples/sample_audit_report.pdf)
→ Audit toolkit and CI integration: [pytest-agent-health](https://github.com/kiyoshisasano/pytest-agent-health)

---

## Quick Start

```bash
pip install agent-failure-debugger
```

### Healthy run

```python
from agent_failure_debugger import diagnose

raw_log = {
    "inputs": {"query": "What was Q3 revenue?"},
    "outputs": {"response": "Q3 revenue was $4.2M based on the latest earnings report."},
    "steps": [
        {"type": "tool", "name": "search_earnings", "inputs": {"quarter": "Q3"},
         "outputs": {"revenue": "$4.2M", "source": "10-Q filing"}, "error": None},
        {"type": "llm", "outputs": {"text": "Q3 revenue was $4.2M based on the latest earnings report."}}
    ]
}

result = diagnose(raw_log, adapter="langchain")
print(result["summary"]["execution_quality"]["status"])  # healthy
print(result["summary"]["failure_count"])                 # 0
```

The tool returns a result on every run. When the agent is healthy, you get confirmation — not silence.

### Degraded run

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
print(result["summary"]["root_cause"])                    # incorrect_output
print(result["summary"]["execution_quality"]["status"])   # degraded
print(result["explanation"]["context_summary"])
# → "Root cause identified: the system produced output misaligned with
#    user intent, requiring correction (confidence: 0.625)."
print(result["explanation"]["risk"]["level"])              # low
print(result["summary"]["fix_count"])                     # 1
```

Same function, same interface. The difference is in the input, not in how you call the tool.

### From matcher output (advanced)

If you already have matcher output (e.g., from a custom integration):

```python
from agent_failure_debugger.pipeline import run_pipeline

result = run_pipeline(matcher_output, use_learning=True)
print(result["summary"])
```

See [Quick Start Guide](docs/quickstart.md) for more usage patterns including `watch()`, multi-run analysis, and direct telemetry.

## Common Mistakes

**⚠ No error is raised for wrong inputs.** The system silently returns zero failures if the adapter cannot extract signals. See [Limitations & FAQ](docs/limitations_faq.md) for common causes and solutions.

## This Tool Cannot

- Verify factual correctness of agent responses
- Detect semantic mismatch (requires embeddings)
- Analyze multi-agent system coordination

See [Limitations & FAQ](docs/limitations_faq.md) for details.

---

## API Details

### Execution quality

Every `diagnose()` result includes execution quality: `healthy`, `degraded`, or `failed`. Degradation indicators (low alignment, weak grounding, redundant tool results) are surfaced before they become failures.

```python
eq = result["summary"]["execution_quality"]
print(eq["status"])       # "healthy" | "degraded" | "failed"
print(eq["indicators"])   # list of degradation concerns (empty if healthy)
```

### Multi-run analysis

```python
from agent_failure_debugger import compare_runs, diff_runs

stability = compare_runs(all_run_results)     # Is the agent stable?
diff = diff_runs(success_runs, failure_runs)   # What separates success from failure?
```

For runnable examples, see [examples/multi_run_stability](examples/multi_run_stability/) and [examples/termination_divergence](examples/termination_divergence/).

### More

Full API documentation including enhanced explanation, individual pipeline steps, external evaluation, direct telemetry, and common mistakes: [Quick Start Guide](docs/quickstart.md).

Input/output format, auto-apply gate, fix safety, and automation guidance: [API Reference](docs/reference.md).

For real-world interpretation examples: [Applied Debugging Examples](https://github.com/kiyoshisasano/llm-failure-atlas/blob/main/docs/applied_debugging_examples.md) and [Operational Playbook](https://github.com/kiyoshisasano/llm-failure-atlas/blob/main/docs/operational_playbook.md) in the Atlas repository.

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
| `integrations/langgraph.py` | LangGraph self-healing health check node |

### Examples

| Directory | Demonstrates |
|---|---|
| `examples/self_healing/` | `create_health_check()`: LangGraph self-healing with informed retry across 3 models |
| `examples/termination_divergence/` | `diff_runs()`: same root cause, different termination modes |
| `examples/multi_run_stability/` | `compare_runs()` → `diff_runs()`: two-step stability and divergence workflow |

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
| [pytest-agent-health](https://github.com/kiyoshisasano/pytest-agent-health) | CI integration — catch silent agent failures in pytest |
| [agent-pld-metrics](https://github.com/kiyoshisasano/agent-pld-metrics) | Behavioral stability framework (PLD) |

---

## Reproducible Examples

For copy-paste-run examples of healthy and degraded runs, see [Quick Start](#quick-start) above.

**With a live agent** (requires `langchain-core` and `langgraph`):

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

Note: `watch()` with `FakeListLLM` demonstrates the callback integration but may not trigger failure patterns — the fake LLM produces no tool calls or user corrections. For failure detection examples, use `diagnose()` with the raw log above.

**Regression test examples:**

12 examples in [llm-failure-atlas](https://github.com/kiyoshisasano/llm-failure-atlas) under `examples/` (10 agent + 2 non-LLM). Each contains `log.json`, `matcher_output.json`, and `expected_debugger_output.json`.

```bash
python -m agent_failure_debugger.main matcher_output.json
```

**Multi-run analysis examples:**

2 examples in this repository under `examples/`. Each contains input fixtures, a runnable script, and `expected_output.json`:

- [termination_divergence](examples/termination_divergence/) — `diff_runs()` comparing silent exit vs error exit
- [multi_run_stability](examples/multi_run_stability/) — `compare_runs()` → `diff_runs()` two-step workflow

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