# agent-failure-debugger

Diagnoses *why* your LLM agent failed, not just *what* failed. Deterministic causal analysis with fix generation.

```
matcher output → root cause → causal path → fix → auto-apply gate
```

---

## Minimal Example

No API key needed. Copy, paste, run:

```python
from langchain_core.language_models import FakeListLLM
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from adapters.callback_handler import watch

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

This simulates an agent that answers a factual question without any data source. Atlas observes the execution and reports:

```
Grounding:  tool_provided_data=False  uncertainty_acknowledged=False
```

The agent produced a confident, detailed answer with no tool data and did not disclose the gap. This would be classified as a risk case — see [Operational Playbook](https://github.com/kiyoshisasano/llm-failure-atlas/blob/main/docs/operational_playbook.md) for how to handle it.

Note: This output depends on runtime signals captured by the observer (telemetry), not static rules alone.

---

## Quick Start

```bash
git clone https://github.com/kiyoshisasano/agent-failure-debugger.git
cd agent-failure-debugger
pip install -r requirements.txt

# Run with sample data
python pipeline.py ../llm-failure-atlas/examples/simple/matcher_output.json --use-learning
```

Output:

```
=== PIPELINE RESULT ===
  Root cause:  premature_model_commitment (confidence: 0.85)
  Failures:    3
  Fixes:       1
  Gate:        auto_apply (score: 0.9218)
  Applied:     no
```

---

## Use as an API

### Full pipeline

```python
from pipeline import run_pipeline

result = run_pipeline(
    matcher_output,
    use_learning=True,
    include_explanation=True,
)

print(result["summary"]["root_cause"])
print(result["summary"]["gate_mode"])
```

### Enhanced explanation

```python
expl = result["explanation"]
print(expl["context_summary"])     # what happened
print(expl["interpretation"])      # why it happened
print(expl["risk"]["level"])       # HIGH / MEDIUM / LOW
print(expl["recommendation"])      # what to do
```

CLI: `python explain.py --enhanced debugger_output.json`

### Individual steps

```python
from pipeline import run_diagnosis, run_fix

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

For real-world interpretation examples, see [Applied Debugging Examples](https://github.com/kiyoshisasano/llm-failure-atlas/blob/main/docs/applied_debugging_examples.md) and [Operational Playbook](https://github.com/kiyoshisasano/llm-failure-atlas/blob/main/docs/operational_playbook.md) in the Atlas repository.

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

## Root Ranking

```
score = 0.5 * confidence + 0.3 * normalized_downstream + 0.2 * (1 - normalized_depth)
```

More downstream impact ranks higher, even with lower confidence. This reflects causal priority.

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

---

## Pipeline Steps

```
matcher_output.json
  → main.py             causal resolution + root ranking
  → abstraction.py      top-k path selection
  → explainer.py        deterministic draft + optional LLM smoothing
  → decision_support.py priority scoring + action plan
  → autofix.py          fix selection + patch generation
  → auto_apply.py       confidence gate
  → execute_fix.py      dependency ordering + staged apply
  → evaluate_fix.py     before/after simulation + regression detection
```

---

## File Structure

| File | Role |
|---|---|
| `pipeline.py` | API entry point (recommended) |
| `main.py` | CLI entry point (diagnosis only) |
| `config.py` | Paths, weights, thresholds |
| `graph_loader.py` | Load failure_graph.yaml |
| `causal_resolver.py` | Normalize, find roots, build paths, rank |
| `formatter.py` | Path scoring + conflict resolution |
| `labels.py` | SIGNAL_MAP (30) + FAILURE_MAP (15) |
| `explainer.py` | Deterministic + optional LLM explanation |
| `decision_support.py` | Failure to action mapping |
| `autofix.py` | Fix selection + patch generation |
| `fix_templates.py` | 15 fix definitions (12 domain + 3 meta) |
| `auto_apply.py` | Confidence gate + auto-apply |
| `execute_fix.py` | Dependency ordering + staged apply |
| `evaluate_fix.py` | Counterfactual simulation |
| `policy_loader.py` | Read-only learning store access |

---

## Graph Source

The canonical `failure_graph.yaml` is in [llm-failure-atlas](https://github.com/kiyoshisasano/llm-failure-atlas). When the Atlas repository is available as a sibling directory (or `ATLAS_ROOT` is set), the debugger loads the graph from Atlas directly. The local copy is a fallback.

```python
from config import GRAPH_PATH
print(GRAPH_PATH)  # shows which graph is loaded
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ATLAS_ROOT` | `../llm-failure-atlas` | Path to Atlas repository |
| `DEBUGGER_ROOT` | `.` | Path to this repository |
| `ATLAS_LEARNING_DIR` | `$ATLAS_ROOT/learning` | Learning store location |

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

This tool implements a single control step within the [PLD](https://github.com/kiyoshisasano/agent-pld-metrics) loop: post-incident causal analysis and intervention decision.

---

## Reproducible Examples

10 examples in [llm-failure-atlas](https://github.com/kiyoshisasano/llm-failure-atlas) under `examples/`. Each contains `log.json`, `matcher_output.json`, and `expected_debugger_output.json`.

```bash
python main.py ../llm-failure-atlas/examples/simple/matcher_output.json
```

---

## License

MIT License. See [LICENSE](LICENSE).