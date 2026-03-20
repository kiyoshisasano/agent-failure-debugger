# agent-failure-debugger

A deterministic pipeline that diagnoses, explains, and fixes failures in LLM-based agent systems.

```
Detection tells you WHAT failed.
This tool tells you WHY — through which causal path, starting from which root.
And then fixes it.
```

---

## Related Repositories

| Repository | Role |
|---|---|
| [llm-failure-atlas](https://github.com/kiyoshisasano/llm-failure-atlas) | Failure pattern definitions, causal graph, matcher, evaluation, KPI |
| [agent-pld-metrics (PLD)](https://github.com/kiyoshisasano/agent-pld-metrics) | Behavioral stability framework this tool applies to |

---

## What It Does

**Input:** Matcher output (detected failures with confidence scores) + causal graph

**Output:**
- Root cause identification with causal ranking
- Full causal path reconstruction
- Deterministic fix generation with safety classification
- Confidence-gated auto-apply with rollback
- Learning-aware priority adjustment

---

## Quickstart

```bash
git clone https://github.com/kiyoshisasano/agent-failure-debugger.git
cd agent-failure-debugger
pip install -r requirements.txt
```

Run with sample data:

```bash
# Diagnosis only
python main.py ../llm-failure-atlas/examples/simple/matcher_output.json

# Full pipeline (recommended)
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

```python
from pipeline import run_pipeline
import json

with open("matcher_output.json") as f:
    matcher_output = json.load(f)

result = run_pipeline(
    matcher_output,
    use_learning=True,   # adjust priority using learning data
    top_k=1,             # number of fixes to generate
    auto_apply=False,    # set True to auto-apply safe fixes
)

print(result["summary"]["root_cause"])   # "premature_model_commitment"
print(result["summary"]["gate_mode"])    # "auto_apply"
```

Individual steps are also available:

```python
from pipeline import run_diagnosis, run_fix

diag = run_diagnosis(matcher_output)
fix_result = run_fix(diag, use_learning=True, top_k=2)
```

---

## Pipeline

```
matcher_output.json   (produced by llm-failure-atlas matcher)
  → main.py             causal resolution + root ranking
  → abstraction.py      top-k path selection + clustering
  → explainer.py        deterministic draft + optional LLM smoothing
  → decision_support.py priority scoring + action plan
  → autofix.py          fix selection + patch generation
  → auto_apply.py       confidence gate → apply / review / proposal
  → execute_fix.py      dependency ordering + staged apply
  → evaluate_fix.py     before/after simulation + regression detection
```

---

## Prerequisite: Matcher

This tool expects matcher output as input. The matcher converts logs into detected failures:

```
log → signals → failure detection (matcher)
```

Pattern definitions are maintained in [llm-failure-atlas](https://github.com/kiyoshisasano/llm-failure-atlas) under `failures/*.yaml`. Pre-generated `matcher_output.json` files are available in each `examples/` directory for immediate use.

---

## Input Format

Matcher output: a JSON array of failure results. Each entry must include `failure_id`, `diagnosed`, and `confidence`.

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

Failures with `"diagnosed": false` are silently excluded.

---

## Output Format

```json
{
  "root_candidates": ["premature_model_commitment"],
  "root_ranking": [{"id": "premature_model_commitment", "score": 0.85}],
  "failures": [
    {"id": "premature_model_commitment", "confidence": 0.7},
    {"id": "semantic_cache_intent_bleeding", "confidence": 0.7,
     "caused_by": ["premature_model_commitment"]},
    {"id": "rag_retrieval_drift", "confidence": 0.6,
     "caused_by": ["semantic_cache_intent_bleeding"]}
  ],
  "causal_paths": [
    ["premature_model_commitment", "semantic_cache_intent_bleeding", "rag_retrieval_drift"]
  ],
  "explanation": "..."
}
```

---

## Root Ranking

```
score = 0.5 × confidence + 0.3 × normalized_downstream + 0.2 × (1 - normalized_depth)
```

A failure with more downstream impact ranks higher, even if its confidence is lower. This reflects causal priority, not detection confidence alone.

---

## Auto-Apply Gate

Fix application is controlled by a deterministic confidence gate:

| Score | Mode | Behavior |
|---|---|---|
| ≥ 0.85 | `auto_apply` | Apply → evaluate → keep or rollback |
| 0.65–0.85 | `staged_review` | Write to patches/, await human approval |
| < 0.65 | `proposal_only` | Present fix proposal only |

Hard blockers (override score, force proposal_only):
- `safety != "high"`
- `review_required == true`
- `fix_type == "workflow_patch"`
- Execution plan has conflicts or failed validation

---

## File Structure

**Core pipeline:**

| File | Responsibility |
|---|---|
| `pipeline.py` | API entry point (recommended) |
| `main.py` | CLI entry point (diagnosis only) |
| `config.py` | Centralized paths, weights, thresholds |
| `graph_loader.py` | Load failure_graph.yaml, exclude planned nodes |
| `causal_resolver.py` | normalize → roots → paths → ranking |
| `formatter.py` | Path scoring + conflict resolution + evidence |
| `labels.py` | SIGNAL_MAP (22 entries) + FAILURE_MAP (12 entries) |
| `abstraction.py` | Top-k path selection + clustering |
| `explainer.py` | Deterministic draft + optional LLM smoothing |
| `decision_support.py` | Failure → action mapping + priority scoring |
| `autofix.py` | Fix selection + patch generation |
| `fix_templates.py` | 12 failure × fix definitions |
| `execute_fix.py` | Dependency ordering + staged apply + rollback |
| `evaluate_fix.py` | Counterfactual simulation + regression detection |
| `auto_apply.py` | Confidence gate + auto-apply + rollback |
| `policy_loader.py` | Read-only access to learning stores |

**CLI wrappers:**

| File | Wraps |
|---|---|
| `explain.py` | `explainer.py` |
| `summarize.py` | `abstraction.py` |
| `advise.py` | `decision_support.py` |
| `apply_fix.py` | `execute_fix.py` (dry-run display) |

**Data:**

| File | Note |
|---|---|
| `failure_graph.yaml` | Causal graph (canonical source is Atlas; see below) |
| `templates/` | Prompt templates for LLM-based explanation |

---

## Graph Sync

`failure_graph.yaml` exists in both repositories. The **canonical source** is always `llm-failure-atlas/failure_graph.yaml`. The copy in this repository must be kept in sync manually.

To verify sync:

```bash
diff failure_graph.yaml ../llm-failure-atlas/failure_graph.yaml
```

If you set `ATLAS_ROOT`, `config.py` can resolve the Atlas graph path directly.

---

## Configuration

Environment variables override defaults:

| Variable | Default | Description |
|---|---|---|
| `ATLAS_ROOT` | `../llm-failure-atlas` | Path to Atlas repository |
| `DEBUGGER_ROOT` | `.` (this repository) | Path to this repository |
| `ATLAS_LEARNING_DIR` | `$ATLAS_ROOT/learning` | Learning store location |

All settings (scoring weights, gate thresholds, KPI targets) are centralized in `config.py`.

---

## Design Principles

- **Graph is not used for diagnosis** — only for causal interpretation
- **Signal names are system-wide contracts** — no redefinition allowed
- **Adding a failure to the Atlas requires no changes to this tool**
- **Learning is suggestion-only** — patterns, graph, and templates are never auto-modified
- **Auto-apply safety hierarchy:** high → auto candidate, medium → review, low → excluded

---

## Relationship to Atlas

This tool depends on [LLM Failure Atlas](https://github.com/kiyoshisasano/llm-failure-atlas):

- `failure_graph.yaml` is sourced from the Atlas
- Node and edge definitions are maintained there
- This tool does not define failures itself

---

## Relationship to PLD

This tool is a concrete application of [Phase Loop Dynamics (PLD)](https://github.com/kiyoshisasano/agent-pld-metrics):

| PLD Phase | Debugger Output |
|---|---|
| Drift | Initiating failure in `root_candidates` |
| Propagation | Downstream failures in `causal_paths` |
| Repair | Fix generation + auto-apply gate |
| Outcome | Leaf node / evaluation decision (keep/rollback) |

PLD provides the behavioral stability framework. This tool provides the causal diagnosis and remediation layer that operates within it.

---

## Reproducible Examples

10 examples are maintained in [llm-failure-atlas](https://github.com/kiyoshisasano/llm-failure-atlas) under `examples/`. Each contains `log.json`, `matcher_output.json`, and `expected_debugger_output.json`.

Run and compare:

```bash
python main.py ../llm-failure-atlas/examples/simple/matcher_output.json
```

Output should match `expected_debugger_output.json` exactly.

---

## License

MIT License. See [LICENSE](LICENSE).
