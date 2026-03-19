# agent-failure-debugger

agent-failure-debugger is a CLI tool that explains **WHY** failures happen.

It converts matcher output into causal explanations using the LLM Failure Atlas graph.

---

## Related Repositories

| Repository | Role |
|---|---|
| [llm-failure-atlas](https://github.com/kiyoshisasano/llm-failure-atlas) | Provides `failure_graph.yaml` and failure pattern definitions |
| [agent-pld-metrics (PLD)](https://github.com/kiyoshisasano/agent-pld-metrics) | Behavioral stability framework this tool applies to |

---

## What It Does

**Input:**
- Matcher output (detected failures with confidence scores)
- Failure graph (from LLM Failure Atlas)

**Output:**
- Root cause candidates
- Root ranking (scored by causal impact)
- Causal relationships
- Causal paths
- Natural language explanation

---

## Core Purpose

```
Detection tells you WHAT failed.
Debugger tells you WHY it failed.
```

---

## Pipeline

```
matcher output
      +
failure_graph.yaml
      ↓
graph_loader     → load graph, exclude planned nodes
causal_resolver  → normalize, find roots, extract paths, rank
formatter        → build explanation from top-ranked path
      ↓
causal explanation
```

![Failure Diagnosis Pipeline](./docs/pipeline.svg)

---

## Prerequisite: Matcher

This tool expects matcher output as input.

The matcher is responsible for:

```
log → signals → failure detection
```

This repository does not include the matcher.  
See [llm-failure-atlas](https://github.com/kiyoshisasano/llm-failure-atlas) for pattern definitions and a reference matcher.

---

## Usage

```bash
python main.py input.json failure_graph.yaml
```

Defaults (if no arguments given):

```
input.json
failure_graph.yaml
```

---

## Input Format

Matcher output: a JSON array of failure results.  
Each entry must include `failure_id`, `diagnosed`, and `confidence`.

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
  },
  {
    "failure_id": "semantic_cache_intent_bleeding",
    "diagnosed": true,
    "confidence": 0.9,
    "signals": {
      "cache_query_intent_mismatch": true,
      "retrieval_skipped_after_cache_hit": true,
      "retrieved_docs_low_intent_alignment": true
    }
  },
  {
    "failure_id": "rag_retrieval_drift",
    "diagnosed": true,
    "confidence": 0.6,
    "signals": {
      "retrieval_skipped_after_cache_hit": true,
      "retrieved_docs_low_intent_alignment": true
    }
  }
]
```

Failures with `"diagnosed": false` are silently excluded.

---

## Output Format

```json
{
  "root_candidates": [
    "premature_model_commitment"
  ],
  "root_ranking": [
    {
      "id": "premature_model_commitment",
      "score": 0.85
    }
  ],
  "failures": [
    {
      "id": "premature_model_commitment",
      "confidence": 0.7
    },
    {
      "id": "semantic_cache_intent_bleeding",
      "confidence": 0.9,
      "caused_by": ["premature_model_commitment"]
    },
    {
      "id": "rag_retrieval_drift",
      "confidence": 0.6,
      "caused_by": ["semantic_cache_intent_bleeding"]
    }
  ],
  "causal_links": [
    {
      "from": "premature_model_commitment",
      "to": "semantic_cache_intent_bleeding",
      "relation": "predisposes"
    },
    {
      "from": "semantic_cache_intent_bleeding",
      "to": "rag_retrieval_drift",
      "relation": "induces"
    }
  ],
  "causal_paths": [
    [
      "premature_model_commitment",
      "semantic_cache_intent_bleeding",
      "rag_retrieval_drift"
    ]
  ],
  "explanation": "Causal path detected: premature_model_commitment predisposed semantic_cache_intent_bleeding, which induced rag_retrieval_drift"
}
```

### Output fields

| Field | Description |
|---|---|
| `root_candidates` | Active failures with no active upstream in the graph |
| `root_ranking` | Root candidates scored by `confidence + downstream impact` |
| `failures` | Enriched failure list with `caused_by` where applicable |
| `causal_links` | Active edges between diagnosed failures |
| `causal_paths` | Full paths from root to leaf through active failures |
| `explanation` | Natural language narrative from top-ranked root path |

---

## Root Ranking

Score formula:

```
score = 0.5 × confidence + 0.3 × normalized_downstream + 0.2 × (1 - normalized_depth)
```

A failure with more downstream impact ranks higher even if its confidence is lower.  
This reflects causal priority, not detection confidence alone.

---

## Reproducible Example

`examples/simple/` contains a complete pipeline trace:

```
examples/simple/
├── log.json                      # input telemetry (all 3 failures diagnosable)
├── matcher_output.json           # what the matcher produces from log.json
└── expected_debugger_output.json # what this tool produces from matcher_output.json
```

Run and compare:

```bash
python main.py examples/simple/matcher_output.json failure_graph.yaml
```

Output should match `examples/simple/expected_debugger_output.json` exactly.

---

## Design Principles

* **No inference beyond graph** — the graph is the sole source of causal structure
* **Graph separation** — `failure_graph.yaml` defines structure; patterns define detection
* **Ranking guides narrative** — explanation starts from the highest-impact root, not the longest path alone
* **Planned nodes are excluded** — `status: planned` nodes in the graph are not used at runtime

---

## File Structure

```
.
├── main.py              # CLI entry point
├── graph_loader.py      # loads graph, excludes planned nodes, builds adjacency
├── causal_resolver.py   # normalize → roots → links → paths → ranking
├── formatter.py         # builds explanation from top-ranked path
├── failure_graph.yaml   # local copy of Atlas graph
├── input.json           # example matcher output (3 failures)
└── examples/
    └── simple/
        ├── log.json
        ├── matcher_output.json
        └── expected_debugger_output.json
```

---

## Relationship to Atlas

This tool depends on [LLM Failure Atlas](https://github.com/kiyoshisasano/llm-failure-atlas):

* `failure_graph.yaml` is sourced from the Atlas
* Node and edge definitions are maintained there
* This tool does not define failures itself

Adding a new failure to the Atlas requires no changes to this tool,  
as long as `failure_graph.yaml` is updated.

---

## Relationship to PLD

Maps directly to [PLD](https://github.com/kiyoshisasano/agent-pld-metrics) phases:

| PLD Phase | Debugger Output |
|---|---|
| Drift | initiating failure in `root_candidates` |
| Propagation | downstream failures in `causal_paths` |
| Outcome | leaf node in the causal chain |

---

## Summary

```
Detection  →  WHAT failed
Debugger   →  WHY it failed, through which causal path, starting from which root
```
