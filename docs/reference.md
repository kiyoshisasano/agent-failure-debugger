# API Reference

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