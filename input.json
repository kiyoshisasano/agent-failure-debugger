[
  {
    "failure_id": "premature_model_commitment",
    "diagnosed": true,
    "confidence": 0.7,
    "threshold": 0.6,
    "signals": {
      "ambiguity_without_clarification": true,
      "assumption_persistence_after_correction": true
    },
    "applied_modifiers": [
      {
        "type": "evidence",
        "signal": "ambiguity_without_clarification",
        "add": 0.3
      },
      {
        "type": "evidence",
        "signal": "assumption_persistence_after_correction",
        "add": 0.4
      }
    ]
  },
  {
    "failure_id": "semantic_cache_intent_bleeding",
    "diagnosed": true,
    "confidence": 0.9,
    "threshold": 0.6,
    "signals": {
      "cache_query_intent_mismatch": true,
      "retrieval_skipped_after_cache_hit": true,
      "retrieved_docs_low_intent_alignment": true
    },
    "applied_modifiers": [
      {
        "type": "evidence",
        "signal": "cache_query_intent_mismatch",
        "add": 0.4
      },
      {
        "type": "evidence",
        "signal": "retrieval_skipped_after_cache_hit",
        "add": 0.3
      },
      {
        "type": "symptom",
        "signal": "retrieved_docs_low_intent_alignment",
        "add": 0.2
      }
    ]
  },
  {
    "failure_id": "rag_retrieval_drift",
    "diagnosed": true,
    "confidence": 0.6,
    "threshold": 0.5,
    "signals": {
      "retrieved_docs_low_intent_alignment": true,
      "retrieval_skipped_after_cache_hit": true
    },
    "applied_modifiers": [
      {
        "type": "evidence",
        "signal": "retrieval_skipped_after_cache_hit",
        "add": 0.4
      },
      {
        "type": "symptom",
        "signal": "retrieved_docs_low_intent_alignment",
        "add": 0.2
      }
    ]
  }
]
