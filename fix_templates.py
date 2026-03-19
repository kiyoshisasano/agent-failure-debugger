"""
fix_templates.py

Phase 16: Deterministic fix templates for each failure.

Each entry maps a failure_id to:
  - fix_type:  prompt_patch | config_patch | guard_patch | workflow_patch
  - target:    the system component to modify
  - patch:     the specific change
  - safety:    high (auto-safe) | medium (needs review) | low (excluded from autofix)
"""

AUTOFIX_MAP = {

    # --- Reasoning layer ---

    "clarification_failure": {
        "fix_type": "prompt_patch",
        "target": "system_prompt",
        "patch": {
            "insert": "before_reasoning",
            "content": (
                "If the user request is ambiguous, ask one clarifying question "
                "before proceeding with an answer."
            )
        },
        "safety": "high"
    },

    "assumption_invalidation_failure": {
        "fix_type": "prompt_patch",
        "target": "reasoning_policy",
        "patch": {
            "insert": "after_observation",
            "content": (
                "If new evidence contradicts your current assumption, "
                "discard it and re-evaluate from scratch."
            )
        },
        "safety": "medium"
    },

    "premature_model_commitment": {
        "fix_type": "prompt_patch",
        "target": "reasoning_policy",
        "patch": {
            "insert": "before_commitment",
            "content": (
                "Generate at least two candidate hypotheses before committing "
                "to a single interpretation."
            )
        },
        "safety": "high"
    },

    "repair_strategy_failure": {
        "fix_type": "workflow_patch",
        "target": "repair_controller",
        "patch": {
            "strategy": "regenerate",
            "disable_patch_mode": True
        },
        "safety": "low"
    },

    # --- Retrieval layer ---

    "semantic_cache_intent_bleeding": {
        "fix_type": "config_patch",
        "target": "cache_config",
        "patch": {
            "intent_similarity_threshold": 0.55,
            "fallback_to_retrieval_on_mismatch": True
        },
        "safety": "high"
    },

    "prompt_injection_via_retrieval": {
        "fix_type": "guard_patch",
        "target": "retrieval_filter",
        "patch": {
            "enable_instruction_filter": True,
            "block_override_patterns": True,
            "strip_executable_instructions": True
        },
        "safety": "medium"
    },

    "context_truncation_loss": {
        "fix_type": "config_patch",
        "target": "context_manager",
        "patch": {
            "priority_truncation": True,
            "preserve_system_and_user_intent": True,
            "max_context_tokens": 8192
        },
        "safety": "medium"
    },

    "rag_retrieval_drift": {
        "fix_type": "guard_patch",
        "target": "retrieval_validator",
        "patch": {
            "enable_alignment_check": True,
            "min_intent_alignment": 0.7,
            "reject_low_relevance_docs": True
        },
        "safety": "high"
    },

    # --- Tool layer ---

    "agent_tool_call_loop": {
        "fix_type": "workflow_patch",
        "target": "tool_controller",
        "patch": {
            "max_repeat_calls": 3,
            "require_progress_between_calls": True
        },
        "safety": "high"
    },

    "tool_result_misinterpretation": {
        "fix_type": "guard_patch",
        "target": "tool_output_validator",
        "patch": {
            "enable_schema_validation": True,
            "require_explicit_parsing": True
        },
        "safety": "high"
    },

    # --- Instruction layer ---

    "instruction_priority_inversion": {
        "fix_type": "guard_patch",
        "target": "instruction_resolver",
        "patch": {
            "enforce_priority_order": [
                "system",
                "developer",
                "user",
                "retrieved"
            ]
        },
        "safety": "high"
    },

    # --- Output layer ---

    "incorrect_output": {
        "fix_type": "workflow_patch",
        "target": "output_validator",
        "patch": {
            "enable_self_check": True,
            "require_intent_alignment": True
        },
        "safety": "high"
    },
}
