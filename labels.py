"""
labels.py

Deterministic label maps for failures and signals.
These provide stable, human-readable descriptions that do not
depend on LLM generation.

Rules:
  - Every failure in the graph must have an entry in FAILURE_MAP.
  - Every signal in any pattern must have an entry in SIGNAL_MAP.
  - Descriptions are factual, concise, and non-speculative.
"""

FAILURE_MAP = {
    "clarification_failure":
        "the system failed to request clarification despite ambiguous input",

    "assumption_invalidation_failure":
        "the system failed to abandon an invalidated hypothesis despite contradicting evidence",

    "premature_model_commitment":
        "the system committed to a single interpretation too early without replanning",

    "semantic_cache_intent_bleeding":
        "the semantic cache returned a result whose intent did not match the current query",

    "rag_retrieval_drift":
        "retrieval behavior degraded, returning documents misaligned with user intent",

    "instruction_priority_inversion":
        "the system failed to maintain correct priority between system and external instructions",

    "prompt_injection_via_retrieval":
        "retrieved content introduced adversarial signals that overrode the intended task",

    "context_truncation_loss":
        "critical information was lost due to context window truncation",

    "agent_tool_call_loop":
        "the agent repeatedly invoked tools without making meaningful state progress",

    "tool_result_misinterpretation":
        "tool outputs were interpreted incorrectly despite correct tool execution",

    "repair_strategy_failure":
        "the system patched output instead of regenerating from corrected assumptions",

    "incorrect_output":
        "the system produced output misaligned with user intent, requiring correction",

    # --- Meta failures (model limitations) ---

    "unmodeled_failure":
        "symptoms are present but no defined failure pattern explains them",

    "insufficient_observability":
        "telemetry is too incomplete for reliable diagnosis",

    "conflicting_signals":
        "observed signals point in contradictory directions, reducing diagnostic confidence",
}

SIGNAL_MAP = {
    # clarification_failure
    "ambiguity_detected_without_clarification":
        "ambiguity was detected but no clarification was requested",

    # clarification_failure, assumption_invalidation_failure
    "no_hypothesis_branching":
        "the system committed to a single hypothesis without branching alternatives",

    # assumption_invalidation_failure
    "contradiction_detected_but_ignored":
        "contradicting evidence was detected but the hypothesis was retained",

    # premature_model_commitment
    "ambiguity_without_clarification":
        "ambiguous input was collapsed without requesting clarification",

    "assumption_persistence_after_correction":
        "the initial assumption persisted even after user correction",

    # semantic_cache_intent_bleeding
    "cache_query_intent_mismatch":
        "the cached query intent did not match the current query intent",

    # semantic_cache_intent_bleeding, rag_retrieval_drift
    "retrieval_skipped_after_cache_hit":
        "retrieval was skipped because of a high-similarity cache hit",

    "retrieved_docs_low_intent_alignment":
        "the retrieved documents had low alignment with user intent",

    # instruction_priority_inversion
    "system_instruction_overridden":
        "the system instruction priority was not maintained",

    "external_instruction_dominant":
        "external instructions dominated over system instructions",

    # prompt_injection_via_retrieval
    "retrieved_context_instruction_override":
        "retrieved content overrode task or system instructions",

    "retrieved_context_adversarial_pattern":
        "an adversarial pattern was detected in retrieved content",

    # context_truncation_loss
    "context_truncated_critical_info":
        "context window truncation removed critical information",

    "retrieval_missing_expected_content":
        "retrieved content covered less than expected",

    # agent_tool_call_loop
    "repeated_tool_call_without_progress":
        "the same tool was invoked repeatedly without meaningful progress",

    "no_replanning_before_repeat":
        "tool calls were repeated without replanning",

    # tool_result_misinterpretation
    "tool_output_misaligned_with_state":
        "the tool executed correctly but state was not updated accordingly",

    "decision_inconsistent_with_tool_output":
        "the agent decision diverged from what the tool returned",

    # repair_strategy_failure
    "output_patched_without_regeneration":
        "output was patched instead of regenerated from corrected assumptions",

    "repair_quality_below_threshold":
        "the repair quality was below the acceptable threshold",

    # incorrect_output
    "output_misaligned_with_intent":
        "the output was misaligned with user intent",

    "user_correction_required":
        "the user had to correct the system output",

    # incorrect_output — grounding risk signals
    "grounding_data_absent":
        "tools returned no usable data but the agent produced a substantial response",

    "grounding_gap_not_acknowledged":
        "tools returned no usable data and the agent did not acknowledge the gap",

    # --- Meta signals (model limitation indicators) ---

    # unmodeled_failure
    "symptoms_present_without_diagnosis":
        "output is misaligned and user corrected, but no known pattern matched",

    "no_known_pattern_matched":
        "no defined failure pattern was diagnosed for this trace",

    # insufficient_observability
    "high_field_absence_rate":
        "more than half of expected telemetry fields are missing",

    "critical_fields_missing":
        "a large number of expected fields are absent from the log",

    # conflicting_signals
    "cache_hit_with_intent_mismatch":
        "cache reported a hit but intent similarity is very low",

    "acceptable_alignment_but_user_corrected":
        "response appeared aligned by metric but user still corrected",
}