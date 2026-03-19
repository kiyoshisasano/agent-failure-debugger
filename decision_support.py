"""
decision_support.py

Phase 15: Decision Support — failure → action conversion.

GPT review fixes applied:
  ① root_norm: single-root case handled (use raw score, not 1.0)
  ② abstraction integration: selected_paths penalty for non-selected failures
  ③ build_plan: sort by priority_score within each timeline
  ④ conflict_resolution_summary: supports multiple conflict groups
"""

from labels import FAILURE_MAP

# ---------------------------------------------------------------------------
# Action Map
# ---------------------------------------------------------------------------

ACTION_MAP = {
    "clarification_failure": {
        "action": "Add explicit clarification step when ambiguity is detected",
        "detail": "Implement ambiguity detection with question generation before proceeding",
        "type": "control_flow",
        "timeline": "short_term",
        "expected_effect": "Prevents premature commitment by resolving ambiguity early",
    },
    "assumption_invalidation_failure": {
        "action": "Implement hypothesis abandonment when contradicting evidence is found",
        "detail": "Add contradiction detection with forced hypothesis re-evaluation",
        "type": "reasoning_constraint",
        "timeline": "short_term",
        "expected_effect": "Enables the system to discard invalid assumptions",
    },
    "premature_model_commitment": {
        "action": "Introduce multi-hypothesis reasoning before commitment",
        "detail": "Generate and evaluate multiple interpretations before selecting one",
        "type": "reasoning_constraint",
        "timeline": "short_term",
        "expected_effect": "Prevents the entire downstream cascade (cache misuse, retrieval drift, tool loops)",
    },
    "semantic_cache_intent_bleeding": {
        "action": "Add intent validation before cache reuse",
        "detail": "Compare cached query intent with current query intent; fall back to retrieval on mismatch",
        "type": "system_guard",
        "timeline": "immediate",
        "expected_effect": "Prevents cache-induced retrieval bypass when intent has changed",
    },
    "rag_retrieval_drift": {
        "action": "Implement retrieval quality monitoring with alignment checks",
        "detail": "Validate retrieved document relevance against user intent before use",
        "type": "system_guard",
        "timeline": "immediate",
        "expected_effect": "Catches retrieval degradation before it affects output",
    },
    "instruction_priority_inversion": {
        "action": "Enforce instruction priority hierarchy (system > user > retrieved)",
        "detail": "Add priority enforcement layer that blocks lower-priority instruction override",
        "type": "security",
        "timeline": "immediate",
        "expected_effect": "Prevents external instructions from overriding system instructions",
    },
    "prompt_injection_via_retrieval": {
        "action": "Add instruction filtering on retrieved content",
        "detail": "Scan retrieved content for instruction-altering patterns; quarantine adversarial content",
        "type": "security",
        "timeline": "immediate",
        "expected_effect": "Blocks adversarial content from influencing system behavior",
    },
    "context_truncation_loss": {
        "action": "Optimize context window allocation with priority-based truncation",
        "detail": "Implement content prioritization so critical information is preserved during truncation",
        "type": "infrastructure",
        "timeline": "long_term",
        "expected_effect": "Ensures critical information survives context window limits",
    },
    "agent_tool_call_loop": {
        "action": "Add max repeat limit with progress validation",
        "detail": "Enforce maximum consecutive tool calls; require state progress between calls",
        "type": "control_logic",
        "timeline": "immediate",
        "expected_effect": "Breaks tool invocation loops when no progress is made",
    },
    "tool_result_misinterpretation": {
        "action": "Add tool output schema validation and post-processing",
        "detail": "Validate tool results against expected schema; add interpretation verification step",
        "type": "parsing",
        "timeline": "short_term",
        "expected_effect": "Ensures tool outputs are correctly interpreted before acting on them",
    },
    "repair_strategy_failure": {
        "action": "Implement regeneration-first repair strategy",
        "detail": "When errors are detected, regenerate from corrected assumptions instead of patching",
        "type": "reasoning_constraint",
        "timeline": "long_term",
        "expected_effect": "Produces higher quality corrections by rebuilding rather than patching",
    },
    "incorrect_output": {
        "action": "Add output verification and self-critique step",
        "detail": "Implement pre-delivery output check that validates alignment with user intent",
        "type": "output_guard",
        "timeline": "immediate",
        "expected_effect": "Catches misaligned output before delivery to user",
    },
}


# ---------------------------------------------------------------------------
# Priority scoring (fix ①: single-root normalization)
# ---------------------------------------------------------------------------

def _compute_priority_score(failure_id: str, debugger_output: dict,
                            selected_nodes: set | None = None) -> float:
    """
    Score a failure for action priority:
      0.4 * root_score_normalized
    + 0.3 * upstream_bonus
    + 0.2 * confidence
    + 0.1 * not_suppressed
    - 0.1 * not_in_selected (fix ②)
    """
    conf_map = {f["id"]: f["confidence"] for f in debugger_output.get("failures", [])}
    root_scores = {r["id"]: r["score"] for r in debugger_output.get("root_ranking", [])}
    primary = debugger_output.get("primary_path") or []

    # Root score normalization (fix ①)
    root_score = root_scores.get(failure_id, 0)
    if len(root_scores) > 1:
        max_root = max(root_scores.values())
        root_norm = root_score / max_root if max_root > 0 else 0
    else:
        # Single root: use raw score directly (not forced to 1.0)
        root_norm = root_score

    # Upstream bonus
    if failure_id in primary:
        position = primary.index(failure_id)
        upstream_bonus = 1.0 / (position + 1)
    else:
        upstream_bonus = 0.0

    # Confidence
    confidence = conf_map.get(failure_id, 0)

    # Suppression penalty
    suppressed = set()
    for c in debugger_output.get("conflicts", []):
        for s in c.get("suppressed", []):
            suppressed.add(s)
    not_suppressed = 0.0 if failure_id in suppressed else 1.0

    score = 0.4 * root_norm + 0.3 * upstream_bonus + 0.2 * confidence + 0.1 * not_suppressed

    # Selected paths penalty (fix ②)
    if selected_nodes is not None and failure_id not in selected_nodes:
        score -= 0.1

    return round(max(score, 0.0), 4)


def _priority_label(score: float) -> str:
    if score >= 0.7:
        return "high"
    elif score >= 0.4:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Action generation (fix ②: uses selected_nodes)
# ---------------------------------------------------------------------------

def generate_actions(debugger_output: dict,
                     abstraction_output: dict | None = None) -> list[dict]:
    """
    Generate recommended actions for each active failure, ranked by priority.
    If abstraction_output is provided, failures not in selected_paths are penalized.
    """
    failures = debugger_output.get("failures", [])
    primary = debugger_output.get("primary_path") or []

    # Build selected nodes set from abstraction (fix ②)
    selected_nodes = None
    if abstraction_output and "selected_paths" in abstraction_output:
        selected_nodes = set()
        for path in abstraction_output["selected_paths"]:
            for node in path:
                selected_nodes.add(node)

    # Suppressed failures
    suppressed = set()
    for c in debugger_output.get("conflicts", []):
        for s in c.get("suppressed", []):
            suppressed.add(s)

    actions = []
    for f in failures:
        fid = f["id"]
        action_def = ACTION_MAP.get(fid)
        if not action_def:
            continue

        score = _compute_priority_score(fid, debugger_output, selected_nodes)
        priority = _priority_label(score)

        # Rationale
        if fid == (primary[0] if primary else None):
            rationale = (
                f"Root cause of the failure cascade. Addressing {fid} "
                f"will prevent downstream failures."
            )
        elif fid in primary:
            pos = primary.index(fid)
            rationale = (
                f"Step {pos + 1} in the primary causal path. "
                f"Fixing this interrupts the cascade at this point."
            )
        elif fid in suppressed:
            rationale = (
                f"Competing cause (suppressed by conflict resolution). "
                f"Lower priority but may be relevant if primary fix is insufficient."
            )
        else:
            rationale = f"Active failure contributing to system degradation."

        actions.append({
            "target_failure": fid,
            "priority": priority,
            "priority_score": score,
            "action": action_def["action"],
            "detail": action_def["detail"],
            "type": action_def["type"],
            "rationale": rationale,
            "expected_effect": action_def["expected_effect"],
        })

    actions.sort(key=lambda a: a["priority_score"], reverse=True)
    return actions


# ---------------------------------------------------------------------------
# Action plan (fix ③: sorted by priority_score within timeline)
# ---------------------------------------------------------------------------

def build_plan(actions: list[dict]) -> dict:
    """
    Structure actions into immediate / short_term / long_term buckets.
    Sorted by priority_score descending within each bucket.
    """
    buckets = {"immediate": [], "short_term": [], "long_term": []}

    for a in actions:
        if a["priority"] == "low":
            continue
        fid = a["target_failure"]
        timeline = ACTION_MAP.get(fid, {}).get("timeline", "short_term")
        buckets[timeline].append({
            "text": f"{a['action']} (target: {fid})",
            "score": a["priority_score"],
        })

    # Sort within each bucket and extract text (fix ③)
    plan = {}
    for timeline, items in buckets.items():
        items.sort(key=lambda x: x["score"], reverse=True)
        plan[timeline] = [item["text"] for item in items]

    return plan


# ---------------------------------------------------------------------------
# Conflict resolution (fix ④: multiple groups)
# ---------------------------------------------------------------------------

def conflict_resolution_summary(debugger_output: dict) -> list[dict]:
    """Summarize all conflict resolutions. Returns list (not single dict)."""
    conflicts = debugger_output.get("conflicts", [])
    if not conflicts:
        return []

    summaries = []
    for c in conflicts:
        summaries.append({
            "group": c.get("group", "unknown"),
            "focus": c["winner"],
            "focus_action": ACTION_MAP.get(c["winner"], {}).get("action", ""),
            "deprioritized": c["suppressed"],
        })
    return summaries


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def decide(debugger_output: dict,
           abstraction_output: dict | None = None) -> dict:
    """
    Full decision support pipeline.
    """
    actions = generate_actions(debugger_output, abstraction_output)
    plan = build_plan(actions)
    conflict_summaries = conflict_resolution_summary(debugger_output)

    result = {
        "recommended_actions": actions,
        "action_plan": plan,
    }
    if conflict_summaries:
        result["conflict_resolutions"] = conflict_summaries

    return result
