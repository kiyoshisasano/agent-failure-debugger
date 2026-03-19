"""
abstraction.py

Phase 14: Abstraction / Pruning / UX layer.

Pipeline position:
  matcher → debugger → formatter → abstraction → explainer

Three capabilities:
  1. top-k causal path selection
  2. failure clustering
  3. explanation simplification (verbose / standard / brief)

This layer does NOT modify diagnostic data.
It only controls what is displayed and how.
"""

from labels import FAILURE_MAP

# ---------------------------------------------------------------------------
# Failure clusters
# ---------------------------------------------------------------------------

CLUSTERS = {
    "reasoning_failure": {
        "label": "Reasoning failure",
        "members": [
            "clarification_failure",
            "assumption_invalidation_failure",
            "premature_model_commitment",
            "repair_strategy_failure",
        ],
    },
    "retrieval_failure": {
        "label": "Retrieval failure",
        "members": [
            "semantic_cache_intent_bleeding",
            "prompt_injection_via_retrieval",
            "context_truncation_loss",
            "rag_retrieval_drift",
        ],
    },
    "tool_failure": {
        "label": "Tool failure",
        "members": [
            "agent_tool_call_loop",
            "tool_result_misinterpretation",
        ],
    },
    "output_failure": {
        "label": "Output failure",
        "members": [
            "incorrect_output",
        ],
    },
    "instruction_failure": {
        "label": "Instruction failure",
        "members": [
            "instruction_priority_inversion",
        ],
    },
}

# Reverse lookup: failure_id → cluster_id
_FAILURE_TO_CLUSTER = {}
for cid, cdef in CLUSTERS.items():
    for member in cdef["members"]:
        _FAILURE_TO_CLUSTER[member] = cid


def get_cluster(failure_id: str) -> str | None:
    return _FAILURE_TO_CLUSTER.get(failure_id)


# ---------------------------------------------------------------------------
# 1. Top-k path selection
# ---------------------------------------------------------------------------

def _score_path_simple(path: list, debugger_output: dict) -> float:
    """Quick score for path ranking using available data."""
    conf_map = {f["id"]: f["confidence"] for f in debugger_output.get("failures", [])}
    root_scores = {r["id"]: r["score"] for r in debugger_output.get("root_ranking", [])}

    avg_conf = sum(conf_map.get(n, 0) for n in path) / len(path) if path else 0
    root_score = root_scores.get(path[0], 0) if path else 0
    length_bonus = len(path) / 10.0

    return avg_conf + root_score * 0.3 + length_bonus


def select_top_k(debugger_output: dict, k: int = 2) -> dict:
    """
    Select top-k paths for display.
    Ensures conflict group winners are always included.
    """
    all_paths = debugger_output.get("causal_paths", [])
    multi_hop = [p for p in all_paths if len(p) >= 2]

    if len(multi_hop) <= k:
        return {
            "selected_paths": multi_hop,
            "suppressed_paths": [],
        }

    # Score and sort
    scored = [(p, _score_path_simple(p, debugger_output)) for p in multi_hop]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Ensure conflict winners are included
    conflict_winners = set()
    for c in debugger_output.get("conflicts", []):
        conflict_winners.add(c.get("winner", ""))

    selected = []
    remaining = []

    for path, score in scored:
        if len(selected) < k:
            selected.append(path)
        elif any(node in conflict_winners for node in path) and \
             not any(node in conflict_winners for p in selected for node in p):
            selected.append(path)
        else:
            remaining.append(path)

    return {
        "selected_paths": selected,
        "suppressed_paths": remaining,
    }


# ---------------------------------------------------------------------------
# 2. Failure clustering
# ---------------------------------------------------------------------------

def cluster_failures(debugger_output: dict) -> list[dict]:
    """
    Group active failures into clusters.
    Returns cluster info with representative (highest confidence member).
    """
    active = {f["id"]: f["confidence"] for f in debugger_output.get("failures", [])}

    result = []
    for cid, cdef in CLUSTERS.items():
        active_members = [m for m in cdef["members"] if m in active]
        if not active_members:
            continue

        # Representative: highest confidence
        representative = max(active_members, key=lambda m: active[m])

        result.append({
            "cluster": cid,
            "label": cdef["label"],
            "members": active_members,
            "representative": representative,
            "representative_description": FAILURE_MAP.get(representative, ""),
        })

    return result


# ---------------------------------------------------------------------------
# 3. Explanation simplification
# ---------------------------------------------------------------------------

def _collapse_cluster_sequence(path: list) -> list[dict]:
    """
    Collapse consecutive same-cluster nodes into cluster entries.
    Returns a list of {'type': 'cluster'|'node', ...} entries.
    """
    if not path:
        return []

    entries = []
    current_cluster = None
    current_members = []

    for node in path:
        cluster = get_cluster(node)
        if cluster == current_cluster and current_cluster is not None:
            current_members.append(node)
        else:
            if current_members:
                if current_cluster:
                    entries.append({
                        "type": "cluster",
                        "cluster": current_cluster,
                        "label": CLUSTERS[current_cluster]["label"],
                        "members": current_members,
                    })
                else:
                    for m in current_members:
                        entries.append({"type": "node", "id": m})
            current_cluster = cluster
            current_members = [node]

    # Flush last group
    if current_members:
        if current_cluster:
            entries.append({
                "type": "cluster",
                "cluster": current_cluster,
                "label": CLUSTERS[current_cluster]["label"],
                "members": current_members,
            })
        else:
            for m in current_members:
                entries.append({"type": "node", "id": m})

    return entries


def simplify_explanation(debugger_output: dict, mode: str = "standard") -> dict:
    """
    Generate simplified explanation based on mode.

    Modes:
      verbose:  all steps, full detail
      standard: clustered primary path + 1-line alternatives
      brief:    root cluster → output cluster only
    """
    primary = debugger_output.get("primary_path") or []
    alternatives = debugger_output.get("alternative_paths", [])
    explanation = debugger_output.get("explanation", "")

    if mode == "verbose":
        return {
            "display_mode": "verbose",
            "summary_explanation": explanation,
            "detailed_explanation": explanation,
        }

    # Collapse primary path into clusters
    collapsed = _collapse_cluster_sequence(primary)

    # Build summary
    if mode == "brief":
        if len(collapsed) >= 2:
            first = collapsed[0]
            last = collapsed[-1]
            first_name = first.get("label", first.get("id", "unknown"))
            last_name = last.get("label", last.get("id", "unknown"))
            summary = (
                f"The failure originated in {first_name.lower()} "
                f"and resulted in {last_name.lower()}."
            )
        elif len(collapsed) == 1:
            name = collapsed[0].get("label", collapsed[0].get("id", "unknown"))
            summary = f"The failure is categorized as {name.lower()}."
        else:
            summary = "No causal path detected."

        return {
            "display_mode": "brief",
            "summary_explanation": summary,
            "detailed_explanation": explanation,
        }

    # Standard mode: clustered path narrative
    parts = []
    for i, entry in enumerate(collapsed):
        if entry["type"] == "cluster":
            label = entry["label"]
            members = entry["members"]
            if len(members) == 1:
                desc = FAILURE_MAP.get(members[0], members[0])
                part = f"{label} ({desc})"
            else:
                first = members[0]
                last = members[-1]
                part = f"{label} ({first} → {last})"
        else:
            desc = FAILURE_MAP.get(entry["id"], entry["id"])
            part = f"{entry['id']} ({desc})"

        if i == 0:
            parts.append(part)
        else:
            parts.append(f"leading to {part}")

    summary = ", ".join(parts) + "." if parts else "No causal path detected."

    # Alternatives (1-line each)
    alt_summaries = []
    for alt in alternatives:
        alt_collapsed = _collapse_cluster_sequence(alt)
        alt_labels = [
            e.get("label", e.get("id", "?")) for e in alt_collapsed
        ]
        alt_summaries.append(" → ".join(alt_labels))

    standard_explanation = f"Primary: {summary}"
    if alt_summaries:
        for alt in alt_summaries:
            standard_explanation += f"\nAlternative: {alt}"

    return {
        "display_mode": "standard",
        "summary_explanation": standard_explanation,
        "detailed_explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def abstract(debugger_output: dict, top_k: int = 2,
             mode: str = "standard") -> dict:
    """
    Full abstraction pipeline:
      1. top-k path selection
      2. failure clustering
      3. explanation simplification

    Returns augmented output (original data preserved).
    """
    path_selection = select_top_k(debugger_output, k=top_k)
    clusters = cluster_failures(debugger_output)
    simplified = simplify_explanation(debugger_output, mode=mode)

    return {
        # Original data (preserved)
        "root_candidates":  debugger_output.get("root_candidates", []),
        "root_ranking":     debugger_output.get("root_ranking", []),
        "failures":         debugger_output.get("failures", []),
        "causal_links":     debugger_output.get("causal_links", []),
        "causal_paths":     debugger_output.get("causal_paths", []),
        "primary_path":     debugger_output.get("primary_path"),
        "conflicts":        debugger_output.get("conflicts", []),
        "evidence":         debugger_output.get("evidence", []),
        # Abstraction layer (new)
        "selected_paths":   path_selection["selected_paths"],
        "suppressed_paths": path_selection["suppressed_paths"],
        "clusters":         clusters,
        **simplified,
    }
