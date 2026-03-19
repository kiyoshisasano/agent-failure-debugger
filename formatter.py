"""
formatter.py

Pipeline:
  1. path scoring → select primary path
  2. conflict resolution → detect exclusivity groups
  3. select alternative paths
  4. build explanation + evidence

Path scoring:
  path_score = 0.4 * avg_confidence
             + 0.2 * normalized_length
             + 0.2 * root_rank_score
             + 0.2 * signal_density

Conflict resolution:
  For each exclusivity group, if multiple members are active,
  the member in the primary path wins and others are suppressed.
"""

RELATION_TEXT = {
    "predisposes":   "predisposed",
    "induces":       "induced",
    "propagates_to": "propagated to",
}


# ---------------------------------------------------------------------------
# Path scoring
# ---------------------------------------------------------------------------

def _signal_density(failure: dict) -> float:
    signals = failure.get("signals", {})
    if not signals:
        return 0.0
    active = sum(1 for v in signals.values() if v)
    return active / len(signals)


def _score_path(path: list, conf_map: dict, failure_map: dict,
                root_scores: dict, max_length: int) -> float:
    avg_conf = sum(conf_map.get(n, 0.0) for n in path) / len(path)
    norm_len = len(path) / max_length if max_length > 0 else 0.0
    root_score = root_scores.get(path[0], 0.0)
    avg_density = sum(
        _signal_density(failure_map[n]) for n in path if n in failure_map
    ) / len(path)

    return 0.4 * avg_conf + 0.2 * norm_len + 0.2 * root_score + 0.2 * avg_density


def select_primary_path(result: dict) -> list | None:
    multi_hop = [p for p in result.get("paths", []) if len(p) >= 2]
    if not multi_hop:
        return None

    conf_map = {f["id"]: f["confidence"] for f in result.get("failures", [])}
    failure_map = {f["id"]: f for f in result.get("failures", [])}
    root_scores = {
        r["id"]: r["score"]
        for r in result.get("root_ranking", [])
    }
    max_length = max(len(p) for p in multi_hop)

    return max(multi_hop,
               key=lambda p: _score_path(p, conf_map, failure_map,
                                         root_scores, max_length))


def select_alternative_paths(result: dict, primary: list | None) -> list:
    if primary is None:
        return []
    multi_hop = [p for p in result.get("paths", []) if len(p) >= 2]
    return [p for p in multi_hop if p != primary]


# ---------------------------------------------------------------------------
# Conflict resolution
# ---------------------------------------------------------------------------

def resolve_conflicts(result: dict, primary: list | None) -> list:
    """
    Detect exclusivity group conflicts among active failures.
    Returns a list of conflict records.

    For soft mode: the group member present in the primary path wins.
    If no member is in the primary path, the highest-confidence member wins.
    """
    relationships = result.get("relationships", [])
    if not relationships:
        return []

    active_ids = {f["id"] for f in result.get("failures", [])}
    conf_map = {f["id"]: f["confidence"] for f in result.get("failures", [])}
    primary_set = set(primary) if primary else set()

    conflicts = []

    for rel in relationships:
        if rel.get("type") != "exclusivity":
            continue

        members_active = [fid for fid in rel["failures"] if fid in active_ids]
        if len(members_active) < 2:
            continue

        # Determine winner: prefer member in primary path, then highest conf
        in_primary = [fid for fid in members_active if fid in primary_set]
        if in_primary:
            winner = in_primary[0]
        else:
            winner = max(members_active, key=lambda fid: conf_map.get(fid, 0.0))

        suppressed = [fid for fid in members_active if fid != winner]

        conflicts.append({
            "group": rel["group"],
            "winner": winner,
            "suppressed": suppressed,
            "mode": rel.get("mode", "soft"),
        })

    return conflicts


# ---------------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------------

def _narrate_path(path: list, edge_map: dict) -> str:
    parts = []
    for i in range(len(path) - 1):
        src = path[i]
        dst = path[i + 1]
        rel = RELATION_TEXT.get(edge_map.get((src, dst), ""), "led to")
        if i == 0:
            parts.append(f"{src} {rel} {dst}")
        else:
            parts.append(f"which {rel} {dst}")
    return ", ".join(parts)


def build_evidence(result: dict, path: list | None) -> list:
    if path is None:
        return []

    failure_map = {f["id"]: f for f in result.get("failures", [])}
    evidence = []
    for node in path:
        f = failure_map.get(node)
        if f is None:
            continue
        active_signals = [
            sig for sig, val in f.get("signals", {}).items() if val
        ]
        if active_signals:
            evidence.append({
                "failure": node,
                "signals": active_signals,
            })
    return evidence


def build_explanation(result: dict, primary: list | None,
                      alternatives: list, conflicts: list) -> str:
    if primary is None:
        return "No causal relationships detected."

    edge_map = {
        (link["from"], link["to"]): link["relation"]
        for link in result["links"]
    }

    # Build suppressed set for annotation
    suppressed_set = set()
    for c in conflicts:
        for s in c["suppressed"]:
            suppressed_set.add(s)

    lines = []

    # Primary path narrative
    primary_narrative = _narrate_path(primary, edge_map)
    if alternatives:
        lines.append(f"Primary causal path: {primary_narrative}")
    else:
        lines.append(f"Causal path detected: {primary_narrative}")

    # Alternative path narratives (annotated if suppressed)
    for alt in alternatives:
        alt_narrative = _narrate_path(alt, edge_map)
        is_suppressed = any(node in suppressed_set for node in alt)
        if is_suppressed:
            lines.append(
                f"Alternative competing path (lower confidence): "
                f"{alt_narrative}"
            )
        else:
            lines.append(
                f"Alternative contributing path: {alt_narrative}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def format_output(result: dict) -> dict:
    # 1. Path scoring
    primary = select_primary_path(result)

    # 2. Conflict resolution
    conflicts = resolve_conflicts(result, primary)

    # 3. Alternative paths
    alternatives = select_alternative_paths(result, primary)

    return {
        "root_candidates":  result["roots"],
        "root_ranking":     result.get("root_ranking", []),
        "failures":         result["failures"],
        "causal_links":     result["links"],
        "causal_paths":     result.get("paths", []),
        "primary_path":     primary,
        "alternative_paths": alternatives,
        "conflicts":        conflicts,
        "explanation":      build_explanation(result, primary, alternatives,
                                             conflicts),
        "evidence":         build_evidence(result, primary),
    }
