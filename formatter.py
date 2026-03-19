"""
formatter.py

build_explanation():
  - Uses path scoring to select the primary path
  - Remaining multi-hop paths become alternative contributing paths
  - Falls back to "No causal relationships detected." if no multi-hop path exists

Path scoring:
  path_score = 0.5 * avg_confidence + 0.3 * normalized_length + 0.2 * root_rank_score

Output fields:
  primary_path       : the single most important causal chain
  alternative_paths  : other active causal chains (may be empty)
  causal_paths       : all paths (structural, unchanged)
  explanation        : human-readable narrative covering primary + alternatives
"""

RELATION_TEXT = {
    "predisposes":   "predisposed",
    "induces":       "induced",
    "propagates_to": "propagated to",
}


def _score_path(path: list, conf_map: dict, root_scores: dict,
                max_length: int) -> float:
    """
    Score a path for primary selection:
      0.5 * avg_confidence + 0.3 * normalized_length + 0.2 * root_rank_score
    """
    avg_conf = sum(conf_map.get(n, 0.0) for n in path) / len(path)
    norm_len = len(path) / max_length if max_length > 0 else 0.0
    root_score = root_scores.get(path[0], 0.0)

    return 0.5 * avg_conf + 0.3 * norm_len + 0.2 * root_score


def select_primary_path(result: dict) -> list | None:
    """
    Return the primary path using path scoring.
    Falls back to None if no multi-hop path exists.
    """
    multi_hop = [p for p in result.get("paths", []) if len(p) >= 2]
    if not multi_hop:
        return None

    conf_map = {f["id"]: f["confidence"] for f in result.get("failures", [])}
    root_scores = {
        r["id"]: r["score"]
        for r in result.get("root_ranking", [])
    }
    max_length = max(len(p) for p in multi_hop)

    return max(multi_hop,
               key=lambda p: _score_path(p, conf_map, root_scores, max_length))


def select_alternative_paths(result: dict, primary: list | None) -> list:
    """
    Return all multi-hop paths that are not the primary path.
    """
    if primary is None:
        return []
    multi_hop = [p for p in result.get("paths", []) if len(p) >= 2]
    return [p for p in multi_hop if p != primary]


def _narrate_path(path: list, edge_map: dict) -> str:
    """Build a human-readable narrative for a single path."""
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


def build_explanation(result: dict, primary: list | None,
                      alternatives: list) -> str:
    if primary is None:
        return "No causal relationships detected."

    edge_map = {
        (link["from"], link["to"]): link["relation"]
        for link in result["links"]
    }

    lines = []

    # Primary path narrative
    primary_narrative = _narrate_path(primary, edge_map)
    if alternatives:
        lines.append(f"Primary causal path: {primary_narrative}")
    else:
        lines.append(f"Causal path detected: {primary_narrative}")

    # Alternative path narratives
    for alt in alternatives:
        alt_narrative = _narrate_path(alt, edge_map)
        lines.append(
            f"Alternative contributing path: {alt_narrative}"
        )

    return "\n".join(lines)


def format_output(result: dict) -> dict:
    primary = select_primary_path(result)
    alternatives = select_alternative_paths(result, primary)

    return {
        "root_candidates":  result["roots"],
        "root_ranking":     result.get("root_ranking", []),
        "failures":         result["failures"],
        "causal_links":     result["links"],
        "causal_paths":     result.get("paths", []),
        "primary_path":     primary,
        "alternative_paths": alternatives,
        "explanation":      build_explanation(result, primary, alternatives),
    }
