"""
formatter.py

build_explanation():
  - Uses root_ranking to select the primary path (top-ranked root, longest path)
  - Remaining multi-hop paths become alternative contributing paths
  - Falls back to "No causal relationships detected." if no multi-hop path exists

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


def select_primary_path(result: dict) -> list | None:
    """
    Return the primary path to narrate:
    - If root_ranking exists, prefer the longest path whose root is top-ranked.
    - Fallback: longest multi-hop path overall.
    - Returns None if no multi-hop path exists.
    """
    multi_hop = [p for p in result.get("paths", []) if len(p) >= 2]
    if not multi_hop:
        return None

    ranking = result.get("root_ranking", [])
    if ranking:
        top_root = ranking[0]["id"]
        from_top = [p for p in multi_hop if p[0] == top_root]
        if from_top:
            return max(from_top, key=len)

    return max(multi_hop, key=len)


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
