"""
formatter.py

build_explanation():
  - Uses root_ranking to select the path to narrate (top-ranked root first)
  - Falls back to longest path if no ranking
  - Falls back to "No causal relationships detected." if no multi-hop path exists
  - Link-enumeration fallback (with active signals) when paths exist but no multi-hop
"""

RELATION_TEXT = {
    "predisposes":   "predisposed",
    "induces":       "induced",
    "propagates_to": "propagated to",
}


def select_path(result: dict) -> list | None:
    """
    Return the path to narrate:
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


def build_explanation(result: dict) -> str:
    path = select_path(result)

    if path is None:
        return "No causal relationships detected."

    edge_map = {
        (l["from"], l["to"]): l["relation"]
        for l in result["links"]
    }

    parts = []
    for i in range(len(path) - 1):
        src = path[i]
        dst = path[i + 1]
        rel = RELATION_TEXT.get(edge_map.get((src, dst), ""), "led to")
        if i == 0:
            parts.append(f"{src} {rel} {dst}")
        else:
            parts.append(f"which {rel} {dst}")

    return "Causal path detected: " + ", ".join(parts)


def format_output(result: dict) -> dict:
    return {
        "root_candidates": result["roots"],
        "root_ranking":    result.get("root_ranking", []),
        "failures":        result["failures"],
        "causal_links":    result["links"],
        "causal_paths":    result.get("paths", []),
        "explanation":     build_explanation(result),
    }
