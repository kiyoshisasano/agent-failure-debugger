"""
formatter.py
Formats resolver output into the final debugger JSON output.

build_explanation():
  - Uses path-based explanation when a multi-hop path (len >= 2) exists
  - Falls back to link enumeration (with active signals) if no multi-hop path
  - Falls back to "no causal relationships" if no links exist
"""

RELATION_TEXT = {
    "predisposes":   "predisposed",
    "induces":       "induced",
    "propagates_to": "propagated to",
}


def build_explanation(result: dict) -> str:
    # Only paths with at least 2 nodes contain an edge
    multi_hop = [p for p in result.get("paths", []) if len(p) >= 2]

    if multi_hop:
        path = max(multi_hop, key=len)
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

    if not result["links"]:
        return "No causal relationships detected."

    failures_by_id = {f["id"]: f for f in result["failures"]}
    parts = []
    for link in result["links"]:
        src = link["from"]
        dst = link["to"]
        src_signals = failures_by_id.get(src, {}).get("signals", {})
        active_signals = [k for k, v in src_signals.items() if v]
        if active_signals:
            parts.append(f"{src} {link['relation']} {dst} ({', '.join(active_signals)})")
        else:
            parts.append(f"{src} {link['relation']} {dst}")
    return "Causal relationships detected: " + "; ".join(parts)


def format_output(result: dict) -> dict:
    return {
        "root_candidates": result["roots"],
        "failures": result["failures"],
        "causal_links": result["links"],
        "causal_paths": result.get("paths", []),
        "explanation": build_explanation(result),
    }
