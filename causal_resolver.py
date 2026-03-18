"""
causal_resolver.py
Resolves causal relationships between active (diagnosed) failures.

Pipeline:
  matcher output → normalize → active_ids → root_candidates + causal_links + enriched_failures
"""


def normalize(matcher_output: list) -> list:
    """
    Convert matcher raw output to debugger internal format.
    Filters out failures where diagnosed is False.
    """
    normalized = []

    for f in matcher_output:
        if not f.get("diagnosed", False):
            continue

        normalized.append({
            "id": f["failure_id"],
            "confidence": f["confidence"],
            "signals": f.get("signals", {}),
        })

    return normalized


def resolve(graph: dict, matcher_output: list) -> dict:
    failures = normalize(matcher_output)
    active_ids = {f["id"] for f in failures}
    backward = graph["backward"]

    # Root candidates: active failures with no active upstream
    roots = []
    for f in failures:
        fid = f["id"]
        upstreams = backward.get(fid, [])
        active_upstreams = [e["from"] for e in upstreams if e["from"] in active_ids]
        if not active_upstreams:
            roots.append(fid)

    # Causal links: edges where both ends are active
    links = []
    for e in graph["edges"]:
        if e["from"] in active_ids and e["to"] in active_ids:
            links.append({
                "from": e["from"],
                "to": e["to"],
                "relation": e["relation"],
                "description": e.get("semantics", {}).get("description", "").strip(),
            })

    # Enrich failures with caused_by
    enriched = []
    for f in failures:
        fid = f["id"]
        causes = [
            e["from"]
            for e in backward.get(fid, [])
            if e["from"] in active_ids
        ]
        item = dict(f)
        if causes:
            item["caused_by"] = causes
        enriched.append(item)

    return {
        "roots": roots,
        "failures": enriched,
        "links": links,
    }
