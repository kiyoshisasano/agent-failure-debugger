"""
causal_resolver.py
Resolves causal relationships between active (diagnosed) failures.

Pipeline:
  matcher output → normalize → active_ids → root_candidates + causal_links + enriched_failures + paths
"""


def normalize(matcher_output: list) -> list:
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


def build_active_forward(graph: dict, active_ids: set) -> dict:
    forward = {}
    for e in graph["edges"]:
        if e["from"] in active_ids and e["to"] in active_ids:
            forward.setdefault(e["from"], []).append(e["to"])
    return forward


def collect_paths(forward: dict, roots: list) -> list:
    paths = []

    def dfs(node, path):
        next_nodes = forward.get(node, [])
        if not next_nodes:
            paths.append(path[:])
            return
        for nxt in next_nodes:
            if nxt in path:  # cycle guard
                continue
            path.append(nxt)
            dfs(nxt, path)
            path.pop()

    for root in roots:
        dfs(root, [root])

    return paths


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

    # Multi-hop paths
    active_forward = build_active_forward(graph, active_ids)
    paths = collect_paths(active_forward, roots)

    return {
        "roots": roots,
        "failures": enriched,
        "links": links,
        "paths": paths,
    }
