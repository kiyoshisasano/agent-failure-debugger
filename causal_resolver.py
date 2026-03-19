"""
causal_resolver.py
Pipeline:
  matcher output → normalize → active_ids → root_candidates + causal_links
                → enriched_failures + paths + root_ranking
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


def compute_root_ranking(failures: list, roots: list, paths: list) -> list:
    """
    Score each root candidate using:
      score = 0.5 * confidence + 0.3 * normalized_downstream + 0.2 * (1 - normalized_depth)

    downstream_count: total nodes downstream across all paths from this root.
    depth: 0 for all roots by definition (roots have no active upstream).
    """
    conf_map = {f["id"]: f["confidence"] for f in failures}

    # Count downstream nodes reachable from each root via paths
    downstream = {fid: 0 for fid in roots}
    for path in paths:
        if not path:
            continue
        root_of_path = path[0]
        if root_of_path in downstream:
            # number of nodes after the root in this path
            downstream[root_of_path] += len(path) - 1

    max_down = max(downstream.values()) if downstream else 1

    # All roots have depth 0 by definition; kept for future multi-root scoring
    max_depth = 1

    ranking = []
    for fid in roots:
        conf = conf_map.get(fid, 0.0)
        down = downstream[fid] / max_down if max_down > 0 else 0.0
        dep  = 0.0  # roots are always depth 0

        score = round(0.5 * conf + 0.3 * down + 0.2 * (1 - dep), 3)

        ranking.append({"id": fid, "score": score})

    ranking.sort(key=lambda x: x["score"], reverse=True)
    return ranking


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
                "to":   e["to"],
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

    # Root ranking
    ranking = compute_root_ranking(enriched, roots, paths)

    return {
        "roots":         roots,
        "root_ranking":  ranking,
        "failures":      enriched,
        "links":         links,
        "paths":         paths,
        "relationships": graph.get("relationships", []),
    }
