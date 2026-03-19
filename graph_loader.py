"""
graph_loader.py
Loads failure_graph.yaml and builds adjacency structures.

Planned nodes and their edges are excluded silently.
(Planned nodes are retained in Atlas for structural purposes
but are not diagnosable with current telemetry.)
"""

import yaml


def load_graph(path: str) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)

    # Exclude planned nodes
    nodes = {
        n["id"]: n
        for n in data["nodes"]
        if n.get("status") != "planned"
    }

    # Exclude edges that reference any excluded node
    edges = [
        e for e in data["edges"]
        if e["from"] in nodes and e["to"] in nodes
    ]

    forward: dict = {}
    backward: dict = {}

    for e in edges:
        forward.setdefault(e["from"], []).append(e)
        backward.setdefault(e["to"], []).append(e)

    return {
        "contracts": data.get("contracts", {}),
        "nodes": nodes,
        "edges": edges,
        "forward": forward,
        "backward": backward,
        "relationships": data.get("relationships", []),
    }
