"""
agent-failure-debugger
Pipeline: matcher output → graph → causal explanation

Usage:
  python main.py [input.json] [failure_graph.yaml]

Defaults:
  input.json          matcher output (list of failure results)
  failure_graph.yaml  Atlas causal graph (resolved via config.py)
"""

import json
import sys

from agent_failure_debugger.graph_loader import load_graph
from agent_failure_debugger.causal_resolver import resolve
from agent_failure_debugger.formatter import format_output
from agent_failure_debugger.config import GRAPH_PATH

_default_graph = str(GRAPH_PATH)


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "input.json"
    graph_path = sys.argv[2] if len(sys.argv) > 2 else _default_graph

    with open(input_path, encoding="utf-8") as f:
        matcher_output = json.load(f)

    graph = load_graph(graph_path)
    result = resolve(graph, matcher_output)
    output = format_output(result)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()