"""
agent-failure-debugger
Pipeline: matcher output → graph → causal explanation

Usage:
  python main.py [input.json] [failure_graph.yaml]

Defaults:
  input.json          matcher output (list of failure results)
  failure_graph.yaml  Atlas causal graph
"""

import json
import sys

from graph_loader import load_graph
from causal_resolver import resolve
from formatter import format_output


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "input.json"
    graph_path = sys.argv[2] if len(sys.argv) > 2 else "failure_graph.yaml"

    with open(input_path) as f:
        matcher_output = json.load(f)

    graph = load_graph(graph_path)
    result = resolve(graph, matcher_output)
    output = format_output(result)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
