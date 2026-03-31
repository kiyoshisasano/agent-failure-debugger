"""
evaluate_fix.py

Phase 18: Before/after evaluation and regression detection.

Performs deterministic counterfactual evaluation:
  1) What failures exist now (before)
  2) If fixes were applied, what would remain (after)
  3) Delta metrics + regression detection
  4) Keep or rollback decision

Usage:
  python evaluate_fix.py debugger_output.json autofix.json
  python evaluate_fix.py debugger_output.json autofix.json --json-only
"""

import json
import sys
from pathlib import Path

from execute_fix import build_execution_plan
from graph_loader import load_graph


DEBUGGER_DIR = Path(__file__).parent
GRAPH_PATH = DEBUGGER_DIR / "failure_graph.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _failure_ids(output: dict) -> set[str]:
    return {f["id"] for f in output.get("failures", [])}


def _descendants(graph: dict, start: str) -> set[str]:
    """All downstream nodes reachable from start via graph edges."""
    forward = {}
    for e in graph.get("edges", []):
        forward.setdefault(e["from"], []).append(e["to"])

    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        for nxt in forward.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _recompute_roots(graph: dict, active_ids: set[str]) -> list[str]:
    """Active nodes with no active upstreams."""
    backward = {}
    for e in graph.get("edges", []):
        backward.setdefault(e["to"], []).append(e["from"])

    roots = []
    for fid in active_ids:
        upstreams = backward.get(fid, [])
        if not any(u in active_ids for u in upstreams):
            roots.append(fid)
    return sorted(roots)


def _filter_paths(paths: list[list[str]], removed: set[str]) -> list[list[str]]:
    return [p for p in paths if not any(node in removed for node in p)]


def _longest_path(paths: list[list[str]]) -> list[str] | None:
    if not paths:
        return None
    return max(paths, key=len)


def _summarize(data: dict) -> dict:
    failures = data.get("failures", [])
    primary = data.get("primary_path") or []
    conflicts = data.get("conflicts", [])
    roots = data.get("root_candidates", [])

    return {
        "failure_ids": [f["id"] for f in failures],
        "failure_count": len(failures),
        "root_candidates": roots,
        "root_count": len(roots),
        "primary_path": primary,
        "primary_path_length": len(primary),
        "conflict_count": len(conflicts),
    }


# ---------------------------------------------------------------------------
# Counterfactual simulation
# ---------------------------------------------------------------------------

def simulate_after_state(before: dict, autofix_output: dict,
                         graph: dict) -> dict:
    """
    Deterministic counterfactual:
    - Fix targets are assumed mitigated
    - Descendants of fixed targets are also mitigated
    - Remaining failures are preserved
    """
    plan = build_execution_plan(autofix_output)
    fixed_targets = [step["target_failure"]
                     for step in plan.get("execution_plan", [])]

    removed = set()
    for fid in fixed_targets:
        removed.add(fid)
        removed |= _descendants(graph, fid)

    after_failures = [f for f in before.get("failures", [])
                      if f["id"] not in removed]
    active_ids = {f["id"] for f in after_failures}
    after_roots = _recompute_roots(graph, active_ids)

    before_paths = before.get("causal_paths", [])
    after_paths = _filter_paths(before_paths, removed)
    after_primary = _longest_path(after_paths)

    after_conflicts = [
        c for c in before.get("conflicts", [])
        if not ({c.get("winner", "")} | set(c.get("suppressed", []))) & removed
    ]

    after_root_ranking = [
        r for r in before.get("root_ranking", []) if r["id"] in set(after_roots)
    ]

    after_evidence = [
        e for e in before.get("evidence", []) if e["failure"] in active_ids
    ]

    return {
        "root_candidates": after_roots,
        "root_ranking": after_root_ranking,
        "failures": after_failures,
        "causal_links": before.get("causal_links", []),
        "causal_paths": after_paths,
        "primary_path": after_primary,
        "alternative_paths": (
            [p for p in after_paths if p != after_primary]
            if after_primary else []
        ),
        "conflicts": after_conflicts,
        "evidence": after_evidence,
        "explanation": (
            f"Predicted mitigation removes failures: {sorted(removed)}. "
            f"Remaining active failures: {sorted(active_ids)}."
        ),
        "_phase18_meta": {
            "fixed_targets": fixed_targets,
            "removed_failures": sorted(removed),
        },
    }


# ---------------------------------------------------------------------------
# Delta metrics
# ---------------------------------------------------------------------------

def compute_delta(before: dict, after: dict) -> dict:
    before_summary = _summarize(before)
    after_summary = _summarize(after)

    before_ids = _failure_ids(before)
    after_ids = _failure_ids(after)

    return {
        "failure_count_delta":
            after_summary["failure_count"] - before_summary["failure_count"],
        "root_count_delta":
            after_summary["root_count"] - before_summary["root_count"],
        "primary_path_length_delta":
            after_summary["primary_path_length"] - before_summary["primary_path_length"],
        "conflict_count_delta":
            after_summary["conflict_count"] - before_summary["conflict_count"],
        "mitigated_failures": sorted(before_ids - after_ids),
        "remaining_failures": sorted(after_ids),
        "new_failures": sorted(after_ids - before_ids),
    }


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------

def detect_regressions(before: dict, after: dict, delta: dict) -> list[dict]:
    regressions = []

    if delta["new_failures"]:
        regressions.append({
            "type": "new_failure_introduced",
            "severity": "hard",
            "detail": f"New failures appeared: {delta['new_failures']}",
        })

    if delta["failure_count_delta"] > 0:
        regressions.append({
            "type": "failure_count_increase",
            "severity": "hard",
            "detail": f"Failure count increased by {delta['failure_count_delta']}",
        })

    if delta["primary_path_length_delta"] > 0:
        regressions.append({
            "type": "path_length_increase",
            "severity": "soft",
            "detail": f"Primary path became longer by {delta['primary_path_length_delta']}",
        })

    if delta["conflict_count_delta"] > 0:
        regressions.append({
            "type": "conflict_increase",
            "severity": "soft",
            "detail": f"Conflict count increased by {delta['conflict_count_delta']}",
        })

    fixed_targets = set(
        after.get("_phase18_meta", {}).get("fixed_targets", []))
    before_roots = set(before.get("root_candidates", []))
    after_roots = set(after.get("root_candidates", []))
    targeted_roots = before_roots & fixed_targets
    if targeted_roots and targeted_roots <= after_roots:
        regressions.append({
            "type": "root_not_mitigated",
            "severity": "hard",
            "detail": f"Targeted root(s) still active: {sorted(targeted_roots)}",
        })

    if not regressions and len(delta["mitigated_failures"]) == 0:
        regressions.append({
            "type": "no_effect",
            "severity": "soft",
            "detail": "No failures were mitigated by the proposed fix set.",
        })

    return regressions


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def decide_keep_or_rollback(regressions: list[dict]) -> str:
    """
    hard regression → rollback
    soft regression only → review
    no regression → keep
    """
    if not regressions:
        return "keep"
    if any(r["severity"] == "hard" for r in regressions):
        return "rollback"
    return "review"


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_report(report: dict):
    print("\n=== PHASE 18: BEFORE / AFTER EVALUATION ===\n")

    b = report["before"]
    a = report["after"]
    d = report["delta"]

    print(f"Before: {b['failure_count']} failures, {b['root_count']} roots, "
          f"path length {b['primary_path_length']}, {b['conflict_count']} conflicts")
    print(f"After:  {a['failure_count']} failures, {a['root_count']} roots, "
          f"path length {a['primary_path_length']}, {a['conflict_count']} conflicts")

    print(f"\nMitigated: {d['mitigated_failures']}")
    print(f"Remaining: {d['remaining_failures']}")
    if d["new_failures"]:
        print(f"NEW (regression): {d['new_failures']}")

    print(f"\nDelta: failures={d['failure_count_delta']:+d} "
          f"roots={d['root_count_delta']:+d} "
          f"path={d['primary_path_length_delta']:+d} "
          f"conflicts={d['conflict_count_delta']:+d}")

    print("\nRegressions:")
    if report["regressions"]:
        for r in report["regressions"]:
            marker = "!!!" if r["severity"] == "hard" else "..."
            print(f"  {marker} [{r['severity']}] {r['type']}: {r['detail']}")
    else:
        print("  None detected.")

    print(f"\nDecision: {report['decision'].upper()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    if len(args) < 2:
        print("Usage: python evaluate_fix.py debugger_output.json autofix.json [--json-only]")
        sys.exit(1)

    with open(args[0], encoding="utf-8") as f:
        before = json.load(f)
    with open(args[1], encoding="utf-8") as f:
        autofix_output = json.load(f)

    graph = load_graph(str(GRAPH_PATH))
    after = simulate_after_state(before, autofix_output, graph)
    delta = compute_delta(before, after)
    regressions = detect_regressions(before, after, delta)
    decision = decide_keep_or_rollback(regressions)

    report = {
        "before": _summarize(before),
        "after": _summarize(after),
        "delta": delta,
        "regressions": regressions,
        "decision": decision,
    }

    if "--json-only" in flags:
        print(json.dumps(report, indent=2))
    else:
        display_report(report)


if __name__ == "__main__":
    main()