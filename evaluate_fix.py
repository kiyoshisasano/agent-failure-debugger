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
    return {
        "failure_ids": [f["id"] for f in failures],
        "failure_count": len(failures),
        "root_count": len(data.get("root_candidates", [])),
        "primary_path": primary,
        "primary_path_length": len(primary),
        "conflict_count": len(data.get("conflicts", [])),
    }


# ---------------------------------------------------------------------------
# Counterfactual simulation
# ---------------------------------------------------------------------------

def simulate_after_state(before: dict, autofix_output: dict,
                         graph: dict) -> dict:
    """
    Simulate the system state after fixes are applied.

    Strategy:
      1. Identify which failures the fixes target
      2. Remove those failures AND their downstream descendants
         (if a root is removed, its entire subtree is removed)
      3. Recompute roots, paths, and conflicts for remaining failures
    """
    import copy
    after = copy.deepcopy(before)

    # Targeted failures
    targeted = set()
    for fix in autofix_output.get("recommended_fixes", []):
        targeted.add(fix["target_failure"])

    # Compute full removal set: targeted + their descendants
    removed = set()
    for t in targeted:
        removed.add(t)
        removed |= _descendants(graph, t)

    # Only remove failures that are actually active
    active_ids = _failure_ids(before)
    actually_removed = removed & active_ids

    # Filter failures
    after["failures"] = [
        f for f in after.get("failures", [])
        if f["id"] not in actually_removed
    ]

    # Update root_candidates
    remaining_ids = _failure_ids(after)
    after["root_candidates"] = _recompute_roots(graph, remaining_ids)

    # Update root_ranking (keep only remaining)
    after["root_ranking"] = [
        r for r in after.get("root_ranking", [])
        if r["id"] in remaining_ids
    ]

    # Update causal_paths
    old_paths = after.get("causal_paths", [])
    after["causal_paths"] = _filter_paths(old_paths, actually_removed)

    # Update primary_path
    remaining_paths = after["causal_paths"]
    after["primary_path"] = _longest_path(remaining_paths) or []

    # Update alternative_paths
    primary = after["primary_path"]
    after["alternative_paths"] = [
        p for p in remaining_paths if p != primary
    ]

    # Update causal_links
    after["causal_links"] = [
        link for link in after.get("causal_links", [])
        if link["from"] not in actually_removed
        and link["to"] not in actually_removed
    ]

    # Update conflicts (remove groups where members were removed)
    new_conflicts = []
    for c in after.get("conflicts", []):
        if c.get("winner") in actually_removed:
            continue
        remaining_suppressed = [
            s for s in c.get("suppressed", [])
            if s not in actually_removed
        ]
        if remaining_suppressed:
            new_conflicts.append({
                **c,
                "suppressed": remaining_suppressed,
            })
    after["conflicts"] = new_conflicts

    return after


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def compute_delta(before: dict, after: dict) -> dict:
    """Compute the difference between before and after states."""
    b = _summarize(before)
    a = _summarize(after)

    mitigated = set(b["failure_ids"]) - set(a["failure_ids"])
    remaining = set(a["failure_ids"])
    new_failures = set(a["failure_ids"]) - set(b["failure_ids"])

    return {
        "failure_count_delta": a["failure_count"] - b["failure_count"],
        "root_count_delta": a["root_count"] - b["root_count"],
        "primary_path_length_delta": a["primary_path_length"] - b["primary_path_length"],
        "conflict_count_delta": a["conflict_count"] - b["conflict_count"],
        "mitigated_failures": sorted(mitigated),
        "remaining_failures": sorted(remaining),
        "new_failures": sorted(new_failures),
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

    # Check if root was mitigated
    b_roots = set(before.get("root_candidates", []))
    a_roots = set(after.get("root_candidates", []))
    if b_roots and b_roots == a_roots and delta["failure_count_delta"] == 0:
        regressions.append({
            "type": "no_effect",
            "severity": "soft",
            "detail": "Fix had no observable effect on failure state",
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

    with open(args[0]) as f:
        before = json.load(f)
    with open(args[1]) as f:
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
