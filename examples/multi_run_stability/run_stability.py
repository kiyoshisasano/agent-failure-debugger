"""
Multi-run stability example: compare_runs → diff_runs workflow.

Simulates 5 runs of the same task where 3 succeed and 2 fail
differently. Demonstrates the two-step analysis:

  1. compare_runs() detects instability
  2. diff_runs() identifies what caused the failures

Usage:
    python run_stability.py
"""

import json
from pathlib import Path

from agent_failure_debugger import compare_runs, diff_runs

# Load run fixtures
here = Path(__file__).parent
runs = []
for i in range(1, 6):
    data = json.loads((here / f"run_{i}.json").read_text(encoding="utf-8"))
    runs.append(data)

# =====================================================================
# Step 1: Are the results stable?
# =====================================================================

print("=" * 60)
print("  Step 1: compare_runs — Stability Analysis")
print("=" * 60)

stability = compare_runs(runs, task_id="flight_booking")

s = stability["stability"]
print(f"\n  Runs analyzed: {stability['run_count']}")
print(f"  Root cause agreement: {s['root_cause_agreement']}")
print(f"  Root cause mode: {s['root_cause_mode']}")
print(f"  Root cause distribution: {s['root_cause_distribution']}")
print(f"  Failure set Jaccard: {s['failure_set_jaccard']}")
print(f"  Stable failures: {s['stable_failures']}")
print(f"  Intermittent failures: {s['intermittent_failures']}")
print(f"\n  Interpretation: {stability['interpretation']}")

# =====================================================================
# Step 2: What separates success from failure?
# =====================================================================

# Separate runs by execution quality
success_runs = [r for r in runs if r["summary"]["execution_quality"]["status"] == "healthy"]
failure_runs = [r for r in runs if r["summary"]["execution_quality"]["status"] != "healthy"]

print(f"\n{'=' * 60}")
print(f"  Step 2: diff_runs — Divergence Analysis")
print(f"  ({len(success_runs)} successful, {len(failure_runs)} failed)")
print(f"{'=' * 60}")

diff = diff_runs(success_runs, failure_runs, task_id="flight_booking")

# Failure set diff
fsd = diff["failure_set_diff"]
print(f"\n  Patterns only in failures:")
for fid, detail in fsd["failure_only"].items():
    print(f"    {fid}: frequency={detail['frequency']}, confidence={detail['mean_confidence']}")

if fsd["success_only"]:
    print(f"\n  Patterns only in successes:")
    for fid, detail in fsd["success_only"].items():
        print(f"    {fid}: frequency={detail['frequency']}, confidence={detail['mean_confidence']}")

print(f"\n  Shared patterns: {', '.join(fsd['shared']) if fsd['shared'] else 'none'}")

# Root cause diff
rcd = diff["root_cause_diff"]
print(f"\n  Root cause shifted: {rcd['shifted']}")
if rcd["shifted"]:
    print(f"    Success: {rcd['success_dominant']['root']}")
    print(f"    Failure: {rcd['failure_dominant']['root']}")

# Signal diff
if diff["signal_diff"]:
    print(f"\n  Top signal differences:")
    for sd in diff["signal_diff"][:3]:
        print(f"    {sd['failure_id']}.{sd['signal']}: "
              f"success={sd['success_true_rate']:.0%} → failure={sd['failure_true_rate']:.0%} "
              f"(delta={sd['delta']:+.0%})")

# Hypothesis
print(f"\n  Hypothesis:")
print(f"    {diff['hypothesis']}")
print()
