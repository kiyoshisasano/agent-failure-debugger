"""
Termination divergence example for diff_runs().

Compares two pipeline results that share the same root cause
(agent_tool_call_loop) but diverge at the termination mode:
  - silent exit (order pipeline: premature_termination)
  - error exit  (travel planner: failed_termination)

Usage:
    python run_diff.py
"""

import json
import os
from pathlib import Path

from agent_failure_debugger import diff_runs

# Load pipeline results from JSON fixtures
here = Path(__file__).parent
silent_exit_run = json.loads((here / "silent_exit_run.json").read_text(encoding="utf-8"))
error_exit_run = json.loads((here / "error_exit_run.json").read_text(encoding="utf-8"))

# The "success" label is relative — here we treat silent exit as the
# baseline and error exit as the failure we want to explain.
result = diff_runs(
    success_runs=[silent_exit_run],
    failure_runs=[error_exit_run],
    task_id="retry_loop_comparison",
)

print("=" * 60)
print("  diff_runs: Termination Divergence")
print("=" * 60)

# 1. Failure set diff
fsd = result["failure_set_diff"]
print(f"\n  Failure-only patterns (in error exit, not in silent exit):")
for fid, detail in fsd["failure_only"].items():
    print(f"    {fid}: frequency={detail['frequency']}, confidence={detail['mean_confidence']}")

print(f"\n  Success-only patterns (in silent exit, not in error exit):")
for fid, detail in fsd["success_only"].items():
    print(f"    {fid}: frequency={detail['frequency']}, confidence={detail['mean_confidence']}")

print(f"\n  Shared patterns: {', '.join(fsd['shared'])}")

# 2. Root cause diff
rcd = result["root_cause_diff"]
print(f"\n  Root cause shifted: {rcd['shifted']}")
print(f"    Silent exit root: {rcd['success_dominant']['root']} (agreement: {rcd['success_dominant']['agreement']})")
print(f"    Error exit root:  {rcd['failure_dominant']['root']} (agreement: {rcd['failure_dominant']['agreement']})")

# 3. Causal path diff
cpd = result["causal_path_diff"]
print(f"\n  Causal path divergence:")
for entry in cpd["success_only_paths"]:
    print(f"    Silent exit path: {' → '.join(entry['path'])}  (terminal: {entry['terminal_node']})")
for entry in cpd["failure_only_paths"]:
    print(f"    Error exit path:  {' → '.join(entry['path'])}  (terminal: {entry['terminal_node']})")

# 4. Hypothesis
print(f"\n  Hypothesis:")
print(f"    {result['hypothesis']}")
print()
