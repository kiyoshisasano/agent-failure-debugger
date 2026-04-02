"""
reliability.py — Cross-run stability analysis.

Measures how consistently the debugger diagnoses the same failures
across multiple runs of the same task.

LLM agents are non-deterministic (temperature, API variability),
but the Atlas matcher is deterministic. Variation in detection
results across runs therefore reflects agent behavior variation,
not matcher instability. This module quantifies that variation.

Usage:
    from agent_failure_debugger.reliability import compare_runs

    results = compare_runs(run_results, task_id="flight_booking_test")
    print(results["stability"]["root_cause_agreement"])
    print(results["interpretation"])
"""

from collections import Counter
import math


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_runs(runs: list, task_id: str | None) -> None:
    """Validate run results before analysis.

    Checks:
      - At least 2 runs provided
      - Each run has required structure (summary with root_cause)
      - If task_id is provided, all runs must carry matching task_id

    Raises:
        ValueError: On invalid input.
    """
    if not isinstance(runs, list) or len(runs) < 2:
        raise ValueError(
            f"compare_runs requires at least 2 runs, got {len(runs) if isinstance(runs, list) else type(runs).__name__}"
        )

    for i, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"runs[{i}] must be a dict, got {type(run).__name__}")
        if "summary" not in run or "root_cause" not in run.get("summary", {}):
            raise ValueError(
                f"runs[{i}] missing summary.root_cause — "
                f"pass the output of run_pipeline() directly"
            )

    # Task ID consistency check
    if task_id is not None:
        missing_task_id = []
        for i, run in enumerate(runs):
            run_task = run.get("task_id")
            if run_task is None:
                missing_task_id.append(i)
            elif run_task != task_id:
                raise ValueError(
                    f"runs[{i}] has task_id='{run_task}' but expected '{task_id}'. "
                    f"All runs must be from the same task."
                )
        if missing_task_id:
            import warnings
            warnings.warn(
                f"task_id='{task_id}' specified but runs {missing_task_id} "
                f"have no task_id field. Cannot verify they belong to the "
                f"same task. Consider adding task_id to your pipeline results.",
                UserWarning,
                stacklevel=3,
            )


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_root(run: dict) -> str:
    """Extract root cause ID from a pipeline result."""
    return run.get("summary", {}).get("root_cause", "unknown")


def _extract_failure_set(run: dict) -> set:
    """Extract the set of diagnosed failure IDs from a pipeline result."""
    diagnosis = run.get("diagnosis", {})
    failures = diagnosis.get("failures", [])
    return {f["id"] for f in failures if isinstance(f, dict) and "id" in f}


def _extract_confidence_map(run: dict) -> dict:
    """Extract failure_id → confidence mapping from a pipeline result."""
    diagnosis = run.get("diagnosis", {})
    failures = diagnosis.get("failures", [])
    return {
        f["id"]: f.get("confidence", 0.0)
        for f in failures
        if isinstance(f, dict) and "id" in f
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _mean_pairwise_jaccard(sets: list[set]) -> float:
    """Average Jaccard similarity across all pairs."""
    n = len(sets)
    if n < 2:
        return 1.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _jaccard(sets[i], sets[j])
            count += 1
    return round(total / count, 4) if count > 0 else 1.0


def _coefficient_of_variation(values: list[float]) -> float:
    """Coefficient of variation (σ/μ). Returns 0.0 if mean is 0."""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0.0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return round(math.sqrt(variance) / mean, 4)


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

def _build_interpretation(
    run_count: int,
    root_agreement: float,
    root_mode: str,
    failure_jaccard: float,
    stable: list,
    intermittent: list,
) -> str:
    """Generate a human-readable summary."""
    parts = []

    # Root cause stability
    if root_agreement == 1.0:
        parts.append(
            f"Root cause is fully stable: '{root_mode}' was identified "
            f"in all {run_count} runs."
        )
    elif root_agreement >= 0.8:
        parts.append(
            f"Root cause is mostly stable: '{root_mode}' was identified "
            f"in {root_agreement:.0%} of runs."
        )
    else:
        parts.append(
            f"Root cause is unstable: '{root_mode}' was the most common "
            f"at {root_agreement:.0%}, but other causes appeared."
        )

    # Failure set stability
    if failure_jaccard >= 0.9:
        parts.append("Detected failure sets are highly consistent across runs.")
    elif failure_jaccard >= 0.7:
        parts.append(
            "Detected failure sets are moderately consistent, "
            "with some variation in peripheral patterns."
        )
    else:
        parts.append(
            "Detected failure sets vary significantly across runs. "
            "The agent's behavior is highly non-deterministic for this input."
        )

    # Intermittent failures
    if intermittent:
        parts.append(
            f"Intermittent failures (not in every run): "
            f"{', '.join(sorted(intermittent))}."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def compare_runs(
    runs: list[dict],
    task_id: str | None = None,
) -> dict:
    """Analyze detection stability across multiple runs of the same task.

    Args:
        runs: List of run_pipeline() outputs for the same input/task.
            Each must contain at minimum: summary.root_cause, diagnosis.failures.
        task_id: Optional task identifier. If provided, runs carrying a
            different task_id will be rejected. This prevents accidental
            comparison of unrelated runs.

    Returns:
        Dict with:
          - run_count: number of runs analyzed
          - task_id: task identifier (if provided)
          - stability: metrics dict (root_cause_agreement, failure_set_jaccard, etc.)
          - interpretation: human-readable summary

    Raises:
        ValueError: If fewer than 2 runs, missing required fields,
            or task_id mismatch.
    """
    _validate_runs(runs, task_id)

    run_count = len(runs)

    # Root cause analysis
    roots = [_extract_root(r) for r in runs]
    root_counts = Counter(roots)
    root_mode = root_counts.most_common(1)[0][0]
    root_agreement = round(root_counts[root_mode] / run_count, 4)

    # Failure set analysis
    failure_sets = [_extract_failure_set(r) for r in runs]
    all_failures = set()
    for fs in failure_sets:
        all_failures |= fs

    failure_jaccard = _mean_pairwise_jaccard(failure_sets)

    stable = sorted(set.intersection(*failure_sets)) if failure_sets else []
    intermittent = sorted(all_failures - set(stable))

    # Confidence variation per failure
    confidence_per_failure = {}
    for fid in all_failures:
        values = []
        for r in runs:
            cmap = _extract_confidence_map(r)
            if fid in cmap:
                values.append(cmap[fid])
        if values:
            confidence_per_failure[fid] = _coefficient_of_variation(values)

    # Interpretation
    interpretation = _build_interpretation(
        run_count, root_agreement, root_mode,
        failure_jaccard, stable, intermittent,
    )

    result = {
        "run_count": run_count,
        "stability": {
            "root_cause_agreement": root_agreement,
            "root_cause_mode": root_mode,
            "root_cause_distribution": dict(root_counts),
            "failure_set_jaccard": failure_jaccard,
            "stable_failures": stable,
            "intermittent_failures": intermittent,
            "confidence_cv": confidence_per_failure,
        },
        "interpretation": interpretation,
    }

    if task_id is not None:
        result["task_id"] = task_id

    return result


# ===========================================================================
# diff_runs — Cross-run differential diagnosis
# ===========================================================================
#
# compare_runs() answers: "Is this agent stable?"
# diff_runs()    answers: "What separates success from failure?"
#
# Typical workflow:
#   1. compare_runs(all_runs) → detect instability
#   2. diff_runs(success_runs, failure_runs) → identify divergence cause
#
# ===========================================================================


# ---------------------------------------------------------------------------
# Extraction helpers (diff-specific)
# ---------------------------------------------------------------------------

def _extract_signal_states(run: dict) -> dict[str, dict[str, bool]]:
    """Extract {failure_id: {signal_name: bool}} from a pipeline result.

    Uses diagnosis.failures which retains signals through
    causal_resolver.normalize().
    """
    diagnosis = run.get("diagnosis", {})
    failures = diagnosis.get("failures", [])
    return {
        f["id"]: dict(f.get("signals", {}))
        for f in failures
        if isinstance(f, dict) and "id" in f
    }


def _extract_primary_path(run: dict) -> list | None:
    """Extract the primary causal path from a pipeline result."""
    diagnosis = run.get("diagnosis", {})
    path = diagnosis.get("primary_path")
    if path and isinstance(path, list):
        return path
    return None


# ---------------------------------------------------------------------------
# Diff validation
# ---------------------------------------------------------------------------

def _validate_diff_groups(
    success_runs: list,
    failure_runs: list,
    task_id: str | None,
) -> None:
    """Validate inputs for diff_runs().

    Checks:
      - Each group has at least 1 run
      - Each run has required structure (summary with root_cause)
      - If task_id is provided, all runs must carry matching task_id

    Raises:
        ValueError: On invalid input.
    """
    if not isinstance(success_runs, list) or len(success_runs) < 1:
        raise ValueError(
            "diff_runs requires at least 1 success run, "
            f"got {len(success_runs) if isinstance(success_runs, list) else type(success_runs).__name__}"
        )
    if not isinstance(failure_runs, list) or len(failure_runs) < 1:
        raise ValueError(
            "diff_runs requires at least 1 failure run, "
            f"got {len(failure_runs) if isinstance(failure_runs, list) else type(failure_runs).__name__}"
        )

    all_runs = list(success_runs) + list(failure_runs)
    for i, run in enumerate(all_runs):
        if not isinstance(run, dict):
            raise ValueError(f"runs[{i}] must be a dict, got {type(run).__name__}")
        if "summary" not in run or "root_cause" not in run.get("summary", {}):
            raise ValueError(
                f"runs[{i}] missing summary.root_cause — "
                f"pass the output of run_pipeline() directly"
            )

    if task_id is not None:
        missing_task_id = []
        for i, run in enumerate(all_runs):
            run_task = run.get("task_id")
            if run_task is None:
                missing_task_id.append(i)
            elif run_task != task_id:
                raise ValueError(
                    f"runs[{i}] has task_id='{run_task}' but expected '{task_id}'. "
                    f"All runs must be from the same task."
                )
        if missing_task_id:
            import warnings
            warnings.warn(
                f"task_id='{task_id}' specified but runs {missing_task_id} "
                f"have no task_id field. Cannot verify they belong to the "
                f"same task. Consider adding task_id to your pipeline results.",
                UserWarning,
                stacklevel=3,
            )


# ---------------------------------------------------------------------------
# Diff computation
# ---------------------------------------------------------------------------

def _compute_failure_set_diff(
    success_runs: list[dict],
    failure_runs: list[dict],
) -> dict:
    """Compute failure pattern set differences between groups.

    Returns failure_only and success_only with frequency and
    mean confidence, plus the shared set.
    """
    # Per-run failure sets
    s_sets = [_extract_failure_set(r) for r in success_runs]
    f_sets = [_extract_failure_set(r) for r in failure_runs]

    s_union = set()
    for s in s_sets:
        s_union |= s
    f_union = set()
    for s in f_sets:
        f_union |= s

    failure_only_ids = f_union - s_union
    success_only_ids = s_union - f_union
    shared_ids = f_union & s_union

    # Detail for failure_only patterns
    failure_only = {}
    for fid in sorted(failure_only_ids):
        confs = []
        count = 0
        for r in failure_runs:
            cmap = _extract_confidence_map(r)
            if fid in cmap:
                count += 1
                confs.append(cmap[fid])
        failure_only[fid] = {
            "frequency": round(count / len(failure_runs), 4),
            "mean_confidence": round(sum(confs) / len(confs), 4) if confs else 0.0,
        }

    # Detail for success_only patterns
    success_only = {}
    for fid in sorted(success_only_ids):
        confs = []
        count = 0
        for r in success_runs:
            cmap = _extract_confidence_map(r)
            if fid in cmap:
                count += 1
                confs.append(cmap[fid])
        success_only[fid] = {
            "frequency": round(count / len(success_runs), 4),
            "mean_confidence": round(sum(confs) / len(confs), 4) if confs else 0.0,
        }

    return {
        "failure_only": failure_only,
        "success_only": success_only,
        "shared": sorted(shared_ids),
    }


def _compute_root_cause_diff(
    success_runs: list[dict],
    failure_runs: list[dict],
) -> dict:
    """Compare dominant root causes between groups."""
    s_roots = Counter(_extract_root(r) for r in success_runs)
    f_roots = Counter(_extract_root(r) for r in failure_runs)

    s_mode, s_count = s_roots.most_common(1)[0]
    f_mode, f_count = f_roots.most_common(1)[0]

    return {
        "success_dominant": {
            "root": s_mode,
            "agreement": round(s_count / len(success_runs), 4),
        },
        "failure_dominant": {
            "root": f_mode,
            "agreement": round(f_count / len(failure_runs), 4),
        },
        "shifted": s_mode != f_mode,
    }


def _compute_signal_diff(
    success_runs: list[dict],
    failure_runs: list[dict],
    shared_failures: list[str],
) -> list[dict]:
    """Compute signal-level firing rate differences for shared patterns.

    Returns top 10 signals sorted by absolute delta descending.
    """
    diffs = []

    for fid in shared_failures:
        # Collect all signal names for this failure across all runs
        all_signals = set()
        for r in success_runs + failure_runs:
            states = _extract_signal_states(r)
            if fid in states:
                all_signals |= set(states[fid].keys())

        for sig in all_signals:
            # Success true rate
            s_true = 0
            s_total = 0
            for r in success_runs:
                states = _extract_signal_states(r)
                if fid in states and sig in states[fid]:
                    s_total += 1
                    if states[fid][sig]:
                        s_true += 1

            # Failure true rate
            f_true = 0
            f_total = 0
            for r in failure_runs:
                states = _extract_signal_states(r)
                if fid in states and sig in states[fid]:
                    f_total += 1
                    if states[fid][sig]:
                        f_true += 1

            s_rate = round(s_true / s_total, 4) if s_total > 0 else 0.0
            f_rate = round(f_true / f_total, 4) if f_total > 0 else 0.0
            delta = round(f_rate - s_rate, 4)

            if abs(delta) > 0.0:
                diffs.append({
                    "failure_id": fid,
                    "signal": sig,
                    "success_true_rate": s_rate,
                    "failure_true_rate": f_rate,
                    "delta": delta,
                })

    diffs.sort(key=lambda d: abs(d["delta"]), reverse=True)
    return diffs[:10]


def _compute_confidence_diff(
    success_runs: list[dict],
    failure_runs: list[dict],
    shared_failures: list[str],
) -> list[dict]:
    """Compute confidence differences for shared patterns."""
    diffs = []

    for fid in shared_failures:
        s_confs = []
        for r in success_runs:
            cmap = _extract_confidence_map(r)
            if fid in cmap:
                s_confs.append(cmap[fid])

        f_confs = []
        for r in failure_runs:
            cmap = _extract_confidence_map(r)
            if fid in cmap:
                f_confs.append(cmap[fid])

        if s_confs and f_confs:
            s_mean = round(sum(s_confs) / len(s_confs), 4)
            f_mean = round(sum(f_confs) / len(f_confs), 4)
            delta = round(f_mean - s_mean, 4)
            if abs(delta) > 0.0:
                diffs.append({
                    "failure_id": fid,
                    "success_mean": s_mean,
                    "failure_mean": f_mean,
                    "delta": delta,
                })

    diffs.sort(key=lambda d: abs(d["delta"]), reverse=True)
    return diffs


def _compute_causal_path_diff(
    success_runs: list[dict],
    failure_runs: list[dict],
) -> dict:
    """Compare primary causal paths between groups.

    Each path entry includes the terminal_node for termination
    mode visibility (e.g. premature_termination vs failed_termination).
    """
    def _path_entries(runs):
        entries = []
        seen = set()
        for r in runs:
            path = _extract_primary_path(r)
            if path:
                key = tuple(path)
                if key not in seen:
                    seen.add(key)
                    entries.append({
                        "path": path,
                        "terminal_node": path[-1],
                    })
        return entries

    s_entries = _path_entries(success_runs)
    f_entries = _path_entries(failure_runs)

    s_keys = {tuple(e["path"]) for e in s_entries}
    f_keys = {tuple(e["path"]) for e in f_entries}

    return {
        "failure_only_paths": [e for e in f_entries if tuple(e["path"]) not in s_keys],
        "success_only_paths": [e for e in s_entries if tuple(e["path"]) not in f_keys],
        "shared_paths": [e for e in f_entries if tuple(e["path"]) in s_keys],
    }


# ---------------------------------------------------------------------------
# Hypothesis generation
# ---------------------------------------------------------------------------

def _build_diff_hypothesis(
    failure_set_diff: dict,
    root_cause_diff: dict,
    signal_diff: list[dict],
    confidence_diff: list[dict],
    causal_path_diff: dict,
) -> str:
    """Generate a deterministic, human-readable hypothesis.

    Priority order: failure_only > root_shift > signal_diff >
    confidence_diff > termination_shift > no_difference.
    """
    parts = []

    # 1. Failure-only patterns
    fo = failure_set_diff.get("failure_only", {})
    if fo:
        high_freq = [
            fid for fid, detail in fo.items()
            if detail["frequency"] >= 0.5
        ]
        if high_freq:
            parts.append(
                f"Failures are associated with patterns not seen in "
                f"successful runs: {', '.join(high_freq)}. "
                f"These are the primary candidates for the failure cause."
            )
        else:
            parts.append(
                f"Patterns {', '.join(sorted(fo.keys()))} appear only in "
                f"failure runs, but at low frequency — may be secondary effects."
            )

    # 2. Root cause shift
    if root_cause_diff.get("shifted"):
        s_root = root_cause_diff["success_dominant"]["root"]
        f_root = root_cause_diff["failure_dominant"]["root"]
        parts.append(
            f"The root cause shifted from '{s_root}' in successful runs "
            f"to '{f_root}' in failed runs."
        )

    # 3. Signal-level divergence
    strong_signals = [d for d in signal_diff if abs(d["delta"]) >= 0.5]
    if strong_signals:
        top = strong_signals[0]
        parts.append(
            f"Within shared pattern '{top['failure_id']}', signal "
            f"'{top['signal']}' fires significantly more in failures "
            f"({top['failure_true_rate']:.0%}) than successes "
            f"({top['success_true_rate']:.0%})."
        )

    # 4. Confidence divergence
    strong_conf = [d for d in confidence_diff if abs(d["delta"]) >= 0.2]
    if strong_conf:
        top = strong_conf[0]
        parts.append(
            f"Pattern '{top['failure_id']}' has notably higher confidence "
            f"in failures ({top['failure_mean']}) vs successes "
            f"({top['success_mean']}), suggesting a threshold-adjacent "
            f"behavior change."
        )

    # 5. Termination mode shift
    fp = causal_path_diff.get("failure_only_paths", [])
    sp = causal_path_diff.get("success_only_paths", [])
    if fp and sp:
        f_terminals = {e["terminal_node"] for e in fp}
        s_terminals = {e["terminal_node"] for e in sp}
        if f_terminals != s_terminals:
            parts.append(
                f"Termination mode differs: successful runs end at "
                f"{', '.join(sorted(s_terminals))}, while failures end at "
                f"{', '.join(sorted(f_terminals))}."
            )

    if not parts:
        parts.append(
            "No structural differences detected between success and failure "
            "runs. The failure may be caused by factors outside the current "
            "telemetry coverage."
        )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main API — diff_runs
# ---------------------------------------------------------------------------

def diff_runs(
    success_runs: list[dict],
    failure_runs: list[dict],
    task_id: str | None = None,
) -> dict:
    """Identify structural differences between successful and failed runs.

    This is the differential diagnosis counterpart to compare_runs().
    While compare_runs() measures stability across homogeneous runs,
    diff_runs() identifies what separates success from failure.

    Typical workflow:
        1. compare_runs(all_runs) → detect instability
        2. Separate runs into success/failure groups
        3. diff_runs(success_runs, failure_runs) → identify divergence cause

    Args:
        success_runs: List of run_pipeline() outputs for successful runs.
            At least 1 required. "Success" is determined by the caller.
        failure_runs: List of run_pipeline() outputs for failed runs.
            At least 1 required.
        task_id: Optional task identifier. If provided, all runs must
            carry matching task_id.

    Returns:
        Dict with:
          - run_counts: number of runs per group
          - task_id: task identifier (if provided)
          - failure_set_diff: pattern presence differences
          - root_cause_diff: dominant root cause comparison
          - signal_diff: signal firing rate differences (top 10)
          - confidence_diff: confidence level differences
          - causal_path_diff: causal path structure differences
          - hypothesis: human-readable divergence explanation

    Raises:
        ValueError: If groups are empty, missing required fields,
            or task_id mismatch.
    """
    _validate_diff_groups(success_runs, failure_runs, task_id)

    # 1. Failure set diff
    failure_set_diff = _compute_failure_set_diff(success_runs, failure_runs)

    # 2. Root cause diff
    root_cause_diff = _compute_root_cause_diff(success_runs, failure_runs)

    # 3. Signal diff (shared patterns only)
    signal_diff = _compute_signal_diff(
        success_runs, failure_runs, failure_set_diff["shared"]
    )

    # 4. Confidence diff (shared patterns only)
    confidence_diff = _compute_confidence_diff(
        success_runs, failure_runs, failure_set_diff["shared"]
    )

    # 5. Causal path diff
    causal_path_diff = _compute_causal_path_diff(success_runs, failure_runs)

    # 6. Hypothesis
    hypothesis = _build_diff_hypothesis(
        failure_set_diff, root_cause_diff, signal_diff,
        confidence_diff, causal_path_diff,
    )

    result = {
        "run_counts": {
            "success": len(success_runs),
            "failure": len(failure_runs),
        },
        "failure_set_diff": failure_set_diff,
        "root_cause_diff": root_cause_diff,
        "signal_diff": signal_diff,
        "confidence_diff": confidence_diff,
        "causal_path_diff": causal_path_diff,
        "hypothesis": hypothesis,
    }

    if task_id is not None:
        result["task_id"] = task_id

    return result