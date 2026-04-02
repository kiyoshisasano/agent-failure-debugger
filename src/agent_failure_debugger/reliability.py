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