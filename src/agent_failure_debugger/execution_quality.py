"""
execution_quality.py — Single-run execution behavior assessment.

Classifies a pipeline run into one of three execution states:

    healthy  — no failures, or only low-severity meta patterns
    degraded — output was produced but quality indicators are weak
    failed   — execution failed to produce usable output

Also classifies termination mode:

    normal        — agent completed and produced output
    silent_exit   — agent stopped without output or error
    error_exit    — agent stopped due to an execution error
    partial_exit  — agent produced output but with significant issues
    unknown       — insufficient telemetry to determine

Design principles:
  - Does NOT add new matcher patterns (Atlas taxonomy stays clean)
  - Uses existing telemetry fields and diagnosis results only
  - Missing fields → unknown (not assumed healthy or failed)
  - Deterministic — same inputs produce same outputs

This module is called from pipeline_summary.py to enrich the
summary with execution behavior information.
"""


# ---------------------------------------------------------------------------
# Telemetry field extraction
# ---------------------------------------------------------------------------

def _get_field(data: dict, dotted_path: str):
    """Traverse a dotted path into a nested dict. Return None if missing."""
    parts = dotted_path.split(".")
    node = data
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


# ---------------------------------------------------------------------------
# Termination mode classification
# ---------------------------------------------------------------------------

# Terminal failure patterns that indicate specific exit modes
_SILENT_EXIT_PATTERNS = {"premature_termination"}
_ERROR_EXIT_PATTERNS = {"failed_termination"}

# Signals that directly indicate termination behavior
_SIGNAL_OUTPUT_PRODUCED = "state.output_produced"
_SIGNAL_CHAIN_ERROR = "state.chain_error_occurred"


def classify_termination(
    diagnosis: dict,
    telemetry: dict | None = None,
) -> dict:
    """Classify how the agent terminated.

    Uses two information sources (in priority order):
      1. Telemetry fields (state.output_produced, state.chain_error_occurred)
      2. Diagnosed failure patterns (premature_termination, failed_termination)

    Args:
        diagnosis: The diagnosis dict from run_pipeline().
        telemetry: Optional raw telemetry dict (adapter output).

    Returns:
        Dict with:
          - mode: "normal" | "silent_exit" | "error_exit" | "partial_exit" | "unknown"
          - reasons: list of evidence strings
    """
    failure_ids = {
        f["id"] for f in diagnosis.get("failures", [])
        if isinstance(f, dict) and "id" in f
    }

    reasons = []

    # --- Telemetry-based classification (highest priority) ---
    output_produced = None
    chain_error = None

    if telemetry:
        output_produced = _get_field(telemetry, _SIGNAL_OUTPUT_PRODUCED)
        chain_error = _get_field(telemetry, _SIGNAL_CHAIN_ERROR)

    # Case 1: Error terminated
    if chain_error is True:
        reasons.append("state.chain_error_occurred is true")
        if output_produced is False:
            reasons.append("state.output_produced is false")
            return {"mode": "error_exit", "reasons": reasons}
        elif output_produced is True:
            # Error occurred but output was still produced
            reasons.append("state.output_produced is true despite error")
            return {"mode": "partial_exit", "reasons": reasons}

    # Case 2: Silent exit (no output, no error)
    if output_produced is False and chain_error is not True:
        reasons.append("state.output_produced is false")
        reasons.append("no execution error detected")
        return {"mode": "silent_exit", "reasons": reasons}

    # Case 3: Normal completion
    if output_produced is True and chain_error is not True:
        reasons.append("state.output_produced is true")
        reasons.append("no execution error detected")
        return {"mode": "normal", "reasons": reasons}

    # --- Fallback: pattern-based classification ---
    if failure_ids & _ERROR_EXIT_PATTERNS:
        reasons.append(
            f"diagnosed pattern: {', '.join(failure_ids & _ERROR_EXIT_PATTERNS)}"
        )
        return {"mode": "error_exit", "reasons": reasons}

    if failure_ids & _SILENT_EXIT_PATTERNS:
        reasons.append(
            f"diagnosed pattern: {', '.join(failure_ids & _SILENT_EXIT_PATTERNS)}"
        )
        return {"mode": "silent_exit", "reasons": reasons}

    # --- Cannot determine ---
    reasons.append("insufficient telemetry for termination classification")
    return {"mode": "unknown", "reasons": reasons}


# ---------------------------------------------------------------------------
# Degradation indicators
# ---------------------------------------------------------------------------

# Thresholds for degradation signals
_ALIGNMENT_DEGRADED_THRESHOLD = 0.5    # below this → weak alignment
_EXPANSION_RATIO_HIGH = 3.0            # above this → possible hallucination
_OBSERVATION_COVERAGE_WEAK = "low"     # coverage level indicating sparse data


def _collect_degradation_indicators(
    diagnosis: dict,
    telemetry: dict | None,
    diagnosis_context: dict | None,
) -> list[dict]:
    """Collect indicators of degraded execution quality.

    Each indicator has:
      - signal: what was observed
      - value: the actual value
      - concern: why this suggests degradation

    Returns empty list if no degradation signals are found.
    """
    indicators = []
    failure_ids = {
        f["id"] for f in diagnosis.get("failures", [])
        if isinstance(f, dict) and "id" in f
    }

    # 1. Low alignment score
    if telemetry:
        alignment = _get_field(telemetry, "response.alignment_score")
        if alignment is not None and alignment < _ALIGNMENT_DEGRADED_THRESHOLD:
            indicators.append({
                "signal": "response.alignment_score",
                "value": alignment,
                "concern": "output alignment with user intent is weak",
            })

    # 2. Weak grounding (only relevant when tools were actually used)
    if telemetry:
        tool_data = _get_field(telemetry, "grounding.tool_provided_data")
        uncertainty_ack = _get_field(
            telemetry, "grounding.uncertainty_acknowledged"
        )

        # Check if tools were actually called
        tool_call_count = _get_field(telemetry, "tools.call_count")
        tools_were_used = (
            tool_call_count is not None and tool_call_count > 0
        )

        if tool_data is False and tools_were_used:
            indicators.append({
                "signal": "grounding.tool_provided_data",
                "value": False,
                "concern": "tools were called but provided no usable data",
            })

        if tool_data is True and uncertainty_ack is False:
            # Tool provided data but agent didn't acknowledge gaps
            expansion = _get_field(telemetry, "grounding.expansion_ratio")
            if expansion is not None and expansion > _EXPANSION_RATIO_HIGH:
                indicators.append({
                    "signal": "grounding.expansion_ratio",
                    "value": expansion,
                    "concern": (
                        "response significantly exceeds source data length, "
                        "suggesting unsupported content generation"
                    ),
                })

    # 3. Low observation coverage
    if diagnosis_context:
        coverage = (
            diagnosis_context
            .get("quality", {})
            .get("coverage", "unknown")
        )
        if coverage == _OBSERVATION_COVERAGE_WEAK:
            missing = (
                diagnosis_context
                .get("quality", {})
                .get("missing_signals", [])
            )
            indicators.append({
                "signal": "observation_coverage",
                "value": coverage,
                "concern": (
                    f"diagnosis based on limited observations "
                    f"({len(missing)} signals missing)"
                ),
            })

    # 4. Unmodeled failure present (symptoms without explanation)
    if "unmodeled_failure" in failure_ids:
        indicators.append({
            "signal": "unmodeled_failure",
            "value": True,
            "concern": (
                "symptoms detected but no known pattern matched — "
                "failure may be outside current taxonomy"
            ),
        })

    # 5. Conflicting signals present
    if "conflicting_signals" in failure_ids:
        indicators.append({
            "signal": "conflicting_signals",
            "value": True,
            "concern": (
                "observed signals point in contradictory directions, "
                "reducing confidence in diagnosis"
            ),
        })

    return indicators


# ---------------------------------------------------------------------------
# Execution quality classification
# ---------------------------------------------------------------------------

def classify_execution_quality(
    diagnosis: dict,
    telemetry: dict | None = None,
    diagnosis_context: dict | None = None,
) -> dict:
    """Classify single-run execution quality.

    Combines termination mode, failure diagnosis, and degradation
    indicators into a unified execution quality assessment.

    The three states:
      healthy  — no significant issues detected
      degraded — output may have been produced but quality is questionable
      failed   — execution did not produce usable output

    Args:
        diagnosis: The diagnosis dict from run_pipeline().
        telemetry: Optional raw telemetry dict (adapter output).
            When available, enables richer degradation detection.
        diagnosis_context: Optional context from diagnose().
            Provides observation coverage information.

    Returns:
        Dict with:
          - status: "healthy" | "degraded" | "failed"
          - termination: termination mode classification
          - indicators: list of degradation indicators (empty if healthy)
          - summary: human-readable one-line assessment
    """
    # Step 1: Termination classification
    termination = classify_termination(diagnosis, telemetry)

    # Step 2: Collect degradation indicators
    indicators = _collect_degradation_indicators(
        diagnosis, telemetry, diagnosis_context
    )

    # Step 3: Determine execution status
    failure_ids = {
        f["id"] for f in diagnosis.get("failures", [])
        if isinstance(f, dict) and "id" in f
    }

    # Meta-only patterns (not domain failures)
    meta_patterns = {
        "unmodeled_failure", "insufficient_observability", "conflicting_signals"
    }
    domain_failures = failure_ids - meta_patterns

    # --- Failed: non-output termination modes ---
    if termination["mode"] in ("error_exit", "silent_exit"):
        status = "failed"
        summary = _build_quality_summary(
            status, termination["mode"], domain_failures, indicators
        )
        return {
            "status": status,
            "termination": termination,
            "indicators": indicators,
            "summary": summary,
        }

    # --- Degraded: output produced but quality concerns exist ---
    if indicators or (domain_failures and termination["mode"] != "normal"):
        status = "degraded"
        summary = _build_quality_summary(
            status, termination["mode"], domain_failures, indicators
        )
        return {
            "status": status,
            "termination": termination,
            "indicators": indicators,
            "summary": summary,
        }

    # --- Degraded: domain failures detected even with normal termination ---
    if domain_failures:
        status = "degraded"
        summary = _build_quality_summary(
            status, termination["mode"], domain_failures, indicators
        )
        return {
            "status": status,
            "termination": termination,
            "indicators": indicators,
            "summary": summary,
        }

    # --- Healthy: no domain failures, no degradation indicators ---
    status = "healthy"
    summary = _build_quality_summary(
        status, termination["mode"], domain_failures, indicators
    )
    return {
        "status": status,
        "termination": termination,
        "indicators": indicators,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def _build_quality_summary(
    status: str,
    termination_mode: str,
    domain_failures: set,
    indicators: list[dict],
) -> str:
    """Build a one-line human-readable summary."""

    if status == "failed":
        if termination_mode == "error_exit":
            return (
                f"Execution failed with error. "
                f"{len(domain_failures)} failure pattern(s) detected."
            )
        elif termination_mode == "silent_exit":
            return (
                f"Execution stopped without output. "
                f"{len(domain_failures)} failure pattern(s) detected."
            )
        return f"Execution failed. {len(domain_failures)} failure pattern(s) detected."

    if status == "degraded":
        concerns = [ind["concern"] for ind in indicators[:2]]
        if concerns:
            detail = "; ".join(concerns)
            return (
                f"Execution completed but quality is degraded: {detail}. "
                f"{len(domain_failures)} failure pattern(s) detected."
            )
        return (
            f"Execution completed with {len(domain_failures)} "
            f"failure pattern(s) detected."
        )

    return "Execution completed normally with no significant issues."