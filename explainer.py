"""
explainer.py

Phase 10: Robust LLM-powered explanation generation.

Architecture:
  debugger output (deterministic)
  → explanation package (filtered subset)
  → deterministic draft (template + labels)
  → LLM smoothing (constrained)
  → validator (faithfulness + coverage + order)

The LLM does NOT diagnose or compose from scratch.
It only smooths a deterministic draft into fluent prose.
"""

import json
import os
from pathlib import Path

from labels import FAILURE_MAP, SIGNAL_MAP

TEMPLATES_DIR = Path(__file__).parent / "templates"


# ---------------------------------------------------------------------------
# Explanation package
# ---------------------------------------------------------------------------

def build_explanation_package(debugger_output: dict) -> dict:
    return {
        "primary_path":     debugger_output.get("primary_path"),
        "alternative_paths": debugger_output.get("alternative_paths", []),
        "conflicts":        debugger_output.get("conflicts", []),
        "evidence":         debugger_output.get("evidence", []),
        "root_ranking":     debugger_output.get("root_ranking", []),
    }


# ---------------------------------------------------------------------------
# Deterministic draft (template renderer)
# ---------------------------------------------------------------------------

def _describe_failure(failure_id: str) -> str:
    return FAILURE_MAP.get(failure_id, failure_id.replace("_", " "))


def _describe_signal(signal_name: str) -> str:
    return SIGNAL_MAP.get(signal_name, signal_name.replace("_", " "))


def _describe_signals(signals: list[str]) -> str:
    descriptions = [_describe_signal(s) for s in signals]
    if len(descriptions) == 0:
        return ""
    if len(descriptions) == 1:
        return descriptions[0]
    return ", ".join(descriptions[:-1]) + ", and " + descriptions[-1]


def render_draft(package: dict) -> dict:
    """
    Build a deterministic explanation draft from the package.
    This is the 'ground truth' structure that the LLM will smooth.
    """
    primary = package.get("primary_path") or []
    evidence_map = {
        e["failure"]: e["signals"]
        for e in package.get("evidence", [])
    }

    # --- Steps ---
    steps = []
    for fid in primary:
        signals = evidence_map.get(fid, [])
        step = {
            "failure": fid,
            "description": _describe_failure(fid),
            "signals": [_describe_signal(s) for s in signals],
        }
        steps.append(step)

    # --- Primary explanation (template) ---
    parts = []
    for i, step in enumerate(steps):
        sig_text = _describe_signals(evidence_map.get(step["failure"], []))
        if i == 0:
            sentence = f"The failure originated from {step['failure']}"
        else:
            sentence = f"This led to {step['failure']}"
        if sig_text:
            sentence += f", where {sig_text}"
        sentence += "."
        parts.append(sentence)
    primary_draft = " ".join(parts)

    # --- Alternative explanations ---
    alt_drafts = []
    for alt_path in package.get("alternative_paths", []):
        alt_parts = []
        for i, fid in enumerate(alt_path):
            desc = _describe_failure(fid)
            if i == 0:
                alt_parts.append(f"{fid} ({desc})")
            else:
                alt_parts.append(f"leading to {fid} ({desc})")
        alt_drafts.append("An alternative path: " + ", ".join(alt_parts) + ".")

    # --- Evidence summary ---
    all_signals = []
    for e in package.get("evidence", []):
        for s in e["signals"]:
            all_signals.append(_describe_signal(s))
    evidence_draft = (
        "This explanation is supported by: " + "; ".join(all_signals) + "."
        if all_signals else "No supporting signals."
    )

    # --- Summary ---
    if len(primary) >= 2:
        summary_draft = (
            f"The failure chain starts with {_describe_failure(primary[0])} "
            f"and results in {_describe_failure(primary[-1])}."
        )
    elif len(primary) == 1:
        summary_draft = f"The failure is: {_describe_failure(primary[0])}."
    else:
        summary_draft = "No causal path detected."

    # --- Confidence note ---
    ranking = package.get("root_ranking", [])
    conflicts = package.get("conflicts", [])
    if ranking:
        top = ranking[0]
        conf_draft = (
            f"The primary path was selected because {top['id']} "
            f"had the highest root ranking score ({top['score']})."
        )
    else:
        conf_draft = "No ranking information available."

    if conflicts:
        c = conflicts[0]
        conf_draft += (
            f" In the {c['group']} conflict group, {c['winner']} "
            f"was preferred over {', '.join(c['suppressed'])}."
        )

    return {
        "summary": summary_draft,
        "primary_explanation": primary_draft,
        "steps": steps,
        "alternative_explanations": alt_drafts,
        "evidence_summary": evidence_draft,
        "confidence_note": conf_draft,
    }


# ---------------------------------------------------------------------------
# Enhanced draft (human-readable context + interpretation + risk)
# ---------------------------------------------------------------------------

def _extract_grounding_signals(package: dict) -> dict:
    """Extract grounding signal states from evidence in the package."""
    grounding = {
        "data_absent": False,
        "gap_not_acknowledged": False,
    }
    for ev in package.get("evidence", []):
        for sig in ev.get("signals", []):
            if sig == "grounding_data_absent":
                grounding["data_absent"] = True
            elif sig == "grounding_gap_not_acknowledged":
                grounding["gap_not_acknowledged"] = True
    return grounding


def _assess_risk(package: dict, grounding: dict) -> tuple:
    """Determine risk level and justification.

    Returns:
        (level, justification) where level is "high"/"medium"/"low".
    """
    ranking = package.get("root_ranking", [])
    top_score = ranking[0]["score"] if ranking else 0.0
    conflicts = package.get("conflicts", [])

    # High risk: unacknowledged grounding gap or high-confidence root
    if grounding["gap_not_acknowledged"]:
        return ("high",
                "The agent produced output without grounded data "
                "and did not disclose the gap.")
    if top_score >= 0.85 and not conflicts:
        return ("high",
                "High-confidence root cause with no competing explanations.")

    # Medium risk: grounding gap acknowledged, or moderate confidence
    if grounding["data_absent"]:
        return ("medium",
                "Data was unavailable but the agent disclosed this. "
                "Output may contain unverified information.")
    if top_score >= 0.65:
        return ("medium",
                "Moderate-confidence diagnosis. "
                "Review recommended before acting on the fix.")

    # Low
    return ("low",
            "Low-confidence or minor issue. "
            "Monitor but no immediate action required.")


def _build_context_summary(package: dict, grounding: dict) -> str:
    """Build a human-readable context summary."""
    parts = []

    ranking = package.get("root_ranking", [])
    if ranking:
        root = ranking[0]
        parts.append(f"Root cause identified: {_describe_failure(root['id'])} "
                     f"(confidence: {root['score']}).")

    primary = package.get("primary_path") or []
    if len(primary) > 1:
        parts.append(f"A causal chain of {len(primary)} failures was detected.")

    if grounding["data_absent"]:
        parts.append("Tools returned no usable data for this task.")
    if grounding["gap_not_acknowledged"]:
        parts.append("The agent did not acknowledge the data gap.")

    conflicts = package.get("conflicts", [])
    if conflicts:
        parts.append("Competing causal explanations exist.")

    return " ".join(parts) if parts else "No significant context to report."


def _build_interpretation(package: dict, grounding: dict) -> str:
    """Build a human-readable interpretation of why the failure occurred."""
    primary = package.get("primary_path") or []
    if not primary:
        return "No causal path was identified."

    root_id = primary[0]
    root_desc = _describe_failure(root_id)

    parts = [f"The failure originated because {root_desc}."]

    if len(primary) > 1:
        terminal = _describe_failure(primary[-1])
        parts.append(f"This cascaded through {len(primary) - 1} "
                     f"intermediate step(s), ultimately resulting in: "
                     f"{terminal}.")

    if grounding["data_absent"] and not grounding["gap_not_acknowledged"]:
        parts.append("The agent acknowledged the data gap but still "
                     "provided a response, which may contain unverified content.")
    elif grounding["gap_not_acknowledged"]:
        parts.append("The agent responded confidently despite having "
                     "no grounded evidence, which may indicate hallucination.")

    return " ".join(parts)


def _build_recommendation(risk_level: str, package: dict) -> str:
    """Build a human-readable recommendation based on risk level."""
    if risk_level == "high":
        return ("Do not auto-apply fixes. Require human review of "
                "both the diagnosis and the proposed fix before action.")
    elif risk_level == "medium":
        return ("Review the proposed fix before applying. "
                "Verify that the root cause assessment is correct.")
    else:
        return ("Low-risk issue. Fix can be applied with standard "
                "review process.")


def render_enhanced_draft(package: dict) -> dict:
    """
    Build an enhanced explanation with context, interpretation,
    risk assessment, and recommendation.

    This extends render_draft() with human-readable sections
    that non-engineers can understand.

    Observation summary is added by the pipeline (not here)
    because observation_quality comes from matcher_output,
    which the explainer does not have access to.
    """
    base_draft = render_draft(package)
    grounding = _extract_grounding_signals(package)
    risk_level, risk_justification = _assess_risk(package, grounding)

    base_draft["context_summary"] = _build_context_summary(package, grounding)
    base_draft["interpretation"] = _build_interpretation(package, grounding)
    base_draft["risk"] = {
        "level": risk_level,
        "justification": risk_justification,
    }
    base_draft["recommendation"] = _build_recommendation(risk_level, package)

    return base_draft


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def load_template(name: str) -> str:
    path = TEMPLATES_DIR / name
    return path.read_text(encoding="utf-8").strip()


def build_prompt(package: dict, draft: dict) -> tuple[str, str]:
    system = load_template("system_prompt.txt")
    user_template = load_template("user_prompt.txt")
    user = user_template.replace(
        "{draft}", json.dumps(draft, indent=2)
    ).replace(
        "{explanation_package}", json.dumps(package, indent=2)
    )
    return system, user


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(system: str, user: str, api_key: str | None = None,
             model: str = "claude-sonnet-4-20250514") -> dict:
    import urllib.request
    import urllib.error

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "No API key. Set ANTHROPIC_API_KEY or pass api_key argument."
        )

    payload = json.dumps({
        "model": model,
        "max_tokens": 1024,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"API error {e.code}: {error_body}") from e

    text = ""
    for block in body.get("content", []):
        if block.get("type") == "text":
            text += block["text"]

    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3].strip()
    if text.startswith("json"):
        text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse LLM response as JSON: {e}\nRaw: {text[:500]}"
        ) from e


# ---------------------------------------------------------------------------
# Validator (Phase 10 strengthened)
# ---------------------------------------------------------------------------

EXPECTED_FIELDS = {
    "summary", "primary_explanation", "steps",
    "alternative_explanations", "evidence_summary", "confidence_note"
}

FORBIDDEN_WORDS = {"maybe", "possibly", "perhaps", "might", "could be",
                   "it seems", "likely that", "I think", "I believe"}


def validate(response: dict, package: dict) -> list[str]:
    violations = []

    # 1. Schema check
    missing = EXPECTED_FIELDS - set(response.keys())
    if missing:
        violations.append(f"Missing fields: {missing}")

    primary = package.get("primary_path") or []
    evidence_map = {
        e["failure"]: e["signals"]
        for e in package.get("evidence", [])
    }

    # 2. Primary path node coverage
    primary_text = response.get("primary_explanation", "")
    for node in primary:
        if node not in primary_text and node.replace("_", " ") not in primary_text:
            violations.append(
                f"Primary path node '{node}' not mentioned in primary_explanation"
            )

    # 3. Alternative count match
    alt_explanations = response.get("alternative_explanations", [])
    alt_paths = package.get("alternative_paths", [])
    if len(alt_explanations) != len(alt_paths):
        violations.append(
            f"alternative_explanations count ({len(alt_explanations)}) "
            f"!= alternative_paths count ({len(alt_paths)})"
        )

    # 4. Signal coverage: all evidence signals must appear somewhere
    #    Uses keyword matching to allow minor LLM paraphrasing
    #    while catching missing signals.
    STOP_WORDS = {"the", "a", "an", "is", "was", "were", "are", "be",
                  "been", "being", "have", "has", "had", "do", "does",
                  "did", "will", "would", "could", "should", "may",
                  "might", "shall", "can", "to", "of", "in", "for",
                  "on", "with", "at", "by", "from", "as", "into",
                  "through", "during", "before", "after", "and", "but",
                  "or", "nor", "not", "no", "so", "if", "than", "that",
                  "this", "it", "its", "because"}
    all_text = json.dumps(response).lower()
    for ev in package.get("evidence", []):
        for sig in ev["signals"]:
            sig_desc = SIGNAL_MAP.get(sig, "").lower()
            sig_name = sig.replace("_", " ").lower()
            # Exact match first (fastest)
            if sig_name in all_text or sig_desc in all_text or sig in all_text:
                continue
            # Keyword fuzzy match: extract content words, check coverage
            keywords = [w for w in sig_desc.split() if w not in STOP_WORDS and len(w) > 2]
            if keywords:
                matched = sum(1 for kw in keywords if kw in all_text)
                coverage = matched / len(keywords)
                if coverage < 0.6:
                    violations.append(f"Signal '{sig}' not reflected in explanation")

    # 5. Forbidden words (speculation check)
    for word in FORBIDDEN_WORDS:
        if word in all_text:
            violations.append(f"Speculative language detected: '{word}'")

    # 6. Causal order check: nodes must appear in primary path order
    for i in range(len(primary) - 1):
        pos_a = primary_text.find(primary[i])
        pos_b = primary_text.find(primary[i + 1])
        if pos_a == -1 or pos_b == -1:
            continue  # already caught by node coverage check
        if pos_a > pos_b:
            violations.append(
                f"Causal order violation: '{primary[i]}' appears after "
                f"'{primary[i+1]}' in primary_explanation"
            )

    # 7. Steps structure check
    steps = response.get("steps", [])
    if steps and len(steps) != len(primary):
        violations.append(
            f"Steps count ({len(steps)}) != primary path length ({len(primary)})"
        )

    return violations


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def explain(debugger_output: dict, api_key: str | None = None,
            model: str = "claude-sonnet-4-20250514",
            use_llm: bool = True,
            enhanced: bool = False) -> dict:
    """
    Full pipeline:
      debugger output → package → draft → (optional LLM) → validated response.

    If use_llm=False, returns the deterministic draft directly.
    If enhanced=True, includes context_summary, interpretation,
    risk assessment, and recommendation in the draft.
    Observation summary is added by pipeline.py, not here.
    """
    package = build_explanation_package(debugger_output)

    if enhanced:
        draft = render_enhanced_draft(package)
    else:
        draft = render_draft(package)

    if not use_llm:
        violations = validate(draft, package)
        return {
            "mode": "deterministic",
            "explanation_package": package,
            "response": draft,
            "validation": {
                "valid": len(violations) == 0,
                "violations": violations,
            },
        }

    system, user = build_prompt(package, draft)
    response = call_llm(system, user, api_key=api_key, model=model)
    violations = validate(response, package)

    return {
        "mode": "hybrid",
        "explanation_package": package,
        "draft": draft,
        "llm_response": response,
        "validation": {
            "valid": len(violations) == 0,
            "violations": violations,
        },
    }