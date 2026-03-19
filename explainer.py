"""
explainer.py (OpenAI version)

Phase 10: Robust LLM-powered explanation generation.
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
    primary = package.get("primary_path") or []
    evidence_map = {
        e["failure"]: e["signals"]
        for e in package.get("evidence", [])
    }

    steps = []
    for fid in primary:
        signals = evidence_map.get(fid, [])
        steps.append({
            "failure": fid,
            "description": _describe_failure(fid),
            "signals": [_describe_signal(s) for s in signals],
        })

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

    all_signals = []
    for e in package.get("evidence", []):
        for s in e["signals"]:
            all_signals.append(_describe_signal(s))
    evidence_draft = (
        "This explanation is supported by: " + "; ".join(all_signals) + "."
        if all_signals else "No supporting signals."
    )

    if len(primary) >= 2:
        summary_draft = (
            f"The failure chain starts with {_describe_failure(primary[0])} "
            f"and results in {_describe_failure(primary[-1])}."
        )
    elif len(primary) == 1:
        summary_draft = f"The failure is: {_describe_failure(primary[0])}."
    else:
        summary_draft = "No causal path detected."

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
# LLM call (OpenAI)
# ---------------------------------------------------------------------------

def call_llm(system: str, user: str, api_key: str | None = None,
             model: str = "gpt-4.1-mini") -> dict:
    import urllib.request
    import urllib.error

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "No API key. Set OPENAI_API_KEY or pass api_key argument."
        )

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": 0
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(f"API error {e.code}: {error_body}") from e

    try:
        text = body["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected API response: {body}")

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
# Validator (Phase 10)
# ---------------------------------------------------------------------------

EXPECTED_FIELDS = {
    "summary", "primary_explanation", "steps",
    "alternative_explanations", "evidence_summary", "confidence_note"
}

FORBIDDEN_WORDS = {"maybe", "possibly", "perhaps", "might", "could be",
                   "it seems", "likely that", "I think", "I believe"}


def validate(response: dict, package: dict) -> list[str]:
    violations = []

    missing = EXPECTED_FIELDS - set(response.keys())
    if missing:
        violations.append(f"Missing fields: {missing}")

    primary = package.get("primary_path") or []
    primary_text = response.get("primary_explanation", "")
    for node in primary:
        if node not in primary_text and node.replace("_", " ") not in primary_text:
            violations.append(
                f"Primary path node '{node}' not mentioned in primary_explanation"
            )

    alt_explanations = response.get("alternative_explanations", [])
    alt_paths = package.get("alternative_paths", [])
    if len(alt_explanations) != len(alt_paths):
        violations.append(
            f"alternative_explanations count ({len(alt_explanations)}) "
            f"!= alternative_paths count ({len(alt_paths)})"
        )

    all_text = json.dumps(response).lower()
    for ev in package.get("evidence", []):
        for sig in ev["signals"]:
            sig_desc = SIGNAL_MAP.get(sig, "").lower()
            sig_name = sig.replace("_", " ").lower()
            if sig_name not in all_text and sig_desc not in all_text and sig not in all_text:
                violations.append(f"Signal '{sig}' not reflected in explanation")

    for word in FORBIDDEN_WORDS:
        if word in all_text:
            violations.append(f"Speculative language detected: '{word}'")

    for i in range(len(primary) - 1):
        pos_a = primary_text.find(primary[i])
        pos_b = primary_text.find(primary[i + 1])
        if pos_a == -1 or pos_b == -1:
            continue
        if pos_a > pos_b:
            violations.append(
                f"Causal order violation: '{primary[i]}' appears after "
                f"'{primary[i+1]}' in primary_explanation"
            )

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
            model: str = "gpt-4.1-mini",
            use_llm: bool = True) -> dict:
    package = build_explanation_package(debugger_output)
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
