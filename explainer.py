"""
explainer.py

Phase 9: LLM-powered explanation generation.

Architecture:
  debugger output (deterministic)
  → explanation package (filtered subset)
  → LLM prompt (constrained)
  → natural language report
  → validator (faithfulness check)

The LLM does NOT diagnose. It only explains the structure
that the debugger has already determined.
"""

import json
import os
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"


# ---------------------------------------------------------------------------
# Explanation package
# ---------------------------------------------------------------------------

def build_explanation_package(debugger_output: dict) -> dict:
    """
    Extract the fields needed for LLM explanation.
    Removes structural data that the LLM doesn't need.
    """
    return {
        "primary_path":     debugger_output.get("primary_path"),
        "alternative_paths": debugger_output.get("alternative_paths", []),
        "conflicts":        debugger_output.get("conflicts", []),
        "evidence":         debugger_output.get("evidence", []),
        "root_ranking":     debugger_output.get("root_ranking", []),
    }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def load_template(name: str) -> str:
    path = TEMPLATES_DIR / name
    return path.read_text().strip()


def build_prompt(package: dict) -> tuple[str, str]:
    """
    Returns (system_prompt, user_prompt).
    """
    system = load_template("system_prompt.txt")
    user_template = load_template("user_prompt.txt")
    user = user_template.replace(
        "{explanation_package}",
        json.dumps(package, indent=2)
    )
    return system, user


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(system: str, user: str, api_key: str | None = None,
             model: str = "claude-sonnet-4-20250514") -> dict:
    """
    Call Anthropic API. Returns parsed JSON response.
    Raises RuntimeError on API or parse failure.
    """
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

    # Extract text from response
    text = ""
    for block in body.get("content", []):
        if block.get("type") == "text":
            text += block["text"]

    # Parse JSON (strip markdown fences if present)
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
# Validator
# ---------------------------------------------------------------------------

EXPECTED_FIELDS = {
    "summary", "primary_explanation", "alternative_explanations",
    "evidence_summary", "confidence_note"
}


def validate(response: dict, package: dict) -> list[str]:
    """
    Check faithfulness of LLM response against the explanation package.
    Returns list of violation messages (empty = valid).
    """
    violations = []

    # 1. Schema check
    missing = EXPECTED_FIELDS - set(response.keys())
    if missing:
        violations.append(f"Missing fields: {missing}")

    # 2. Collect allowed failure names from package
    allowed_failures = set()
    primary = package.get("primary_path") or []
    for node in primary:
        allowed_failures.add(node)
    for alt in package.get("alternative_paths", []):
        for node in alt:
            allowed_failures.add(node)
    for conflict in package.get("conflicts", []):
        allowed_failures.add(conflict.get("winner", ""))
        for s in conflict.get("suppressed", []):
            allowed_failures.add(s)

    # Collect allowed signal names
    allowed_signals = set()
    for ev in package.get("evidence", []):
        for sig in ev.get("signals", []):
            allowed_signals.add(sig)

    # 3. Check that primary_explanation mentions primary path nodes
    primary_text = response.get("primary_explanation", "")
    for node in primary:
        # Convert underscore names to either underscore or space form
        if node not in primary_text and node.replace("_", " ") not in primary_text:
            violations.append(
                f"Primary path node '{node}' not mentioned in primary_explanation"
            )

    # 4. Check alternative count matches
    alt_explanations = response.get("alternative_explanations", [])
    alt_paths = package.get("alternative_paths", [])
    if len(alt_explanations) != len(alt_paths):
        violations.append(
            f"alternative_explanations count ({len(alt_explanations)}) "
            f"!= alternative_paths count ({len(alt_paths)})"
        )

    return violations


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def explain(debugger_output: dict, api_key: str | None = None,
            model: str = "claude-sonnet-4-20250514") -> dict:
    """
    Full pipeline: debugger output → explanation package → LLM → validated response.

    Returns:
      {
        "explanation_package": {...},
        "llm_response": {...},
        "validation": {"valid": bool, "violations": [...]},
      }
    """
    package = build_explanation_package(debugger_output)
    system, user = build_prompt(package)
    response = call_llm(system, user, api_key=api_key, model=model)
    violations = validate(response, package)

    return {
        "explanation_package": package,
        "llm_response": response,
        "validation": {
            "valid": len(violations) == 0,
            "violations": violations,
        },
    }
