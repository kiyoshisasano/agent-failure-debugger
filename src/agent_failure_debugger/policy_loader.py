"""
policy_loader.py

Phase 20: Read-only access to learning stores.

Loads:
  - fix_effectiveness.json   → effectiveness scores per (failure, fix_type)
  - threshold_policy.json    → proposed threshold adjustments

Source: llm-failure-atlas/learning/
Location: agent-failure-debugger/ (root)

Design rules:
  - Read-only: never writes to learning stores
  - Graceful degradation: missing files → empty defaults
  - No modification of matcher/graph/templates (principle #7)
"""

import json
import os

from llm_failure_atlas.resource_loader import get_learning_dir

# Learning stores from atlas package resources
LEARNING_DIR = str(get_learning_dir())


def _load_json(filename: str) -> dict:
    """Load a JSON file from learning dir, returning {} on any failure."""
    path = os.path.join(LEARNING_DIR, filename)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load_policies() -> dict:
    """Load all policy data from learning stores."""
    return {
        "fix_effectiveness": _load_json("fix_effectiveness.json"),
        "threshold_policy": _load_json("threshold_policy.json"),
    }


def get_fix_effectiveness(failure: str, fix_type: str) -> float:
    """
    Return effectiveness score for a (failure, fix_type) pair.
    Returns 0.0 if no data exists.
    """
    data = _load_json("fix_effectiveness.json")
    entry = data.get(failure, {}).get(fix_type, {})
    return entry.get("effectiveness_score", 0.0)


def scale_effectiveness(raw: float) -> float:
    """
    Normalize raw effectiveness_score to spread the useful range.

    Raw scores cluster in 0.6–1.0 due to the formula in update_policy.py.
    This rescales so that:
      0.5 → 0.0 (mediocre = no boost)
      1.0 → 1.0 (perfect = full boost)
      <0.5 → 0.0 (poor = clamped)
    """
    return max(0.0, min(1.0, (raw - 0.5) * 2))


def get_best_effectiveness(failure: str) -> float:
    """
    Return the highest effectiveness score across all fix types for a failure.
    Returns 0.0 if no data exists.
    """
    data = _load_json("fix_effectiveness.json")
    entries = data.get(failure, {})
    if not entries:
        return 0.0
    return max(e.get("effectiveness_score", 0.0) for e in entries.values())


def get_fix_record(failure: str, fix_type: str) -> dict | None:
    """
    Return the full effectiveness record for a (failure, fix_type) pair.
    Returns None if no data exists.
    """
    data = _load_json("fix_effectiveness.json")
    return data.get(failure, {}).get(fix_type)


def get_threshold_proposals() -> list[dict]:
    """
    Return threshold adjustment proposals from calibration learning.
    Each entry: {failure, field, current, proposed, reason}
    """
    data = _load_json("threshold_policy.json")
    return data.get("proposals", [])