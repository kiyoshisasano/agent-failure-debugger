"""
config.py

Centralized configuration for agent-failure-debugger.

All paths and settings in one place. Uses llm-failure-atlas package
for graph and learning resources.

Environment variables:
  LLM_FAILURE_ATLAS_GRAPH_PATH    Override graph location
  LLM_FAILURE_ATLAS_LEARNING_DIR  Override learning store location
"""

import os
from pathlib import Path

from llm_failure_atlas.resource_loader import get_graph_path, get_learning_dir

# ---------------------------------------------------------------------------
# Repository roots
# ---------------------------------------------------------------------------

DEBUGGER_ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Key paths
# ---------------------------------------------------------------------------

# Graph: loaded from atlas package (single source of truth)
GRAPH_PATH = get_graph_path()
TEMPLATES_DIR = DEBUGGER_ROOT / "templates"

# Learning stores — from atlas package resources
LEARNING_DIR = get_learning_dir()

# Runtime output
PATCHES_DIR = DEBUGGER_ROOT / "patches"

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

# decision_support.py — priority scoring (no learning)
PRIORITY_WEIGHTS = {
    "root_norm": 0.4,
    "upstream_bonus": 0.3,
    "confidence": 0.2,
    "not_suppressed": 0.1,
}

# decision_support.py — priority scoring (with learning)
PRIORITY_WEIGHTS_LEARNING = {
    "root_norm": 0.35,
    "upstream_bonus": 0.25,
    "confidence": 0.15,
    "not_suppressed": 0.1,
    "effectiveness": 0.15,
}

# autofix.py — final_fix_score
AUTOFIX_WEIGHTS = {
    "priority": 0.6,
    "effectiveness": 0.4,
}

# auto_apply.py — gate scoring
GATE_WEIGHTS = {
    "priority": 0.35,
    "effectiveness": 0.30,
    "root_confidence": 0.20,
    "regression_safety": 0.15,
}

# auto_apply.py — gate thresholds
GATE_THRESHOLDS = {
    "auto_apply": 0.85,
    "staged_review": 0.65,
}

# safety promotion
SAFETY_PROMOTION_THRESHOLD = 0.9  # effectiveness >= this + 0 rollbacks

# ---------------------------------------------------------------------------
# KPI targets
# ---------------------------------------------------------------------------

KPI_TARGETS = {
    "threshold_boundary_rate": {"target": 0.05, "direction": "lower"},
    "fix_dominance": {"target": 0.60, "direction": "lower"},
    "failure_monotonicity": {"target": 0.90, "direction": "higher"},
    "rollback_rate": {"target": 0.10, "direction": "lower"},
    "no_regression_rate": {"target": 0.95, "direction": "higher"},
    "causal_consistency_rate": {"target": 0.90, "direction": "higher"},
}

KPI_WINDOW_SIZE = 30