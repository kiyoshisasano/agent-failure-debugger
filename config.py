"""
config.py

Centralized configuration for agent-failure-debugger.

All paths and settings in one place. Reads from environment
variables with sensible defaults for local development.

Environment variables:
  ATLAS_ROOT          Path to llm-failure-atlas repository
  DEBUGGER_ROOT       Path to agent-failure-debugger repository (this repo)
  ATLAS_LEARNING_DIR  Override learning store location
  ATLAS_VALIDATION_DIR Override validation data location
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository roots
# ---------------------------------------------------------------------------

DEBUGGER_ROOT = Path(os.environ.get(
    "DEBUGGER_ROOT",
    Path(__file__).parent
))

ATLAS_ROOT = Path(os.environ.get(
    "ATLAS_ROOT",
    DEBUGGER_ROOT.parent / "llm-failure-atlas"
))

# ---------------------------------------------------------------------------
# Key paths
# ---------------------------------------------------------------------------

# Debugger
GRAPH_PATH = DEBUGGER_ROOT / "failure_graph.yaml"
TEMPLATES_DIR = DEBUGGER_ROOT / "templates"

# Atlas — learning stores
LEARNING_DIR = Path(os.environ.get(
    "ATLAS_LEARNING_DIR",
    ATLAS_ROOT / "learning"
))

# Atlas — validation
VALIDATION_DIR = Path(os.environ.get(
    "ATLAS_VALIDATION_DIR",
    ATLAS_ROOT / "validation"
))

# Atlas — examples
EXAMPLES_DIR = ATLAS_ROOT / "examples"

# Atlas — evaluation
EVALUATION_DIR = ATLAS_ROOT / "evaluation"

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
