"""
diagnose.py — Single entry point for the debugger.

Accepts a raw log, runs the adapter + matcher + pipeline,
and returns the full diagnosis with explanation.

Usage:
    from diagnose import diagnose

    result = diagnose(raw_log, adapter="langchain")
    print(result["summary"]["root_cause"])
    print(result["explanation"]["interpretation"])

Requires llm-failure-atlas as a sibling directory (or ATLAS_ROOT set).
"""

import json
import sys
import tempfile
from pathlib import Path

# Resolve atlas path
_debugger_root = Path(__file__).parent
_atlas_root_env = __import__("os").environ.get("ATLAS_ROOT")
if _atlas_root_env:
    _atlas_root = Path(_atlas_root_env)
else:
    _atlas_root = _debugger_root.parent / "llm-failure-atlas"

if _atlas_root.exists():
    sys.path.insert(0, str(_atlas_root))
    sys.path.insert(0, str(_atlas_root / "adapters"))

sys.path.insert(0, str(_debugger_root))

# Adapter name → (module, class) mapping
_ADAPTERS = {
    "langchain": ("langchain_adapter", "LangChainAdapter"),
    "langsmith": ("langsmith_adapter", "LangSmithAdapter"),
    "crewai": ("crewai_adapter", "AtlasCrewListener"),
    "redis_help_demo": ("redis_help_demo_adapter", "RedisHelpDemoAdapter"),
}


def _load_adapter(name: str):
    """Import and instantiate an adapter by name."""
    if name not in _ADAPTERS:
        available = ", ".join(sorted(_ADAPTERS.keys()))
        raise ValueError(
            f"Unknown adapter '{name}'. Available: {available}"
        )
    module_name, class_name = _ADAPTERS[name]
    module = __import__(module_name)
    cls = getattr(module, class_name)
    return cls()


def diagnose(raw_log: dict, adapter: str, **pipeline_kwargs) -> dict:
    """
    Run the full pipeline: adapt → match → diagnose → explain.

    Args:
        raw_log: Raw log/response from the agent or service.
        adapter: Adapter name ("langchain", "langsmith", "crewai", "redis_help_demo").
        **pipeline_kwargs: Passed to run_pipeline (e.g. use_learning, top_k).

    Returns:
        Dict with: diagnosis, fix, summary, explanation.

    Raises:
        ValueError: If adapter name is unknown.
        FileNotFoundError: If atlas repository is not found.
    """
    if not _atlas_root.exists():
        raise FileNotFoundError(
            f"Atlas not found at {_atlas_root}. "
            "Clone llm-failure-atlas as a sibling directory, "
            "or set ATLAS_ROOT."
        )

    # Step 1: Adapt raw log to matcher input
    adapter_instance = _load_adapter(adapter)
    matcher_input = adapter_instance.build_matcher_input(raw_log)

    # Step 2: Run matcher on all patterns
    from matcher import run as run_matcher
    failures_dir = _atlas_root / "failures"

    # Write matcher input to temp file (matcher expects file path)
    fd, tmp_path = tempfile.mkstemp(suffix=".json")
    import os
    os.close(fd)  # close fd immediately, write via open()
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(matcher_input, f)

        matcher_output = []
        for pattern_file in sorted(failures_dir.glob("*.yaml")):
            result = run_matcher(str(pattern_file), tmp_path)
            matcher_output.append(result)
    finally:
        os.unlink(tmp_path)

    # Step 3: Run debugger pipeline
    from pipeline import run_pipeline

    defaults = {
        "use_learning": True,
        "include_explanation": True,
    }
    defaults.update(pipeline_kwargs)

    pipeline_result = run_pipeline(matcher_output, **defaults)

    # Add raw telemetry for inspection
    pipeline_result["telemetry"] = matcher_input
    pipeline_result["matcher_output"] = [
        {
            "failure_id": r["failure_id"],
            "diagnosed": r["diagnosed"],
            "confidence": r["confidence"],
        }
        for r in matcher_output
        if r.get("diagnosed")
    ]

    return pipeline_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Diagnose agent failures from a raw log file."
    )
    parser.add_argument("log", help="Path to raw log JSON file")
    parser.add_argument(
        "--adapter", required=True,
        choices=sorted(_ADAPTERS.keys()),
        help="Adapter to use for log conversion",
    )
    args = parser.parse_args()

    with open(args.log, encoding="utf-8") as f:
        raw = json.load(f)

    result = diagnose(raw, adapter=args.adapter)

    s = result.get("summary", {})
    expl = result.get("explanation", {})

    print(f"Root cause:     {s.get('root_cause', 'none')}")
    print(f"Confidence:     {s.get('root_confidence', 0)}")
    print(f"Failures:       {s.get('failure_count', 0)}")
    print(f"Gate:           {s.get('gate_mode', '-')}")

    if expl:
        print(f"\nContext:        {expl.get('context_summary', '-')}")
        print(f"Interpretation: {expl.get('interpretation', '-')}")
        risk = expl.get("risk", {})
        print(f"Risk:           {risk.get('level', '-').upper()}")
        print(f"Action:         {expl.get('recommendation', '-')}")