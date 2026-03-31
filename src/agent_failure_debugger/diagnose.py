"""
diagnose.py — Single entry point for the debugger.

Accepts a raw log, runs the adapter + matcher + pipeline,
and returns the full diagnosis with explanation.

Usage:
    from agent_failure_debugger import diagnose

    result = diagnose(raw_log, adapter="langchain")
    print(result["summary"]["root_cause"])
    print(result["explanation"]["interpretation"])
"""

import json
import os
import tempfile
from pathlib import Path

from llm_failure_atlas.adapters.langchain_adapter import LangChainAdapter
from llm_failure_atlas.adapters.langsmith_adapter import LangSmithAdapter
from llm_failure_atlas.adapters.crewai_adapter import CrewAIAdapter
from llm_failure_atlas.adapters.redis_help_demo_adapter import RedisHelpDemoAdapter

# Adapter name → class mapping
_ADAPTERS = {
    "langchain": LangChainAdapter,
    "langsmith": LangSmithAdapter,
    "crewai": CrewAIAdapter,
    "redis_help_demo": RedisHelpDemoAdapter,
}


def _load_adapter(name: str):
    """Import and instantiate an adapter by name."""
    if name not in _ADAPTERS:
        available = ", ".join(sorted(_ADAPTERS.keys()))
        raise ValueError(
            f"Unknown adapter '{name}'. Available: {available}"
        )
    cls = _ADAPTERS[name]
    return cls()


def _build_diagnosis_context(raw_log: dict, adapter_name: str,
                             matcher_input: dict,
                             matcher_output: list) -> dict:
    """
    Build a diagnosis context dict that consolidates:
      - raw: original log data
      - observed: adapter-produced telemetry
      - quality: observation quality from matcher
      - metadata: adapter name, timestamp, counts

    This is the internal contract between diagnose() and pipeline.
    Not a public API — structure may change.
    """
    import time

    # Collect observation quality across all patterns
    observed = []
    missing = []
    for entry in matcher_output:
        oq = entry.get("observation_quality", {})
        for sig_name, info in oq.items():
            if info.get("observed"):
                if sig_name not in observed:
                    observed.append(sig_name)
            elif info.get("missing"):
                if sig_name not in missing:
                    missing.append(sig_name)

    total = len(observed) + len(missing)
    if total > 0:
        ratio = len(observed) / total
        if ratio >= 0.8:
            coverage = "high"
        elif ratio >= 0.5:
            coverage = "medium"
        else:
            coverage = "low"
    else:
        coverage = "unknown"

    diagnosed_count = sum(1 for r in matcher_output if r.get("diagnosed"))

    return {
        "raw": raw_log,
        "observed": matcher_input,
        "quality": {
            "observed_signals": sorted(observed),
            "missing_signals": sorted(missing),
            "coverage": coverage,
        },
        "metadata": {
            "adapter": adapter_name,
            "timestamp": time.time(),
            "pattern_count": len(matcher_output),
            "diagnosed_count": diagnosed_count,
        },
    }


def diagnose(raw_log: dict, adapter: str, **pipeline_kwargs) -> dict:
    """
    Run the full pipeline: adapt → match → diagnose → explain.

    Args:
        raw_log: Raw log/response from the agent or service.
        adapter: Adapter name ("langchain", "langsmith", "crewai", "redis_help_demo").
        **pipeline_kwargs: Passed to run_pipeline (e.g. use_learning, top_k).

    Returns:
        Dict with: diagnosis, fix, summary, explanation, telemetry, matcher_output.

    Raises:
        ValueError: If adapter name is unknown.
    """
    # Step 1: Adapt raw log to matcher input
    adapter_instance = _load_adapter(adapter)
    matcher_input = adapter_instance.build_matcher_input(raw_log)

    # Step 2: Run matcher on all patterns
    from llm_failure_atlas.matcher import run as run_matcher
    from llm_failure_atlas.resource_loader import get_patterns_dir
    failures_dir = Path(get_patterns_dir())

    fd, tmp_path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(matcher_input, f)

        matcher_output = []
        for pattern_file in sorted(failures_dir.glob("*.yaml")):
            result = run_matcher(str(pattern_file), tmp_path)
            matcher_output.append(result)
    finally:
        os.unlink(tmp_path)

    # Step 3: Build diagnosis context
    context = _build_diagnosis_context(
        raw_log, adapter, matcher_input, matcher_output
    )

    # Step 4: Run debugger pipeline
    from agent_failure_debugger.pipeline import run_pipeline

    defaults = {
        "use_learning": True,
        "include_explanation": True,
    }
    defaults.update(pipeline_kwargs)

    pipeline_result = run_pipeline(
        matcher_output, diagnosis_context=context, **defaults
    )

    # Add telemetry and diagnosed failures for inspection
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