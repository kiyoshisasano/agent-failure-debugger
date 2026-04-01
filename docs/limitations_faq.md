# Limitations, Constraints, and FAQ

## What This Tool Can Do

- **Detect 17 failure patterns** from agent telemetry using deterministic rules (no ML, no LLM calls)
- **Identify root causes** via a causal graph (17 nodes, 15 edges)
- **Propose fixes** from predefined templates with confidence-based gating
- **Explain failures** with context, risk level, and recommendations
- **Work with multiple frameworks** via adapters (LangChain, LangSmith, CrewAI, Redis demo)
- **Run entirely offline** — no API keys needed for detection and diagnosis

## What This Tool Cannot Do

- **Cannot detect failures it has no pattern for.** There are 17 patterns. If the failure doesn't match any of them, it won't be detected. The `unmodeled_failure` meta-pattern flags when symptoms are present but no domain pattern matches.
- **Cannot verify factual correctness.** It uses keyword/threshold heuristics, not embedding similarity or LLM judgment. It can detect that the response doesn't align with the query (via word overlap), but cannot judge whether the content is factually correct.
- **Cannot detect semantic mismatch.** A tool returning data for a completely different topic is not detectable with current heuristics. This would require embedding-based comparison.
- **Cannot analyze multi-agent coordination.** Atlas covers single-agent runtime failures. For multi-agent systems, see [MAST](https://github.com/multi-agent-systems-failure-taxonomy/MAST).
- **Cannot observe what the adapter doesn't provide.** If a field is missing from telemetry, the corresponding signal defaults to False and the confidence contribution receives 0.6× decay.
- **Cannot guarantee fix correctness.** Fixes are templates, not verified patches. Always review before applying in production.

## Adapter Coverage

Not all adapters produce all telemetry fields:

| Adapter | `state` fields | `grounding` fields | Full detection |
|---|---|---|---|
| `callback_handler` (watch) | ✅ | ✅ | ✅ |
| `langchain` | ✅ | ✅ | ✅ |
| `langsmith` | ✅ | ✅ | ✅ |
| `crewai` | ❌ | ❌ | Partial |
| `redis_help_demo` | ❌ | ❌ | Partial |

When `state` or `grounding` fields are missing, patterns like `agent_tool_call_loop` may not fire even when the behavior is occurring.

## Detection Limitations

**Confidence scores are not probabilities.** They are rule-based evidence accumulations. A score of 0.7 means "multiple signals agree" — it does not mean "70% chance of being correct."

**Observation quality matters.** When the adapter can't extract a field, the signal is marked "unobserved" and its confidence contribution is reduced by 40% (0.6× decay). Low observation coverage is flagged by the `insufficient_observability` meta-pattern.

**Threshold sensitivity.** Detection flips at specific threshold values. There are no "fuzzy" transitions. See `evaluation/sensitivity_eval.py` for the exact transition points of each pattern.

**No temporal analysis.** Telemetry is a summarized snapshot, not an ordered event trace. The system cannot detect patterns that depend on the exact sequence of steps.

## FAQ

### Q: I get "0 failures detected" but my agent clearly failed

Three common causes:

1. **Insufficient telemetry.** The adapter needs enough data to extract signals. A minimal log with just one LLM step won't trigger most patterns. Provide tool calls, retriever results, and input/output pairs.

2. **Wrong adapter.** Each adapter expects a specific input format. Using `langchain` adapter with a LangSmith export (or vice versa) produces empty telemetry. No error is raised. See [Adapter Formats](adapter_formats.md).

3. **Unmodeled failure.** The failure type might not be in the 17 patterns. Check if `unmodeled_failure` is detected — it fires when symptoms are present but no domain pattern matches.

### Q: How do I test if detection works without a real agent?

Use direct telemetry (bypass adapters). See the "Advanced / debugging" section in the [Quick Start Guide](quickstart.md).

### Q: Can I add my own failure patterns?

Yes. Set the `LLM_FAILURE_ATLAS_PATTERNS_DIR` environment variable to a directory containing your custom YAML patterns. Your patterns will be used instead of the bundled defaults.

### Q: Can I use this in production?

Atlas is an experimental debugging tool, not a production monitoring system. It is designed for development-time diagnosis, regression testing, and offline log analysis. The auto-apply gate exists but should not be used in production without human review.

### Q: Does this call any external APIs?

No. Detection, diagnosis, and fix generation are entirely local. The optional `explainer.py` can use an LLM for enhanced explanations, but this is off by default.

### Q: What Python versions are supported?

Python 3.11 and above. This is required for the `importlib.resources` API used for bundled resource loading.