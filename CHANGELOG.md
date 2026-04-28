# Changelog

All notable changes to `agent-failure-debugger` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-04-28

### Added
- `summary.execution_quality.utilisation` — chunk-utilisation summary computed from Atlas adapter output (`retrieval.retrieved_ids` and `retrieval.used_chunk_ids`). Aggregates the per-chunk text-overlap proxy into a run-level signal: `{ratio, used_count, retrieved_count, method}`.
- `_compute_utilisation()` helper in `execution_quality.py` — returns `None` when retrieval did not occur or when the adapter does not yet emit utilisation data, distinguishing absence-of-retrieval from zero-utilisation cases.

### Notes
- The utilisation field is paired with the proxy method name (currently `text_overlap_proxy`), so consumers know it is an approximation rather than a direct observation. See `examples/rag_chunk_diagnosis/` in the Atlas repository for a walkthrough.
- Existing `execution_quality` fields (`status`, `termination`, `indicators`, `summary`) are unchanged. The `utilisation` field is added to all four return paths.

## [0.3.1] - 2026-04-04

### Fixed
- Self-healing demo: `tool_flaky` scenario threshold tuning so the first call fails and the retry succeeds as expected.
- Claude model name in self-healing demo aligned with current API naming.

## [0.3.0] - 2026-04-04

### Added
- Self-healing health-check node (`create_health_check`) for LangGraph integration. Adds automatic failure detection and informed retry to any LangGraph agent. On retry, the diagnosis is injected into the conversation as a `HumanMessage` so the LLM sees why it failed.
- `task_incomplete` detection: classifies cases where the agent produced output but admitted failure (tools failed and the agent gave up) as `failed` rather than `degraded`.

## [0.2.1] - 2026-04-03

### Added
- Health-check positioning improvements: clearer separation between CI-side detection (`pytest-agent-health`) and production-side healing (`create_health_check`).

### Fixed
- Gemini adapter consumption issues.
- `grounding.tool_result_diversity` consumption — used by execution-quality assessment to detect cases where multiple tool calls returned identical (and likely useless) data.

## [0.2.0] - 2026-04-03

### Added
- Execution behavior diagnosis: three-state classification (`healthy` / `degraded` / `failed`) replacing the prior failure-only model.
- Multi-run analysis: `compare_runs()` for stability assessment, `diff_runs()` for differential diagnosis between success and failure groups.
- Termination mode classification: `normal` / `silent_exit` / `error_exit` / `partial_exit` / `unknown`.

### Fixed
- Callback handler trace assembly issues.

## [0.1.0] - 2026-03

### Added
- Initial PyPI release.
- Causal analysis pipeline: matcher output → root cause → causal path → explanation → fix proposal.
- Auto-apply gate with confidence-based proposal_only / apply modes.
- Reliability assessment alongside diagnosis output.

[0.4.0]: https://github.com/kiyoshisasano/agent-failure-debugger/releases/tag/v0.4.0