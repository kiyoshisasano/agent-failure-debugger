# Multi-Run Stability Example

An LLM agent runs the same task 5 times. Three runs succeed, two fail.
This example demonstrates the two-step workflow:

1. `compare_runs()` — detect instability across all 5 runs
2. `diff_runs()` — identify what separates success from failure

## Files

| File | Content |
|---|---|
| `run_1.json` | Successful run (healthy, no failures) |
| `run_2.json` | Failed run: tool loop → silent exit |
| `run_3.json` | Successful run (healthy, no failures) |
| `run_4.json` | Failed run: premature commitment → incorrect output |
| `run_5.json` | Successful run (healthy, no failures) |
| `expected_output.json` | Full compare_runs() + diff_runs() output for verification |
| `run_stability.py` | Runnable script |

## Run

```bash
pip install agent-failure-debugger
python run_stability.py
```

## Scenario

A flight booking agent processes "Change my flight to tomorrow morning":

- **Run 1, 3, 5**: Agent handles the request correctly (no failures detected)
- **Run 2**: Agent enters a tool retry loop and exits silently
- **Run 4**: Agent commits to wrong interpretation, produces incorrect output

## Output

### Step 1: compare_runs — Stability Analysis

```
Runs analyzed: 5
Root cause agreement: 0.6
Root cause distribution: {'unknown': 3, 'agent_tool_call_loop': 1, 'premature_model_commitment': 1}
Failure set Jaccard: 0.3
Intermittent failures: agent_tool_call_loop, incorrect_output,
                       premature_model_commitment, premature_termination

Interpretation: Root cause is unstable: 'unknown' was the most common
at 60%, but other causes appeared. The agent's behavior is highly
non-deterministic for this input.
```

Root cause agreement of 0.6 means the agent is unstable — different runs produce different failure modes. This triggers Step 2.

### Step 2: diff_runs — Divergence Analysis

Runs are separated by `execution_quality.status`: healthy (1, 3, 5) vs non-healthy (2, 4).

```
Patterns only in failures:
  agent_tool_call_loop:       frequency=0.5
  incorrect_output:           frequency=0.5
  premature_model_commitment: frequency=0.5
  premature_termination:      frequency=0.5

Root cause shifted: True
  Success: unknown
  Failure: agent_tool_call_loop

Hypothesis: Failures are associated with patterns not seen in successful
runs. The root cause shifted from 'unknown' in successful runs to
'agent_tool_call_loop' in failed runs.
```

## Interpretation

The two failed runs have different failure modes (tool loop vs wrong interpretation), but neither pattern appears in successful runs. This confirms the agent is non-deterministic — the same input sometimes works and sometimes triggers one of two distinct failure paths. The 0.5 frequency on each pattern means each failure mode appeared in 1 of the 2 failed runs.

Actions:
- **Tool loop (Run 2)**: add max retry limit with progress validation
- **Wrong interpretation (Run 4)**: add ambiguity detection with clarification step
- Both fixes are independent — they address different failure paths from the same input