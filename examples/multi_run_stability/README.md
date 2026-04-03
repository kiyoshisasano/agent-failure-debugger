# Termination Divergence Example

Two systems share the same root cause (`agent_tool_call_loop`) but terminate differently:

- **Order pipeline**: payment retry loop → silent exit (no output, no error)
- **Travel planner**: flight API retry loop → error exit (timeout exception)

`diff_runs()` identifies what separates these two outcomes.

## Files

| File | Content |
|---|---|
| `silent_exit_run.json` | Pipeline result: tool loop → premature_termination (silent) |
| `error_exit_run.json` | Pipeline result: tool loop → failed_termination (error) |
| `expected_output.json` | Full diff_runs() output for verification |
| `run_diff.py` | Runnable script |

## Run

```bash
pip install agent-failure-debugger
python run_diff.py
```

## Output

```
Failure-only patterns (in error exit, not in silent exit):
  failed_termination: frequency=1.0, confidence=0.7

Success-only patterns (in silent exit, not in error exit):
  premature_termination: frequency=1.0, confidence=0.75

Shared patterns: agent_tool_call_loop

Root cause shifted: False
  Silent exit root: agent_tool_call_loop (agreement: 1.0)
  Error exit root:  agent_tool_call_loop (agreement: 1.0)

Causal path divergence:
  Silent exit path: agent_tool_call_loop → premature_termination
  Error exit path:  agent_tool_call_loop → failed_termination

Hypothesis:
  Failures are associated with patterns not seen in successful runs:
  failed_termination. Termination mode differs: successful runs end
  at premature_termination, while failures end at failed_termination.
```

## Interpretation

The root cause is the same (`agent_tool_call_loop`) — the divergence is in the downstream termination mode. This tells you:

- The **fix target** is the same: break the tool retry loop (add max retries, progress validation)
- The **error handling** differs: the order pipeline silently gives up, the travel planner throws an exception
- Both need the same root fix, but the travel planner additionally needs graceful error recovery