# Termination Divergence Example

Two systems share the same root cause (`agent_tool_call_loop`) but terminate differently:

- **Order pipeline**: payment retry loop → silent exit (no output, no error)
- **Travel planner**: flight API retry loop → error exit (timeout exception)

`diff_runs()` identifies what separates these two outcomes.

## Run

```bash
pip install agent-failure-debugger
python run_diff.py
```

## What to expect

```
failure_only in failures: failed_termination
success_only (not in failures): premature_termination
root cause shifted: False (both are agent_tool_call_loop)
termination mode: premature_termination → failed_termination
```

The root cause is the same — the *divergence* is in the downstream termination mode.
This tells you the fix target is the same (tool loop), but the error handling path differs.
