# Multi-Run Stability Example

An LLM agent runs the same task 5 times. Three runs succeed, two fail.
This example demonstrates the two-step workflow:

1. `compare_runs()` — detect instability across all 5 runs
2. `diff_runs()` — identify what separates success from failure

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

## What to expect

```
Step 1: compare_runs → root_cause_agreement < 1.0 (unstable)
Step 2: diff_runs   → failure_only patterns in failed runs
                    → hypothesis explains the divergence
```
