"""
Self-Healing Agent Demo
=======================

Demonstrates the self-healing health check node with a tool that fails
intermittently, causing agents to loop or produce incorrect output.

The health check detects the failure, injects diagnostic feedback
into the conversation, and the agent retries with awareness of
what went wrong.

Three scenarios:
  1. tool_flaky    — Tool fails on first call, succeeds on retry
  2. tool_loop     — Tool returns same data for different queries
  3. healthy       — Everything works (control case)

Usage:
    # Run with default model (gpt-4o-mini)
    python self_healing_demo.py

    # Run with specific model
    python self_healing_demo.py --model claude

    # Run all models
    python self_healing_demo.py --model all

Requirements:
    pip install agent-failure-debugger[langchain] langgraph langchain-openai
    Optional: langchain-anthropic langchain-google-genai
"""

import argparse
import json
import os
import sys
from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

from agent_failure_debugger.integrations.langgraph import (
    create_health_check,
    RETRYABLE_PATTERNS,
)


# ---------------------------------------------------------------------------
# Tools with controllable behavior
# ---------------------------------------------------------------------------

# Global state for controlling tool behavior
_tool_call_counter = {}


def _reset_tool_state():
    global _tool_call_counter
    _tool_call_counter = {}


@tool
def get_company_revenue(company: str) -> str:
    """Get the annual revenue for a company."""
    _tool_call_counter.setdefault("get_company_revenue", 0)
    _tool_call_counter["get_company_revenue"] += 1

    data = {
        "Apple": "Revenue: $394.3 billion (FY2024)",
        "Microsoft": "Revenue: $245.1 billion (FY2024)",
        "Google": "Revenue: $328.2 billion (FY2024)",
    }
    return data.get(company, f"No data found for {company}")


@tool
def get_company_revenue_flaky(company: str) -> str:
    """Get the annual revenue for a company."""
    _tool_call_counter.setdefault("get_company_revenue_flaky", 0)
    _tool_call_counter["get_company_revenue_flaky"] += 1

    call_num = _tool_call_counter["get_company_revenue_flaky"]

    # Fail on first call, succeed on subsequent calls
    if call_num <= 2:
        return "Error: Service temporarily unavailable. Please try again."

    data = {
        "Apple": "Revenue: $394.3 billion (FY2024)",
        "Microsoft": "Revenue: $245.1 billion (FY2024)",
        "Google": "Revenue: $328.2 billion (FY2024)",
    }
    return data.get(company, f"No data found for {company}")


@tool
def get_company_revenue_loop(company: str) -> str:
    """Get the annual revenue for a company. Supports: Apple, Microsoft, Google."""
    # Always returns the same data regardless of input
    # Simulates a broken API that ignores parameters
    return "Revenue: $394.3 billion (FY2024) — Apple Inc."


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

def get_llm(model_name: str):
    """Get LLM instance by model name."""
    if model_name == "gpt":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    elif model_name == "claude":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.3)
    elif model_name == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_self_healing_graph(llm, tools_list):
    """Build a LangGraph agent with self-healing health check."""

    llm_with_tools = llm.bind_tools(tools_list)

    def agent_node(state: MessagesState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "check"

    # Create health check with diagnosis logging
    diagnoses = []

    def log_diagnosis(result):
        diagnoses.append(result)

    health_check, route = create_health_check(
        max_retries=2,
        retry_on_degraded=False,
        on_diagnosis=log_diagnosis,
        verbose=True,
    )

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools_list))
    workflow.add_node("health_check", health_check)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue,
                                   {"tools": "tools", "check": "health_check"})
    workflow.add_edge("tools", "agent")
    workflow.add_conditional_edges("health_check", route,
                                   {"retry": "agent", "end": END})

    return workflow.compile(), diagnoses


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS = {
    "healthy": {
        "description": "Normal operation — tool works correctly",
        "query": "What is Apple's annual revenue?",
        "tools": [get_company_revenue],
        "expected_status": "healthy",
    },
    "tool_flaky": {
        "description": "Tool fails on first call, succeeds on retry",
        "query": "What is Apple's annual revenue?",
        "tools": [get_company_revenue_flaky],
        "expected_status": "retry then healthy",
    },
    "tool_loop": {
        "description": "Tool returns identical data for different queries (broken API)",
        "query": "Compare the revenue of Apple, Microsoft, and Google.",
        "tools": [get_company_revenue_loop],
        "expected_status": "degraded (tool_result_diversity)",
    },
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_scenario(scenario_name: str, model_name: str):
    """Run a single scenario with a specific model."""
    scenario = SCENARIOS[scenario_name]

    print(f"\n{'#'*60}")
    print(f"  Scenario: {scenario_name}")
    print(f"  Model: {model_name}")
    print(f"  {scenario['description']}")
    print(f"  Expected: {scenario['expected_status']}")
    print(f"{'#'*60}")

    _reset_tool_state()

    try:
        llm = get_llm(model_name)
    except (ImportError, Exception) as e:
        print(f"  ⏭ Skipping {model_name}: {e}")
        return None

    graph, diagnoses = build_self_healing_graph(llm, scenario["tools"])

    try:
        result = graph.invoke({
            "messages": [
                SystemMessage(content=(
                    "You are a financial research assistant. "
                    "Always use the available tools to look up data. "
                    "Never answer from memory — always call a tool first."
                )),
                HumanMessage(content=scenario["query"]),
            ]
        })
    except Exception as e:
        print(f"  ❌ Execution error: {e}")
        return None

    # Print final output
    final_msg = result["messages"][-1]
    content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
    print(f"\n  Agent output: {content[:300]}...")

    # Print health check metadata
    check = result.get("__health_check", {})
    print(f"\n  Health check result: {json.dumps(check, indent=4)}")

    # Summarize diagnoses
    print(f"\n  Total diagnoses run: {len(diagnoses)}")
    for i, diag in enumerate(diagnoses):
        eq = diag.get("summary", {}).get("execution_quality", {})
        failures = [d["failure_id"] for d in diag.get("matcher_output", [])]
        print(f"    [{i+1}] status={eq.get('status', '?')} failures={failures}")

    return {
        "scenario": scenario_name,
        "model": model_name,
        "output": content[:500],
        "health_check": check,
        "diagnosis_count": len(diagnoses),
        "diagnoses": [
            {
                "status": d.get("summary", {}).get("execution_quality", {}).get("status"),
                "failures": [m["failure_id"] for m in d.get("matcher_output", [])],
            }
            for d in diagnoses
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Self-Healing Agent Demo")
    parser.add_argument(
        "--model", default="gpt",
        choices=["gpt", "claude", "gemini", "all"],
        help="Which model to use (default: gpt)",
    )
    parser.add_argument(
        "--scenario", default="all",
        choices=list(SCENARIOS.keys()) + ["all"],
        help="Which scenario to run (default: all)",
    )
    args = parser.parse_args()

    models = ["gpt", "claude", "gemini"] if args.model == "all" else [args.model]
    scenarios = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]

    results = []
    for scenario in scenarios:
        for model in models:
            result = run_scenario(scenario, model)
            if result:
                results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {len(results)} runs completed")
    print(f"{'='*60}")
    for r in results:
        check = r["health_check"]
        status = check.get("execution_status", check.get("status", "?"))
        print(f"  {r['scenario']:15s} | {r['model']:7s} | {status:10s} | "
              f"diagnoses={r['diagnosis_count']}")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()