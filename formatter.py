"""
formatter.py
Formats resolver output into the final debugger JSON output.

build_explanation():
  - Does not assume linear topology (handles branching)
  - Lists each causal link as an independent statement
  - Includes active signals from the source failure when available
"""


def build_explanation(result: dict) -> str:
    if not result["links"]:
        return "No causal relationships detected."

    failures_by_id = {f["id"]: f for f in result["failures"]}
    parts = []

    for link in result["links"]:
        src = link["from"]
        dst = link["to"]
        relation = link["relation"]

        src_signals = failures_by_id.get(src, {}).get("signals", {})
        active_signals = [k for k, v in src_signals.items() if v]

        if active_signals:
            parts.append(f"{src} {relation} {dst} ({', '.join(active_signals)})")
        else:
            parts.append(f"{src} {relation} {dst}")

    return "Causal relationships detected: " + "; ".join(parts)


def format_output(result: dict) -> dict:
    return {
        "root_candidates": result["roots"],
        "failures": result["failures"],
        "causal_links": result["links"],
        "explanation": build_explanation(result),
    }
